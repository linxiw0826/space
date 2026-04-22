"""
MoPE forward-patch helper.

Extracted from train_space.py so that both the training entry-point and the
lmms-eval inference plugin (src/eval/models/qwen3_vl_mope.py) can share the
same patch logic without importing train_space.py (which has module-level
imports that crash outside the training environment).

Sidecar mechanism
-----------------
Training path:
    The outer model is called with ``mope_frames=<tensor>`` kwarg; the patched
    forward picks it up directly.

Inference path (lmms-eval):
    lmms-eval's generate pipeline does not forward extra kwargs into the model.
    Before calling ``model.generate()``, the eval plugin sets
    ``inner_model._pending_mope_frames = <tensor>``.  The patched forward reads
    it from the sidecar attribute at the top of each call, then the finally
    block in the plugin resets it to None.
"""

import types
import torch  # noqa: F401 — required by callers that may not import torch independently


def _patch_model_for_mope(model) -> None:
    """Monkey-patch inner_model.forward() to inject MoPE embeddings at inference and training.

    Because refs/ is read-only, we cannot edit modeling_qwen3_vl.py directly.
    Instead we wrap the forward method of ``model.model`` (the inner VL model)
    to inject MoPE at the ``get_image_features`` call site — before visual
    embeddings are fused with geometry embeddings.

    Fusion equation (additive, applied per-token via broadcast):
        image_embeds = image_embeds + mope_bias   # mope_bias: [1, llm_dim]
    where mope_bias is derived from MoPEProjector output squeezed to [1, llm_dim].

    This is mathematically equivalent to injecting after geometry fusion because
    addition is commutative:
        (raw_visual + mope) + geo  ==  raw_visual + mope + geo

    The MoPE frames tensor is expected to be passed as a keyword argument
    ``mope_frames`` with shape [B, C, T, H, W] through the outer model's
    forward call.  When ``mope_frames`` is None or absent, the patch is a
    no-op (pure GUIDE behavior is preserved).

    Sidecar support: if ``mope_frames`` kwarg is None, the patch will also
    check ``self._pending_mope_frames`` on the inner model instance.  This
    allows lmms-eval to inject frames without modifying the generate pipeline.

    IMPORTANT: This patch intercepts at the *inner model* (model.model) level,
    not at the outer model.  The outer Qwen3VLForConditionalGeneration.forward()
    passes **kwargs through to model.model.forward(), so adding mope_frames as
    a kwarg to the outer call propagates automatically.

    Insertion point in the original source:
        File:   qwenvl/model/modeling_qwen3_vl.py
        Method: Qwen3VLModel.forward()
        Wraps:  self.get_image_features(pixel_values, grid_thw)
        Effect: MoPE bias is added to each shard in image_embeds_list before
                the result is used for geometry fusion and scatter into
                inputs_embeds.
    """
    inner_model = model.model
    original_forward = inner_model.forward

    def patched_forward(self, *args, mope_frames=None, **kwargs):
        # ---------------------------------------------------------------
        # Sidecar support: inference callers set _pending_mope_frames
        # instead of passing mope_frames kwarg through generate().
        # ---------------------------------------------------------------
        _mope_encoder = getattr(self, '_mope_encoder', None)
        _mope_projector = getattr(self, '_mope_projector', None)

        if mope_frames is None:
            mope_frames = getattr(self, '_pending_mope_frames', None)

        # ---------------------------------------------------------------
        # Unified injection path: wrap get_image_features to add MoPE
        # bias to each visual-feature shard before geometry fusion.
        # Works regardless of whether the geometry encoder is active.
        # ---------------------------------------------------------------
        if _mope_encoder is not None and mope_frames is not None:
            original_get_image_features = self.get_image_features

            def _mope_get_image_features(pixel_values, grid_thw):
                image_embeds_list, deepstack = original_get_image_features(pixel_values, grid_thw)
                mope_feats = _mope_encoder(mope_frames)
                mope_embeds = _mope_projector(mope_feats)        # [B, 1, llm_dim]
                B_mope = mope_embeds.shape[0]
                assert B_mope == 1, (
                    f"MoPE additive fusion requires per_device_train_batch_size=1; got {B_mope}. "
                    f"Increase gradient_accumulation_steps instead."
                )
                mope_bias = mope_embeds.squeeze(1)  # [1, llm_dim]
                image_embeds_list = [e + mope_bias.to(e.dtype) for e in image_embeds_list]
                return image_embeds_list, deepstack

            self.get_image_features = _mope_get_image_features

        try:
            output = original_forward(*args, **kwargs)
        finally:
            # Restore original so state is clean for the next call.
            if _mope_encoder is not None and mope_frames is not None:
                self.get_image_features = original_get_image_features

        return output

    inner_model.forward = types.MethodType(patched_forward, inner_model)
    print("[Space Sensing] Patched inner_model.forward() with MoPE injection.")
