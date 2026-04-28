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
                mope_embeds = _mope_projector(mope_feats)   # [B, 1, llm_dim]
                mope_bias = mope_embeds.squeeze(1)          # [B, llm_dim]
                B_mope = mope_bias.shape[0]
                n_total = len(image_embeds_list)
                assert n_total % B_mope == 0, (
                    f"MoPE: image_embeds_list length ({n_total}) not divisible by "
                    f"batch size ({B_mope}). Unequal images-per-sample not supported."
                )
                imgs_per_sample = n_total // B_mope
                new_embeds = []
                for b in range(B_mope):
                    bias = mope_bias[b:b+1]  # [1, llm_dim]
                    for e in image_embeds_list[b * imgs_per_sample:(b + 1) * imgs_per_sample]:
                        new_embeds.append(e + bias.to(e.dtype))
                return new_embeds, deepstack

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


def _patch_model_for_mope_concat(model) -> None:
    """E-02b: prepend MoPE patch tokens to the LLM input sequence via layer hooks.

    Registers a forward_pre_hook on EVERY decoder layer.  At layer 0 the hook
    prepends mope_embeds to hidden_states.  At all layers the hook extends
    attention_mask (4-D or 2-D) and position_embeddings / position_ids to match
    the extended sequence length, so RoPE and attention patterns are correct.

    Injection condition: pixel_values must be present in the inner-model forward
    call (prefill only).  Decode steps that arrive without pixel_values are
    passed through unchanged; the MoPE tokens are already stored in the KV cache
    from the prefill step.

    Training correctness: the training data collator must prepend N_mope -100
    entries to each labels row so that loss computation over the extended logit
    sequence remains correct (see MoPECollatorWrapper with mope_num_tokens > 0).
    """
    inner_model = model.model
    original_forward = inner_model.forward

    def patched_forward(self, *args, mope_frames=None, **kwargs):
        _mope_encoder = getattr(self, '_mope_encoder', None)
        _mope_projector = getattr(self, '_mope_projector', None)

        if mope_frames is None:
            mope_frames = getattr(self, '_pending_mope_frames', None)

        # Only inject during prefill (pixel_values present).
        pixel_values = kwargs.get('pixel_values', None)
        should_inject = (
            _mope_encoder is not None
            and _mope_projector is not None
            and mope_frames is not None
            and pixel_values is not None
        )

        if not should_inject:
            return original_forward(*args, **kwargs)

        # Compute MoPE token embeddings.
        mope_feats = _mope_encoder(mope_frames)      # [B, N_mope, 768]
        mope_embeds = _mope_projector(mope_feats)    # [B, N_mope, llm_dim]
        N_mope = mope_embeds.shape[1]

        def _lm_pre_hook(module, args, kwargs):
            # Prepend MoPE tokens to inputs_embeds: [B, L, D] → [B, N+L, D]
            inputs_embeds = kwargs.get('inputs_embeds', None)
            if inputs_embeds is not None:
                kwargs['inputs_embeds'] = torch.cat(
                    [mope_embeds.to(inputs_embeds.dtype), inputs_embeds], dim=1
                )

            # Extend position_ids (last dim): [..., L] → [..., N+L] (prepend zeros).
            # RoPE at position 0 is identity (cos=1, sin=0), preserving Q/K values.
            pos_ids = kwargs.get('position_ids', None)
            if pos_ids is not None:
                z = torch.zeros(
                    *pos_ids.shape[:-1], N_mope,
                    dtype=pos_ids.dtype, device=pos_ids.device,
                )
                kwargs['position_ids'] = torch.cat([z, pos_ids], dim=-1)

            # Extend 2-D attention_mask [B, L] → [B, N+L]: prepend ones for MoPE.
            attn_mask = kwargs.get('attention_mask', None)
            if attn_mask is not None and attn_mask.dim() == 2:
                ones = attn_mask.new_ones(attn_mask.shape[0], N_mope)
                kwargs['attention_mask'] = torch.cat([ones, attn_mask], dim=1)

            # Extend visual_pos_masks [B, L] → [B, N+L]: prepend False for MoPE.
            vis_mask = kwargs.get('visual_pos_masks', None)
            if vis_mask is not None:
                false_prefix = vis_mask.new_zeros(
                    vis_mask.shape[0], N_mope, dtype=torch.bool
                )
                kwargs['visual_pos_masks'] = torch.cat([false_prefix, vis_mask], dim=1)

            return args, kwargs

        lm = getattr(self, 'language_model', None)
        if lm is None:
            return original_forward(*args, **kwargs)

        handle = lm.register_forward_pre_hook(_lm_pre_hook, with_kwargs=True, prepend=True)
        try:
            output = original_forward(*args, **kwargs)
        finally:
            handle.remove()

        return output

    inner_model.forward = types.MethodType(patched_forward, inner_model)
    print("[Space Sensing] Patched inner_model.forward() with MoPE concat injection (E-02b).")
