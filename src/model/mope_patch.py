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

        _first_layer_done = [False]

        def _layer_hook(module, args, layer_kwargs):
            hidden_states = args[0]  # [B, T or T+N_mope, dim]

            new_hs = hidden_states
            # Layer 0 only: prepend MoPE token embeddings.
            if not _first_layer_done[0]:
                _first_layer_done[0] = True
                new_hs = torch.cat(
                    [mope_embeds.to(hidden_states.dtype), hidden_states], dim=1
                )

            T_hs = new_hs.shape[1]

            # Extend 4-D causal attention mask if present and not yet extended.
            attn_mask = layer_kwargs.get('attention_mask', None)
            if attn_mask is not None:
                if attn_mask.dim() == 4:
                    T_mask = attn_mask.shape[-1]
                    if T_mask < T_hs:
                        n_ext = T_hs - T_mask
                        B_m = attn_mask.shape[0]
                        dt, dv = attn_mask.dtype, attn_mask.device
                        min_val = torch.finfo(dt).min
                        # MoPE-to-MoPE: all attend (0).
                        # MoPE-to-original: causal mask (MoPE comes first, -inf for future).
                        # Original-to-MoPE: all attend (0).
                        top = torch.cat([
                            torch.zeros(B_m, 1, n_ext, n_ext, dtype=dt, device=dv),
                            torch.full((B_m, 1, n_ext, T_mask), min_val, dtype=dt, device=dv),
                        ], dim=-1)
                        bot = torch.cat([
                            torch.zeros(B_m, 1, T_mask, n_ext, dtype=dt, device=dv),
                            attn_mask,
                        ], dim=-1)
                        layer_kwargs['attention_mask'] = torch.cat([top, bot], dim=-2)
                elif attn_mask.dim() == 2:
                    # Flash-Attention style 2-D mask [B, T]: prepend ones for MoPE.
                    T_mask = attn_mask.shape[1]
                    if T_mask < T_hs:
                        n_ext = T_hs - T_mask
                        ones = attn_mask.new_ones(attn_mask.shape[0], n_ext)
                        layer_kwargs['attention_mask'] = torch.cat([ones, attn_mask], dim=1)

            # Extend pre-computed rotary position embeddings (cos, sin).
            pos_emb = layer_kwargs.get('position_embeddings', None)
            if pos_emb is not None and isinstance(pos_emb, (tuple, list)) and len(pos_emb) == 2:
                cos, sin = pos_emb
                if cos.dim() >= 2:
                    T_pos = cos.shape[-2]
                    if T_pos < T_hs:
                        n_ext = T_hs - T_pos
                        # Use position=0 values (cos(0)=1, sin(0)=0) so RoPE
                        # degrades to identity for MoPE tokens instead of zeroing
                        # out Q/K vectors.  cos/sin shape: (..., seq_len, head_dim).
                        pos0_cos = cos[..., :1, :].expand(
                            *cos.shape[:-2], n_ext, cos.shape[-1]
                        )
                        pos0_sin = sin[..., :1, :].expand(
                            *sin.shape[:-2], n_ext, sin.shape[-1]
                        )
                        layer_kwargs['position_embeddings'] = (
                            torch.cat([pos0_cos, cos], dim=-2),
                            torch.cat([pos0_sin, sin], dim=-2),
                        )

            # Extend position_ids if passed separately (Qwen3 3-D RoPE).
            pos_ids = layer_kwargs.get('position_ids', None)
            if pos_ids is not None:
                T_pos = pos_ids.shape[-1]
                if T_pos < T_hs:
                    n_ext = T_hs - T_pos
                    shape = list(pos_ids.shape)
                    shape[-1] = n_ext
                    z_ids = torch.zeros(shape, dtype=pos_ids.dtype, device=pos_ids.device)
                    layer_kwargs['position_ids'] = torch.cat([z_ids, pos_ids], dim=-1)

            return (new_hs,) + args[1:], layer_kwargs

        layers = getattr(self, 'layers', None)
        if layers is None:
            return original_forward(*args, **kwargs)

        handles = [
            layer.register_forward_pre_hook(_layer_hook, with_kwargs=True, prepend=True)
            for layer in layers
        ]

        try:
            output = original_forward(*args, **kwargs)
        finally:
            for h in handles:
                h.remove()

        return output

    inner_model.forward = types.MethodType(patched_forward, inner_model)
    print("[Space Sensing] Patched inner_model.forward() with MoPE concat injection (E-02b).")
