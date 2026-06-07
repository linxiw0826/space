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


def _compute_pooled_text(inner_model, args, kwargs):
    """Pool the question-text token embeddings for the E-10 content-driven gate.

    Extracts ``input_ids`` from the inner-model forward call (kwarg or first
    positional arg), embeds it via ``get_input_embeddings()``, masks out the
    visual placeholder tokens (image_token_id / video_token_id), and mean-pools
    the remaining (text) positions per sample.

    Args:
        inner_model: the Qwen3VLModel instance (has .config and
                     .get_input_embeddings()).
        args:        positional args passed to the patched inner forward.
        kwargs:      keyword args passed to the patched inner forward.

    Returns:
        pooled_text: Float tensor [B, llm_dim], or None if input_ids is
                     unavailable (safe fallback → gate degrades to g=1).
    """
    # ------------------------------------------------------------------
    # Locate input_ids: normally kwarg, fallback to first positional arg.
    # (Qwen3VLModel.forward signature: forward(input_ids, attention_mask, ...))
    # ------------------------------------------------------------------
    input_ids = kwargs.get('input_ids', None)
    if input_ids is None and len(args) > 0:
        a0 = args[0]
        if hasattr(a0, 'dim') and a0.dtype in (torch.long, torch.int, torch.int64, torch.int32):
            input_ids = a0
    if input_ids is None:
        return None  # safe fallback — no text condition available → g=1

    try:
        embed_fn = inner_model.get_input_embeddings()
        text_embeds = embed_fn(input_ids)                       # [B, L, llm_dim]

        cfg = getattr(inner_model, 'config', None)
        image_token_id = getattr(cfg, 'image_token_id', None)
        video_token_id = getattr(cfg, 'video_token_id', None)

        # Build a boolean mask of NON-visual (i.e. text) positions.
        text_mask = torch.ones_like(input_ids, dtype=torch.bool)   # [B, L]
        if image_token_id is not None:
            text_mask &= (input_ids != image_token_id)
        if video_token_id is not None:
            text_mask &= (input_ids != video_token_id)

        mask = text_mask.unsqueeze(-1).to(text_embeds.dtype)        # [B, L, 1]
        denom = mask.sum(dim=1).clamp_min(1.0)                      # [B, 1]
        pooled = (text_embeds * mask).sum(dim=1) / denom           # [B, llm_dim]

        # Safety: if a sample had zero text tokens (all visual), its pooled row
        # is meaningless; fall back to g=1 for the whole batch only when EVERY
        # sample is degenerate. Otherwise per-sample pooled is fine (denom>=1).
        if not text_mask.any():
            return None
        return pooled
    except Exception as exc:  # robust fallback — never break the forward pass
        print(f"[MoPE Router] _compute_pooled_text failed ({type(exc).__name__}: {exc}); gate -> g=1.")
        return None


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
    """E-02b: prepend MoPE patch tokens to the LLM input sequence (Qwen3VLTextModel level).

    Wraps ``inner_model.forward()`` (Qwen3VLModel) so that on each prefill call a
    ``forward_pre_hook`` is temporarily registered on ``inner_model.language_model``
    (Qwen3VLTextModel).  The hook runs just before Qwen3VLTextModel.forward() and
    prepends N_mope learned tokens to ``inputs_embeds``, extending the sequence
    from [B, L, D] to [B, N_mope+L, D].  Matching extensions are applied to
    ``position_ids``, ``attention_mask`` (2-D only), and ``visual_pos_masks``
    so that RoPE and causal attention over the extended sequence are correct.

    Injection condition: ``pixel_values`` must be present in the outer kwargs
    (prefill only).  Decode steps without ``pixel_values`` are passed through
    unchanged; MoPE tokens are already in the KV cache from the prefill step.

    ``self.language_model(...)`` in Qwen3VLModel.forward() (line ~1888) passes
    all inputs as keyword arguments, so the hook reliably finds ``inputs_embeds``
    in ``kwargs``.  A positional-arg fallback is also provided (``args[0]``) for
    defensive robustness.

    Training correctness: the training data collator must prepend N_mope -100
    entries to each labels row so that loss computation over the extended logit
    sequence remains correct (see MoPECollatorWrapper with mope_num_tokens > 0).
    """
    inner_model = model.model
    original_forward = inner_model.forward

    # Mutable counter for diagnostic throttle (shared across hook closures).
    _diag_state = {"calls": 0}

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

        # ---------------------------------------------------------------
        # [MoPE E02b DEBUG] Diagnostic print — fires on the first 3 calls
        # to confirm whether the injection condition is met and mope_frames
        # is reaching patched_forward.  Auto-silences after 3 prints.
        # ---------------------------------------------------------------
        if _diag_state["calls"] < 3:
            _diag_state["calls"] += 1
            pv_shape = pixel_values.shape if pixel_values is not None else None
            mf_shape = mope_frames.shape if mope_frames is not None else None
            enc_ok = _mope_encoder is not None
            proj_ok = _mope_projector is not None
            print(
                f"[MoPE E02b DEBUG] call#{_diag_state['calls']}: "
                f"should_inject={should_inject}, "
                f"encoder_ok={enc_ok}, projector_ok={proj_ok}, "
                f"mope_frames={mf_shape}, pixel_values={pv_shape}"
            )

        if not should_inject:
            return original_forward(*args, **kwargs)

        # Compute MoPE token embeddings.
        mope_feats = _mope_encoder(mope_frames)      # [B, N_mope, 768]
        mope_embeds = _mope_projector(mope_feats)    # [B, N_mope, llm_dim]

        # Guard: NaN/Inf in mope_embeds (e.g. after bf16 optimizer corrupts projector weights).
        if not mope_embeds.isfinite().all():
            print(
                f"[MoPE E02b WARNING] mope_embeds non-finite at call#{_diag_state['calls']}, "
                f"count={( ~mope_embeds.isfinite()).sum().item()} — zeroing to skip this batch",
                flush=True,
            )
            mope_embeds = torch.zeros_like(mope_embeds)

        # Clamp gradient flowing back to projector to prevent bf16 overflow.
        # 784 prepended tokens accumulate gradients from all subsequent positions;
        # without clamping the per-element gradient can exceed bf16 max (65504).
        if mope_embeds.requires_grad:
            mope_embeds.register_hook(
                lambda g: torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-1.0, 1.0)
                if g is not None else g
            )

        N_mope = mope_embeds.shape[1]

        def _lm_pre_hook(module, args, kwargs):
            # ------------------------------------------------------------------
            # Locate inputs_embeds: normally in kwargs (Qwen3VLModel.forward()
            # calls self.language_model(inputs_embeds=..., ...) at ~line 1888
            # using keyword args).  Positional-arg fallback handles any edge case
            # where the 3-D tensor ends up in args[0].
            # ------------------------------------------------------------------
            inputs_embeds = kwargs.get('inputs_embeds', None)
            _from_args = False
            if inputs_embeds is None and len(args) > 0:
                _a0 = args[0]
                if hasattr(_a0, 'dim') and _a0.dim() == 3:
                    inputs_embeds = _a0
                    _from_args = True

            if _diag_state["calls"] <= 3:
                ie_shape = inputs_embeds.shape if inputs_embeds is not None else None
                print(
                    f"[MoPE E02b DEBUG]   _lm_pre_hook fired: "
                    f"inputs_embeds={ie_shape} (from_args={_from_args}), "
                    f"N_mope={N_mope}, "
                    f"kwargs_keys={list(kwargs.keys())[:8]}"
                )

            if inputs_embeds is not None:
                new_embeds = torch.cat(
                    [mope_embeds.to(inputs_embeds.dtype), inputs_embeds], dim=1
                )
                if _from_args:
                    args = (new_embeds,) + args[1:]
                else:
                    kwargs['inputs_embeds'] = new_embeds

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
            if attn_mask is not None and hasattr(attn_mask, 'dim') and attn_mask.dim() == 2:
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


def _patch_model_for_mope_crossattn(model) -> None:
    """E-02c: per-shard cross-attention fusion at the get_image_features call site.

    Injection point is identical to E-02a (_patch_model_for_mope): wraps
    ``inner_model.forward()`` and temporarily replaces ``self.get_image_features``
    inside that call so each image-feature shard is updated before geometry fusion.

    The only difference from E-02a is the fusion operation applied to each shard:
    - E-02a: MoPEProjector(mope_feats) → [B, 1, llm_dim] → broadcast-add bias
    - E-02c: MoPEProjectorCrossAttn(mope_feats[b:b+1], e.unsqueeze(0)) per shard
             → [1, N_img_i, llm_dim] → squeeze → replace shard

    MoPEProjectorCrossAttn.forward signature:
        (mope_features: [B, N_mope, mope_dim], image_embeds: [B, N_img, llm_dim])
        -> [B, N_img, llm_dim]

    Sidecar support: identical to E-02a — if ``mope_frames`` kwarg is None the
    patch reads ``self._pending_mope_frames`` from the inner model instance.
    """
    inner_model = model.model
    original_forward = inner_model.forward

    def patched_forward(self, *args, mope_frames=None, **kwargs):
        _mope_encoder = getattr(self, '_mope_encoder', None)
        _mope_projector = getattr(self, '_mope_projector', None)

        if mope_frames is None:
            mope_frames = getattr(self, '_pending_mope_frames', None)

        if _mope_encoder is not None and mope_frames is not None:
            original_get_image_features = self.get_image_features

            # ---------------------------------------------------------------
            # E-10 (Router v1) content-driven gate condition.
            # Only computed when the projector actually uses a gate, so the
            # ungated E-02c/E-03a path is byte-for-byte unchanged.
            #
            # pooled_text = mean over NON-image / NON-video token positions of
            #   the text embeddings get_input_embeddings()(input_ids).  The
            #   visual placeholder tokens MUST be masked out (they have not yet
            #   been scattered with visual features at get_image_features time —
            #   pooling them in would dilute the question-text condition).
            # Shape: [B, llm_dim].  Falls back to None (g=1) if input_ids is
            #   unavailable or a sample has no usable text positions.
            # ---------------------------------------------------------------
            pooled_text = None
            if getattr(_mope_projector, 'use_gate', False):
                pooled_text = _compute_pooled_text(self, args, kwargs)

            def _mope_get_image_features(pixel_values, grid_thw):
                image_embeds_list, deepstack = original_get_image_features(pixel_values, grid_thw)
                mope_feats = _mope_encoder(mope_frames)   # [B, N_mope, 768]
                B_mope = mope_feats.shape[0]
                n_total = len(image_embeds_list)
                assert n_total % B_mope == 0, (
                    f"MoPE crossattn: image_embeds_list length ({n_total}) not divisible by "
                    f"batch size ({B_mope})."
                )
                imgs_per_sample = n_total // B_mope
                _use_gate = getattr(_mope_projector, 'use_gate', False)
                new_embeds = []
                for b in range(B_mope):
                    mf = mope_feats[b:b+1]   # [1, N_mope, 768]
                    # Per-sample text condition for the gate, aligned to shard b.
                    cond_b = None
                    if _use_gate and pooled_text is not None:
                        cond_b = pooled_text[b:b+1]   # [1, llm_dim]
                    for e in image_embeds_list[b * imgs_per_sample:(b + 1) * imgs_per_sample]:
                        if _use_gate:
                            new_e = _mope_projector(
                                mf, e.unsqueeze(0), cond_text=cond_b
                            ).squeeze(0)            # [N_img_i, llm_dim]
                        else:
                            # Ungated E-02c/E-03a path — identical to before.
                            new_e = _mope_projector(mf, e.unsqueeze(0)).squeeze(0)
                        new_embeds.append(new_e)
                return new_embeds, deepstack

            self.get_image_features = _mope_get_image_features

        try:
            output = original_forward(*args, **kwargs)
        finally:
            if _mope_encoder is not None and mope_frames is not None:
                self.get_image_features = original_get_image_features

        return output

    inner_model.forward = types.MethodType(patched_forward, inner_model)
    print("[Space Sensing] Patched inner_model.forward() with MoPE cross-attention injection (E-02c).")


def _patch_model_for_mope_qformer(model) -> None:
    """E-02d: prepend Q-Former-compressed MoPE tokens (32) to the LLM input sequence.

    Injection point is identical to E-02b (_patch_model_for_mope_concat): wraps
    ``inner_model.forward()`` and registers a ``forward_pre_hook`` on
    ``inner_model.language_model`` that prepends MoPE token embeddings to
    ``inputs_embeds`` and extends ``position_ids``, ``attention_mask``, and
    ``visual_pos_masks`` accordingly.

    The only difference from E-02b is the projector output shape:
    - E-02b: MoPEProjectorConcat(mope_feats) → [B, 784, llm_dim]  (N_mope=784)
    - E-02d: MoPEProjectorQFormer(mope_feats) → [B, 32, llm_dim]   (num_queries=32)

    No NaN/Inf guard or gradient clamp hook is applied: Q-Former outputs only 32
    tokens (vs. 784 in E-02b), so gradient accumulation pressure is substantially
    lower, and the out_proj zero-initialization provides the same stable start.

    Injection condition: ``pixel_values`` must be present (prefill only); decode
    steps are passed through unchanged because MoPE tokens are already in the KV cache.
    """
    inner_model = model.model
    original_forward = inner_model.forward

    _diag_state = {"calls": 0}

    def patched_forward(self, *args, mope_frames=None, **kwargs):
        _mope_encoder = getattr(self, '_mope_encoder', None)
        _mope_projector = getattr(self, '_mope_projector', None)

        if mope_frames is None:
            mope_frames = getattr(self, '_pending_mope_frames', None)

        pixel_values = kwargs.get('pixel_values', None)
        should_inject = (
            _mope_encoder is not None
            and _mope_projector is not None
            and mope_frames is not None
            and pixel_values is not None
        )

        if _diag_state["calls"] < 3:
            _diag_state["calls"] += 1
            pv_shape = pixel_values.shape if pixel_values is not None else None
            mf_shape = mope_frames.shape if mope_frames is not None else None
            enc_ok = _mope_encoder is not None
            proj_ok = _mope_projector is not None
            print(
                f"[MoPE E02d DEBUG] call#{_diag_state['calls']}: "
                f"should_inject={should_inject}, "
                f"encoder_ok={enc_ok}, projector_ok={proj_ok}, "
                f"mope_frames={mf_shape}, pixel_values={pv_shape}"
            )

        if not should_inject:
            return original_forward(*args, **kwargs)

        mope_feats = _mope_encoder(mope_frames)      # [B, N_mope, 768]
        mope_embeds = _mope_projector(mope_feats)    # [B, 32, llm_dim]

        N_mope = mope_embeds.shape[1]

        def _lm_pre_hook(module, args, kwargs):
            inputs_embeds = kwargs.get('inputs_embeds', None)
            _from_args = False
            if inputs_embeds is None and len(args) > 0:
                _a0 = args[0]
                if hasattr(_a0, 'dim') and _a0.dim() == 3:
                    inputs_embeds = _a0
                    _from_args = True

            if _diag_state["calls"] <= 3:
                ie_shape = inputs_embeds.shape if inputs_embeds is not None else None
                print(
                    f"[MoPE E02d DEBUG]   _lm_pre_hook fired: "
                    f"inputs_embeds={ie_shape} (from_args={_from_args}), "
                    f"N_mope={N_mope}, "
                    f"kwargs_keys={list(kwargs.keys())[:8]}"
                )

            if inputs_embeds is not None:
                new_embeds = torch.cat(
                    [mope_embeds.to(inputs_embeds.dtype), inputs_embeds], dim=1
                )
                if _from_args:
                    args = (new_embeds,) + args[1:]
                else:
                    kwargs['inputs_embeds'] = new_embeds

            pos_ids = kwargs.get('position_ids', None)
            if pos_ids is not None:
                z = torch.zeros(
                    *pos_ids.shape[:-1], N_mope,
                    dtype=pos_ids.dtype, device=pos_ids.device,
                )
                kwargs['position_ids'] = torch.cat([z, pos_ids], dim=-1)

            attn_mask = kwargs.get('attention_mask', None)
            if attn_mask is not None and hasattr(attn_mask, 'dim') and attn_mask.dim() == 2:
                ones = attn_mask.new_ones(attn_mask.shape[0], N_mope)
                kwargs['attention_mask'] = torch.cat([ones, attn_mask], dim=1)

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
    print("[Space Sensing] Patched inner_model.forward() with MoPE Q-Former injection (E-02d).")
