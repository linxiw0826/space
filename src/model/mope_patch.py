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

import logging
import types

import torch  # noqa: F401 — required by callers that may not import torch independently
import torch.nn.functional as F  # noqa: F401 — used by the LFP auxiliary loss

logger = logging.getLogger(__name__)


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


def _patch_model_for_mope_crossattn(model, feed_features: bool = True) -> None:
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

    paper-2 Stage 1 (mope_feed_features bypass, spec Q6/S5)
    -------------------------------------------------------
    ``feed_features=False`` (the "prediction-only" E-15 arm) skips the
    cross-attn residual fusion and returns the original ``image_embeds_list``
    untouched — i.e. the LLM no longer sees MoPE features injected. The MoPE
    encoder still runs ONCE per forward and its output ``x_vis [B,784,768]``
    is cached on ``self._mope_cached_x_vis`` so the LFP target path can reuse
    it without a second encoding (risk-8). When ``feed_features=True`` (the
    default, = E-03a / E-16) the residual fusion runs exactly as before and
    the cache is still populated for the LFP path. In both cases the NTP path
    is unaffected by the cache (it is a detached no_grad tensor from the
    frozen encoder).
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
                # -----------------------------------------------------------
                # Stage 1 LFP: cache the frozen-encoder latent so the LFP
                # target path reuses it WITHOUT a second encoding (risk-8).
                # Detached + no_grad (encoder is @torch.no_grad and frozen):
                # safe to hold across the inner forward; the cache is cleared
                # in the finally block below so it never leaks across calls.
                # -----------------------------------------------------------
                self._mope_cached_x_vis = mope_feats.detach()
                B_mope = mope_feats.shape[0]
                n_total = len(image_embeds_list)
                assert n_total % B_mope == 0, (
                    f"MoPE crossattn: image_embeds_list length ({n_total}) not divisible by "
                    f"batch size ({B_mope})."
                )
                # -----------------------------------------------------------
                # mope_feed_features bypass (paper-2 Stage 1, spec Q6/S5):
                #   False -> skip the residual fusion, return image_embeds
                #   untouched (the LLM no longer sees MoPE features injected).
                #   The encoder still ran once above (its output is cached for
                #   the LFP target path), so the "prediction-only" arm still
                #   has a valid MoPE latent to predict.
                # -----------------------------------------------------------
                if not feed_features:
                    return image_embeds_list, deepstack
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
                # NOTE: we intentionally do NOT clear ``self._mope_cached_x_vis``
                # here. The outer LFP forward-patch reads it AFTER this inner
                # forward returns (the finally block runs before return), so
                # clearing here would race the consumer. The cache is a detached
                # no_grad tensor (frozen encoder), so it carries no autograd
                # state and is simply overwritten on the next forward — no leak.

        return output

    inner_model.forward = types.MethodType(patched_forward, inner_model)
    print(
        f"[Space Sensing] Patched inner_model.forward() with MoPE cross-attention "
        f"injection (E-02c; feed_features={feed_features})."
    )


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


# =============================================================================
# paper-2 Stage 1: LFP (Latent Future Prediction) auxiliary head.
#
# implements D-12 (论文2 辅助信号). See src/model/lfp_head.py header for the
# full decision-pointer. This patch wraps the OUTER Qwen3VLForConditionalGeneration
# forward (not the inner model) because the LFP head needs the LLM last hidden
# state, which is computed inside the outer forward (modeling_qwen3_vl.py:2056)
# and is NOT exposed in the returned Qwen3VLCausalLMOutputWithPast.
#
# Spec: state/analyses/20260619_stage1_lfp_head_integration.md (§一集成点, §二 S2-S4,
# §三架构图, §四风险点).
#
# Byte-equivalence guarantee (risk-3, the highest-risk item)
# ----------------------------------------------------------
# When ``mope_lfp_enable=False`` (the default) this patch is NEVER INSTALLED
# (train_space.py only calls _patch_model_for_lfp when the flag is True), so
# the outer forward is the untouched original and the E-03a path is
# byte-for-byte unchanged.
#
# When the patch IS installed, it is installed in ADDITION to the inner
# crossattn patch (which handles the feed-features injection). The LFP patch
# wraps the outer forward and, on every training forward that has labels:
#   1. calls the ORIGINAL outer forward once to get the full NTP loss + logits
#      + the last hidden states (via output_hidden_states=True, which the
#      original forward does NOT set, so we inline-replicate the NTP path —
#      see below);
#   2. computes the LFP auxiliary loss and adds it to the NTP loss.
# At eval (no labels, generate path) the patch routes to the original forward
# unchanged, so eval is byte-for-byte E-03a (the head is training-only, risk-6).
#
# To keep the NTP loss EXACTLY byte-equivalent to the original (rather than
# re-deriving logits+loss from the returned logits, which can drift due to the
# ``logits_to_keep`` slicing and the fp32 loss path), the patch calls the
# ORIGINAL outer forward to get ``loss`` (the real NTP CE, computed by
# modeling_qwen3_vl.py:2062-2064 exactly as before) and the cached MoPE latent.
# It then needs the last hidden state to feed the LFP head. The cleanest way
# that avoids re-deriving NTP is to ALSO call the inner model with
# output_hidden_states once — but that doubles the LLM forward cost. Instead we
# use the single-call approach: we inline-replicate the outer forward's tail
# (modeling_qwen3_vl.py:2041-2064) so that we (a) call self.model(...) ONCE,
# (b) take hidden_states = outputs[0], (c) compute logits + ntp_loss by calling
# the SAME self.lm_head / self.loss_function the original uses, with the SAME
# logits_to_keep slice. This reproduces the original numerics bit-for-bit and
# only adds the LFP term on top.
# =============================================================================


def _build_lfp_target(cached_x_vis, context_frames, target_pool):
    """Build the LFP target from the cached frozen-MoPE latent (spec S2, risk-4).

    Args:
        cached_x_vis:   [B, 784, 768] detached frozen-encoder latent (from the
                        inner crossattn patch's cache). 784 = 4 time-bins x 196
                        spatial (8 frames, tubelet_size=2, 224x224).
        context_frames: number of leading frames treated as "context" (default
                        4 -> 4 / 4 time-bin split). The FUTURE half-bin count is
                        ``max(1, total_bins - context_bins)`` where
                        ``total_bins = 784 // 196 = 4`` and
                        ``context_bins = context_frames // tubelet_size``.
        target_pool:    "mean" (default) -> [B, 768]; "token" -> [B, N_future, 768].

    Returns:
        target: the (detached) future-frame latent, or None if the split is
                degenerate (no future bins).
    """
    # 784 = 4 time-bins x 196 spatial. MoPE patch grid is row-major
    # (idx = t * H * W + h * W + w), so the first dim is time. Splitting along
    # the token dim at time-bin granularity (multiples of 196) is correct and
    # never breaks a time-bin across context/future (risk-4).
    SPATIAL = 196
    TUBELET = 2
    B, N, D = cached_x_vis.shape
    n_bins = N // SPATIAL
    context_bins = max(0, context_frames // TUBELET)
    future_bins = n_bins - context_bins
    if future_bins <= 0:
        return None
    start = context_bins * SPATIAL
    future_latent = cached_x_vis[:, start:, :]   # [B, future_bins*196, 768]
    if target_pool == "mean":
        return future_latent.mean(dim=1)         # [B, 768]
    # "token": keep per-token (used by future Stage-1 extension; not in the
    # default Stage-1 config which uses "mean").
    return future_latent                          # [B, N_future, 768]


def _extract_pred_source(hidden_states, labels, pred_source):
    """Extract the [B, llm_dim] prediction-source vector from the LLM hidden states.

    Args:
        hidden_states: [B, L, llm_dim] last hidden state from the inner model.
        labels:        [B, L] label tensor (with -100 at non-answer positions).
        pred_source:   "last_answer" -> for each row, the hidden state at the
                       index of the LAST token whose label != -100.

    Returns:
        h: [B, llm_dim], or None if no valid answer position exists.
    """
    B, L, D = hidden_states.shape
    if pred_source == "last_answer":
        if labels is None:
            return None
        valid = labels != -100                       # [B, L]
        has_any = valid.any(dim=1)                   # [B]
        if not has_any.any():
            return None
        # Last valid index per row: flip, find first valid, invert back.
        # For rows with no valid token we clamp to 0 and zero the row afterwards.
        flipped_valid = valid.flip(dims=[1])
        flipped_first = flipped_valid.float().argmax(dim=1)   # [B]
        last_idx = (L - 1) - flipped_first                    # [B]
        gather_idx = last_idx.view(B, 1, 1).expand(B, 1, D)
        h = hidden_states.gather(1, gather_idx).squeeze(1)    # [B, D]
        # Zero out rows that had no valid answer token.
        h = h * has_any.unsqueeze(-1).to(h.dtype)
        return h
    # Other pred_source values (e.g. "mean_text") are reserved for future
    # Stage-1 extensions; the default config uses "last_answer".
    return None


# =============================================================================
# paper-2 Stage 1 (FINAL, decisions.md 2026-06-27): token-shift TRUE-FUTURE
# prediction (Cambrian-S faithful), replacing the BLOCK-1 distillation target.
#
# Spec: state/analyses/20260622_stage1_tokenshift_spec.md (§S1-S4, §F1-F5).
#
# Why token-shift (one line): the MoPE latent is computed BIDIRECTIONALLY over
# all 8 frames, so "predict the current bin" leaks future information into the
# target. "Predict the NEXT bin" puts the target on frames the LLM hidden has
# not attended to (LLM is strictly causal, video tokens are laid out in temporal
# order — spec F1), giving a genuine future-prediction objective with no leak.
#
# Pipeline:
#   1. _extract_per_frame_video_hidden: pool the LLM last hidden state PER VIDEO
#      FRAME (one mean-pool per contiguous run of video_token_id), temporal order.
#   2. _group_llm_frames_to_bins: uniformly bin the F LLM frames into n_bins time
#      bins (so the LLM side aligns to the MoPE 4-bin / tubelet-2 structure even
#      when the LLM frame count F != 8 — spec F5 / §2.3).
#   3. _build_tokenshift_pairs: source = LLM bins [0..K-2], target = frozen-MoPE
#      latent bins [1..K-1]; pair source[t] -> target[t] (= bin t hidden predicts
#      bin t+1 latent). With 4 bins this is 3 pairs.
# =============================================================================


def _extract_per_frame_video_hidden(hidden_states, input_ids, visual_token_ids):
    """Pool the LLM last hidden state PER VISUAL FRAME, in temporal order (spec S1).

    Qwen3VL lays out each frame as an independent run
    ``<vision_start> <visual_token x N> <vision_end>`` in the LLM sequence (spec
    F1), where N (= tokens-per-frame) depends on the frame resolution and is
    therefore NOT a fixed constant. We must NOT assume 8 frames or a fixed token
    count: we split on contiguous runs of ANY visual placeholder token (a run
    break = a gap > 1 between consecutive visual-token indices) and mean-pool
    each run into one per-frame vector.

    IMPORTANT (training-vs-eval token id, the E-16 no-DIAG bug fix): the SPAR /
    VSI-590K training pipeline converts each video into a list of IMAGES
    (data_processor.py:743-755 replaces <video> with N <image> tokens and drops
    the "video" key), so during TRAINING each frame is an IMAGE token
    (image_token_id=151655), NOT a video token (151656). At eval (VSI-Bench) the
    same frames may instead be video tokens. We therefore key on the SET of
    visual token ids (image_token_id AND video_token_id) so the per-frame
    extraction works for both layouts. The temporal-order + causal-isolation
    argument (spec F1) holds identically for the multi-image layout: images are
    emitted in temporal order, contiguously, and the LLM is strictly causal, so
    frame t's hidden cannot attend to frame t+1's tokens.

    Args:
        hidden_states:    [B, L, llm_dim] LLM last hidden state.
        input_ids:        [B, L] token ids.
        visual_token_ids: int OR iterable of ints (e.g. (image_token_id,
                          video_token_id)). None entries are ignored.

    Returns:
        per_frame_hidden: [B, F, llm_dim] per-frame mean-pooled hidden, in
                          TEMPORAL ORDER (frame 0 .. frame F-1). F is the number
                          of frames in this batch's sequence. If samples disagree
                          on F (different resolutions / frame counts), they are
                          truncated to the per-batch minimum F (defensive).
                          Returns None if input_ids is None, no valid visual
                          token id is given, or no visual tokens are present.
    """
    if input_ids is None or visual_token_ids is None:
        return None
    # Normalize to a list of valid (non-None) ids.
    if isinstance(visual_token_ids, int):
        id_list = [visual_token_ids]
    else:
        id_list = [t for t in visual_token_ids if t is not None]
    if not id_list:
        return None
    is_video = torch.zeros_like(input_ids, dtype=torch.bool)   # [B, L]
    for tid in id_list:
        is_video |= (input_ids == tid)
    B, L, D = hidden_states.shape
    per_frame = []
    for b in range(B):
        mask_b = is_video[b]                         # [L]
        idx = torch.nonzero(mask_b, as_tuple=False).squeeze(-1)   # [M]
        if idx.numel() == 0:
            return None
        # Segment boundaries: a contiguous run breaks where the index gap > 1.
        breaks = (idx[1:] - idx[:-1]) > 1            # [M-1] bool
        seg_starts = torch.cat([idx[:1], idx[1:][breaks]])
        seg_ends = torch.cat([idx[:-1][breaks], idx[-1:]])
        frames_b = []
        for s, e in zip(seg_starts.tolist(), seg_ends.tolist()):
            frames_b.append(hidden_states[b, s:e + 1, :].mean(dim=0))   # [D]
        per_frame.append(torch.stack(frames_b, dim=0))                  # [F_b, D]
    # Same training template -> F should match across the batch; truncate to the
    # minimum defensively in case resolutions differ.
    F_min = min(f.shape[0] for f in per_frame)
    if F_min == 0:
        return None
    per_frame = torch.stack([f[:F_min] for f in per_frame], dim=0)      # [B, F_min, D]
    return per_frame


def _group_llm_frames_to_bins(per_frame_hidden, F, n_bins=4, tubelet=2):
    """Uniformly group F LLM per-frame hidden vectors into n_bins time bins (spec S3).

    The MoPE latent always has 4 time bins (8 frames / tubelet 2), but the LLM
    frame count F is set by the video_processor (fps/min/max frames) and may be
    < 8 (spec F5). We therefore align by TIME SEGMENT, not by frame index: bin b
    covers LLM frames ``[b*F//n_bins, (b+1)*F//n_bins)`` and is the mean of the
    frames in that segment. This is the correct abstraction under heterogeneous
    frame counts — bin b's hidden strictly precedes bin b+1's frames, so the
    LLM-causal isolation (spec F1) is preserved. When F == 8 this degenerates to
    "2 frames per bin", exactly matching the MoPE tubelet-2 structure.

    Note: ``tubelet`` is accepted for signature parity with the MoPE latent
    layout but is not used by uniform binning (the binning is purely proportional
    to F); it is kept so callers can document the tubelet-2 correspondence.

    Args:
        per_frame_hidden: [B, F, D] per-frame hidden (temporal order).
        F:                int number of LLM frames.
        n_bins:           int target bin count (= MoPE time bins, default 4).
        tubelet:          int (unused; documents the 2-frames-per-bin intent).

    Returns:
        bin_hidden: [B, n_bins, D].
    """
    del tubelet  # documented-but-unused (see docstring)
    B, _, D = per_frame_hidden.shape
    bins = []
    for b in range(n_bins):
        s = (b * F) // n_bins
        e = ((b + 1) * F) // n_bins
        if e <= s:                 # empty bin (F < n_bins): borrow one frame
            e = s + 1
        e = min(e, F)
        bins.append(per_frame_hidden[:, s:e, :].mean(dim=1))   # [B, D]
    return torch.stack(bins, dim=1)                            # [B, n_bins, D]


def _build_tokenshift_pairs(cached_x_vis, per_frame_hidden, tubelet=2, spatial=196):
    """Build the token-shift (source, target) pairs for TRUE future prediction (spec S2).

    MoPE latent layout (spec F3): cached_x_vis[:, t*196:(t+1)*196] = time-bin t,
        so ``cached_x_vis.view(B, n_bins, 196, D).mean(2)`` gives per-bin latents
        ``latent_bins [B, n_bins, 768]`` in temporal order.
    LLM side: per_frame_hidden [B, F, llm_dim] is grouped into the SAME n_bins
        time bins via uniform binning (spec S3).

    token-shift (Cambrian-S roll(-1) equivalent, done with slices to avoid any
    cross-sample mixing):
        source_bins = llm_bin_hidden[:, :-1]   # bins 0..K-2  (the "seen" bins)
        target_bins = latent_bins[:,   1:]     # bins 1..K-1  (the "future" bins)
        pair source_bins[t] -> target_bins[t]  (= bin t hidden predicts bin t+1
        latent). Because the LLM is strictly causal (spec F1), bin t's hidden
        cannot see bin t+1's tokens -> genuine future prediction.

    Args:
        cached_x_vis:     [B, n_bins*196, 768] detached frozen-MoPE latent.
        per_frame_hidden: [B, F, llm_dim] per-frame LLM hidden (temporal order).
        tubelet:          int (passed through to the binning, documents intent).
        spatial:          int spatial tokens per bin (196 for ViT-B / 8 frames).

    Returns:
        (source_bins, target_bins):
            source_bins: [B, K-1, llm_dim] LLM-side per-bin hidden (causal).
            target_bins: [B, K-1, 768]     frozen-MoPE per-bin latent.
        or (None, None) if the latent has < 2 bins (no shift possible).
    """
    B, N, D = cached_x_vis.shape
    n_bins = N // spatial
    if n_bins < 2:
        return None, None
    latent_bins = cached_x_vis.view(B, n_bins, spatial, D).mean(dim=2)   # [B, n_bins, 768]
    F = per_frame_hidden.shape[1]
    llm_bin_hidden = _group_llm_frames_to_bins(per_frame_hidden, F, n_bins, tubelet)  # [B, n_bins, llm_dim]
    source_bins = llm_bin_hidden[:, :-1, :]   # [B, K-1, llm_dim]
    target_bins = latent_bins[:, 1:, :]       # [B, K-1, 768]
    return source_bins, target_bins


def _patch_model_for_lfp(model, lfp_cfg) -> None:
    """Wrap the OUTER Qwen3VLForConditionalGeneration.forward with an LFP head.

    The patch is TRAINING-ONLY: it activates the LFP auxiliary loss only when
    the forward call carries ``labels`` (training) AND the inner crossattn
    patch populated ``_mope_cached_x_vis`` (i.e. MoPE frames were supplied and
    the frozen encoder ran once this step). On eval / generate calls (no
    labels) the patch routes to the original outer forward unchanged, so the
    inference path is byte-for-byte E-03a (the head is never called).

    The NTP loss is reproduced BIT-FOR-BIT by inlining the original outer
    forward's tail (modeling_qwen3_vl.py:2041-2064): call ``self.model(...)``
    once, take ``hidden_states = outputs[0]``, slice with the original
    ``logits_to_keep`` semantics, apply ``self.lm_head``, and call
    ``self.loss_function`` with the SAME arguments the original uses. The only
    addition is the LFP term ``lfp_weight * L_lfp`` added to the NTP loss.

    Args:
        model:    the top-level Qwen3VLForConditionalGeneration (outer model).
        lfp_cfg:  a namespace/dataclass-like object with the following fields:
            - enable (bool):          must be True (caller only invokes when on)
            - weight (float):         lambda (default 0.1)
            - mse_weight (float):     MSE term weight (default 1.0)
            - cos_weight (float):     cosine term weight (default 1.0)
            - context_frames (int):   leading context frames (default 4)
            - target_pool (str):      "mean" | "token" (default "mean")
            - pred_source (str):      "last_answer" (default)
    """
    outer = model
    original_forward = outer.forward
    inner_model = outer.model

    # ------------------------------------------------------------------
    # W1 guard: warn (rank0) if the split degenerates. With the default
    # mope_all_frames=8 and tubelet_size=2 there are 4 time-bins; if
    # context_frames >= 8 (>= 4 context bins) there is NO future bin and
    # _build_lfp_target will always return None, silently disabling the LFP
    # auxiliary loss — training degrades to plain E-03a. This is a config
    # foot-gun: emit a warning so the silent degradation is visible.
    # (distributed rank0: the master process is rank 0; on single-node CPU
    #  torch.distributed is not initialized and we just log.)
    # ------------------------------------------------------------------
    try:
        import torch.distributed as dist
        is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
    except Exception:
        is_rank0 = True
    # ``pred_source == "per_frame_video"`` selects the FINAL token-shift path
    # (decisions.md 2026-06-27); any other value keeps the legacy distillation
    # path (kept for back-compat / reproducing the old runs).
    _use_tokenshift = str(getattr(lfp_cfg, "pred_source", "per_frame_video")) == "per_frame_video"
    # The context_frames foot-gun only applies to the legacy distillation path;
    # token-shift derives its 3 pairs from the 4 latent bins directly.
    if (not _use_tokenshift) and int(lfp_cfg.context_frames) >= 8 and is_rank0:
        logger.warning(
            "LFP target empty (context_frames>=8), LFP loss disabled — "
            "training degrades to E-03a. Lower mope_lfp_context_frames "
            "(default 4 -> 4 context / 4 future time-bins)."
        )

    # Diagnostic throttle for the token-shift lambda-magnitude probe (spec §二):
    # print mse/cos per-bin, L_lfp, L_ntp, the ratio, and lambda*L_lfp for the
    # first 3 training steps (rank0 only), then auto-silence.
    _lfp_diag_state = {"calls": 0}

    def lfp_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        cache_position=None,
        logits_to_keep=0,
        geometry_encoder_inputs=None,
        **kwargs,
    ):
        # ----------------------------------------------------------------
        # LFP is TRAINING-ONLY. Eval / generate never passes labels, so we
        # route straight to the original outer forward — byte-for-byte E-03a
        # (head not built, head not called). This also keeps generate() safe.
        # ----------------------------------------------------------------
        if labels is None:
            return original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                geometry_encoder_inputs=geometry_encoder_inputs,
                **kwargs,
            )

        # ----------------------------------------------------------------
        # Inline-replicate modeling_qwen3_vl.py:2041-2064 to get BOTH the NTP
        # loss AND the last hidden states in ONE inner forward call. This is
        # the exact code path of the original (same self.model call, same
        # logits_to_keep slice, same self.lm_head, same self.loss_function),
        # so the NTP numerics are bit-for-bit identical to E-03a.
        # ----------------------------------------------------------------
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            geometry_encoder_inputs=geometry_encoder_inputs,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs[0]

        # logits_to_keep handling — verbatim from modeling_qwen3_vl.py:2059-2060.
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # NTP CE — verbatim from modeling_qwen3_vl.py:2062-2064.
        loss = self.loss_function(
            logits=logits,
            labels=labels,
            vocab_size=self.config.text_config.vocab_size,
        )

        # ----------------------------------------------------------------
        # LFP auxiliary loss (spec S2 + S3, risk-1/4/7/8).
        # ----------------------------------------------------------------
        lfp_head = getattr(inner_model, '_mope_lfp_head', None)
        cached_x_vis = getattr(inner_model, '_mope_cached_x_vis', None)

        # ----------------------------------------------------------------
        # One-time unconditional entry probe (rank0). Prints WHY the LFP block
        # was or was not entered, so a 5-step dry run confirms the wiring. It
        # fires once per process and then auto-silences.
        # ----------------------------------------------------------------
        if not _lfp_diag_state.get("entry_probed", False):
            try:
                import torch.distributed as _dist
                _probe_rank0 = (not _dist.is_initialized()) or _dist.get_rank() == 0
            except Exception:
                _probe_rank0 = True
            if _probe_rank0:
                _lfp_diag_state["entry_probed"] = True
                print(
                    f"[LFP PROBE] entry: use_tokenshift={_use_tokenshift}, "
                    f"lfp_head_is_None={lfp_head is None}, "
                    f"cached_x_vis_is_None={cached_x_vis is None}",
                    flush=True,
                )

        if lfp_head is not None and cached_x_vis is not None and _use_tokenshift:
            # ----------------------------------------------------------------
            # FINAL token-shift path (Cambrian-S faithful, decisions.md
            # 2026-06-27). source = LLM bins 0..K-2 (causal hidden), target =
            # frozen-MoPE latent bins 1..K-1 -> bin t hidden predicts bin t+1
            # latent (3 pairs for 4 bins). spec S1-S4.
            # ----------------------------------------------------------------
            # Key on BOTH image and video placeholder ids: SPAR/VSI-590K training
            # feeds frames as IMAGES (data_processor converts <video> -> N <image>),
            # so the per-frame runs are image tokens at train time; eval may use
            # video tokens. See _extract_per_frame_video_hidden docstring.
            _cfg = getattr(self, 'config', None) or getattr(inner_model, 'config', None)
            image_token_id = getattr(_cfg, 'image_token_id', None)
            video_token_id = getattr(_cfg, 'video_token_id', None)

            per_frame_h = _extract_per_frame_video_hidden(
                hidden_states, input_ids, (image_token_id, video_token_id)
            )
            if not _lfp_diag_state.get("shape_probed", False):
                try:
                    import torch.distributed as _dist
                    _probe_rank0 = (not _dist.is_initialized()) or _dist.get_rank() == 0
                except Exception:
                    _probe_rank0 = True
                if _probe_rank0:
                    _lfp_diag_state["shape_probed"] = True
                    _pf_shape = None if per_frame_h is None else tuple(per_frame_h.shape)
                    print(
                        f"[LFP PROBE] image_token_id={image_token_id}, "
                        f"video_token_id={video_token_id}, "
                        f"per_frame_h={_pf_shape}",
                        flush=True,
                    )
            if per_frame_h is not None:
                source_bins, target_bins = _build_tokenshift_pairs(
                    cached_x_vis, per_frame_h, tubelet=2, spatial=196,
                )
                if not _lfp_diag_state.get("src_probed", False):
                    try:
                        import torch.distributed as _dist
                        _probe_rank0 = (not _dist.is_initialized()) or _dist.get_rank() == 0
                    except Exception:
                        _probe_rank0 = True
                    if _probe_rank0:
                        _lfp_diag_state["src_probed"] = True
                        _sb = None if source_bins is None else tuple(source_bins.shape)
                        _tb = None if target_bins is None else tuple(target_bins.shape)
                        print(
                            f"[LFP PROBE] source_bins={_sb}, target_bins={_tb}",
                            flush=True,
                        )
                if source_bins is not None and source_bins.shape[1] > 0:
                    B_ts, Km1, llm_dim = source_bins.shape
                    target_dim = target_bins.shape[-1]
                    pred_bins = lfp_head(
                        source_bins.reshape(B_ts * Km1, llm_dim).to(hidden_states.dtype)
                    ).reshape(B_ts, Km1, target_dim)
                    # frozen encoder output -> detach for safety (risk-1/4).
                    target_bins = target_bins.detach().to(pred_bins.dtype)
                    # fp32 upcast for the small auxiliary head, reduction='none'
                    # then mean over pairs (Cambrian-S per-pair averaging, S2).
                    mse_per = F.mse_loss(
                        pred_bins.float(), target_bins.float(), reduction='none'
                    ).mean(dim=-1)                                              # [B, K-1]
                    cos_per = 1.0 - F.cosine_similarity(
                        pred_bins.float(), target_bins.float(), dim=-1
                    )                                                          # [B, K-1]
                    L_lfp = (
                        lfp_cfg.mse_weight * mse_per.mean()
                        + lfp_cfg.cos_weight * cos_per.mean()
                    )
                    loss_ntp = loss
                    loss = loss + lfp_cfg.weight * L_lfp

                    # --- lambda-magnitude diagnostic (spec §二): rank0, 1st 3 steps.
                    # Recompute rank0 at call time (dist may not have been
                    # initialized when the patch was installed).
                    try:
                        import torch.distributed as _dist
                        _diag_rank0 = (not _dist.is_initialized()) or _dist.get_rank() == 0
                    except Exception:
                        _diag_rank0 = True
                    if _lfp_diag_state["calls"] < 3 and _diag_rank0:
                        _lfp_diag_state["calls"] += 1
                        print(
                            f"[LFP DIAG] step={_lfp_diag_state['calls']}: "
                            f"mse_per_bin={mse_per.mean().item():.4f}, "
                            f"cos_per_bin={cos_per.mean().item():.4f}, "
                            f"L_lfp={L_lfp.item():.4f}, "
                            f"L_ntp={loss_ntp.item():.4f}, "
                            f"ratio_L_lfp/L_ntp={(L_lfp.item() / max(loss_ntp.item(), 1e-8)):.4f}, "
                            f"lambda*L_lfp={lfp_cfg.weight * L_lfp.item():.4f}",
                            flush=True,
                        )

        elif lfp_head is not None and cached_x_vis is not None:
            # ----------------------------------------------------------------
            # LEGACY distillation path (BLOCK-1, kept for back-compat only;
            # pred_source != "per_frame_video"). Not the default any more.
            # ----------------------------------------------------------------
            # target = future time-bin latent (risk-4: split at time-bin
            # granularity, never mid-bin). Already detached/no_grad (frozen
            # encoder); double-detach for safety (risk-1).
            target = _build_lfp_target(
                cached_x_vis,
                lfp_cfg.context_frames,
                lfp_cfg.target_pool,
            )
            if target is not None:
                target = target.detach().to(hidden_states.dtype)
                # pred source: [B, llm_dim] from the last hidden state.
                # Use the FULL (un-sliced) hidden_states so last_answer index
                # aligns with the label positions.
                h = _extract_pred_source(hidden_states, labels, lfp_cfg.pred_source)
                if h is not None:
                    h = h.to(hidden_states.dtype)
                    pred = lfp_head(h)   # [B, target_dim] or [B, N, target_dim]
                    pred = pred.to(target.dtype)
                    # Align shapes: mean-pool pred over token dim if needed.
                    if pred.dim() == 3 and target.dim() == 2:
                        pred = pred.mean(dim=1)
                    if pred.dim() == 2 and target.dim() == 3:
                        target = target.mean(dim=1)
                    # fp32 upcast for the auxiliary loss (bf16 path), matching
                    # the NTP loss_function's internal fp32 upcast style —
                    # reduces gradient noise from bf16 matmul on the small head.
                    mse = F.mse_loss(pred.float(), target.float())
                    cos = 1.0 - F.cosine_similarity(pred.float(), target.float(), dim=-1).mean()
                    L_lfp = (
                        lfp_cfg.mse_weight * mse
                        + lfp_cfg.cos_weight * cos
                    )
                    loss = loss + lfp_cfg.weight * L_lfp

        # Re-assemble the return object (verbatim fields, only ``loss`` changed).
        # Lazy import so this module stays importable outside the training env.
        try:
            from qwenvl.model.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast
        except Exception:
            Qwen3VLCausalLMOutputWithPast = None

        if Qwen3VLCausalLMOutputWithPast is not None:
            return Qwen3VLCausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=getattr(outputs, 'past_key_values', None),
                rope_deltas=getattr(outputs, 'rope_deltas', None),
            )
        # Fallback (e.g. non-Qwen3VL backbone): return a plain tuple matching
        # the HF convention (loss, logits, *rest).
        return (
            loss,
            logits,
            getattr(outputs, 'past_key_values', None),
            getattr(outputs, 'rope_deltas', None),
        )

    outer.forward = types.MethodType(lfp_forward, outer)
    _mode = "token-shift (predict-next-bin)" if _use_tokenshift else "legacy distillation"
    print(
        f"[Space Sensing] Patched outer model.forward() with LFP auxiliary head "
        f"[{_mode}] (weight={lfp_cfg.weight}, mse_w={lfp_cfg.mse_weight}, "
        f"cos_w={lfp_cfg.cos_weight}, pred_source={lfp_cfg.pred_source}, "
        f"align_strategy={getattr(lfp_cfg, 'align_strategy', 'uniform')}, "
        f"ctx_frames={lfp_cfg.context_frames}, target_pool={lfp_cfg.target_pool})."
    )
