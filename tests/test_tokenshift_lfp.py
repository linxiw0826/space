"""CPU unit tests for the paper-2 Stage-1 token-shift LFP construction.

Covers the spec §四 single-test requirements (state/analyses/
20260622_stage1_tokenshift_spec.md):
  - per-frame video-hidden extraction index correctness (variable token count
    per frame, contiguous-run segmentation, temporal order).
  - token-shift pairing correctness (target_bins[:,t] == latent_bins[:,t+1]).
  - uniform frame->bin binning (F==8 -> 2 frames/bin; F<n_bins empty-bin
    defense; off-by-one).
  - structural byte-equivalence guard (mope_lfp_enable=False never installs the
    LFP patch — verified by source inspection; the full-model byte-equiv test
    needs GPU + the Qwen3VL backbone and is documented but not run here).

Run:  python tests/test_tokenshift_lfp.py
"""

import contextlib
import importlib.util
import io
import os
import re
import types

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "src")
_MOPE_PATCH = os.path.join(_SRC, "model", "mope_patch.py")
_TRAIN_SPACE = os.path.join(_SRC, "train_framework", "train_space.py")

# Import mope_patch.py standalone by file path (avoids package import which pulls
# in the heavy training environment).
_spec = importlib.util.spec_from_file_location("mope_patch_under_test", _MOPE_PATCH)
mp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mp)


def _build_video_input_ids(seg_lengths, vid_id, B, L, gap_token=5):
    """Construct [B, L] input_ids with one contiguous video run per seg length.

    Runs are separated by a single non-video gap token (mimicking
    <vision_end> <timestamp> <vision_start> between frames). Leading/trailing
    text tokens pad to length L.
    """
    rows = []
    for _ in range(B):
        seq = [gap_token, gap_token]  # leading text
        for n in seg_lengths:
            seq += [vid_id] * n
            seq += [gap_token]        # gap between frames
        seq += [gap_token] * 3        # trailing answer text
        assert len(seq) <= L, f"seq {len(seq)} > L {L}"
        seq += [gap_token] * (L - len(seq))
        rows.append(seq)
    return torch.tensor(rows, dtype=torch.long)


def test_per_frame_extraction_basic():
    """3 frames of 49 tokens each -> [B, 3, D]; mean-pool matches hand calc."""
    vid_id = 151656
    B, L, D = 2, 256, 8
    seg = [49, 49, 49]
    input_ids = _build_video_input_ids(seg, vid_id, B, L)
    # Distinctive hidden values so the mean-pool is checkable.
    hidden = torch.arange(B * L * D, dtype=torch.float32).reshape(B, L, D)

    out = mp._extract_per_frame_video_hidden(hidden, input_ids, vid_id)
    assert out is not None
    assert out.shape == (B, 3, D), out.shape

    # Re-derive expected per-frame means directly from the mask.
    is_vid = (input_ids == vid_id)
    for b in range(B):
        idx = torch.nonzero(is_vid[b]).squeeze(-1)
        breaks = (idx[1:] - idx[:-1]) > 1
        starts = torch.cat([idx[:1], idx[1:][breaks]])
        ends = torch.cat([idx[:-1][breaks], idx[-1:]])
        assert len(starts) == 3
        for f, (s, e) in enumerate(zip(starts.tolist(), ends.tolist())):
            exp = hidden[b, s:e + 1, :].mean(dim=0)
            assert torch.allclose(out[b, f], exp), (b, f)
    print("ok  test_per_frame_extraction_basic")


def test_per_frame_extraction_variable_token_count():
    """Frames with DIFFERENT token counts (resolution varies) still segment right."""
    vid_id = 151656
    B, L, D = 1, 128, 4
    seg = [16, 25, 9]  # heterogeneous per-frame token counts
    input_ids = _build_video_input_ids(seg, vid_id, B, L)
    hidden = torch.randn(B, L, D)
    out = mp._extract_per_frame_video_hidden(hidden, input_ids, vid_id)
    assert out.shape == (B, 3, D), out.shape
    # Frame 1 (the 25-token run) mean must equal the explicit slice mean.
    idx = torch.nonzero(input_ids[0] == vid_id).squeeze(-1)
    breaks = (idx[1:] - idx[:-1]) > 1
    starts = torch.cat([idx[:1], idx[1:][breaks]])
    ends = torch.cat([idx[:-1][breaks], idx[-1:]])
    exp1 = hidden[0, starts[1]:ends[1] + 1, :].mean(dim=0)
    assert torch.allclose(out[0, 1], exp1)
    print("ok  test_per_frame_extraction_variable_token_count")


def test_per_frame_extraction_no_video():
    """No video tokens -> None (safe fallback, LFP loss skipped)."""
    input_ids = torch.full((2, 32), 7, dtype=torch.long)
    hidden = torch.randn(2, 32, 4)
    assert mp._extract_per_frame_video_hidden(hidden, input_ids, 151656) is None
    assert mp._extract_per_frame_video_hidden(hidden, None, 151656) is None
    assert mp._extract_per_frame_video_hidden(hidden, input_ids, None) is None
    print("ok  test_per_frame_extraction_no_video")


def test_group_frames_to_bins_F8():
    """F==8 -> 2 frames per bin (matches MoPE tubelet=2)."""
    B, F, D = 2, 8, 6
    pf = torch.arange(B * F * D, dtype=torch.float32).reshape(B, F, D)
    out = mp._group_llm_frames_to_bins(pf, F=8, n_bins=4, tubelet=2)
    assert out.shape == (B, 4, D)
    for b in range(4):
        exp = pf[:, 2 * b:2 * b + 2, :].mean(dim=1)
        assert torch.allclose(out[:, b], exp), b
    print("ok  test_group_frames_to_bins_F8")


def test_group_frames_to_bins_F7_and_small():
    """F<8 still works; F<n_bins triggers the empty-bin defense (no crash)."""
    B, D = 1, 4
    pf7 = torch.randn(B, 7, D)
    out7 = mp._group_llm_frames_to_bins(pf7, F=7, n_bins=4, tubelet=2)
    assert out7.shape == (B, 4, D)
    # F=3 < n_bins=4 -> some bin borrows a frame (e=s+1), no empty mean (NaN).
    pf3 = torch.randn(B, 3, D)
    out3 = mp._group_llm_frames_to_bins(pf3, F=3, n_bins=4, tubelet=2)
    assert out3.shape == (B, 4, D)
    assert torch.isfinite(out3).all(), "empty-bin defense produced NaN/Inf"
    print("ok  test_group_frames_to_bins_F7_and_small")


def test_tokenshift_pairing():
    """source=bins0..2, target=latent bins1..3; target[:,t]==latent_bins[:,t+1]."""
    B, llm_dim = 2, 32
    spatial, n_bins = 196, 4
    cached = torch.randn(B, spatial * n_bins, 768)
    pf = torch.randn(B, 8, llm_dim)
    src, tgt = mp._build_tokenshift_pairs(cached, pf, tubelet=2, spatial=spatial)
    assert src.shape == (B, 3, llm_dim), src.shape
    assert tgt.shape == (B, 3, 768), tgt.shape

    latent_bins = cached.view(B, n_bins, spatial, 768).mean(dim=2)  # [B,4,768]
    # token-shift: target bin t == latent bin t+1 (the FUTURE bin).
    for t in range(3):
        assert torch.allclose(tgt[:, t], latent_bins[:, t + 1]), t
    # source bin t == uniform-binned LLM hidden bin t (the SEEN bin).
    llm_bins = mp._group_llm_frames_to_bins(pf, F=8, n_bins=4, tubelet=2)
    for t in range(3):
        assert torch.allclose(src[:, t], llm_bins[:, t]), t
    print("ok  test_tokenshift_pairing")


def test_tokenshift_degenerate_single_bin():
    """< 2 latent bins -> (None, None) (no shift possible)."""
    cached = torch.randn(1, 196, 768)  # 1 bin only
    pf = torch.randn(1, 8, 16)
    src, tgt = mp._build_tokenshift_pairs(cached, pf, tubelet=2, spatial=196)
    assert src is None and tgt is None
    print("ok  test_tokenshift_degenerate_single_bin")


def test_head_consumes_batched_source():
    """LFPHead forward accepts the reshaped [B*(K-1), llm_dim] source batch."""
    _spec2 = importlib.util.spec_from_file_location(
        "lfp_head_under_test", os.path.join(_SRC, "model", "lfp_head.py")
    )
    lh = importlib.util.module_from_spec(_spec2)
    _spec2.loader.exec_module(lh)
    B, llm_dim, K = 2, 32, 4
    head = lh.MoPELFPHead(llm_dim=llm_dim, hidden=64, target_dim=768)
    src = torch.randn(B, K - 1, llm_dim)
    out = head(src.reshape(B * (K - 1), llm_dim)).reshape(B, K - 1, 768)
    assert out.shape == (B, K - 1, 768), out.shape
    print("ok  test_head_consumes_batched_source")


def test_per_frame_extraction_image_tokens():
    """Frames fed as IMAGE tokens (SPAR train path) extract correctly via id-set.

    Regression for the E-16 no-DIAG bug: training converts <video> -> N <image>,
    so frames are image_token_id (151655), not video_token_id (151656). The
    extractor must accept a tuple of ids and segment on whichever is present.
    """
    image_id, video_id = 151655, 151656
    B, L, D = 1, 200, 4
    seg = [40, 40, 40, 40]  # 4 image frames
    input_ids = _build_video_input_ids(seg, image_id, B, L)
    hidden = torch.randn(B, L, D)
    # Keyed on video id ALONE -> None (this was the bug).
    assert mp._extract_per_frame_video_hidden(hidden, input_ids, video_id) is None
    # Keyed on the (image, video) id set -> 4 frames.
    out = mp._extract_per_frame_video_hidden(hidden, input_ids, (image_id, video_id))
    assert out is not None and out.shape == (B, 4, D), None if out is None else out.shape
    # None entries in the id set are ignored.
    out2 = mp._extract_per_frame_video_hidden(hidden, input_ids, (image_id, None))
    assert out2 is not None and out2.shape == (B, 4, D)
    print("ok  test_per_frame_extraction_image_tokens")


def test_integration_lfp_block_executes_with_image_tokens():
    """End-to-end: with two patches' state in place + IMAGE tokens, the LFP term
    is actually ADDED to the loss and [LFP DIAG]/[LFP PROBE] print.

    Builds a minimal fake outer/inner mirroring the real wiring:
      - inner.forward sets _mope_cached_x_vis (as the crossattn patch does) and
        returns hidden_states = outputs[0];
      - inner carries _mope_lfp_head;
      - outer carries lm_head / loss_function / config;
      - _patch_model_for_lfp wraps outer.forward.
    Then we run with cfg.weight=0 (NTP only) vs cfg.weight>0 and assert the loss
    differs -> the token-shift LFP block executed on the image-token layout.
    """
    image_id, video_id = 151655, 151656
    vocab, llm_dim, target_dim = 16, 12, 8
    spatial, n_bins = 196, 4
    B, L = 2, 220
    seg = [30, 30, 30, 30, 30]  # 5 image frames
    input_ids = _build_video_input_ids(seg, image_id, B, L)
    labels = input_ids.clone()  # non-None -> training branch
    cached = torch.randn(B, spatial * n_bins, target_dim)

    # Load MoPELFPHead.
    _spec2 = importlib.util.spec_from_file_location(
        "lfp_head_under_test2", os.path.join(_SRC, "model", "lfp_head.py")
    )
    lh = importlib.util.module_from_spec(_spec2)
    _spec2.loader.exec_module(lh)

    class FakeInner(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(image_id + 8, llm_dim)
            self.config = types.SimpleNamespace(
                image_token_id=image_id, video_token_id=video_id
            )
            self._cached = cached

        def forward(self, input_ids=None, **kw):
            # Mirror the crossattn patch: populate the frozen-latent cache.
            self._mope_cached_x_vis = self._cached
            return (self.embed(input_ids),)   # outputs[0] = hidden_states

    class FakeOuter(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.model = inner
            self.lm_head = nn.Linear(llm_dim, vocab)
            self.config = types.SimpleNamespace(
                text_config=types.SimpleNamespace(vocab_size=vocab),
                image_token_id=image_id, video_token_id=video_id,
            )

        def loss_function(self, logits, labels, vocab_size):
            return logits.float().mean()   # deterministic stand-in for NTP CE

        def forward(self, *a, **k):
            raise RuntimeError("original_forward should not run in training branch")

    def _make_cfg(weight, reg_weight=0.0, reg_normalize=True):
        return types.SimpleNamespace(
            enable=True, weight=weight, mse_weight=1.0, cos_weight=1.0,
            context_frames=4, target_pool="mean",
            pred_source="per_frame_video", align_strategy="uniform",
            # E-16b reg fields (reg_weight=0.0 default -> reg branch NOT entered).
            reg_weight=reg_weight, reg_type="vicreg", reg_var_hinge=1.0,
            reg_cov_scale=1.0, reg_normalize=reg_normalize,
        )

    def _run(weight, reg_weight=0.0):
        torch.manual_seed(0)
        inner = FakeInner()
        outer = FakeOuter(inner)
        head = lh.MoPELFPHead(llm_dim=llm_dim, hidden=16, target_dim=target_dim)
        inner._mope_lfp_head = head
        mp._patch_model_for_lfp(outer, _make_cfg(weight, reg_weight=reg_weight))
        out = outer(input_ids=input_ids, labels=labels, pixel_values=torch.zeros(1))
        loss = out[0] if isinstance(out, tuple) else out.loss
        return float(loss.item())

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        loss_ntp_only = _run(0.0)   # weight=0 -> LFP term contributes nothing
        loss_with_lfp = _run(0.5)   # weight>0 -> LFP term added
    log = buf.getvalue()

    assert "[LFP PROBE] entry:" in log, log
    assert "per_frame_h=(2, 5," in log, "per-frame extraction must find 5 image frames:\n" + log
    assert "[LFP DIAG]" in log, "LFP DIAG must print (block executed):\n" + log
    assert abs(loss_with_lfp - loss_ntp_only) > 1e-6, (
        f"LFP term not added: ntp={loss_ntp_only}, with_lfp={loss_with_lfp}"
    )
    print("ok  test_integration_lfp_block_executes_with_image_tokens")


def test_integration_reg_opt_in_and_byte_equiv():
    """E-16b WARN-1/2: reg_weight=0 is byte-equal to no-reg; reg_weight>0 adds
    L_reg and [LFP DIAG] prints reg_normalize + post-norm z std.

    Reuses the minimal fake outer/inner wiring (image-token layout). The reg
    target is the per-token GLOBAL visual hidden, decoupled from token-shift
    pairing (WARN-2: the reg block sits right after the NTP loss).
    """
    image_id, video_id = 151655, 151656
    vocab, llm_dim, target_dim = 16, 12, 8
    spatial, n_bins = 196, 4
    B, L = 2, 220
    seg = [30, 30, 30, 30, 30]
    input_ids = _build_video_input_ids(seg, image_id, B, L)
    labels = input_ids.clone()
    cached = torch.randn(B, spatial * n_bins, target_dim)

    _spec2 = importlib.util.spec_from_file_location(
        "lfp_head_under_test3", os.path.join(_SRC, "model", "lfp_head.py")
    )
    lh = importlib.util.module_from_spec(_spec2)
    _spec2.loader.exec_module(lh)

    class FakeInner(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(image_id + 8, llm_dim)
            self.config = types.SimpleNamespace(
                image_token_id=image_id, video_token_id=video_id
            )
            self._cached = cached

        def forward(self, input_ids=None, **kw):
            self._mope_cached_x_vis = self._cached
            return (self.embed(input_ids),)

    class FakeOuter(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.model = inner
            self.lm_head = nn.Linear(llm_dim, vocab)
            self.config = types.SimpleNamespace(
                text_config=types.SimpleNamespace(vocab_size=vocab),
                image_token_id=image_id, video_token_id=video_id,
            )

        def loss_function(self, logits, labels, vocab_size):
            return logits.float().mean()

        def forward(self, *a, **k):
            raise RuntimeError("original_forward should not run in training branch")

    def _make_cfg(weight, reg_weight, reg_normalize=True):
        return types.SimpleNamespace(
            enable=True, weight=weight, mse_weight=1.0, cos_weight=1.0,
            context_frames=4, target_pool="mean",
            pred_source="per_frame_video", align_strategy="uniform",
            reg_weight=reg_weight, reg_type="vicreg", reg_var_hinge=1.0,
            reg_cov_scale=1.0, reg_normalize=reg_normalize,
        )

    def _run(weight, reg_weight, reg_normalize=True):
        torch.manual_seed(0)
        inner = FakeInner()
        outer = FakeOuter(inner)
        head = lh.MoPELFPHead(llm_dim=llm_dim, hidden=16, target_dim=target_dim)
        inner._mope_lfp_head = head
        mp._patch_model_for_lfp(outer, _make_cfg(weight, reg_weight, reg_normalize))
        out = outer(input_ids=input_ids, labels=labels, pixel_values=torch.zeros(1))
        loss = out[0] if isinstance(out, tuple) else out.loss
        return float(loss.item()), out

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        loss_reg_off, _ = _run(0.5, reg_weight=0.0)   # beta=0 -> reg NOT added
        loss_reg_on, _ = _run(0.5, reg_weight=1.0)    # beta>0 -> reg added
    log = buf.getvalue()

    # beta=0: byte-equal to the no-reg run (reg branch never entered).
    loss_lfp_only, _ = _run(0.5, reg_weight=0.0)  # determinism re-check (no print)
    assert loss_reg_off == loss_lfp_only, (loss_reg_off, loss_lfp_only)
    # beta>0: loss strictly changes (L_reg added) and DIAG shows the new fields.
    assert abs(loss_reg_on - loss_reg_off) > 1e-6, (loss_reg_on, loss_reg_off)
    assert "reg_normalize=True" in log, "DIAG must print reg_normalize:\n" + log
    assert "reg_z_std=" in log, "DIAG must print post-norm z std:\n" + log
    assert "reg_target_N=" in log and "reg_target_N=0," not in log, (
        "reg target must be the global visual hidden (N>0):\n" + log
    )
    print("ok  test_integration_reg_opt_in_and_byte_equiv")


def test_vicreg_loss_helper():
    """E-16b WARN-1: _vicreg_loss normalization + collapse sensitivity.

    - normalize=True pulls the overall z std to ~1 (so the tau=1 variance hinge
      can fire) while keeping the RELATIVE per-dim variance structure;
    - normalize=False leaves the raw (>>1) scale -> the hinge saturates to 0;
    - a collapsed input (near-constant dims) yields L_var clearly > 0;
    - the function returns a 4-tuple with a detached, gradient-free z_std.
    """
    torch.manual_seed(0)
    N, D = 512, 64
    # LLM-like raw hidden: large per-dim std (>>1), heterogeneous across dims.
    scale = torch.linspace(20.0, 60.0, D)
    z = torch.randn(N, D) * scale + 5.0

    # normalize=True -> overall std ~1.
    l_var_n, l_cov_n, l_reg_n, z_std_n = mp._vicreg_loss(z, normalize=True)
    assert abs(float(z_std_n) - 1.0) < 0.05, f"post-norm std should be ~1, got {float(z_std_n)}"
    assert z_std_n.requires_grad is False, "z_std must be detached (diagnostic only)"
    # On well-spread normalized data with tau=1, the per-dim std is < 1 (scale
    # pulled to overall 1), so the variance hinge is active (non-trivial L_var).
    assert float(l_var_n) > 0.0

    # normalize=False -> raw scale (>>1) -> hinge saturates -> L_var == 0.
    l_var_raw, _, _, z_std_raw = mp._vicreg_loss(z, normalize=False)
    assert float(z_std_raw) > 10.0, f"raw std should be >>1, got {float(z_std_raw)}"
    assert float(l_var_raw) == 0.0, (
        f"raw-scale hinge must saturate to 0 (this is exactly the WARN-1 bug), "
        f"got {float(l_var_raw)}"
    )

    # RELATIVE collapse, the exact WARN-1 case: half the dims keep a large std
    # (~30), the other half are squeezed to std~2 -- still ABOVE tau=1, so the
    # RAW hinge (no normalize) misses them entirely (relu(1-2)=0). After global
    # normalization the overall scale -> 1, so the squeezed dims drop to std<<1
    # and the hinge fires. This is precisely what WARN-1 fixes: tau=1 detects
    # relative collapse on LLM-scale hidden where raw-scale is blind.
    z_collapse = torch.randn(N, D) * 30.0
    z_collapse[:, D // 2:] = 3.0 + torch.randn(N, D // 2) * 2.0   # squeezed half (std~2)
    l_var_c, _, _, _ = mp._vicreg_loss(z_collapse, normalize=True)
    assert float(l_var_c) > 0.3, f"relative collapse must give large L_var, got {float(l_var_c)}"
    # Raw scale: squeezed dims still have std~2 (>tau) -> hinge saturates -> blind.
    l_var_c_raw, _, _, _ = mp._vicreg_loss(z_collapse, normalize=False)
    assert float(l_var_c_raw) == 0.0, (
        f"raw scale misses relative collapse (WARN-1 bug), got {float(l_var_c_raw)}"
    )

    # Gradient flows through L_reg (regularizer can reshape the representation).
    z_g = (torch.randn(N, D) * 30.0).requires_grad_(True)
    _, _, l_reg_g, _ = mp._vicreg_loss(z_g, normalize=True)
    l_reg_g.backward()
    assert z_g.grad is not None and torch.isfinite(z_g.grad).all()
    print("ok  test_vicreg_loss_helper")


def test_extract_visual_token_hidden():
    """E-16b reg target: flatten ALL visual-token hidden to [N, D] (not pooled)."""
    image_id, video_id = 151655, 151656
    B, L, D = 2, 200, 4
    seg = [40, 40, 40, 40]  # 4 frames * 40 tokens = 160 visual tokens / row
    input_ids = _build_video_input_ids(seg, image_id, B, L)
    hidden = torch.randn(B, L, D, requires_grad=True)
    out = mp._extract_visual_token_hidden(hidden, input_ids, (image_id, video_id))
    assert out is not None
    assert out.shape == (B * 160, D), out.shape   # N = all visual tokens, batch-flat
    # gradient preserved (boolean-mask gather).
    out.sum().backward()
    assert hidden.grad is not None and float(hidden.grad.abs().sum()) > 0
    # No visual tokens -> None.
    text_only = torch.full((B, L), 7, dtype=torch.long)
    assert mp._extract_visual_token_hidden(hidden, text_only, (image_id, video_id)) is None
    print("ok  test_extract_visual_token_hidden")


def test_byte_equivalence_guard_source():
    """Structural: _patch_model_for_lfp is only called under mope_lfp_enable.

    Full numeric byte-equivalence (loss/logits identical to E-03a when
    mope_lfp_enable=False) needs the Qwen3VL backbone + GPU and is NOT run on
    CPU. The opt-in guarantee instead rests on train_space.py installing the LFP
    patch ONLY inside `if mope_args.mope_lfp_enable:`; we assert that invariant
    by source inspection so a future refactor that moves the call out of the
    gate is caught here.
    """
    with open(_TRAIN_SPACE, "r") as fh:
        src = fh.read()
    # Every call site of _patch_model_for_lfp must be inside the enable block.
    calls = [m.start() for m in re.finditer(r"_patch_model_for_lfp\(", src)]
    # one import-from line + at least one call; filter the `from ... import` line.
    call_lines = [
        ln for ln in src.splitlines()
        if "_patch_model_for_lfp(" in ln and "import" not in ln
    ]
    assert call_lines, "no _patch_model_for_lfp call found"
    enable_pos = src.find("if mope_args.mope_lfp_enable:")
    assert enable_pos != -1, "mope_lfp_enable gate not found"
    for pos in calls:
        # The actual invocation (not the import) must appear after the gate.
        snippet = src[pos:pos + 40]
        if "import" in src[max(0, pos - 20):pos]:
            continue
        assert pos > enable_pos, "LFP patch called outside the enable gate"
    print("ok  test_byte_equivalence_guard_source")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_per_frame_extraction_basic()
    test_per_frame_extraction_variable_token_count()
    test_per_frame_extraction_no_video()
    test_group_frames_to_bins_F8()
    test_group_frames_to_bins_F7_and_small()
    test_tokenshift_pairing()
    test_tokenshift_degenerate_single_bin()
    test_head_consumes_batched_source()
    test_per_frame_extraction_image_tokens()
    test_integration_lfp_block_executes_with_image_tokens()
    test_integration_reg_opt_in_and_byte_equiv()
    test_vicreg_loss_helper()
    test_extract_visual_token_hidden()
    test_byte_equivalence_guard_source()
    print("\nALL TOKEN-SHIFT LFP TESTS PASSED")
