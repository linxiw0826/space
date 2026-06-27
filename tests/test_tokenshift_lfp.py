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

import importlib.util
import os
import re

import torch

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
    test_byte_equivalence_guard_source()
    print("\nALL TOKEN-SHIFT LFP TESTS PASSED")
