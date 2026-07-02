"""CPU unit tests for the E-16e feed cross-attn causal-over-bins mask.

Covers the E-16e single-variable requirement (causal mask on the FEED
cross-attn, adapted from Video-CCAM get_ccam):
  - flag OFF (attn_mask=None): the projector output is byte-identical to the
    original bidirectional E-16 MoPEProjectorCrossAttn forward.
  - mask construction: _build_feed_causal_mask puts -inf on all K/V tokens of
    bins > b_f and 0 on bins <= b_f, with b_f = the faithful frame->bin mapping
    (== _group_llm_frames_to_bins base ranges; f//2 when F==8).
  - mask ON behaviour: after softmax the attention weight to every future-bin
    K/V token is ~0, and perturbing future-bin MoPE features leaves the masked
    output UNCHANGED (invariance) while it DOES change the unmasked output
    (bidirectional leakage — the exact thing E-16e removes).

CPU / tiny-tensor, no GPU, no model download.

Run:  python tests/test_feed_causal_mask.py   (or: pytest tests/test_feed_causal_mask.py)
"""

import importlib.util
import math
import os

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "src")
_MOPE_PATCH = os.path.join(_SRC, "model", "mope_patch.py")
_MOPE_PROJECTOR = os.path.join(_SRC, "model", "mope_projector.py")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mp = _load("mope_patch_under_test_e16e", _MOPE_PATCH)
mproj = _load("mope_projector_under_test_e16e", _MOPE_PROJECTOR)


def _make_projector(mope_dim=8, llm_dim=6, seed=0):
    """MoPEProjectorCrossAttn with a NON-zero out_proj (so the residual actually
    depends on the attention — the zero-init default would make every output
    equal image_embeds and hide any attention bug)."""
    torch.manual_seed(seed)
    p = mproj.MoPEProjectorCrossAttn(mope_dim=mope_dim, llm_dim=llm_dim).double()
    with torch.no_grad():
        p.out_proj.weight.normal_(std=0.5)
        p.out_proj.bias.normal_(std=0.5)
        # k/v/norm already randomly initialised; force a distinctive norm.
        p.norm.weight.normal_(mean=1.0, std=0.1)
        p.norm.bias.normal_(std=0.1)
    return p


def _reference_e16_forward(p, mope_feats, image_embeds):
    """Re-implement the ORIGINAL bidirectional E-16 forward exactly (no mask)."""
    x = p.norm(mope_feats)
    K = p.k_proj(x)
    V = p.v_proj(x)
    Q = image_embeds
    scale = Q.shape[-1] ** -0.5
    attn = torch.softmax(Q @ K.transpose(-2, -1) * scale, dim=-1)
    out = attn @ V
    out = p.out_proj(out)
    out = out.to(image_embeds.dtype)
    return image_embeds + out


def test_off_is_byte_equivalent_to_e16():
    """attn_mask=None (and not passing it at all) == original E-16 forward."""
    p = _make_projector()
    torch.manual_seed(1)
    mope_feats = torch.randn(1, 16, 8, dtype=torch.double)
    e = torch.randn(1, 5, 6, dtype=torch.double)

    ref = _reference_e16_forward(p, mope_feats, e)
    out_default = p(mope_feats, e)                       # attn_mask defaulted
    out_none = p(mope_feats, e, attn_mask=None)          # explicit None

    assert torch.equal(out_default, ref), (out_default - ref).abs().max()
    assert torch.equal(out_none, ref), (out_none - ref).abs().max()
    print("ok  test_off_is_byte_equivalent_to_e16")


def test_frame_to_bin_matches_group_llm_frames():
    """The mask's frame->bin assignment == _group_llm_frames_to_bins base ranges.

    Build the mask for spatial=1 (so kv_bin[j] == j and each 'bin' is one K/V
    token), then read frame f's allowed set: the max unmasked column is exactly
    b_f. Compare against the base ranges the LFP head uses.
    """
    for F, n_bins in [(8, 4), (6, 4), (3, 4), (5, 4), (8, 3)]:
        n_mope = n_bins  # spatial=1 -> one K/V token per bin -> n_mope == n_bins
        mask = mp._build_feed_causal_mask(F, n_mope, device="cpu",
                                          dtype=torch.double, spatial=1)
        assert mask.shape == (F, n_bins)
        # Faithful base ranges (identical formula to _group_llm_frames_to_bins).
        expected_bin = [0] * F
        for b in range(n_bins):
            s = (b * F) // n_bins
            e = ((b + 1) * F) // n_bins
            for f in range(s, e):
                expected_bin[f] = b
        for f in range(F):
            row = mask[f]
            visible = (row == 0.0)
            masked = torch.isinf(row) & (row < 0)
            # exactly bins <= b_f visible, bins > b_f masked.
            b_f = expected_bin[f]
            for j in range(n_bins):
                if j <= b_f:
                    assert bool(visible[j]), (F, n_bins, f, j)
                else:
                    assert bool(masked[j]), (F, n_bins, f, j)
        # F==8, n_bins==4 must reduce to the tubelet-2 rule b_f = f // 2.
        if F == 8 and n_bins == 4:
            assert expected_bin == [0, 0, 1, 1, 2, 2, 3, 3]
    print("ok  test_frame_to_bin_matches_group_llm_frames")


def test_mask_values_784_layout():
    """The real 784 = 4 bins x 196 layout: frame f -> bin f//2, kv j -> bin j//196."""
    F, n_mope, spatial = 8, 784, 196
    mask = mp._build_feed_causal_mask(F, n_mope, device="cpu",
                                      dtype=torch.double, spatial=spatial)
    assert mask.shape == (F, n_mope)
    for f in range(F):
        b_f = f // 2
        kv_bin = torch.arange(n_mope) // spatial
        for b in range(4):
            cols = (kv_bin == b)
            if b <= b_f:
                assert torch.all(mask[f, cols] == 0.0), (f, b)
            else:
                assert torch.all(torch.isinf(mask[f, cols])), (f, b)
    # No row is fully masked (bin 0 always visible -> softmax safe).
    assert torch.all((mask == 0.0).any(dim=1))
    print("ok  test_mask_values_784_layout")


def test_mask_on_zeros_future_bin_attention_weights():
    """With the mask, softmax attention weight to every future-bin K/V is ~0."""
    p = _make_projector()
    spatial, n_bins = 3, 4
    n_mope = spatial * n_bins            # 12
    F = 8
    torch.manual_seed(2)
    mope_feats = torch.randn(1, n_mope, 8, dtype=torch.double)
    e = torch.randn(1, 5, 6, dtype=torch.double)
    mask = mp._build_feed_causal_mask(F, n_mope, device="cpu",
                                      dtype=torch.double, spatial=spatial)
    kv_bin = torch.arange(n_mope) // spatial

    x = p.norm(mope_feats)
    K = p.k_proj(x)
    Q = e
    scale = Q.shape[-1] ** -0.5
    for f in range(F):
        b_f = f // 2
        scores = Q @ K.transpose(-2, -1) * scale + mask[f].view(1, 1, -1)
        attn = torch.softmax(scores, dim=-1)             # [1, N_img, n_mope]
        future_cols = kv_bin > b_f
        if future_cols.any():
            assert attn[:, :, future_cols].abs().max() < 1e-12, (f, b_f)
        # visible cols carry all the mass.
        vis_cols = kv_bin <= b_f
        assert torch.allclose(attn[:, :, vis_cols].sum(-1),
                              torch.ones(1, 5, dtype=torch.double))
    print("ok  test_mask_on_zeros_future_bin_attention_weights")


def test_masked_output_invariant_to_future_bins():
    """Perturbing FUTURE-bin MoPE features leaves the masked output UNCHANGED
    (E-16e no-leak), but CHANGES the unmasked E-16 output (the leak E-16e fixes).
    """
    p = _make_projector()
    spatial, n_bins = 3, 4
    n_mope = spatial * n_bins
    F = 8
    torch.manual_seed(3)
    mope_feats = torch.randn(1, n_mope, 8, dtype=torch.double)
    e = torch.randn(1, 5, 6, dtype=torch.double)
    mask = mp._build_feed_causal_mask(F, n_mope, device="cpu",
                                      dtype=torch.double, spatial=spatial)

    # Pick an early frame (f=1 -> bin 0): bins 1,2,3 are its future.
    f = 1
    b_f = f // 2
    kv_bin = torch.arange(n_mope) // spatial
    future = kv_bin > b_f

    # Perturb only the future-bin K/V rows.
    mope_pert = mope_feats.clone()
    mope_pert[:, future, :] += 10.0 * torch.randn_like(mope_pert[:, future, :])

    mask_f = mask[f].view(1, 1, -1)
    out_masked = p(mope_feats, e, attn_mask=mask_f)
    out_masked_pert = p(mope_pert, e, attn_mask=mask_f)
    assert torch.allclose(out_masked, out_masked_pert, atol=1e-10), \
        "masked output must be invariant to future-bin features"

    # Same perturbation WITHOUT the mask (E-16 bidirectional) DOES change output.
    out_bidir = p(mope_feats, e)
    out_bidir_pert = p(mope_pert, e)
    assert not torch.allclose(out_bidir, out_bidir_pert, atol=1e-6), \
        "bidirectional (no-mask) output should leak future-bin features"
    print("ok  test_masked_output_invariant_to_future_bins")


def test_build_mask_degenerate_layout():
    """n_mope < spatial -> < 1 bin -> None (mask defensively disabled)."""
    assert mp._build_feed_causal_mask(8, 100, device="cpu",
                                      dtype=torch.double, spatial=196) is None
    print("ok  test_build_mask_degenerate_layout")


# ---------------------------------------------------------------------------
# EVAL-case tests: the per-QUERY-TOKEN construction (_build_feed_causal_mask_tokens)
# that unifies the train (T=1 per shard) and eval (single T=F <video> shard) paths.
# ---------------------------------------------------------------------------

def test_tokens_mask_reduces_to_per_frame_when_T1():
    """TRAIN layout: with frames_in_shard=1, each shard's per-token mask must be
    the SAME single row as _build_feed_causal_mask for that global frame,
    replicated across the shard's spatial query tokens. This proves the unified
    per-token path is byte-identical to the old per-frame-row path at train."""
    F, n_mope, spatial = 8, 784, 196
    per_frame = mp._build_feed_causal_mask(F, n_mope, device="cpu",
                                           dtype=torch.double, spatial=spatial)
    S = 5   # spatial query tokens per LLM frame (arbitrary)
    for f in range(F):
        m = mp._build_feed_causal_mask_tokens(
            n_tokens=S, frames_in_shard=1, frame_offset=f, total_frames=F,
            n_mope=n_mope, device="cpu", dtype=torch.double, spatial=spatial,
        )
        assert m.shape == (S, n_mope)
        # every one of the S query tokens equals per-frame row f.
        for k in range(S):
            assert torch.equal(m[k], per_frame[f]), (f, k)
    print("ok  test_tokens_mask_reduces_to_per_frame_when_T1")


def test_eval_single_shard_per_token_mask_is_causal_not_inert():
    """EVAL layout: ONE shard carrying T=4 frames x S spatial tokens must produce
    a per-token mask that is (a) NOT all-zero (the inert-mask bug) and (b) causal:
    tokens of frame f get -inf on K/V bins > b_f and 0 on bins <= b_f."""
    T, S = 4, 6                     # 4 frames, 6 spatial tokens/frame -> 24 query tokens
    spatial, n_bins = 3, 4          # K/V: 3 tokens per bin, 4 bins
    n_mope = spatial * n_bins       # 12
    n_tokens = T * S
    m = mp._build_feed_causal_mask_tokens(
        n_tokens=n_tokens, frames_in_shard=T, frame_offset=0, total_frames=T,
        n_mope=n_mope, device="cpu", dtype=torch.double, spatial=spatial,
    )
    assert m.shape == (n_tokens, n_mope)
    # NOT inert: at least one -inf must be present (else eval == bidirectional).
    assert torch.isinf(m).any(), "eval per-token mask must NOT be all-zero (inert)"

    kv_bin = torch.arange(n_mope) // spatial
    # frame f -> bin b_f via the same base ranges (F==n_bins==4 -> b_f == f).
    expected_bin = [0] * T
    for b in range(n_bins):
        s = (b * T) // n_bins
        e = ((b + 1) * T) // n_bins
        for f in range(s, e):
            expected_bin[f] = b
    assert expected_bin == [0, 1, 2, 3]
    for k in range(n_tokens):
        f = k // S                       # frame-major ordering
        b_f = expected_bin[f]
        row = m[k]
        for j in range(n_mope):
            if kv_bin[j].item() <= b_f:
                assert row[j] == 0.0, (k, f, j)
            else:
                assert torch.isinf(row[j]) and row[j] < 0, (k, f, j)
    print("ok  test_eval_single_shard_per_token_mask_is_causal_not_inert")


def test_eval_single_shard_matches_per_frame_semantics():
    """The eval single-shard per-token mask must equal, row-by-row, the per-frame
    table row of the token's frame — i.e. splitting a video into T image shards
    (train) vs. feeding it as ONE T-frame shard (eval) yield the SAME mask, so an
    E-16e checkpoint sees the SAME causal feed in both regimes."""
    T, S = 4, 3
    spatial, n_bins = 196, 4
    n_mope = spatial * n_bins
    per_frame = mp._build_feed_causal_mask(T, n_mope, device="cpu",
                                           dtype=torch.double, spatial=spatial)
    m = mp._build_feed_causal_mask_tokens(
        n_tokens=T * S, frames_in_shard=T, frame_offset=0, total_frames=T,
        n_mope=n_mope, device="cpu", dtype=torch.double, spatial=spatial,
    )
    for k in range(T * S):
        f = k // S
        assert torch.equal(m[k], per_frame[f]), (k, f)
    print("ok  test_eval_single_shard_matches_per_frame_semantics")


def test_single_frame_sample_mask_is_inert_guard_condition():
    """The inert-mask guard condition: total_frames==1 with n_bins>1 -> the single
    frame maps to the LAST bin -> the mask masks NOTHING (no -inf). This is exactly
    what _feed_mask_diag warns about; assert the construction indeed yields an
    all-zero mask so the guard's `not any_masked` trigger is well-founded."""
    spatial, n_bins = 196, 4
    n_mope = spatial * n_bins
    m = mp._build_feed_causal_mask_tokens(
        n_tokens=50, frames_in_shard=1, frame_offset=0, total_frames=1,
        n_mope=n_mope, device="cpu", dtype=torch.double, spatial=spatial,
    )
    assert not torch.isinf(m).any(), "single-frame sample -> inert (all-zero) mask"
    assert torch.all(m == 0.0)
    print("ok  test_single_frame_sample_mask_is_inert_guard_condition")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_off_is_byte_equivalent_to_e16()
    test_frame_to_bin_matches_group_llm_frames()
    test_mask_values_784_layout()
    test_mask_on_zeros_future_bin_attention_weights()
    test_masked_output_invariant_to_future_bins()
    test_build_mask_degenerate_layout()
    test_tokens_mask_reduces_to_per_frame_when_T1()
    test_eval_single_shard_per_token_mask_is_causal_not_inert()
    test_eval_single_shard_matches_per_frame_semantics()
    test_single_frame_sample_mask_is_inert_guard_condition()
    print("\nALL E-16e FEED-CAUSAL MASK TESTS PASSED")
