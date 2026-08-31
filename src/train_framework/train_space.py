"""
Space Sensing Training Entry Point.

Forked from:
  refs/Let_Geometry_GUIDE/qwen-vl-finetune/qwenvl/train/train_qwen.py

Changes vs. upstream GUIDE:
  1. Imports MoPEArguments in addition to the base GUIDE argument dataclasses.
  2. After model loading (and optional LORA setup), attaches MoPEEncoder and
     MoPEProjector to the model when --use_mope is set.
  3. Patches model.model.forward() to inject MoPE embeddings into image/video
     tokens after the GUIDE geometry fusion step.

MoPE Fusion Design:
  - We wrap get_image_features (Qwen3VLModel.get_image_features) to inject a
    per-clip scene-prior bias before GUIDE geometry fusion.
  - mope_bias = mope_projector(mope_encoder(frames)).squeeze(1)  # [1, llm_dim]
  - image_embeds_list = [e + mope_bias for e in image_embeds_list]  # broadcast
  - Mathematically equivalent to post-geometry fusion (addition is commutative):
        (raw_visual + mope_bias) + geo == raw_visual + mope_bias + geo
  - MoPE contributes a per-clip bias that is uniform across all visual tokens,
    analogous to a clip-level scene prior.

Fusion insertion point (wraps get_image_features in model.model.forward):
  File: refs/.../qwenvl/model/modeling_qwen3_vl.py
  Method: Qwen3VLModel.forward()  (line ~1744 calls get_image_features)
  At: self.get_image_features(pixel_values, image_grid_thw)
  Before: torch.cat + _process_geometry_features

  The patch replaces self.get_image_features with a wrapper that calls the
  original, adds mope_bias to each element of image_embeds_list, and returns
  the modified (image_embeds_list, deepstack) unchanged in structure.

  Because modifying refs/ is forbidden, we use runtime forward-method
  monkey-patching after model creation (see src/model/mope_patch.py).
"""

# === E-10 NCCL hard-pin (must run before any torch/deepspeed import) ===========
# A dependency import can otherwise ctypes-load the stray system libnccl.so.2
# (2.28.9, CUDA-13 build) first, which then satisfies torch's NCCL symbols and
# crashes multi-GPU DeepSpeed broadcast with "driver insufficient". LD_PRELOAD
# can't fix an already-dlopen'd lib, so we dlopen the pip 2.21.5 FIRST (RTLD_GLOBAL).
import os as _os, sys as _sys, glob as _glob, ctypes as _ctypes
for _cand in (
    _glob.glob(_os.path.join(_sys.prefix, "lib", "python*", "site-packages",
                             "nvidia", "nccl", "lib", "libnccl.so.2"))
    or ["/data1/miniconda3/envs/space/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2"]
):
    if _os.path.isfile(_cand):
        try:
            _ctypes.CDLL(_cand, mode=_ctypes.RTLD_GLOBAL)
            print(f"[E10-nccl-pin] preloaded {_cand}", flush=True)
        except OSError as _e:
            print(f"[E10-nccl-pin] WARN could not preload {_cand}: {_e}", flush=True)
        break
# ==============================================================================

import os
import logging
import pathlib
import torch
import transformers
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: ensure GUIDE qwenvl package is importable when this script is
# run from /u/lwu9/Space_sensing/projects/space/src/.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent

# qwenvl lives in src/qwenvl, added via _SRC_ROOT below

# Space Sensing src root (for src.model.mope_encoder / mope_projector)
_SRC_ROOT = _THIS_DIR.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

# ---------------------------------------------------------------------------
# Imports from GUIDE (unchanged)
# ---------------------------------------------------------------------------
from qwenvl.train.trainer import replace_qwen2_vl_attention_class  # noqa: E402
import qwenvl.train.sampler  # noqa: F401, E402

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)
from qwenvl.model.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from qwenvl.data.data_processor import make_supervised_data_module
from transformers import AutoProcessor, Trainer, AutoConfig
from transformers.trainer_utils import get_last_checkpoint

# ---------------------------------------------------------------------------
# Imports from Space Sensing (new)
# ---------------------------------------------------------------------------
from src.train_framework.argument import (
    ModelArguments,
    MoPEArguments,
    DataArguments,
    TrainingArguments,
)
from src.train_framework.checkpoint_rotation import (
    is_complete_checkpoint,
    predelete_for_two_slot_rotation,
)

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


# ---------------------------------------------------------------------------
# E-10 (Router v1) gate process logging callback.
#
# Zero side effects when the gate is disabled: train() only adds this callback
# when mope_use_gate is true; even if added, the callback no-ops if it cannot
# find a projector with a _gate_buf / use_gate. It never all-gathers (rank0 uses
# only its LOCAL buffer), never touches model numerics, and clears _gate_buf on
# every rank to keep memory bounded.
# ---------------------------------------------------------------------------
class GateStatsCallback(transformers.TrainerCallback):
    """Periodically logs windowed statistics of the E-10 content-driven gate g.

    The MoPEProjectorCrossAttn.forward stashes ``g.detach().float()`` into
    ``projector._gate_buf`` on every training forward.  This callback drains
    that buffer at a fixed step interval and prints one summary line via
    ``rank0_print`` (rank0 only, into the tee'd LOG_FILE):

        [E10-gate] step=<s> g_mean=.. g_std=.. g_min=.. g_max=.. n=<n> gate_grad_norm=..

    ``gate_grad_norm`` is the L2 norm of the gate-MLP parameter gradients,
    captured best-effort in ``on_pre_optimizer_step`` (grads still live there);
    falls back to -1.0 when no grads are available.
    """

    def __init__(self, gate_log_interval: int = 25, projector=None):
        self.gate_log_interval = max(1, int(gate_log_interval))
        # Resolve the projector module; no-op if it has no usable gate buffer.
        if projector is None or not hasattr(projector, "_gate_buf") \
                or not getattr(projector, "use_gate", False):
            self._projector = None
        else:
            self._projector = projector
        self._last_gate_gnorm = -1.0

    def _gate_params(self):
        """Yield the gate-MLP parameters (empty if no gate)."""
        proj = self._projector
        if proj is None or not hasattr(proj, "gate_mlp"):
            return []
        return list(proj.gate_mlp.parameters())

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        # Grads are still populated here. Compute sum-of-squares L2 norm over the
        # gate-MLP params; best-effort — any param with grad=None is skipped, and
        # if none have grads we record -1.0. All local (no cross-rank comm).
        if self._projector is None:
            return
        sq = 0.0
        seen = False
        for p in self._gate_params():
            g = p.grad
            if g is None:
                continue
            seen = True
            sq += float(g.detach().float().pow(2).sum().item())
        self._last_gate_gnorm = (sq ** 0.5) if seen else -1.0

    def on_step_end(self, args, state, control, **kwargs):
        if self._projector is None:
            return
        gs = state.global_step
        if not (gs < 3 or gs % self.gate_log_interval == 0):
            return
        buf = self._projector._gate_buf
        # Always clear (every rank) to bound memory — independent of rank0.
        if len(buf) == 0:
            # Nothing accumulated this window; skip printing (no error).
            return
        try:
            cat = torch.cat(buf)
            n = int(cat.numel())
            g_mean = cat.mean().item()
            g_std = cat.std().item() if n > 1 else 0.0
            g_min = cat.min().item()
            g_max = cat.max().item()
        finally:
            buf.clear()
        if local_rank == 0:
            rank0_print(
                f"[E10-gate] step={gs} "
                f"g_mean={g_mean:.4f} g_std={g_std:.4f} "
                f"g_min={g_min:.4f} g_max={g_max:.4f} "
                f"n={n} gate_grad_norm={self._last_gate_gnorm:.4f}"
            )


# ---------------------------------------------------------------------------
# E-10b (Router v1.1) rich gate diagnostic callback.
#
# Goal (change B): one `[gate-diag]` line every mope_gate_diag_every steps that
# captures everything needed to localise gate collapse in a single training run
# (so v2/v3 do not have to be re-run). Reads ONLY local projector buffers /
# trainer snapshot; never all-gathers, never touches numerics. All buffers it
# reads are plain Python lists (invisible to state_dict / optimizer / FSDP).
#
# Fields per line (all distinguish Stage1/Stage2 via the `stage` tag, derived
# from whether the LLM is trainable):
#   g distribution : g_mean g_std g_min g_max + histogram bins
#   gate logit     : logit_mean logit_absmean (saturation monitor)
#   gate weights   : w0_norm wlast_norm blast (content-sensitivity = is the
#                    INPUT weight still alive — the v1 collapse smoking gun)
#   gate grad      : gate_grad_norm (≈0 => gate not learning). E-10b v2.1: this
#                    is the LIVE-GRAPH logit grad norm d(loss)/d(gate_logits),
#                    captured by a backward hook in _gate_anticollapse_loss
#                    (reliable under DeepSpeed ZeRO-2, unlike the param .grad
#                    which is reduced/unavailable and gave the -1 sentinel).
#   loss decomp    : task_loss H_marg H_cond MI MI_w lambda_t l_z(raw/weighted).
#                    E-10b v2.1 uses a mutual-information objective (marginal
#                    entropy − mean per-sample entropy, RIM/IMSAT); z-loss is OFF
#                    by default (coef=0) so l_z_w=0 unless explicitly enabled.
#   residual ratio : mean(‖g·mope_out‖/‖image_embeds‖) (branch influence)
#   task-label split: SKIPPED — no static/dynamic label is plumbed into the
#                    forward (would require label pass-through, which we avoid
#                    to keep the gate content-driven, D-15 b). Logged as
#                    `axis_split=skip(no_task_labels)`.
# ---------------------------------------------------------------------------
class GateDiagCallback(transformers.TrainerCallback):
    """E-10b: rich per-window gate diagnostics into the tee'd LOG_FILE (rank0)."""

    _HIST_EDGES = (0.0, 0.1, 0.3, 0.7, 0.9, 1.0)

    def __init__(self, diag_every: int = 20, projector=None, trainer_ref=None):
        self.diag_every = max(1, int(diag_every))
        if projector is None or not getattr(projector, "use_gate", False):
            self._projector = None
        else:
            self._projector = projector
        self._trainer_ref = trainer_ref      # set after Trainer construction
        self._stage = "?"

    def set_trainer(self, trainer_ref):
        self._trainer_ref = trainer_ref

    # E-10b v2.1: the gate_grad_norm printed in [gate-diag] is now the live-graph
    # logit grad norm captured by the backward hook in _gate_anticollapse_loss and
    # read off the trainer (trainer._last_gate_logit_gnorm). The old
    # on_pre_optimizer_step param-grad sampler was removed because under DeepSpeed
    # ZeRO-2 the gate_mlp param .grad is reduced/unavailable at that callback,
    # which produced the misleading gate_grad_norm=-1 sentinel.

    def on_train_begin(self, args, state, control, **kwargs):
        # Stage tag: Stage 2 trains the LLM (warm-start joint); Stage 1 does not.
        model = kwargs.get("model", None)
        try:
            llm_trainable = any(
                p.requires_grad for p in model.language_model.parameters()
            )
            self._stage = "stage2" if llm_trainable else "stage1"
        except Exception:
            self._stage = "?"

    @staticmethod
    def _histogram(vals):
        edges = GateDiagCallback._HIST_EDGES
        bins = [0] * (len(edges) - 1)
        n = len(vals)
        for v in vals:
            for j in range(len(edges) - 1):
                hi = edges[j + 1]
                # last bin is closed on the right so g==1.0 lands in [0.9,1.0].
                if v < hi or (j == len(edges) - 2 and v <= hi):
                    bins[j] += 1
                    break
        if n == 0:
            return "0/0/0/0/0"
        return "/".join(f"{b / n:.2f}" for b in bins)

    def on_step_end(self, args, state, control, **kwargs):
        proj = self._projector
        if proj is None:
            return
        gs = state.global_step
        if not (gs < 3 or gs % self.diag_every == 0):
            # Still drain the projector diag buffers to bound memory.
            proj._gate_buf_diag.clear()
            proj._gate_logit_buf.clear()
            proj._resid_ratio_buf.clear()
            return

        # Read this callback's OWN g window (_gate_buf_diag), independent of the
        # _gate_buf that GateStatsCallback drains — so a colliding step never
        # robs this line of its data.
        gbuf = proj._gate_buf_diag
        lbuf = proj._gate_logit_buf
        rbuf = proj._resid_ratio_buf
        if len(gbuf) == 0:
            return
        try:
            gcat = torch.cat(gbuf)
            n = int(gcat.numel())
            g_mean = gcat.mean().item()
            g_std = gcat.std().item() if n > 1 else 0.0
            g_min = gcat.min().item()
            g_max = gcat.max().item()
            hist = self._histogram(gcat.tolist())
            if len(lbuf) > 0:
                lcat = torch.cat(lbuf)
                logit_mean = lcat.mean().item()
                logit_absmean = lcat.abs().mean().item()
            else:
                logit_mean = logit_absmean = float("nan")
            resid_ratio = (sum(rbuf) / len(rbuf)) if rbuf else float("nan")
        finally:
            gbuf.clear()
            lbuf.clear()
            rbuf.clear()

        # gate_mlp weight norms (content-sensitivity smoking gun).
        try:
            w0 = float(proj.gate_mlp[0].weight.detach().float().norm().item())
            wlast = float(proj.gate_mlp[-1].weight.detach().float().norm().item())
            blast = float(proj.gate_mlp[-1].bias.detach().float().item())
        except Exception:
            w0 = wlast = blast = float("nan")

        # Loss decomposition snapshot from the trainer (most recent micro-batch).
        gd = {}
        gate_grad_norm = -1.0
        if self._trainer_ref is not None:
            gd = getattr(self._trainer_ref, "_gate_diag", {}) or {}
            # E-10b v2.1: real gate-logit grad norm captured by the live-graph
            # backward hook in _gate_anticollapse_loss (logit grad calibre, see
            # note there). Falls back to -1 sentinel if no backward seen yet.
            gate_grad_norm = float(
                getattr(self._trainer_ref, "_last_gate_logit_gnorm", -1.0)
            )
        task_loss = gd.get("task_loss", float("nan"))
        l_z_raw = gd.get("l_z_raw", float("nan"))
        l_z_w = gd.get("l_z_weighted", float("nan"))
        H_marg = gd.get("H_marg", float("nan"))
        H_cond = gd.get("H_cond", float("nan"))
        MI = gd.get("MI", float("nan"))
        MI_w = gd.get("MI_weighted", float("nan"))
        lam_t = gd.get("lambda_t", float("nan"))

        if local_rank == 0:
            rank0_print(
                f"[gate-diag] stage={self._stage} step={gs} "
                f"g_mean={g_mean:.4f} g_std={g_std:.4f} g_min={g_min:.4f} g_max={g_max:.4f} "
                f"g_hist[0,.1,.3,.7,.9,1]={hist} n={n} "
                f"logit_mean={logit_mean:.4f} logit_absmean={logit_absmean:.4f} "
                f"w0_norm={w0:.4f} wlast_norm={wlast:.4f} blast={blast:.4f} "
                f"gate_grad_norm={gate_grad_norm:.6f} "
                f"task_loss={task_loss:.4f} "
                f"H_marg={H_marg:.4f} H_cond={H_cond:.4f} MI={MI:.4f} "
                f"MI_w={MI_w:.6f} lambda_t={lam_t:.6f} "
                f"l_z_raw={l_z_raw:.6f} l_z_w={l_z_w:.6f} "
                f"resid_ratio={resid_ratio:.6f} "
                f"axis_split=skip(no_task_labels)"
            )


# ---------------------------------------------------------------------------
# Helpers (verbatim from GUIDE)
# ---------------------------------------------------------------------------

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


# ---------------------------------------------------------------------------
# MoPE integration helpers (new)
# ---------------------------------------------------------------------------

from model.mope_patch import (  # noqa: E402
    _patch_model_for_mope,
    _patch_model_for_mope_concat,
    _patch_model_for_mope_crossattn,
    _patch_model_for_mope_qformer,
)


def _attach_mope_to_model(model, mope_args: MoPEArguments):
    """Instantiate MoPEEncoder + MoPEProjector and attach them to ``model``.

    Encoder is frozen; projector is trainable.

    We attach the modules to ``model.model`` (the inner
    Qwen3VLModel / Qwen2_5_VLModel) so that they are accessible inside the
    patched forward method as ``self._mope_encoder`` / ``self._mope_projector``.
    """
    from model.mope_projector import MoPEProjector

    inner_model = model.model  # Qwen3VLModel / Qwen2_5_VLModel

    # Encoder backbone selection (paper-2 D-26/D-27 drop-in). Default 'mope' is
    # byte-for-byte identical to every prior experiment; 'videomae' swaps in a
    # frozen plain VideoMAE ViT-B stand-in (E-03a_vmae / E-16_vmae). Both emit
    # [B,784,768] time-major latents, so nothing downstream changes.
    _encoder_type = getattr(mope_args, "mope_encoder_type", "mope")
    if _encoder_type == "videomae":
        from model.mope_encoder import VideoMAEEncoder
        rank0_print("[Space Sensing] Attaching VideoMAEEncoder (frozen, drop-in) ...")
        encoder = VideoMAEEncoder(
            checkpoint_path=mope_args.mope_checkpoint_path,
            all_frames=mope_args.mope_all_frames,
        )
    elif _encoder_type == "mope":
        from model.mope_encoder import MoPEEncoder
        rank0_print("[Space Sensing] Attaching MoPEEncoder (frozen) ...")
        encoder = MoPEEncoder(
            checkpoint_path=mope_args.mope_checkpoint_path,
            all_frames=mope_args.mope_all_frames,
        )
    else:
        raise ValueError(
            f"Unknown --mope_encoder_type '{_encoder_type}'. "
            f"Must be 'mope' (default) or 'videomae'."
        )
    # Register as sub-modules so they participate in .to(device), .parameters(), etc.
    inner_model.add_module("_mope_encoder", encoder)

    rank0_print("[Space Sensing] Attaching MoPEProjector (trainable) ...")
    # Qwen3VLConfig stores LLM hidden_size under text_config; Qwen2.5-VL puts it
    # directly on config.  Fall back gracefully so both model families work.
    _cfg = model.config
    _llm_hidden = getattr(_cfg, "hidden_size", None) or getattr(
        getattr(_cfg, "text_config", None), "hidden_size", None
    )
    if _llm_hidden is None:
        raise AttributeError(
            "Cannot determine LLM hidden_size from model.config. "
            "Set --mope_llm_dim explicitly."
        )
    llm_dim = _llm_hidden if mope_args.mope_llm_dim <= 0 else mope_args.mope_llm_dim
    rank0_print(
        f"[Space Sensing] MoPE llm_dim = {llm_dim} "
        f"({'auto from config' if mope_args.mope_llm_dim <= 0 else 'manual'}), "
        f"fusion_mode = {mope_args.mope_fusion_mode}"
    )
    if mope_args.mope_fusion_mode == "concat":
        from model.mope_projector import MoPEProjectorConcat
        projector = MoPEProjectorConcat(mope_dim=768, llm_dim=llm_dim)
    elif mope_args.mope_fusion_mode == "crossattn":
        from model.mope_projector import MoPEProjectorCrossAttn
        # E-10 (Router v1): when --mope_use_gate is set, instantiate the projector
        # with the learned content-driven gate. The gate submodule lives inside the
        # projector, so it is covered by `for p in projector.parameters(): requires_grad
        # = True` below and by --freeze_mope_projector handling later, giving correct
        # trainability in both 2-stage stages with no extra parameter group.
        # PENDING[D-15]: gate_mode "oracle_task" (真值静/动标签硬门控 = E-10-oracle
        # 路由上界, D-15 a) 留第二步; 本任务只实现 learned 主路径.
        # E-10b A1: gate_init_bias defaults to 0.0 (g≈0.5, non-saturated) so the
        # gate starts in its maximal-slope regime instead of the E-10 saturated
        # +4.0 (g≈0.98). E-10 behaviour is recoverable with --mope_gate_init_bias 4.0.
        # E-16t: enable the Cosmos-style per-bin temporal PE on the feed K/V when
        # --mope_feed_temporal_pe is set. Default False -> no module, byte-for-byte E-16.
        projector = MoPEProjectorCrossAttn(
            mope_dim=768,
            llm_dim=llm_dim,
            use_gate=mope_args.mope_use_gate,
            gate_mode=mope_args.mope_gate_mode,
            gate_init_bias=mope_args.mope_gate_init_bias,
            gate_lastw_std=mope_args.mope_gate_lastw_std,
            use_temporal_pe=getattr(mope_args, "mope_feed_temporal_pe", False),
        )
    elif mope_args.mope_fusion_mode == "qformer":
        from model.mope_projector import MoPEProjectorQFormer
        projector = MoPEProjectorQFormer(
            mope_dim=768,
            llm_dim=llm_dim,
            num_queries=mope_args.mope_qformer_num_queries,
        )
    else:
        projector = MoPEProjector(mope_dim=768, llm_dim=llm_dim)
    inner_model.add_module("_mope_projector", projector)

    # E-10b: enable the gate anti-collapse loss + rich training diagnostics on the
    # projector only for the E-10b run (--mope_gate_anticollapse). Default False
    # => E-10's training forward is byte-for-byte unchanged (no diagnostic buffers
    # populated, no z-loss graph buffer). Safe on non-crossattn projectors too
    # (they have no use_gate; the flags are simply unused).
    if mope_args.mope_use_gate and mope_args.mope_gate_anticollapse \
            and hasattr(projector, "_diag_enabled"):
        projector._diag_enabled = True
        projector._anticollapse_enabled = True
        rank0_print(
            "[Space Sensing] E-10b gate anti-collapse + diagnostics ENABLED on projector."
        )

    # Ensure projector is trainable (encoder is already frozen internally).
    for p in projector.parameters():
        p.requires_grad = True

    # -----------------------------------------------------------------
    # paper-2 Stage 1: LFP auxiliary head (implements D-12).
    # Only built when mope_lfp_enable=True (opt-in). When False the head is
    # never constructed and the training path is byte-for-byte E-03a. The
    # head lives on model.model as _mope_lfp_head (mirrors _mope_projector)
    # so it participates in .to(device)/.parameters()/state_dict/FSDP.
    # Trainability is managed jointly with --freeze_mope_projector later in
    # the trainability block (both 2-stage stages train it, aligned with the
    # projector). The head has its OWN decoupled input projection so the
    # "prediction-only" arm (mope_feed_features=False) is strictly orthogonal
    # to the cross-attn projector (risk-6).
    # -----------------------------------------------------------------
    if mope_args.mope_lfp_enable:
        from model.lfp_head import MoPELFPHead
        lfp_head = MoPELFPHead(
            llm_dim=llm_dim,
            hidden=mope_args.mope_lfp_hidden,
            target_dim=mope_args.mope_lfp_target_dim,
        )
        inner_model.add_module("_mope_lfp_head", lfp_head)
        for p in lfp_head.parameters():
            p.requires_grad = True
        rank0_print(
            f"[Space Sensing] LFP auxiliary head attached "
            f"(implements D-12; hidden={mope_args.mope_lfp_hidden}, "
            f"target_dim={mope_args.mope_lfp_target_dim}, "
            f"weight={mope_args.mope_lfp_weight}, "
            f"context_frames={mope_args.mope_lfp_context_frames}, "
            f"target_pool={mope_args.mope_lfp_target_pool}, "
            f"pred_source={mope_args.mope_lfp_pred_source})."
        )

    rank0_print(
        f"[Space Sensing] MoPE attached: encoder frozen, "
        f"projector dim 768 -> {llm_dim} (trainable)"
    )


def _load_mope_weights_from_checkpoint(inner_model, ckpt_dir: str) -> bool:
    """Load _mope_encoder / _mope_projector weights from a Stage 1 checkpoint.

    Checkpoint keys are formatted as ``model._mope_encoder.*`` and
    ``model._mope_projector.*``.  When loading into *inner_model* (the
    Qwen3VLModel / Qwen2_5_VLModel) the ``model.`` prefix must be stripped.

    Supports both safetensors (including sharded) and pytorch_model*.bin.
    Returns True if at least one MoPE tensor was loaded, False otherwise.
    """
    ckpt_path = Path(ckpt_dir)
    state_dict: dict = {}

    # Try safetensors first (preferred, used by HF save_pretrained).
    try:
        from safetensors.torch import load_file as _st_load_file

        shards = sorted(ckpt_path.glob("*.safetensors"))
        for shard in shards:
            shard_data = _st_load_file(str(shard))
            for k, v in shard_data.items():
                # Also pick up the paper-2 Stage 1 LFP head keys when present
                # (implements D-12) so a Stage-1 LFP checkpoint warm-starts the
                # head in Stage 2. Checkpoints without LFP keys (the default,
                # e.g. E-00b Stage-1) simply contribute none here.
                if (
                    k.startswith("model._mope_encoder.")
                    or k.startswith("model._mope_projector.")
                    or k.startswith("model._mope_lfp_head.")
                ):
                    state_dict[k[len("model."):]] = v
    except ImportError:
        pass

    # Fall back to PyTorch .bin files.
    if not state_dict:
        bin_files = sorted(ckpt_path.glob("pytorch_model*.bin"))
        for bin_file in bin_files:
            bin_data = torch.load(str(bin_file), map_location="cpu", weights_only=True)
            for k, v in bin_data.items():
                if (
                    k.startswith("model._mope_encoder.")
                    or k.startswith("model._mope_projector.")
                    or k.startswith("model._mope_lfp_head.")
                ):
                    state_dict[k[len("model."):]] = v

    if not state_dict:
        rank0_print(
            f"[Space Sensing] WARNING: no MoPE keys found in checkpoint at {ckpt_path}"
        )
        return False

    msg = inner_model.load_state_dict(state_dict, strict=False)
    # Explicitly report the projector keys actually loaded so the warm-gate probe
    # can confirm at runtime that out_proj (the warm dynamic-branch weight, e.g.
    # E-03a ~1.147) really came in — not just the encoder. Shard layout is handled
    # by the *.safetensors glob loop above (every shard is read + prefix-filtered),
    # so projector tensors are found no matter which shard holds them.
    _proj_keys = sorted(k for k in state_dict if k.startswith("_mope_projector."))
    rank0_print(
        f"[Space Sensing] Loaded {len(state_dict)} MoPE tensors from checkpoint "
        f"at {ckpt_path} ({len(_proj_keys)} projector keys): {_proj_keys}.  "
        f"Missing keys (expected): {msg.missing_keys}"
    )
    return True


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, MoPEArguments, DataArguments, TrainingArguments)
    )
    model_args, mope_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load config (verbatim from GUIDE)
    # ------------------------------------------------------------------
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    custom_config_attrs = []

    if hasattr(data_args, "vgllm_resize"):
        setattr(config, "vgllm_resize", data_args.vgllm_resize)
        custom_config_attrs.append("vgllm_resize")
    if hasattr(data_args, "vgllm_resize_mode"):
        setattr(config, "vgllm_resize_mode", data_args.vgllm_resize_mode)
        custom_config_attrs.append("vgllm_resize_mode")
    if hasattr(data_args, "vgllm_target_size"):
        setattr(config, "vgllm_target_size", data_args.vgllm_target_size)
        custom_config_attrs.append("vgllm_target_size")
    if hasattr(data_args, "use_geometry_inputs"):
        setattr(config, "use_geometry_inputs", data_args.use_geometry_inputs)
        custom_config_attrs.append("use_geometry_inputs")
    if hasattr(data_args, "use_patch_size_alin"):
        setattr(config, "use_patch_size_alin", data_args.use_patch_size_alin)
        custom_config_attrs.append("use_patch_size_alin")

    if hasattr(model_args, "use_geometry_encoder"):
        setattr(config, "use_geometry_encoder", model_args.use_geometry_encoder)
        custom_config_attrs.append("use_geometry_encoder")
    if hasattr(model_args, "use_feature_fusion_module"):
        setattr(config, "use_feature_fusion_module", model_args.use_feature_fusion_module)
        custom_config_attrs.append("use_feature_fusion_module")
    if hasattr(model_args, "use_geometry_deepstack_only"):
        setattr(config, "use_geometry_deepstack_only", model_args.use_geometry_deepstack_only)
        custom_config_attrs.append("use_geometry_deepstack_only")
    if hasattr(model_args, "feature_fusion_method"):
        setattr(config, "feature_fusion_method", model_args.feature_fusion_method)
        custom_config_attrs.append("feature_fusion_method")
    if hasattr(model_args, "fusion_num_layers"):
        setattr(config, "fusion_num_layers", model_args.fusion_num_layers)
        custom_config_attrs.append("fusion_num_layers")
    if hasattr(model_args, "use_camera_method") and model_args.use_camera_method is not None:
        setattr(config, "use_camera_method", model_args.use_camera_method)
        custom_config_attrs.append("use_camera_method")
    if hasattr(model_args, "geometry_deepstack_indexes") and model_args.geometry_deepstack_indexes is not None:
        indexes = [int(x.strip()) for x in model_args.geometry_deepstack_indexes.split(",")]
        setattr(config, "geometry_deepstack_indexes", indexes)
        custom_config_attrs.append("geometry_deepstack_indexes")
    if hasattr(model_args, "geometry_deepstack_indexes_pro") and model_args.geometry_deepstack_indexes_pro is not None:
        geo_ds_entries = []
        for pair in model_args.geometry_deepstack_indexes_pro.split(","):
            pair = pair.strip()
            if ":" not in pair:
                raise ValueError(
                    f"geometry_deepstack_indexes_pro must use 'vggt:llm' format, got '{pair}'."
                )
            vggt_part, llm_part = pair.split(":")
            vggt_idx = int(vggt_part.strip())
            llm_layers = [int(x.strip()) for x in llm_part.split("-")]
            geo_ds_entries.append([vggt_idx, llm_layers])
        setattr(config, "geometry_deepstack_indexes_pro", geo_ds_entries)
        custom_config_attrs.append("geometry_deepstack_indexes_pro")
    if hasattr(model_args, "use_deepstack_importance_gate") and model_args.use_deepstack_importance_gate is not None:
        gate_val = model_args.use_deepstack_importance_gate.strip()
        if gate_val.lower() == "all":
            setattr(config, "use_deepstack_importance_gate", "all")
        else:
            gate_layers = [int(x.strip()) for x in gate_val.split(",")]
            setattr(config, "use_deepstack_importance_gate", gate_layers)
        custom_config_attrs.append(f"use_deepstack_importance_gate={getattr(config, 'use_deepstack_importance_gate')}")
    if hasattr(model_args, "use_deepstack_global_gate") and model_args.use_deepstack_global_gate is not None:
        gate_val = model_args.use_deepstack_global_gate.strip()
        if gate_val.lower() == "all":
            setattr(config, "use_deepstack_global_gate", "all")
        else:
            gate_layers = [int(x.strip()) for x in gate_val.split(",")]
            setattr(config, "use_deepstack_global_gate", gate_layers)
        custom_config_attrs.append(f"use_deepstack_global_gate={getattr(config, 'use_deepstack_global_gate')}")
    if hasattr(model_args, "use_deepstack_camera_adaln") and model_args.use_deepstack_camera_adaln is not None:
        gate_val = model_args.use_deepstack_camera_adaln.strip()
        if gate_val.lower() == "all":
            setattr(config, "use_deepstack_camera_adaln", "all")
        else:
            gate_layers = [int(x.strip()) for x in gate_val.split(",")]
            setattr(config, "use_deepstack_camera_adaln", gate_layers)
        custom_config_attrs.append(
            f"use_deepstack_camera_adaln={getattr(config, 'use_deepstack_camera_adaln')}"
        )

    if custom_config_attrs:
        rank0_print(f"Setting custom config attributes: {custom_config_attrs}")

    # ------------------------------------------------------------------
    # Load model (verbatim from GUIDE)
    # ------------------------------------------------------------------
    _model_type = getattr(config, 'model_type', '').lower()
    _path_lower = model_args.model_name_or_path.lower()
    _path_name  = Path(model_args.model_name_or_path.rstrip("/")).name.lower()

    _is_qwen3_moe = ('qwen3' in _model_type and 'moe' in _model_type) or \
                    ('qwen3' in _path_lower and 'a' in _path_name)
    _is_qwen3     = ('qwen3' in _model_type) or ('qwen3' in _path_lower)
    _is_qwen25    = ('qwen2.5' in _model_type or 'qwen2_5' in _model_type) or \
                    ('qwen2.5' in _path_lower)

    if _is_qwen3_moe:
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif _is_qwen3:
        geometry_encoder_path = model_args.geometry_encoder_path if model_args.use_geometry_encoder else None
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
            geometry_encoder_path=geometry_encoder_path,
        )
        data_args.model_type = "qwen3vl"
    elif _is_qwen25:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2vl"

    print(f"Initialized model: {model_args.model_name_or_path} ({model.__class__.__name__})")

    if data_args.model_type == "qwen3vl":
        from qwenvl.model.processing_qwen3_vl import Qwen3VLProcessor
        processor = Qwen3VLProcessor.from_pretrained(model_args.model_name_or_path)
    else:
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # ------------------------------------------------------------------
    # MoPE: attach encoder + projector BEFORE LoRA wrapping / param freeze
    # so that MoPEProjector parameters are visible when we set requires_grad.
    # ------------------------------------------------------------------
    if mope_args.use_mope:
        if not mope_args.mope_checkpoint_path:
            raise ValueError(
                "--use_mope is set but --mope_checkpoint_path is not provided."
            )

        # Override MOPE_ROOT in sys.path if an explicit encoder path was given.
        if mope_args.mope_encoder_path:
            mope_path = os.path.abspath(mope_args.mope_encoder_path)
            if mope_path not in sys.path:
                sys.path.insert(0, mope_path)
                rank0_print(f"[Space Sensing] Added mope_encoder_path to sys.path: {mope_path}")

        _attach_mope_to_model(model, mope_args)

        # Phase 2 two-stage: load pre-trained projector weights from Stage 1 checkpoint.
        if mope_args.freeze_mope_projector:
            _loaded = _load_mope_weights_from_checkpoint(
                model.model, model_args.model_name_or_path
            )
            if not _loaded:
                raise RuntimeError(
                    f"--freeze_mope_projector is set but no MoPE weights found in "
                    f"checkpoint at {model_args.model_name_or_path}. "
                    f"Ensure the checkpoint was produced by a Stage 1 experiment "
                    f"(E-00b / E-00c)."
                )
            rank0_print("[Space Sensing] Stage 2: loaded pre-trained projector weights from Stage 1 checkpoint.")
        elif mope_args.load_mope_projector_from_ckpt:
            # E-10b warm-gate probe: load the projector (k/v/out_proj) from
            # --model_name_or_path but keep it TRAINABLE. The projector is attached
            # AFTER from_pretrained (line ~769), so HF's from_pretrained silently
            # dropped the checkpoint's `model._mope_projector.*` keys as unexpected
            # and the fresh projector is currently zero-init. Without this load the
            # warm-gate probe would start with out_proj≈0 (dead dynamic branch),
            # defeating its whole premise. _load_mope_weights_from_checkpoint uses
            # strict=False, so a non-gated checkpoint (e.g. E-03a, no gate_mlp keys)
            # loads k/v/out_proj while the freshly-constructed gate_mlp KEEPS its
            # constructor MI-seed init (it is simply absent from the loaded
            # state_dict). Unlike the freeze branch above this does NOT freeze the
            # projector (warm-start joint training) and is non-fatal if the
            # checkpoint has no MoPE keys (warns and proceeds with the init).
            _loaded = _load_mope_weights_from_checkpoint(
                model.model, model_args.model_name_or_path
            )
            if _loaded:
                rank0_print(
                    "[Space Sensing] Warm-gate probe: loaded MoPE projector "
                    "(k/v/out_proj) from --model_name_or_path; gate_mlp (if any) "
                    "keeps its fresh init (strict=False); projector stays trainable."
                )
            else:
                rank0_print(
                    "[Space Sensing] WARNING: --load_mope_projector_from_ckpt set but "
                    f"no MoPE projector keys found at {model_args.model_name_or_path}; "
                    "projector keeps its zero-init (dynamic branch would start dead)."
                )

    # ------------------------------------------------------------------
    # Trainability setup (verbatim from GUIDE, with MoPE projector unfreeze)
    # ------------------------------------------------------------------
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

        # Re-enable MoPEProjector after LoRA wrapping froze everything
        # (unless freeze_mope_projector is set for two-stage training).
        if mope_args.use_mope:
            projector = model.model._mope_projector
            for p in projector.parameters():
                p.requires_grad = not mope_args.freeze_mope_projector
            _proj_status = "frozen (two-stage)" if mope_args.freeze_mope_projector else "re-enabled"
            rank0_print(f"[Space Sensing] MoPEProjector {_proj_status} after LoRA freeze.")
            # paper-2 Stage 1: the LFP head trains in BOTH 2-stage stages
            # (aligned with the projector, implements D-12). It is only present
            # when mope_lfp_enable=True; default False -> this block is skipped.
            if hasattr(model.model, "_mope_lfp_head"):
                for p in model.model._mope_lfp_head.parameters():
                    p.requires_grad = True
                rank0_print(
                    "[Space Sensing] MoPELFPHead trainable after LoRA freeze "
                    "(both stages train it, aligned with the projector)."
                )
    else:
        set_model(model_args, model)

        # MoPEEncoder is frozen internally; only MoPEProjector is trainable.
        if mope_args.use_mope:
            # Encoder: ensure all params are frozen.
            for p in model.model._mope_encoder.parameters():
                p.requires_grad = False
            # Projector: trainable by default, frozen when freeze_mope_projector is set
            # (Phase 2 two-stage: projector pre-trained in Stage 1, LLM trains in Stage 2).
            for p in model.model._mope_projector.parameters():
                p.requires_grad = not mope_args.freeze_mope_projector
            _proj_status = "frozen (two-stage)" if mope_args.freeze_mope_projector else "trainable"
            rank0_print(f"[Space Sensing] MoPEEncoder frozen, MoPEProjector {_proj_status}.")
            # paper-2 Stage 1: the LFP head trains in BOTH 2-stage stages
            # (aligned with the projector, implements D-12). It is only present
            # when mope_lfp_enable=True; default False -> this block is skipped.
            if hasattr(model.model, "_mope_lfp_head"):
                for p in model.model._mope_lfp_head.parameters():
                    p.requires_grad = True
                rank0_print(
                    "[Space Sensing] MoPELFPHead trainable "
                    "(both stages train it, aligned with the projector)."
                )

        is_rank0 = (
            (not torch.distributed.is_available())
            or (not torch.distributed.is_initialized())
            or torch.distributed.get_rank() == 0
        )
        if is_rank0:
            if hasattr(model, "visual") and hasattr(model.visual, "print_trainable_parameters"):
                model.visual.print_trainable_parameters()
            lm = getattr(model, "language_model", None) or getattr(model, "model", None)
            if lm is not None and hasattr(lm, "print_trainable_parameters"):
                lm.print_trainable_parameters()

            # [DIAG] Trainable parameter breakdown.
            total_p = sum(p.numel() for p in model.parameters())
            trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
            _lm_mod = getattr(model, "language_model", None) or getattr(model.model, "language_model", None)
            lm_trainable = sum(p.numel() for p in _lm_mod.parameters() if p.requires_grad) if _lm_mod else -1
            lm_total = sum(p.numel() for p in _lm_mod.parameters()) if _lm_mod else -1
            proj_trainable = -1
            if mope_args.use_mope and hasattr(model.model, "_mope_projector"):
                proj_trainable = sum(p.numel() for p in model.model._mope_projector.parameters() if p.requires_grad)
            print(
                f"[Space Sensing DIAG] Trainable params: {trainable_p:,} / {total_p:,}\n"
                f"  LLM: {lm_trainable:,} / {lm_total:,} trainable\n"
                f"  MoPE projector: {proj_trainable:,} trainable"
            )

    # ------------------------------------------------------------------
    # E-02b / E-02d: Register nan_to_num gradient hooks on ALL trainable
    # params for modes that prepend tokens to the LLM input sequence.
    # With prepended tokens and tune_mm_llm=True, LLM attention/MLP
    # backward can overflow bf16 to Inf.  clip_grad_norm_ then computes
    # Inf×0 = NaN (IEEE 754), and AdamW propagates that as 0×NaN = NaN
    # even at lr=0, corrupting weights before step 2.
    # Zeroing Inf/NaN before the optimizer sees them prevents this.
    # Applies to: concat (784 tokens, E-02b) and qformer (32 tokens, E-02d).
    # ------------------------------------------------------------------
    if mope_args.use_mope and mope_args.mope_fusion_mode in ("concat", "qformer"):
        _n_hooked = 0
        for _p in model.parameters():
            if _p.requires_grad:
                _p.register_hook(
                    lambda g: torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                    if g is not None else g
                )
                _n_hooked += 1
        rank0_print(
            f"[Space Sensing] E-02b/E-02d: registered nan_to_num grad hooks on "
            f"{_n_hooked} trainable tensors (prevents bf16 Inf → AdamW NaN)."
        )

    # ------------------------------------------------------------------
    # Patch forward to inject MoPE (after parameter freeze to avoid
    # accidentally making patch-internal references trainable).
    # ------------------------------------------------------------------
    if mope_args.use_mope:
        if mope_args.mope_fusion_mode == "concat":
            _patch_model_for_mope_concat(model)
        elif mope_args.mope_fusion_mode == "crossattn":
            # paper-2 Stage 1: mope_feed_features bypass (spec Q6/S5). Default
            # True = E-03a residual fusion unchanged. False = skip the residual
            # (LLM no longer sees MoPE features injected) but still cache the
            # frozen-encoder latent for the LFP target path. Orthogonal to
            # mope_lfp_enable.
            # E-16e: mope_feed_causal_mask (default False) adds a causal-over-bins
            # mask to the feed cross-attn so frame f only attends to MoPE K/V of
            # bins <= b_f. False = byte-for-byte E-16 (no mask). Only active when
            # feed_features=True.
            _patch_model_for_mope_crossattn(
                model,
                feed_features=mope_args.mope_feed_features,
                feed_causal_mask=mope_args.mope_feed_causal_mask,
            )
        elif mope_args.mope_fusion_mode == "qformer":
            _patch_model_for_mope_qformer(model)
        else:
            _patch_model_for_mope(model)

        # ----------------------------------------------------------------
        # paper-2 Stage 1: LFP outer forward-patch (implements D-12).
        # Installed ONLY when mope_lfp_enable=True (opt-in). When False the
        # outer forward is untouched and the E-03a path is byte-for-byte
        # unchanged. Requires the inner crossattn patch above to populate the
        # frozen-MoPE latent cache (_mope_cached_x_vis) used as the prediction
        # target, so we only attach it when the fusion mode is crossattn.
        # ----------------------------------------------------------------
        if mope_args.mope_lfp_enable:
            if mope_args.mope_fusion_mode != "crossattn":
                raise RuntimeError(
                    "mope_lfp_enable=True requires mope_fusion_mode=crossattn "
                    "(the inner crossattn patch caches the frozen-MoPE latent "
                    "that the LFP target path reuses). Got "
                    f"mope_fusion_mode={mope_args.mope_fusion_mode!r}."
                )
            from model.mope_patch import _patch_model_for_lfp

            # Lightweight config container closed over by the patch closure.
            class _LFPCfg:
                enable = True
                weight = float(mope_args.mope_lfp_weight)
                mse_weight = float(mope_args.mope_lfp_mse_weight)
                cos_weight = float(mope_args.mope_lfp_cos_weight)
                # pred_source="per_frame_video" (default) -> FINAL token-shift
                # path; any other value -> legacy distillation (uses the two
                # fields below). align_strategy is reserved (only "uniform" now).
                pred_source = str(mope_args.mope_lfp_pred_source)
                align_strategy = str(mope_args.mope_lfp_align_strategy)
                # Legacy distillation fields (back-compat; ignored by token-shift).
                context_frames = int(mope_args.mope_lfp_context_frames)
                target_pool = str(mope_args.mope_lfp_target_pool)
                # E-16b anti-collapse regularizer (VICReg). reg_weight=0.0
                # (default) -> branch not entered, byte-equivalent to E-16.
                reg_weight = float(mope_args.mope_lfp_reg_weight)
                reg_type = str(mope_args.mope_lfp_reg_type)
                reg_var_hinge = float(mope_args.mope_lfp_reg_var_hinge)
                reg_cov_scale = float(mope_args.mope_lfp_reg_cov_scale)
                # WARN-1 fix: scale-normalize the VICReg target to O(1) so the
                # variance hinge (tau=1) fires on LLM raw hidden. Default True.
                reg_normalize = bool(mope_args.mope_lfp_reg_normalize)

            _patch_model_for_lfp(model, _LFPCfg)

    # ------------------------------------------------------------------
    # Data + training (verbatim from GUIDE)
    # ------------------------------------------------------------------
    data_module = make_supervised_data_module(processor, data_args=data_args)

    if mope_args.use_mope:
        from train_framework.data.mope_data_wrapper import (
            MoPEDatasetWrapper,
            MoPECollatorWrapper,
        )
        data_module["train_dataset"] = MoPEDatasetWrapper(
            data_module["train_dataset"],
            mope_all_frames=mope_args.mope_all_frames,
        )
        # For concat fusion, prepend N_mope -100 labels so loss shape matches
        # the extended logit sequence produced by _patch_model_for_mope_concat.
        if mope_args.mope_fusion_mode == "concat":
            n_mope_tokens = mope_args.mope_concat_num_tokens
        elif mope_args.mope_fusion_mode == "qformer":
            n_mope_tokens = mope_args.mope_qformer_num_queries
        else:
            n_mope_tokens = 0
        data_module["data_collator"] = MoPECollatorWrapper(
            data_module["data_collator"],
            mope_num_tokens=n_mope_tokens,
        )
        rank0_print(
            f"[Space Sensing] MoPE data pipeline: {mope_args.mope_all_frames} frames/sample, "
            f"fusion={mope_args.mope_fusion_mode}, label_pad={n_mope_tokens}"
        )
        training_args.remove_unused_columns = False

    # Resolve the gate projector once (None unless mope_use_gate); the trainer
    # closes over it + mope_args for the E-10b anti-collapse loss / diagnostics.
    _gate_proj = None
    if mope_args.use_mope and mope_args.mope_use_gate:
        _gate_proj = getattr(getattr(model, "model", None), "_mope_projector", None)

    # [DIAG] Subclass to print actual loss.item() for first 20 micro-batches.
    # Covers ~5 optimizer steps (grad_accum=4) to capture the step-1→step-2 transition.
    class _DiagTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.args.predelete_oldest_checkpoint and (
                self.args.save_total_limit is None or self.args.save_total_limit < 2
            ):
                raise ValueError(
                    "predelete_oldest_checkpoint requires save_total_limit>=2"
                )
            self._diag_calls = 0
            # E-10b gate anti-collapse config (closed over from mope_args).
            self._gate_proj = _gate_proj
            self._gate_ac = bool(
                mope_args.mope_use_gate and mope_args.mope_gate_anticollapse
            )
            self._gate_zc = float(mope_args.mope_gate_zloss_coef)
            self._gate_ec = float(mope_args.mope_gate_entropy_coef)
            self._gate_ewarm = max(0, int(mope_args.mope_gate_entropy_warmup_steps))
            # Last-window diagnostic snapshot for GateDiagCallback to print.
            self._gate_diag = {}
            # E-10b v2.1: real gate-logit grad norm captured by the backward hook
            # registered in _gate_anticollapse_loss (change 2). -1 = not yet seen.
            self._last_gate_logit_gnorm = -1.0

        def _save_checkpoint(self, model, trial):
            if self.args.predelete_oldest_checkpoint:
                error = None
                deleted = []
                if self.is_world_process_zero():
                    try:
                        deleted = predelete_for_two_slot_rotation(
                            self.args.output_dir,
                            self.state.global_step,
                        )
                    except Exception as exc:  # coordinated across all ranks below
                        error = f"{type(exc).__name__}: {exc}"

                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    payload = [error]
                    torch.distributed.broadcast_object_list(payload, src=0)
                    error = payload[0]
                    if error is None:
                        torch.distributed.barrier()

                if error is not None:
                    raise RuntimeError(
                        f"two-slot checkpoint pre-delete failed: {error}"
                    )
                if self.is_world_process_zero():
                    print(
                        "[checkpoint-two-slot] "
                        f"step={self.state.global_step} deleted={deleted or 'none'}",
                        flush=True,
                    )

            return super()._save_checkpoint(model, trial)

        def _gate_anticollapse_loss(self, base_loss):
            """E-10b v2.1: build the gate anti-collapse loss bundle on the live graph.

            Drains projector._gate_logit_graph_buf (one live-graph fp32 logit per
            cross-attn forward this micro-batch) and computes, with gradients, a
            MUTUAL-INFORMATION objective (RIM/IMSAT form) plus an OPTIONAL weak
            z-loss guard:

              MI (maximised)   : MI = H_marg - H_cond
                  H_marg  = binary entropy of g_bar = mean(sigmoid(logit))
                            (marginal / batch-usage entropy)
                  H_cond  = mean over samples of the per-sample binary entropy
                            of g_i = sigmoid(logit_i)  (conditional entropy)
                  -> loss term = -lambda_t * MI         (maximise MI)
                     lambda_t  = entropy_coef * min(1, step/T_warm)
                  At the all-0.5 collapse point MI = 0 (its GLOBAL MINIMUM /
                  unstable saddle), so the MI gradient pushes the gate AWAY from
                  the constant g=0.5 — every sample sharpens (H_cond down) while
                  batch usage stays spread (H_marg up) => content-dependent
                  divergence. This replaces the v2 batch-mean entropy term, whose
                  stable max at g_bar=0.5 had zero gradient and could not pry the
                  gate open.

              z-loss (guard)   : L_z = mean(logit^2). DEFAULT OFF. Only added to
                  the loss when mope_gate_zloss_coef > 0. v2 added it always and
                  its unique minimum at logit=0 (g=0.5) actively pulled the gate
                  TOWARD the constant 0.5, fighting the per-sample confidence the
                  MI term needs. Kept for diagnostics (l_z_raw) regardless.

            Also registers a backward hook on the live-graph logits tensor so the
            GateDiagCallback can report a real, non-negative gate_grad_norm (see
            note at the hook below).

            Returns (extra_loss_tensor_or_None, diag_dict). When the gate is
            disabled / anti-collapse off / no logits were stashed, returns
            (None, {}) and the task loss is unchanged.
            """
            proj = self._gate_proj
            if proj is None:
                return None, {}
            buf = getattr(proj, "_gate_logit_graph_buf", None)
            if not buf:
                # Always drain to bound memory even if anti-collapse is off.
                if buf is not None:
                    buf.clear()
                return None, {}
            # Concatenate all live-graph logits of this micro-batch -> [N].
            logits = torch.cat(buf)                       # fp32, graph intact
            buf.clear()                                   # drain (memory-bounded)

            if not self._gate_ac:
                return None, {}

            # --- gate_grad_norm instrumentation (change 2) -------------------
            # Register a hook on the LIVE-GRAPH logits tensor. On backward this
            # fires with grad = d(total_loss)/d(gate_logits); we stash its norm
            # on the trainer for GateDiagCallback to print as gate_grad_norm.
            # CALIBRE: this is the *logit* gradient norm (signal arriving AT the
            # gate output), NOT the gate_mlp *parameter* gradient norm. We use it
            # because under DeepSpeed ZeRO-2 the gate_mlp param .grad is reduced /
            # may be unavailable at on_pre_optimizer_step (yielding the -1
            # sentinel); the activation-level grad of the live logits tensor is
            # always populated during backward, before any partition/reduce, so
            # it is a reliable, real measure of whether the gate is receiving
            # gradient. A non-zero value => the gate is being driven by the loss.
            if logits.requires_grad:
                def _logit_grad_hook(grad):
                    try:
                        self._last_gate_logit_gnorm = float(
                            grad.detach().float().norm().item()
                        )
                    except Exception:
                        self._last_gate_logit_gnorm = -1.0
                    return None
                logits.register_hook(_logit_grad_hook)

            # z-loss (diagnostic always; added to loss only if coef > 0).
            l_z = (logits ** 2).mean()

            # All entropy/MI math in fp32 (logits already fp32). Clamp before log.
            g = torch.sigmoid(logits)

            # Marginal (batch-usage) entropy: H_marg = H(g_bar).
            g_bar = g.mean().clamp(1e-6, 1.0 - 1e-6)
            H_marg = -(g_bar * torch.log(g_bar)
                       + (1.0 - g_bar) * torch.log(1.0 - g_bar))

            # Conditional (per-sample) entropy: mean_i H(g_i).
            g_c = g.clamp(1e-6, 1.0 - 1e-6)
            H_cond_i = -(g_c * torch.log(g_c) + (1.0 - g_c) * torch.log(1.0 - g_c))
            H_cond = H_cond_i.mean()

            # Mutual information in [0, ln2]; 0 at the all-0.5 collapse point.
            MI = H_marg - H_cond

            # Linear warm-up of lambda_t.
            step = float(self.state.global_step)
            if self._gate_ewarm > 0:
                warm = min(1.0, step / float(self._gate_ewarm))
            else:
                warm = 1.0
            lam_t = self._gate_ec * warm

            # Entropy term: maximise MI => subtract lam_t * MI.
            ent_term = -lam_t * MI
            extra = ent_term
            # z-loss guard: OFF by default (coef=0). Only add when coef > 0.
            if self._gate_zc > 0.0:
                zloss_term = self._gate_zc * l_z
                extra = extra + zloss_term
                l_z_weighted = float(zloss_term.detach().float().item())
            else:
                l_z_weighted = 0.0
            extra = extra.to(base_loss.dtype)

            diag = {
                "task_loss": float(base_loss.detach().float().item()),
                "l_z_raw": float(l_z.detach().float().item()),
                "l_z_weighted": l_z_weighted,
                "H_marg": float(H_marg.detach().float().item()),
                "H_cond": float(H_cond.detach().float().item()),
                "MI": float(MI.detach().float().item()),
                "MI_weighted": float((lam_t * MI).detach().float().item()),
                "lambda_t": float(lam_t),
                "n_logits": int(logits.numel()),
            }
            return extra, diag

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            _rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            lbl = inputs.get("labels", None)
            has_mope = "mope_frames" in inputs
            has_pv   = inputs.get("pixel_values", None) is not None
            if self._diag_calls < 20 and _rank == 0:
                _n_valid = (lbl != -100).sum().item() if lbl is not None else -1
                print(
                    f"[DIAG TRAINER] call#{self._diag_calls + 1}: "
                    f"labels={tuple(lbl.shape) if lbl is not None else None} "
                    f"valid={_n_valid} "
                    f"mope_frames={has_mope} pixel_values={has_pv}",
                    flush=True,
                )
            _super_kwargs = {} if num_items_in_batch is None else {"num_items_in_batch": num_items_in_batch}
            _result = super().compute_loss(model, inputs, return_outputs=True, **_super_kwargs)
            _loss, _outputs = _result
            if torch.isnan(_loss).any() or torch.isinf(_loss).any():
                if _rank == 0:
                    print(
                        f"[DIAG TRAINER] NaN/Inf loss at call#{self._diag_calls + 1}, "
                        f"replacing with zero to protect Adam states",
                        flush=True,
                    )
                _loss = torch.nan_to_num(_loss, nan=0.0, posinf=0.0, neginf=0.0)

            # ----------------------------------------------------------------
            # E-10b: add the gate anti-collapse loss bundle (z-loss + warmed-up
            # Bernoulli entropy). The drain of _gate_logit_graph_buf ALWAYS runs
            # (memory hygiene) but the extra loss is only added when
            # mope_gate_anticollapse=True; otherwise _loss is unchanged
            # (E-10 / E-03a path byte-for-byte). PENDING[D-15]: regularises the
            # content-driven (b) scalar gate, no task-label supervision.
            # ----------------------------------------------------------------
            _extra, _gdiag = self._gate_anticollapse_loss(_loss)
            if _extra is not None:
                _loss = _loss + _extra
            if _gdiag:
                # Stash the most-recent decomposition for GateDiagCallback.
                self._gate_diag = _gdiag

            if self._diag_calls < 20 and _rank == 0:
                self._diag_calls += 1
                _logits_shape = None
                if hasattr(_outputs, 'logits') and _outputs.logits is not None:
                    _logits_shape = tuple(_outputs.logits.shape)
                print(
                    f"[DIAG TRAINER] call#{self._diag_calls}: "
                    f"loss={_loss.item():.8f} "
                    f"loss_isnan={_loss.isnan().item()} "
                    f"logits={_logits_shape}",
                    flush=True,
                )
            else:
                self._diag_calls += 1
            return (_loss, _outputs) if return_outputs else _loss

    trainer = _DiagTrainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    # E-10 (Router v1): attach the gate process-logging callback only when the
    # learned content-driven gate is enabled. Zero side effects otherwise (the
    # callback is never added, so the ungated / eval paths are untouched).
    if mope_args.use_mope and mope_args.mope_use_gate:
        _gate_projector = getattr(getattr(model, "model", None), "_mope_projector", None)
        trainer.add_callback(
            GateStatsCallback(
                gate_log_interval=mope_args.gate_log_interval,
                projector=_gate_projector,
            )
        )
        rank0_print(
            f"[Space Sensing] E-10 GateStatsCallback attached "
            f"(gate_log_interval={mope_args.gate_log_interval})."
        )
        # E-10b (Router v1.1): attach the rich [gate-diag] diagnostic callback.
        # Gated on --mope_gate_anticollapse so E-10's logs stay byte-for-byte
        # unchanged; the E-10b script sets it True and gets the rich diagnostics.
        if mope_args.mope_gate_anticollapse:
            _gate_diag_cb = GateDiagCallback(
                diag_every=mope_args.mope_gate_diag_every,
                projector=_gate_projector,
                trainer_ref=trainer,
            )
            trainer.add_callback(_gate_diag_cb)
            rank0_print(
                f"[Space Sensing] E-10b GateDiagCallback attached "
                f"(diag_every={mope_args.mope_gate_diag_every}, "
                f"anticollapse={mope_args.mope_gate_anticollapse}, "
                f"zloss_coef={mope_args.mope_gate_zloss_coef}, "
                f"entropy_coef={mope_args.mope_gate_entropy_coef}, "
                f"entropy_warmup={mope_args.mope_gate_entropy_warmup_steps}, "
                f"gate_init_bias={mope_args.mope_gate_init_bias})."
            )

    # Respect the CLI-provided --resume_from_checkpoint (used as-is, no
    # completeness check). Only auto-detect a checkpoint when it was not
    # explicitly passed, and only accept a *complete* checkpoint; otherwise
    # train from scratch.
    resume_ckpt = training_args.resume_from_checkpoint
    if not resume_ckpt and not training_args.overwrite_output_dir:
        last_ckpt = get_last_checkpoint(training_args.output_dir)
        if last_ckpt is not None and is_complete_checkpoint(last_ckpt):
            resume_ckpt = last_ckpt
            logging.info(f"checkpoint found, resume training from {resume_ckpt}")
    if resume_ckpt:
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
