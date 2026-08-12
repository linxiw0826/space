"""Head-free Part A checkpoint loader for matched lmms-eval runs."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.qwen3_vl_my import Qwen3_VL_MY

from parta.checkpoint import TRAINING_CHECKPOINT_SCHEMA, load_head_free_artifact
from parta.provenance import sha256_file


@register_model("qwen3_vl_parta")
class Qwen3VLPartA(Qwen3_VL_MY):
    """Load A0 or independently audited A1-O-drop without constructing a state head."""

    def __init__(self, *, parta_artifact: str, parta_arm: str,
                 head_free_audit: str = "", **kwargs) -> None:
        if parta_arm not in {"a0", "a1o_drop"}:
            raise ValueError("parta_arm must be a0 or a1o_drop")
        super().__init__(**kwargs)
        if hasattr(self.model, "parta_state_head"):
            raise RuntimeError("Part A eval model unexpectedly instantiated state head")
        artifact = Path(parta_artifact)
        if parta_arm == "a1o_drop":
            audit = json.loads(Path(head_free_audit).read_text(encoding="utf-8"))
            if (audit.get("status") != "complete_passed"
                    or audit.get("independent_model_construction") is not True
                    or audit.get("forward_passed") is not True
                    or audit.get("head_free_artifact_sha256") != sha256_file(artifact)):
                raise ValueError("A1-O-drop independent audit is absent or mismatched")
            load_head_free_artifact(self.model, artifact)
        else:
            payload = torch.load(artifact, map_location="cpu", weights_only=False)
            if (payload.get("schema_version") != TRAINING_CHECKPOINT_SCHEMA
                    or payload.get("contract", {}).get("arm") != "a0"):
                raise ValueError("A0 training checkpoint contract mismatch")
            incompatible = self.model.load_state_dict(payload.get("model", {}), strict=False)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise ValueError("A0 checkpoint is incompatible with head-free GUIDE model")
