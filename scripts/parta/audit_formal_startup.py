#!/usr/bin/env python3
"""Derive the matched step-zero A0/A1-O startup contract from frozen artifacts."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import torch
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts/parta"))
from parta.provenance import atomic_json_dump, sha256_file, stable_sha256  # noqa: E402
from run_t0_a import _checkpoint_artifact_provenance  # noqa: E402

def checkpoint_artifact(path: Path) -> dict:
    return _checkpoint_artifact_provenance(path)

def validate_startup_input(name: str, item: dict) -> None:
    path = Path(str(item.get("path", ""))).resolve()
    if name in {"guide", "vggt"}:
        actual = _checkpoint_artifact_provenance(path)
        if item.get("artifact") != actual or item.get("artifact_sha256") != actual["artifact_sha256"]:
            raise ValueError("formal startup model artifact manifest mismatch")
    elif not path.is_file() or item.get("sha256") != sha256_file(path):
        raise ValueError("formal startup input hash mismatch")
def t0_a_model_state_digest(state: dict) -> str:
    import hashlib
    records = []
    for key, value in sorted(state.items()):
        raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        records.append((key, list(value.shape), str(value.dtype), hashlib.sha256(raw).hexdigest()))
    return stable_sha256(records)

def validate_single_checkpoint_artifact(artifact: dict, checkpoint: Path) -> None:
    record = {"name": checkpoint.name, "size_bytes": checkpoint.stat().st_size,
              "sha256": sha256_file(checkpoint)}
    payload = {"mode": "no_index_explicit_manifest", "config_files": [],
               "ordered_shards": [record]}
    if (artifact.get("mode") != payload["mode"] or artifact.get("index") is not None
            or artifact.get("config_files") != [] or artifact.get("ordered_shards") != [record]
            or artifact.get("artifact_sha256") != stable_sha256(payload)):
        raise ValueError("T0-A checkpoint artifact manifest is invalid")

def validate_startup_configs(configs: dict, authoritative_digest: str) -> None:
    ignored = {"arm", "state_head_enabled"}
    common = {arm: {k: v for k, v in cfg.items() if k not in ignored} for arm, cfg in configs.items()}
    init = {arm: cfg.get("initialization_sha256") for arm, cfg in configs.items()}
    if (set(configs) != {"a0", "a1o"} or common["a0"] != common["a1o"]
            or any(cfg.get("start_step") != 0 for cfg in configs.values())
            or set(init.values()) != {authoritative_digest}):
        raise ValueError("formal startup arms are not matched at step zero")

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a0-config", type=Path, required=True); p.add_argument("--a1o-config", type=Path, required=True)
    p.add_argument("--matched-fairness", type=Path, required=True); p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--guide", type=Path, required=True); p.add_argument("--vggt", type=Path, required=True)
    p.add_argument("--t0-a-checkpoint", type=Path, required=True)
    p.add_argument("--t0-a-report", type=Path, required=True)
    p.add_argument("--t0-a-provenance", type=Path, required=True)
    p.add_argument("--t0-a-run-status", type=Path, required=True)
    p.add_argument("--t0-a-resolved-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); args = p.parse_args()
    configs = {arm: json.loads(path.read_text()) for arm, path in (("a0", args.a0_config), ("a1o", args.a1o_config))}
    t0_report = json.loads(args.t0_a_report.read_text()); t0_prov = json.loads(args.t0_a_provenance.read_text())
    t0_status = json.loads(args.t0_a_run_status.read_text()); t0_resolved = json.loads(args.t0_a_resolved_config.read_text())
    fairness = json.loads(args.matched_fairness.read_text()); freeze = json.loads(args.freeze.read_text())
    checkpoint = torch.load(args.t0_a_checkpoint, map_location="cpu", weights_only=False)
    state_digest = t0_a_model_state_digest(checkpoint.get("model", {}))
    checkpoint_artifact = t0_prov.get("a1_checkpoint_artifact", {})
    validate_single_checkpoint_artifact(checkpoint_artifact, args.t0_a_checkpoint)
    declared_digest = checkpoint.get("parameter_sha256", checkpoint.get("initialization_sha256", state_digest))
    resolved_identity = stable_sha256(t0_resolved)
    if (t0_report.get("status") != "complete_passed"
            or t0_prov.get("status") != "complete_passed" or t0_status.get("status") != "complete"
            or checkpoint.get("optimizer_steps") != 0
            or checkpoint.get("checkpoint_role") != "initialization_no_optimizer_updates"
            or declared_digest != state_digest
            or t0_prov.get("a1_checkpoint_optimizer_steps") != 0
            or t0_prov.get("a1_checkpoint_role") != "initialization_no_optimizer_updates"
            or t0_prov.get("a1_checkpoint_state_sha256") != state_digest
            or t0_status.get("checkpoint_sha256") != state_digest
            or t0_prov.get("resolved_config_sha256") != resolved_identity
            or t0_status.get("resolved_config_sha256") != resolved_identity
            or Path(str(t0_prov.get("resolved_config_path", ""))).resolve()
               != args.t0_a_resolved_config.resolve()
            or Path(str(t0_status.get("resolved_config_path", ""))).resolve()
               != args.t0_a_resolved_config.resolve()):
        raise ValueError("T0-A is not the authoritative zero-step initialization")
    ignored = {"arm", "state_head_enabled"}
    common = {arm: {k: v for k, v in cfg.items() if k not in ignored} for arm, cfg in configs.items()}
    init = {arm: cfg.get("initialization_sha256") for arm, cfg in configs.items()}
    validate_startup_configs(configs, state_digest)
    expected_identities = {
        "matched_contract_sha256": sha256_file(args.matched_fairness),
        "freeze_artifact_sha256": sha256_file(args.freeze),
        "guide_artifact_sha256": _checkpoint_artifact_provenance(args.guide)["artifact_sha256"],
        "vggt_artifact_sha256": _checkpoint_artifact_provenance(args.vggt)["artifact_sha256"],
    }
    for arm, config in configs.items():
        if any(config.get(key) != value for key, value in expected_identities.items()):
            raise ValueError(f"{arm} startup identity differs from frozen inputs")
        if config.get("manifest_sha256") != fairness.get("manifest_sha256"):
            raise ValueError(f"{arm} startup manifest differs from matched fairness contract")
    if (freeze.get("matched_contract_sha256") != expected_identities["matched_contract_sha256"]
            or freeze.get("guide_artifact_sha256") != expected_identities["guide_artifact_sha256"]
            or freeze.get("vggt_artifact_sha256") != expected_identities["vggt_artifact_sha256"]):
        raise ValueError("freeze artifact does not bind startup identities")
    if (t0_prov.get("guide_checkpoint_artifact", {}).get("artifact_sha256")
            != expected_identities["guide_artifact_sha256"]
            or t0_prov.get("vggt_checkpoint_artifact", {}).get("artifact_sha256")
               != expected_identities["vggt_artifact_sha256"]):
        raise ValueError("T0-A provenance differs from GUIDE/VGGT startup artifacts")
    inputs = {"a0_config": args.a0_config, "a1o_config": args.a1o_config,
              "matched_fairness": args.matched_fairness, "freeze": args.freeze,
              "guide": args.guide, "vggt": args.vggt}
    inputs.update({"t0_a_checkpoint": args.t0_a_checkpoint, "t0_a_report": args.t0_a_report,
                   "t0_a_provenance": args.t0_a_provenance,
                   "t0_a_run_status": args.t0_a_run_status,
                   "t0_a_resolved_config": args.t0_a_resolved_config})
    producer = Path(__file__).resolve()
    atomic_json_dump({"schema_version": "parta_formal_startup_audit_v1", "status": "complete_passed",
                      "producer": {"path": str(producer), "sha256": sha256_file(producer),
                                   "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()},
                      "inputs": {name: ({"path": str(path.resolve()),
                                         "artifact": _checkpoint_artifact_provenance(path),
                                         "artifact_sha256": _checkpoint_artifact_provenance(path)["artifact_sha256"]}
                                        if name in {"guide", "vggt"} else
                                        {"path": str(path.resolve()), "sha256": sha256_file(path)})
                                 for name, path in inputs.items()},
                      "arms": {arm: {"start_step": 0, "initialization_sha256": init[arm]} for arm in configs},
                      "matched_common_sha256": stable_sha256(common["a0"])}, args.output)
if __name__ == "__main__": main()
