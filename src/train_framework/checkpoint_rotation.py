"""Fail-safe checkpoint helpers for bounded-space training runs."""

import os
import re
import shutil
import json
from pathlib import Path


_CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")


def is_complete_checkpoint(ckpt_dir):
    """Return whether a Trainer/DeepSpeed checkpoint finished writing."""
    ckpt_dir = os.fspath(ckpt_dir)
    if not os.path.isfile(os.path.join(ckpt_dir, "trainer_state.json")):
        return False
    latest_file = os.path.join(ckpt_dir, "latest")
    if os.path.isfile(latest_file):
        try:
            with open(latest_file) as handle:
                tag = handle.read().strip()
        except (OSError, ValueError):
            return False
        if not tag or not os.path.isdir(os.path.join(ckpt_dir, tag)):
            return False
    return True


def validate_deepspeed_resume_checkpoint(ckpt_dir, expected_world_size):
    """Fail closed before handing a checkpoint to Trainer/DeepSpeed."""
    ckpt_dir = Path(ckpt_dir)
    expected_world_size = int(expected_world_size)
    if expected_world_size <= 0:
        raise ValueError("expected_world_size must be positive")
    if not is_complete_checkpoint(ckpt_dir):
        raise RuntimeError(f"incomplete Trainer/DeepSpeed checkpoint: {ckpt_dir}")
    try:
        state = json.loads((ckpt_dir / "trainer_state.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid trainer_state.json: {ckpt_dir}") from exc
    match = _CHECKPOINT_DIR_RE.fullmatch(ckpt_dir.name)
    if match and int(state.get("global_step", -1)) != int(match.group(1)):
        raise RuntimeError(
            f"resume global_step does not match directory: {ckpt_dir} "
            f"state={state.get('global_step')}"
        )
    latest = ckpt_dir / "latest"
    if not latest.is_file():
        raise RuntimeError(f"DeepSpeed resume checkpoint has no latest tag: {ckpt_dir}")
    tag = latest.read_text().strip()
    tag_dir = ckpt_dir / tag
    model_shards = [path for path in tag_dir.glob("*model_states.pt") if path.stat().st_size > 0]
    optimizer_shards = [path for path in tag_dir.glob("*optim_states.pt") if path.stat().st_size > 0]
    if not model_shards or len(optimizer_shards) != expected_world_size:
        raise RuntimeError(
            "incomplete DeepSpeed shards: "
            f"path={ckpt_dir}, model={len(model_shards)}, "
            f"optimizer={len(optimizer_shards)}, expected_optimizer={expected_world_size}"
        )
    return {
        "global_step": int(state["global_step"]),
        "tag": tag,
        "model_shards": len(model_shards),
        "optimizer_shards": len(optimizer_shards),
    }


def predelete_for_two_slot_rotation(output_dir, upcoming_step):
    """Free one checkpoint slot while retaining the newest complete recovery.

    This runs *before* DeepSpeed writes ``checkpoint-<upcoming_step>``. The
    newest existing checkpoint must be complete, and an already-existing
    destination is rejected so stale shards cannot mix into a new checkpoint.
    """
    output_dir = Path(output_dir)
    upcoming = output_dir / f"checkpoint-{int(upcoming_step)}"
    if upcoming.exists():
        raise RuntimeError(
            f"refusing two-slot save because destination already exists: {upcoming}"
        )

    checkpoints = []
    if output_dir.is_dir():
        for path in output_dir.iterdir():
            match = _CHECKPOINT_DIR_RE.fullmatch(path.name)
            if match and path.is_dir():
                checkpoints.append((int(match.group(1)), path))
    checkpoints.sort(key=lambda item: item[0])

    if len(checkpoints) <= 1:
        return []

    newest_step, newest = checkpoints[-1]
    if not is_complete_checkpoint(newest):
        raise RuntimeError(
            "refusing to delete older checkpoints because the newest recovery "
            f"point is incomplete: step={newest_step} path={newest}"
        )

    deleted = []
    for _, path in checkpoints[:-1]:
        shutil.rmtree(path)
        deleted.append(str(path))
    return deleted
