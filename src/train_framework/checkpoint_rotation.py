"""Fail-safe checkpoint helpers for bounded-space training runs."""

import os
import re
import shutil
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
