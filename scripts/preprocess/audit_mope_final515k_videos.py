"""Fail-closed full-video audit for the MoPE-final515k training manifest.

Every unique ``mope_video`` is checked in an isolated subprocess so a corrupt
or blocked FFmpeg decode can be killed without hanging the audit.  The worker
replays the owner-provided OpenCV full-decode contract, then verifies that the
bounded Decord loader sees the same frame count, 4x4 indices, and RGB pixels.
No sample is skipped or replaced.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def sample_indices_4x4(total: int) -> list[int]:
    if total <= 0:
        raise ValueError("video has no frames")
    indices: list[int] = []
    for group_id in range(4):
        start = int(np.floor(group_id * total / 4))
        end = max(int(np.floor((group_id + 1) * total / 4)), start + 1)
        indices.extend(np.rint(np.linspace(start, end - 1, 4)).astype(np.int64).tolist())
    return [int(np.clip(index, 0, total - 1)) for index in indices[:16]]


def audit_one(video: Path) -> dict:
    import cv2
    import decord

    started = time.monotonic()
    capture = cv2.VideoCapture(str(video))
    cv_frames = 0
    try:
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            cv_frames += 1
    finally:
        capture.release()
    if cv_frames <= 0:
        raise RuntimeError("OpenCV decoded zero frames")

    indices = sample_indices_4x4(cv_frames)
    wanted = set(indices)
    cv_selected: dict[int, np.ndarray] = {}
    capture = cv2.VideoCapture(str(video))
    try:
        for frame_index in range(cv_frames):
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in wanted:
                cv_selected[frame_index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()
    missing = sorted(wanted.difference(cv_selected))
    if missing:
        raise RuntimeError(f"OpenCV second pass missed indices={missing}")
    cv_batch = np.stack([cv_selected[index] for index in indices])

    reader = decord.VideoReader(str(video), ctx=decord.cpu(0), num_threads=1)
    decord_frames = len(reader)
    if decord_frames != cv_frames:
        return {
            "status": "frame_count_mismatch",
            "video": str(video),
            "opencv_frames": cv_frames,
            "decord_frames": decord_frames,
            "seconds": time.monotonic() - started,
        }
    decord_indices = sample_indices_4x4(decord_frames)
    decord_batch = reader.get_batch(decord_indices).asnumpy()
    difference = np.abs(cv_batch.astype(np.int16) - decord_batch.astype(np.int16))
    pixel_equal = bool(np.array_equal(cv_batch, decord_batch))
    return {
        "status": "pass" if pixel_equal else "pixel_mismatch",
        "video": str(video),
        "opencv_frames": cv_frames,
        "decord_frames": decord_frames,
        "indices": indices,
        "pixel_equal": pixel_equal,
        "max_abs_pixel_difference": int(difference.max(initial=0)),
        "mean_abs_pixel_difference": float(difference.mean()),
        "seconds": time.monotonic() - started,
    }


def _worker(video: str) -> int:
    try:
        result = audit_one(Path(video))
    except Exception as exc:
        result = {
            "status": "decode_error",
            "video": video,
            "error_type": type(exc).__name__,
            "reason": str(exc),
        }
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "pass" else 1


def _run_isolated(script: Path, video: Path, timeout: int) -> dict:
    command = [sys.executable, str(script), "--worker-video", str(video)]
    try:
        completed = subprocess.run(
            command, text=True, capture_output=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "video": str(video),
            "timeout_seconds": timeout,
            "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
        }
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return {
            "status": "worker_failure",
            "video": str(video),
            "returncode": completed.returncode,
            "stderr_tail": completed.stderr[-2000:],
        }
    try:
        result = json.loads(lines[-1])
    except json.JSONDecodeError:
        return {
            "status": "worker_failure",
            "video": str(video),
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
        }
    result["returncode"] = completed.returncode
    if completed.stderr:
        result["stderr_tail"] = completed.stderr[-2000:]
    return result


def load_unique_videos(manifest: Path) -> tuple[list[Path], int]:
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"expected non-empty JSON list: {manifest}")
    missing = [row.get("id", "UNKNOWN") for row in rows if not row.get("mope_video")]
    if missing:
        raise ValueError(f"manifest rows missing mope_video; examples={missing[:10]}")
    videos = sorted({Path(row["mope_video"]).resolve() for row in rows})
    absent = [str(video) for video in videos if not video.is_file()]
    if absent:
        raise FileNotFoundError(f"manifest videos missing; examples={absent[:10]}")
    return videos, len(rows)


def load_passed_state(path: Path) -> dict[str, dict]:
    """Load completed PASS records; tolerate an interrupted final JSONL line."""
    passed: dict[str, dict] = {}
    if not path.is_file():
        return passed
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if item.get("status") == "pass" and item.get("video"):
            passed[str(item["video"])] = item
    return passed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--worker-video")
    args = parser.parse_args()
    if args.worker_video:
        raise SystemExit(_worker(args.worker_video))
    if not args.manifest or not args.report:
        parser.error("--manifest and --report are required")
    if args.workers <= 0 or args.timeout_seconds <= 0:
        parser.error("workers and timeout must be positive")

    videos, rows = load_unique_videos(args.manifest)
    if args.limit > 0:
        videos = videos[: args.limit]
    script = Path(__file__).resolve()
    state_path = args.report.with_suffix(args.report.suffix + ".states.jsonl")
    results_by_video = load_passed_state(state_path)
    selected_video_strings = {str(video) for video in videos}
    results_by_video = {
        video: result for video, result in results_by_video.items()
        if video in selected_video_strings
    }
    remaining = [video for video in videos if str(video) not in results_by_video]
    print(
        f"[start] rows={rows} unique_videos={len(videos)} "
        f"resumed_passes={len(results_by_video)} remaining={len(remaining)} "
        f"workers={args.workers} timeout={args.timeout_seconds}s "
        f"state={state_path}",
        flush=True,
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_handle = state_path.open("a", encoding="utf-8")
    printed_failures = 0
    executor = ThreadPoolExecutor(max_workers=args.workers)
    interrupted = False
    try:
        pending = {
            executor.submit(_run_isolated, script, video, args.timeout_seconds): video
            for video in remaining
        }
        for future in as_completed(pending):
            result = future.result()
            results_by_video[result["video"]] = result
            state_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            state_handle.flush()
            completed_count = len(results_by_video)
            should_print = completed_count == 1 or completed_count % 10 == 0
            if result["status"] != "pass" and printed_failures < 20:
                printed_failures += 1
                should_print = True
            if should_print:
                print(
                    f"[{completed_count}/{len(videos)}] status={result['status']} "
                    f"video={result['video']}",
                    flush=True,
                )
    except KeyboardInterrupt:
        interrupted = True
        print(
            f"INTERRUPTED completed={len(results_by_video)}/{len(videos)} "
            f"state={state_path}; rerun the same command to resume",
            flush=True,
        )
    finally:
        state_handle.close()
        executor.shutdown(wait=not interrupted, cancel_futures=interrupted)
    if interrupted:
        raise SystemExit(130)

    results = list(results_by_video.values())
    results.sort(key=lambda item: item["video"])
    failures = [item for item in results if item["status"] != "pass"]
    slowest = sorted(
        (item for item in results if item.get("seconds") is not None),
        key=lambda item: item["seconds"], reverse=True,
    )[:20]
    report = {
        "schema_version": "mope_final515k_video_audit_v1",
        "status": "complete_passed" if not failures else "failed",
        "manifest": str(args.manifest.resolve()),
        "manifest_rows": rows,
        "unique_videos_audited": len(videos),
        "workers": args.workers,
        "timeout_seconds": args.timeout_seconds,
        "state_jsonl": str(state_path.resolve()),
        "sampling": "4x4_uniform_segments_rint",
        "decoder_contract": "owner_opencv_full_decode_vs_decord_indexed_rgb_exact",
        "failure_count": len(failures),
        "failures": failures,
        "slowest": slowest,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(f".{args.report.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, args.report)
    print(
        f"STATUS={report['status']} videos={len(videos)} failures={len(failures)} "
        f"report={args.report}",
        flush=True,
    )
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
