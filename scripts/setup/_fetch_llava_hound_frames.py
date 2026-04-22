"""
_fetch_llava_hound_frames.py

Parse llava_hound_64k.json, extract required video_ids, then selectively
download video frames from HuggingFace ShareGPTVideo/train_video_and_instruction
using streaming mode to avoid downloading the full dataset.

Supports resumable progress via a JSON progress file.

Usage:
    python3 _fetch_llava_hound_frames.py \
        --ann_json /path/to/llava_hound_64k.json \
        --output_dir /path/to/llava_hound/frames \
        --progress_file /path/to/.done/llava_hound_frames_progress.json
"""

import os

# Must be set before any huggingface import
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import json
import logging
import subprocess
import tempfile
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Progress file helpers (atomic write)
# ---------------------------------------------------------------------------

def load_progress(progress_file: str) -> set:
    """Return the set of already-completed video_ids from the progress file."""
    p = Path(progress_file)
    if not p.exists():
        return set()
    try:
        with open(p, "r") as f:
            data = json.load(f)
        return set(data.get("done", []))
    except Exception as e:
        logger.warning(f"Could not read progress file {progress_file}: {e}. Starting fresh.")
        return set()


def save_progress(progress_file: str, done_ids: set) -> None:
    """Atomically write the current set of done video_ids to the progress file."""
    p = Path(progress_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = str(p) + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump({"done": sorted(done_ids)}, f)
        os.rename(tmp_path, str(p))
    except Exception as e:
        logger.warning(f"Could not write progress file {progress_file}: {e}")


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------

def parse_annotation(ann_json: str) -> dict:
    """
    Parse llava_hound_64k.json and return a mapping:
        video_id -> set of frame filenames needed

    Expected image path format inside each annotation item:
        llava_hound/frames/{video_id}/{frame_file}
    We extract video_id as path.split("/")[2].
    """
    logger.info(f"Loading annotation file: {ann_json}")
    with open(ann_json, "r") as f:
        data = json.load(f)

    # data may be a list of items, or a dict with a key like "data"
    if isinstance(data, dict):
        items = data.get("data", data.get("annotations", list(data.values())[0]))
    else:
        items = data

    video_frames: dict[str, set] = {}  # video_id -> set of frame filenames

    skipped = 0
    for item in items:
        # image/video paths may live under different keys
        images = item.get("image", item.get("images", item.get("video", None)))
        if images is None:
            skipped += 1
            continue

        if isinstance(images, str):
            images = [images]

        for img_path in images:
            parts = img_path.replace("\\", "/").split("/")
            # Format: llava_hound/frames/{video_id}  (3 parts, no frame filename)
            #      or llava_hound/frames/{video_id}/{frame_file}  (4 parts)
            if len(parts) < 3:
                logger.warning(
                    f"Unexpected image path format (expected >=3 parts): {img_path!r}"
                )
                continue
            video_id = parts[2]
            if video_id not in video_frames:
                video_frames[video_id] = set()
            if len(parts) >= 4:
                video_frames[video_id].add(parts[3])

    if skipped:
        logger.warning(f"Skipped {skipped} annotation items with no image/video field.")

    logger.info(f"Parsed {len(video_frames)} unique video_ids from {len(items)} items.")
    return {vid: sorted(frames) for vid, frames in video_frames.items()}


# ---------------------------------------------------------------------------
# Frame extraction from a video file via ffmpeg
# ---------------------------------------------------------------------------

def extract_frames_ffmpeg(video_path: str, output_dir: str) -> bool:
    """
    Use ffmpeg to extract frames at 1 fps into output_dir as %04d.jpg.
    Returns True on success.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", "fps=1",
        os.path.join(output_dir, "%04d.jpg"),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            logger.warning(
                f"ffmpeg failed for {video_path}: {result.stderr[-500:]}"
            )
            return False
        return True
    except FileNotFoundError:
        logger.warning("ffmpeg not found on PATH. Cannot extract frames from video files.")
        return False
    except subprocess.TimeoutExpired:
        logger.warning(f"ffmpeg timed out for {video_path}")
        return False


# ---------------------------------------------------------------------------
# Save PIL / bytes image
# ---------------------------------------------------------------------------

def save_image(img_obj, dest_path: str) -> bool:
    """Save a PIL Image or raw bytes to dest_path."""
    try:
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        # PIL Image
        if hasattr(img_obj, "save"):
            img_obj.save(dest_path)
            return True
        # raw bytes
        if isinstance(img_obj, (bytes, bytearray)):
            with open(dest_path, "wb") as f:
                f.write(img_obj)
            return True
        logger.warning(f"Unknown image object type: {type(img_obj)} for {dest_path}")
        return False
    except Exception as e:
        logger.warning(f"Could not save image to {dest_path}: {e}")
        return False


# ---------------------------------------------------------------------------
# Main download logic
# ---------------------------------------------------------------------------

def fetch_frames(ann_json: str, output_dir: str, progress_file: str) -> None:
    # 1. Load progress
    done_ids = load_progress(progress_file)
    logger.info(f"Progress file: {len(done_ids)} video_ids already completed.")

    # 2. Parse annotation
    video_frames = parse_annotation(ann_json)
    total_needed = len(video_frames)
    pending_ids = {vid for vid in video_frames if vid not in done_ids}

    logger.info(
        f"Total video_ids needed: {total_needed}  |  "
        f"Already done: {len(done_ids)}  |  "
        f"To download: {len(pending_ids)}"
    )

    if not pending_ids:
        logger.info("All video_ids already downloaded. Nothing to do.")
        return

    # 3. Load dataset in streaming mode
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error(
            "The 'datasets' library is not installed. "
            "Install with: pip install datasets"
        )
        raise

    logger.info("Loading ShareGPTVideo/train_video_and_instruction in streaming mode ...")
    ds = load_dataset(
        "ShareGPTVideo/train_video_and_instruction",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    # 4. Inspect first few items to understand field names
    field_names_printed = False
    inspected = 0
    MAX_INSPECT = 3

    total_done_this_run = 0
    total_to_do = len(pending_ids)

    # We need to scan the entire stream until all pending_ids are resolved
    remaining = set(pending_ids)

    for item in ds:
        # Debug: print field names for the first MAX_INSPECT items
        if not field_names_printed and inspected < MAX_INSPECT:
            logger.info(f"[DEBUG] Dataset item keys (item #{inspected}): {list(item.keys())}")
            inspected += 1
            if inspected >= MAX_INSPECT:
                field_names_printed = True

        if not remaining:
            logger.info("All pending video_ids processed. Stopping stream early.")
            break

        # Resolve video_id from item
        item_vid = (
            item.get("video_id")
            or item.get("id")
            or item.get("video_name")
            or item.get("clip_id")
        )
        if item_vid is None:
            # Try to derive from a video path field
            for key in ("video", "video_path", "file"):
                if key in item and isinstance(item[key], str):
                    item_vid = Path(item[key]).stem
                    break

        if item_vid not in remaining:
            continue

        # Resolve output directory for this video_id
        vid_output_dir = os.path.join(output_dir, item_vid)
        frame_files = video_frames[item_vid]

        success = False

        # Case A: item contains pre-extracted frames as PIL images or bytes
        frames_field = None
        for key in ("frames", "images", "image_list"):
            if key in item:
                frames_field = key
                break

        if frames_field is not None:
            frames = item[frames_field]
            if not isinstance(frames, (list, tuple)):
                frames = [frames]
            Path(vid_output_dir).mkdir(parents=True, exist_ok=True)
            saved = 0
            for i, frame in enumerate(frames):
                # Use the i-th expected frame filename if available
                if i < len(frame_files):
                    fname = frame_files[i]
                else:
                    fname = f"{i + 1:04d}.jpg"
                dest = os.path.join(vid_output_dir, fname)
                if save_image(frame, dest):
                    saved += 1
            if saved > 0:
                success = True
            else:
                logger.warning(
                    f"video_id={item_vid}: frames field '{frames_field}' "
                    f"found but no images could be saved."
                )

        # Case B: item contains a video file path → use ffmpeg
        elif any(k in item for k in ("video", "video_path", "file")):
            for key in ("video", "video_path", "file"):
                if key in item and isinstance(item[key], str):
                    video_path = item[key]
                    if os.path.exists(video_path):
                        success = extract_frames_ffmpeg(video_path, vid_output_dir)
                    else:
                        logger.warning(
                            f"video_id={item_vid}: video path {video_path!r} "
                            f"does not exist on disk."
                        )
                    break

        else:
            logger.warning(
                f"video_id={item_vid}: cannot determine frame/video field from keys "
                f"{list(item.keys())}. Skipping."
            )

        if success:
            remaining.discard(item_vid)
            done_ids.add(item_vid)
            total_done_this_run += 1

            # Persist progress after each completed video_id
            save_progress(progress_file, done_ids)

            # Print progress every 100 completions
            if total_done_this_run % 100 == 0:
                logger.info(
                    f"[{total_done_this_run}/{total_to_do}] "
                    f"video_ids downloaded this run  "
                    f"(total done: {len(done_ids)}/{total_needed})"
                )

    if remaining:
        logger.warning(
            f"{len(remaining)} video_ids were NOT found in the dataset stream: "
            f"{sorted(remaining)[:20]}{'...' if len(remaining) > 20 else ''}"
        )

    total_done_overall = len(done_ids)
    logger.info(
        f"Completed: {total_done_overall}/{total_needed} video_ids "
        f"({total_done_this_run} newly downloaded this run)"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Selectively download LLaVA-Hound video frames from HuggingFace."
    )
    parser.add_argument(
        "--ann_json",
        required=True,
        help="Path to llava_hound_64k.json annotation file.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for frames (e.g. .../llava_hound/frames).",
    )
    parser.add_argument(
        "--progress_file",
        required=True,
        help=(
            "Path to JSON progress file. Format: {\"done\": [\"id1\", ...]}. "
            "Created automatically if it does not exist."
        ),
    )
    args = parser.parse_args()

    if not os.path.isfile(args.ann_json):
        logger.error(f"ann_json not found: {args.ann_json}")
        raise FileNotFoundError(args.ann_json)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    fetch_frames(
        ann_json=args.ann_json,
        output_dir=args.output_dir,
        progress_file=args.progress_file,
    )


if __name__ == "__main__":
    main()
