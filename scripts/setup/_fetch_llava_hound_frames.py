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
    )

    # 4. Stream through dataset, saving all frames for each needed video_id.
    #    Each dataset item is one frame (WebDataset format: jpeg + __key__ + __url__).
    #    __key__ format: "{video_id}/{frame_idx}" — first part is the video_id.
    #    We scan the full dataset to collect ALL frames per video (not just the first).
    #    Checkpoint: save progress every 1000 frames in case of interruption.

    inspected = 0
    MAX_INSPECT = 3

    frames_saved_this_run = 0
    saved_ids: set = set()   # video_ids with ≥1 frame saved this run

    for item in ds:
        # Debug: print actual field values for first few items
        if inspected < MAX_INSPECT:
            debug_vals = {
                k: (f"<bytes len={len(v)}>" if isinstance(v, (bytes, bytearray))
                    else str(v)[:120])
                for k, v in item.items()
            }
            logger.info(f"[DEBUG] item #{inspected}: {debug_vals}")
            inspected += 1

        # Resolve video_id from item fields
        item_vid = (
            item.get("video_id")
            or item.get("id")
            or item.get("video_name")
            or item.get("clip_id")
        )
        if item_vid is None:
            for key in ("video", "video_path", "file"):
                if key in item and isinstance(item[key], str):
                    item_vid = Path(item[key]).stem
                    break

        # WebDataset format: __key__ is "{video_id}/{frame_idx}"
        if item_vid is None and "__key__" in item:
            key_str = str(item["__key__"])
            candidate = key_str.split("/")[0]
            if candidate in pending_ids or candidate in done_ids:
                item_vid = candidate
            elif key_str in pending_ids or key_str in done_ids:
                item_vid = key_str

        if item_vid is None:
            continue
        if item_vid in done_ids:
            continue  # fully completed in a previous run
        if item_vid not in pending_ids and item_vid not in saved_ids:
            continue  # not needed

        vid_output_dir = os.path.join(output_dir, item_vid)
        frame_files = video_frames.get(item_vid, [])
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
                fname = frame_files[i] if i < len(frame_files) else f"{i + 1:04d}.jpg"
                dest = os.path.join(vid_output_dir, fname)
                if not os.path.exists(dest) and save_image(frame, dest):
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
                            f"video_id={item_vid}: video path {video_path!r} does not exist."
                        )
                    break

        # Case C: WebDataset per-frame format — one jpeg per item
        elif "jpeg" in item:
            jpeg_data = item["jpeg"]
            if isinstance(jpeg_data, (bytes, bytearray)):
                key_str = str(item.get("__key__", ""))
                key_parts = key_str.split("/")
                frame_part = key_parts[-1] if len(key_parts) > 1 else key_str
                try:
                    frame_num = int(frame_part)
                    fname = f"{frame_num:04d}.jpg"
                except (ValueError, TypeError):
                    existing = len(list(Path(vid_output_dir).glob("*.jpg"))) if Path(vid_output_dir).exists() else 0
                    fname = f"{existing + 1:04d}.jpg"
                dest = os.path.join(vid_output_dir, fname)
                if not os.path.exists(dest) and save_image(jpeg_data, dest):
                    success = True

        else:
            logger.warning(
                f"video_id={item_vid}: unrecognised item keys {list(item.keys())}. Skipping."
            )

        if success:
            saved_ids.add(item_vid)
            frames_saved_this_run += 1

            # Checkpoint every 1000 frames
            if frames_saved_this_run % 1000 == 0:
                for vid in saved_ids:
                    done_ids.add(vid)
                save_progress(progress_file, done_ids)
                logger.info(
                    f"[checkpoint] {frames_saved_this_run} frames saved this run, "
                    f"{len(saved_ids)} videos started, "
                    f"{len(done_ids)}/{total_needed} total done"
                )

    # Final: mark all videos with ≥1 frame as done
    for vid in saved_ids:
        done_ids.add(vid)
    save_progress(progress_file, done_ids)

    not_found = pending_ids - done_ids
    if not_found:
        logger.warning(
            f"{len(not_found)} video_ids NOT found in dataset stream: "
            f"{sorted(not_found)[:20]}{'...' if len(not_found) > 20 else ''}"
        )

    logger.info(
        f"Completed: {len(done_ids)}/{total_needed} video_ids "
        f"({frames_saved_this_run} frames saved this run)"
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
