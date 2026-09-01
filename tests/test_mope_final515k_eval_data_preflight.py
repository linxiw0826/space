import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/preprocess/preflight_mope_final515k_eval_data.py"


VSI_TYPES = (
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
)


def run_preflight(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        text=True,
        capture_output=True,
        check=False,
    )


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_vsibench_metadata_preflight_and_balanced_smoke(tmp_path):
    root = tmp_path / "vsibench"
    annotation = root / "test.jsonl"
    video = root / "scannet" / "scene.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"fixture")
    records = []
    for index, question_type in enumerate(VSI_TYPES):
        record = {
            "dataset": "scannet",
            "scene_name": "scene",
            "question": f"question {index}",
            "question_type": question_type,
            "ground_truth": "A" if index < 6 else "1",
        }
        if index < 6:
            record["options"] = ["A", "B"]
        records.append(record)
    write_jsonl(annotation, records)
    smoke = tmp_path / "smoke.jsonl"
    report = tmp_path / "report.json"

    result = run_preflight(
        "--dataset", "vsibench", "--annotation", annotation,
        "--video-root", root, "--expected-rows", "10", "--expected-videos", "1",
        "--smoke-output", smoke, "--report", report,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(report.read_text())
    assert payload["status"] == "complete_passed"
    assert payload["annotation_rows"] == 10
    assert payload["unique_videos"] == 1
    assert payload["decoded_videos"] == 0
    selected = [json.loads(line) for line in smoke.read_text().splitlines()]
    assert [item["question_type"] for item in selected] == list(VSI_TYPES)


def test_vsibench_preflight_rejects_missing_required_type(tmp_path):
    root = tmp_path / "vsibench"
    video = root / "scannet" / "scene.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"fixture")
    annotation = root / "test.jsonl"
    records = [{
        "dataset": "scannet", "scene_name": "scene", "question": "q",
        "question_type": "object_counting", "ground_truth": "1",
    }]
    write_jsonl(annotation, records)

    result = run_preflight(
        "--dataset", "vsibench", "--annotation", annotation, "--video-root", root,
    )

    assert result.returncode == 2
    assert "lacks required question types" in result.stderr


def test_preflight_rejects_wrong_expected_cardinality(tmp_path):
    root = tmp_path / "vsibench"
    video = root / "scannet" / "scene.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"fixture")
    annotation = root / "test.jsonl"
    records = []
    for index, question_type in enumerate(VSI_TYPES):
        record = {
            "dataset": "scannet", "scene_name": "scene", "question": "q",
            "question_type": question_type, "ground_truth": index,
        }
        if index < 6:
            record["options"] = ["A", "B"]
        records.append(record)
    write_jsonl(annotation, records)

    report = tmp_path / "failed_report.json"
    result = run_preflight(
        "--dataset", "vsibench", "--annotation", annotation, "--video-root", root,
        "--expected-rows", "5130", "--report", report,
    )

    assert result.returncode == 2
    assert "row count mismatch: expected=5130, actual=10" in result.stderr
    payload = json.loads(report.read_text())
    assert payload["status"] == "failed"
    assert payload["error_type"] == "ValueError"
    assert "row count mismatch" in payload["error"]


def test_vlm4d_array_preflight_resolves_urls_and_selects_sources(tmp_path):
    root = tmp_path / "VLM4D"
    records = []
    for source in ("davis", "ego4d", "youtube-vos"):
        video = root / "videos_real" / source / f"{source}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"fixture")
        records.append({
            "id": source,
            "question": "Which option?",
            "choices": {"A": "one", "B": "two", "C": "three", "D": "four"},
            "answer": "two",
            "question_type": "multiple-choice",
            "video": (
                "https://huggingface.co/datasets/shijiezhou/VLM4D/resolve/main/"
                f"videos_real/{source}/{source}.mp4"
            ),
        })
    annotation = tmp_path / "real_mc.json"
    annotation.write_text(json.dumps(records))
    smoke = tmp_path / "smoke.jsonl"

    result = run_preflight(
        "--dataset", "vlm4d", "--annotation", annotation,
        "--video-root", root, "--smoke-output", smoke,
    )

    assert result.returncode == 0, result.stderr
    selected = [json.loads(line) for line in smoke.read_text().splitlines()]
    assert len(selected) == 3
    assert {Path(item["video"]).parent.name for item in selected} == {
        "davis", "ego4d", "youtube-vos",
    }


def test_vlm4d_smoke_count_four_covers_sources_and_adds_new_video(tmp_path):
    root = tmp_path / "VLM4D"
    records = []
    for source, names in {
        "davis": ("first", "extra"),
        "ego4d": ("first",),
        "youtube-vos": ("first",),
    }.items():
        for name in names:
            video = root / "videos_real" / source / f"{name}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"fixture")
            records.append({
                "id": f"{source}-{name}",
                "question": "Which option?",
                "choices": {"A": "one", "B": "two", "C": "three", "D": "four"},
                "answer": "two",
                "question_type": "multiple-choice",
                "video": f"https://host/resolve/main/videos_real/{source}/{name}.mp4",
            })
    annotation = tmp_path / "real_mc.json"
    annotation.write_text(json.dumps(records))
    smoke = tmp_path / "smoke.jsonl"

    result = run_preflight(
        "--dataset", "vlm4d", "--annotation", annotation,
        "--video-root", root, "--smoke-count", "4", "--smoke-output", smoke,
    )

    assert result.returncode == 0, result.stderr
    selected = [json.loads(line) for line in smoke.read_text().splitlines()]
    assert len(selected) == 4
    assert {Path(item["video"]).parent.name for item in selected} == {
        "davis", "ego4d", "youtube-vos",
    }
    assert len({item["video"] for item in selected}) == 4


def test_vlm4d_accepts_numeric_answer_and_choice_values(tmp_path):
    root = tmp_path / "VLM4D"
    records = []
    for source in ("davis", "ego4d", "youtube-vos"):
        video = root / "videos_real" / source / "numeric.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"fixture")
        records.append({
            "question": "How many?", "choices": {"A": 0, "B": 1, "C": 2, "D": 3},
            "answer": 1, "question_type": "multiple-choice",
            "video": f"https://host/resolve/main/videos_real/{source}/numeric.mp4",
        })
    annotation = tmp_path / "real_mc.json"
    annotation.write_text(json.dumps(records))

    result = run_preflight(
        "--dataset", "vlm4d", "--annotation", annotation, "--video-root", root,
    )

    assert result.returncode == 0, result.stderr


def test_vlm4d_preflight_rejects_unmapped_answer(tmp_path):
    root = tmp_path / "VLM4D"
    video = root / "videos_real" / "davis" / "a.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"fixture")
    annotation = tmp_path / "real_mc.json"
    annotation.write_text(json.dumps([{
        "question": "q", "choices": {"A": "one", "B": "two", "C": "three", "D": "four"},
        "answer": "five", "question_type": "multiple-choice",
        "video": "https://host/resolve/main/videos_real/davis/a.mp4",
    }]))

    result = run_preflight(
        "--dataset", "vlm4d", "--annotation", annotation, "--video-root", root,
    )

    assert result.returncode == 2
    assert "answer does not match any choice" in result.stderr


def test_vlm4d_preflight_requires_all_and_only_canonical_sources(tmp_path):
    root = tmp_path / "VLM4D"
    video = root / "videos_real" / "davis" / "a.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"fixture")
    annotation = tmp_path / "real_mc.json"
    annotation.write_text(json.dumps([{
        "question": "q", "choices": {"A": "one", "B": "two", "C": "three", "D": "four"},
        "answer": "one", "question_type": "multiple-choice",
        "video": "https://host/resolve/main/videos_real/davis/a.mp4",
    }]))

    result = run_preflight(
        "--dataset", "vlm4d", "--annotation", annotation, "--video-root", root,
    )

    assert result.returncode == 2
    assert "VLM4D video sources mismatch" in result.stderr


def test_final515k_indices_have_exact_contract():
    import importlib.util

    spec = importlib.util.spec_from_file_location("eval_preflight", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    indices = module.final515k_indices(101)
    assert len(indices) == 16
    assert indices[:4] == [0, 8, 16, 24]
    assert all(0 <= index < 101 for index in indices)


def test_rgb_parity_converts_opencv_bgr_and_reports_difference():
    import importlib.util
    import numpy as np

    spec = importlib.util.spec_from_file_location("eval_preflight_parity", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rgb = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    bgr = rgb[..., ::-1]
    module.assert_rgb_frame_parity(rgb, bgr, 7)

    broken = bgr.copy()
    broken[0, 0, 0] = 99
    try:
        module.assert_rgb_frame_parity(rgb, broken, 7)
    except ValueError as exc:
        message = str(exc)
        assert "frame 7 RGB mismatch" in message
        assert "max_abs_diff=" in message
    else:
        raise AssertionError("pixel mismatch was not rejected")
