import json

import pytest

from scripts.preprocess.audit_mope_final515k_videos import (
    load_unique_videos,
    sample_indices_4x4,
)


def test_audit_sampler_matches_final515k_contract():
    assert sample_indices_4x4(100) == [
        0, 8, 16, 24, 25, 33, 41, 49,
        50, 58, 66, 74, 75, 83, 91, 99,
    ]


def test_manifest_video_inventory_is_unique_and_complete(tmp_path):
    video = tmp_path / "scene.mp4"
    video.touch()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([
        {"id": "q1", "mope_video": str(video)},
        {"id": "q2", "mope_video": str(video)},
    ]))
    videos, rows = load_unique_videos(manifest)
    assert videos == [video.resolve()]
    assert rows == 2


def test_manifest_video_inventory_rejects_missing_sidecar(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"id": "q1"}]))
    with pytest.raises(ValueError, match="q1"):
        load_unique_videos(manifest)
