import json

import pytest

from scripts.preprocess.migrate_vsi590k_spar_final515k import migrate


def _row(dataset, scene):
    return {
        "id": f"{dataset}-{scene}",
        "image": ["frame_0.jpg"],
        "spar_info": json.dumps({"dataset": dataset, "scene_name": scene}),
    }


def test_migration_resolves_flat_and_preferred_nested_video(tmp_path):
    flat = tmp_path / "scannet" / "scene1.mp4"
    nested = tmp_path / "procthor" / "scene2" / "raw_navigation_camera__0.mp4"
    flat.parent.mkdir(parents=True)
    nested.parent.mkdir(parents=True)
    flat.touch()
    nested.touch()

    original = [_row("scannet", "scene1"), _row("procthor", "scene2")]
    result = migrate(original, tmp_path)

    assert result[0]["mope_video"] == str(flat.resolve())
    assert result[1]["mope_video"] == str(nested.resolve())
    assert "mope_video" not in original[0]
    assert result[0]["image"] == original[0]["image"]


def test_migration_fails_closed_when_source_video_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="dataset=scannet scene=missing"):
        migrate([_row("scannet", "missing")], tmp_path)
