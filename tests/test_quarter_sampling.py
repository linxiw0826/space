import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from scripts.preprocess.sample_vsi590k_source_stratified_quarter import sample


def row(source: str, index: int) -> dict:
    return {"id": f"{source}-{index}", "mope_video": f"/data2/wlx/data/VSI-590K/{source}/{index}.mp4"}


def test_sampling_is_deterministic_and_stratified():
    rows = [row("adt", i) for i in range(8)] + [row("scannet", i) for i in range(12)]
    first, stats = sample(rows, 0.25, 123)
    second, _ = sample(rows, 0.25, 123)
    assert [r["id"] for r in first] == [r["id"] for r in second]
    assert stats["adt"]["selected_rows"] == 2
    assert stats["scannet"]["selected_rows"] == 3
