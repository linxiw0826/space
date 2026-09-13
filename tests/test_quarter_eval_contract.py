import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
WRAPPERS = [
    "eval_baseline_quarter_vsibench.sh", "eval_baseline_quarter_vlm4d.sh",
    "eval_mope73_projector_quarter_vsibench.sh", "eval_mope73_projector_quarter_vlm4d.sh",
    "eval_mope73_projector_lora_quarter_vsibench.sh", "eval_mope73_projector_lora_quarter_vlm4d.sh",
]

def test_quarter_eval_wrappers_exist_and_parse():
    for name in WRAPPERS:
        path = ROOT / "scripts/idea1_feature/eval" / name
        assert path.is_file()
        assert subprocess.run(["bash", "-n", str(path)], check=False).returncode == 0

def test_mope_feature_loader_disables_weights_only_default():
    source = (ROOT / "refs/mope-jepa-native-final515k/extract_native_mope_features_final515k.py").read_text()
    assert "weights_only=False" in source

def test_quarter_mope_eval_dry_run_contract(tmp_path):
    manifest = tmp_path / "vsi590k_spar_590k_quarter_stratified.json"
    manifest.write_text("[]")
    env = os.environ.copy()
    env.update({"SPACE_ROOT": str(ROOT), "SPACE_OUTPUT_ROOT": str(tmp_path / "output"),
                "SPACE_LOG_ROOT": str(tmp_path / "logs"), "MOPE_NEW_ALLOW_MISSING_ASSETS": "1",
                "VSI590K_SPAR_ANN": str(manifest), "DRY_RUN": "1"})
    result = subprocess.run(["bash", str(ROOT / "scripts/idea1_feature/eval/eval_mope73_projector_quarter_vsibench.sh")],
                            cwd=ROOT, env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 0
    assert "checkpoint-73.pth" in result.stdout
    assert "mope73_projector_quarter_lr1e5_4b" in result.stdout
