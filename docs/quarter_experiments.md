# VSI-590K quarter experiments

The reproducible quarter manifest is generated with
`scripts/preprocess/sample_vsi590k_source_stratified_quarter.py` using
row-level sampling, seed `20260912`, fraction `0.25`. The server artifact is
`/data2/wlx/data/vsi590k_processed/vsi590k_spar_590k_quarter_stratified.json`
(`87,217` rows, with source proportions preserved).

All three experiments use the same manifest, one epoch, per-device batch 2,
gradient accumulation 6 (effective global batch 48), cosine-with-warmup and
warmup ratio 0.03 on GPUs `1,3,5,6`.

* `baseline_quarter_4b`: initialized from `guide_reproduced/4b`, no MoPE.
* `mope73_projector_quarter_lr1e5_4b`: initialized from the quarter baseline;
  frozen MoPE encoder and Qwen, trainable projector at `1e-5`.
* `mope73_projector_lora_quarter_lr1e5_4b`: same initialization and MoPE;
  projector at `1e-5` plus LoRA at `5e-6` on the final eight language layers,
  q/k/v/o, rank 8, alpha 16, dropout 0.05.

The MoPE encoder is the PhysBench-posttrained artifact
`checkpoint-73-posttrain-physbench-fullmope-epoch44.pth`, derived from
checkpoint-73. Its internal metadata still reports `epoch=73`; the filename
is the authoritative identifier for this post-training artifact. It uses the
final515k 3D-sincos recipe:
16 frames, 4 groups x 4 frames, input 224, temporal pooling, output
`[B, 8, 768]`. Quarter eval wrappers are named
`eval_{baseline,mope73_projector,mope73_projector_lora}_quarter_{vsibench,vlm4d}.sh`.
They enforce isolated result paths and the PhysBench-posttrained checkpoint for
MoPE runs. No feature scaling is applied in these experiments.

## Naming and path contract

Training, evaluation, log, and result identifiers must remain aligned:

| experiment | training script | checkpoint | eval scripts |
|---|---|---|---|
| `baseline_quarter_4b` | `train_baseline_quarter_4b.sh` | `/data2/wlx/output/train/baseline_quarter_4b` | `eval_baseline_quarter_{vsibench,vlm4d}.sh` |
| `mope73_projector_quarter_lr1e5_4b` | `train_mope73_projector_quarter_lr1e5.sh` | `/data2/wlx/output/train/mope73_projector_quarter_lr1e5_4b` | `eval_mope73_projector_quarter_{vsibench,vlm4d}.sh` |
| `mope73_projector_lora_quarter_lr1e5_4b` | `train_mope73_projector_lora_quarter_lr1e5.sh` | `/data2/wlx/output/train/mope73_projector_lora_quarter_lr1e5_4b` | `eval_mope73_projector_lora_quarter_{vsibench,vlm4d}.sh` |

Each eval writes to the matching `/data2/wlx/output/eval/{vsibench,vlm4d}/<experiment>` directory and uses an `<experiment>_*` log prefix. Wrappers must use canonical `/data2/wlx/output` and `/data2/wlx/logs` paths, never a path derived from the source checkout. Quarter wrappers must set the quarter-stratified manifest internally; variables inherited from `activate.sh` must not override explicit checkpoint, output, manifest, or log paths.

Verify before launch:

```bash
git diff --check
bash -n scripts/idea1_feature/train/train_*quarter*.sh scripts/idea1_feature/eval/eval_*quarter*.sh
test -f /data2/wlx/data/vsi590k_processed/vsi590k_spar_590k_quarter_stratified.json
```
