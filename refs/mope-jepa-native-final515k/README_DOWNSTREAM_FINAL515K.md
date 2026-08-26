# MoPE final515k downstream feature extraction

## Paths on MOPE server

- Model code: `/data2/mope-jepa-native-final515k`
- Latest symlink: `/data2/mope-jepa-native-latest`
- Inference script: `/data2/mope-jepa-native-final515k/extract_native_mope_features_final515k.py`
- Checkpoint: `/data2/mope-jepa-assets/jepa_checkpoints/native_mope_b_dense8_moe8_top1_shared1_anchor1_final515k_3dpos_ep100_warm3_cos_lr75e6_min25e6/checkpoint-50.pth`

## Model

Final MoPE native encoder:

- 12 encoder blocks
- dense blocks: 0-7
- MoE blocks: 8-11
- 8 routed experts, top-1 routing
- 1 shared expert
- 3D positional setting from the final515k run
- default feature output: `8 x 768` tokens with `--pool temporal`
- architecture positional encoding: encoder `pos_embed_type=3d_sincos`, predictor `predictor_pos_embed_type=3d_sincos`

## Sampling

The downstream script uses the same intended input form as final training:

- 16 RGB frames
- 4 temporal groups x 4 frames per group
- resize + center crop to 224 x 224
- robust OpenCV sequential decode; if a video has fewer than 16 decodable frames, the last frame is repeated to pad to 16

## Example

```bash
source /data2/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate /data2/envs/mope-jepa
cd /data2/mope-jepa-native-final515k

CUDA_VISIBLE_DEVICES=0 python extract_native_mope_features_final515k.py \
  --video /path/to/video.mp4 \
  --save-dir /path/to/save/features \
  --pool temporal
```

The saved `.pt` file contains:

- `feature`: tensor, usually shape `[8, 768]`
- `meta`: video path, checkpoint path, sampling indices, decoded frame count, and feature shape

To use a newer pretraining checkpoint later, keep the same code and only replace `--ckpt` or overwrite the checkpoint file path above.
