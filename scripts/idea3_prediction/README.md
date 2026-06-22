# scripts/idea3_prediction/ — 论文2 Stage 1 预测头脚本（待实现）

> 占位目录。论文2 主线 = **「辅助预测头范式」**（详见 `state/paper2_design.md`）。

## Stage 1 三路对照实验（待实现）

| 实验 | 一句话 | 状态 |
|------|--------|------|
| **E-15**（预测 only） | 给 LLM 加辅助 MoPE-latent 预测头（Cambrian-P 式：2 层 MLP off LLM hidden、MSE+cosine、推理丢头），不喂 MoPE 特征。验证"预测 > 喂"。 | ❌ 待实现 |
| **E-16**（喂 + 预测） | E-03a 的喂 MoPE 特征 + E-15 的预测头，两 flag (`mope_feed_features` / `mope_lfp_enable`) 正交。 | ❌ 待实现 |
| **E-03a**（喂 only，基线） | = `scripts/idea1_feature/train/train_e03a_mope_crossattn_two_stage.sh`。已有产物。 | ✅ 已完成 |

## 设计约束

- 两 flag 默认关 = 字节级等价 E-03a（ReviewAgent 重点核：NTP 逐位等价 / time-bin 索引 784=4×196 / 三对照正交 / encoder 单次复用）。
- 推理丢头（Cambrian-P 式），路径同 E-03a；不做 Cambrian-S 式"推理留头当 surprise"。
- spec 详见 `state/analyses/20260619_stage1_lfp_head_integration.md`。

## 实现后填充

待 CodeAgent 写完 `train_e15_*.sh` / `train_e16_*.sh` / `eval_e15_*.sh` / `eval_e16_*.sh` 后，本目录的 `train/` 与 `eval/` 子目录将各获 2 个脚本。
