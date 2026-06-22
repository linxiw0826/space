# src/ — 目录职责说明

> 生成时间：2026-06-22（基于 `state/analyses/20260622_src_structure_audit.md` 的 §A 目录用途清单 + §C 活/死判定 + §E 重复梳理）。
>
> **目的**：消除"多套 `data/` / `train/` 像是重复"的命名假象。它们**职责不同、并非重复**。重组时只删了 `src/data/` 与 `src/train/` 两个空桩（见文末"已删"节）。

---

## 活跃顶层目录

| 目录 | 装什么 | 活/死 | 关键文件 |
|------|--------|-------|----------|
| `analysis/` | 离线探针 / 分析脚本（oracle 路由上限、MoPE 分层 probe、VLM4D 方向标签构建） | 活（独立工具，**无跨目录 import**，仅 stdlib+numpy） | `oracle_route_headroom.py`、`mope_probe_layers.py`、`mope_layer_features.py`、`build_vlm4d_direction_labels.py` |
| `eval/` | **lmms-eval 插件包**（包名 `src.eval`，被 `LMMS_EVAL_PLUGINS="src.eval"` 写死）。注册 6 个 `qwen3_vl_mope*` model type | 活（推理入口必经） | `models/qwen3_vl_mope.py`（925 行，所有模型类本体）+ 5 个 1 行 re-export 桩（`qwen3_vl_mope_{concat,crossattn,qformer,router,zeroshot}.py`） |
| `model/` | 本项目核心 MoPE 模块 | 活（训练 + 推理都 import） | `mope_encoder.py`、`mope_projector.py`、`mope_patch.py` |
| `preprocess/` | VSI-590K 数据预处理（生 JSON/JSONL） | 活（独立脚本，仅 stdlib + cv2/tqdm） | `preprocess_vsi590k.py` |
| `train_framework/` | **本项目训练入口层**：调用 qwenvl 提供的 trainer，加 MoPE 支持 | 活（训练唯一入口） | `train_space.py`（torchrun 入口）、`argument.py`、`data/mope_data_wrapper.py`（MoPE 数据集/collator 包装器，**无 `__init__.py`**） |
| `qwenvl/` | GUIDE `qwen-vl-finetune/qwenvl/` 的工作副本 + 本项目小改。**不是只读 vendor** | 活（被 train_space/eval 深度依赖） | `data/__init__.py`（**data_dict 真正定义点，第 97 行**）、`data/data_processor.py`、`model/modeling_qwen3_vl.py`（项目改过 RoPE/gate）、`train/trainer.py`+`sampler.py`、`model/vggt/*`（VGGT-1B 子树，~50 文件） |
| `vendor/` | 第三方代码副本 | **半活** | `lmms-eval/`（582 文件，**含 3 个项目活改动**，见下）+ `mope/`（5 文件，纯死代码，被 sys.path 动态加载） |

---

## 三套 `data/` 的真相（**不是重复**）

曾经存在三个名字带 `data` 的目录，职责完全不同：

| 路径（重组后） | 职责 | 是不是活 |
|----------------|------|----------|
| `qwenvl/data/__init__.py:97` | **dataset registry**（`data_dict` 字典：vsi590k_spar / vsi590k_video / spar_234k / llava_hound_64k 等），被 `data_processor` 经 `--dataset_use` 查表 | 活（训练必经） |
| `train_framework/data/mope_data_wrapper.py` | **dataset/collator wrapper**（`MoPEDatasetWrapper` / `MoPECollatorWrapper`，E-02+/E-10 包装数据集注入 `mope_frames`）。**无 `__init__.py`，连包都不是** | 活（训练 E-02+ 用） |
| ~~`src/data/`~~ | 原本是纯文档桩（只一个 `__init__.py`，内容是指向 **不存在** 文件的说明文字） | 死 → **2026-06-22 已删** |

**关键纠错**：旧文档（`setup_pythonpath.sh` 旧注释、`project_map.md` Part 3/4）声称"`src/train_framework/data/__init__.py` shadow GUIDE 的 data、定义 data_dict"——**这是 STALE**。该目录从未有 `__init__.py`，data_dict 唯一定义点是 `qwenvl/data/__init__.py:97`。已在本次重组修正。

---

## 三套 `train/` 的真相（**不是重复**）

| 路径（重组后） | 职责 | 是不是活 |
|----------------|------|----------|
| `train_framework/train_space.py` | **本项目训练入口**（torchrun 调它） | 活 |
| `qwenvl/train/{trainer,sampler}.py` | GUIDE 提供的 trainer / sampler，被 train_space import | 活（被 import） |
| `qwenvl/train/{train_qwen,argument}.py` | GUIDE 原训练入口，本项目用 `train_space.py` 取代了它们（理论可删，但属于 qwenvl 副本完整性，**不动**） | 半死（GUIDE 遗留，本项目不直接用） |
| ~~`src/train/`~~ | 原本是空桩（只一个 `__init__.py`，1 行注释） | 死 → **2026-06-22 已删** |

---

## `vendor/` 的半活属性

vendor 看似只读第三方代码，但 `vendor/lmms-eval/` 相对 GUIDE 的副本有 **3 个被改/新增的活文件**：

| 文件 | 状态 | 说明 |
|------|------|------|
| `lmms_eval/models/simple/qwen3_vl_my.py` | MODIFIED vs GUIDE | `Qwen3_VL_MY` 基类被项目改过（所有 eval mope 模型继承它） |
| `lmms_eval/tasks/vsibench/utils.py` | MODIFIED vs GUIDE | vsibench 评测指标逻辑（本项目活代码，所有 VSI eval 跑它） |
| `lmms_eval/tasks/vlm4d/utils.py` | NEW（GUIDE 无） | VLM4D 评测任务（本项目新增） |

`vendor/mope/`（5 文件）则是 `refs/mope/` 的**纯净副本**（逐文件一致），被 `mope_encoder.py:16-21` 经 `sys.path.insert` 动态加载，代码本身不动。

**重组铁律**：vendor 整体只读，不可改名 / 不可移动 / 不可"去重"删除 GUIDE 也有的文件（会误删上面 3 个活改动）。

---

## PYTHONPATH 四段（不可乱改）

`scripts/_common/setup_pythonpath.sh` 设的 PYTHONPATH：

```
${SPACE_ROOT}/src/train_framework : ${SPACE_ROOT} : ${GUIDE_ROOT}=${SPACE_ROOT}/src : ${MOPE_ROOT}=${SPACE_ROOT}/src/vendor/mope
```

| 段 | 提供 | 去掉会破什么 |
|----|-------|--------------|
| 1. `src/train_framework` | `train_framework.*`（但代码实际用 `src.train_framework.*`，靠段 2 解析） | 当前**不破**（grep 无 `import train_framework` 写法）。历史意图是 shadow data，但 shadow 不存在 → **冗余段**，保留仅为向后兼容 |
| 2. `${SPACE_ROOT}`（项目根） | `src.*`（含 `src.train_framework.*`、`src.eval.*`、`src.model.*`、`src.qwenvl.*`） | **CRITICAL 破**：train_space.py:98 `from src.train_framework.argument`、eval `LMMS_EVAL_PLUGINS="src.eval"` |
| 3. `${GUIDE_ROOT}=${SPACE_ROOT}/src` | `qwenvl.*`、`model.*`、`eval.*`、`analysis.*`、`preprocess.*` 顶层包 | **CRITICAL 破**：`from qwenvl.model...`、`from model.mope_patch` 全靠这层 |
| 4. `${MOPE_ROOT}=src/vendor/mope` | `models.*`（MoPE timm 注册） | 训练/推理会破，但 `mope_encoder.py:16-21` 已自己 sys.path.insert 兜底 → 实际冗余 |

---

## 已删（2026-06-22 重组）

- `src/data/` — 纯文档桩（1 文件，0 真实 import，已 grep 确认）
- `src/train/` — 空桩（1 文件 1 行注释，0 真实 import，已 grep 确认）

详见 `state/analyses/20260622_src_structure_audit.md` §G.1（可自由删）+ §H.3（必配套文档修正）+ `state/analyses/20260622_repo_reorg_manifest.md`（本次重组 manifest）。
