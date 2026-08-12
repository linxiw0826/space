# Part A A1-O Architecture

更新时间：2026-08-10（D-62代码完成与最终静态Review PASS；D-63 execution boundary同步）
状态：D-62三源v2 train/val、engineering subset与coverage-matrix代码已完成且最终静态Review PASS；
执行服务器CPU/GPU均未运行，Gate@CONFIG未批准，正式训练HOLD。v1三分split/四源Gate仅为历史实现。

> **当前覆盖声明**：本轮正式registry固定ADT、Hypersim、ScanNet++ V2，共`375,682 QA /
> 1,355 scenes`。不等待ScanNet；未来接入必须新版本+manifest+Preflight。D-62代码与静态Review
> 已完成，但本文接口尚无执行服务器runtime证据，不得写成CPU/GPU PASS。

> **执行边界**：canonical代码服务器`/u/lwu9/Space_sensing/projects/space`只做开发、只读分析和
> ReviewAgent静态审查，不运行Python、`py_compile`、pytest或CPU/GPU runtime。静态Review通过后
> 同步已审commit到当前执行服务器`user@112.91.161.190:12013`的`/data2/wlx/projects/space`
> （数据根`/data2/wlx/data`），再经单独授权运行CPU验证；GPU阶段仍需对应Gate。CPU证据必须绑定
> 执行服务器commit、命令、exit status、环境和artifact路径。

## 1. 范围与边界

当前代码覆盖论文2 Part A首批工程链：

- ADT、Hypersim、ScanNet++ V2 canonical/exact数据与三源统一scene split；
- 通用多源dataset、CPU state collator、source-balanced采样和actual-input-only对象选择；
- Qwen visual tap、A1-O普通clip-level set-slot side head与D-58 Hungarian loss；
- T0-A、T0-B、matched A0/A1-O训练、全状态checkpoint/resume与A1-O-drop独立无头审计；
- T0-A、三源T0-B、fixed-train-subset overfit/matched runner、独立resource profile与
  checkpoint/resume/head-free/validator coverage matrix（取代独立GUIDE smoke phase）；
- matched A0/A1-O-drop VSI-Bench评测与结果比较。

不包含当前正式registry中的ScanNet、A1-P、A1-OP、persistent slots、Relation Graph、Spatial Memory、
MoPE或A2--A4。ScanNet adapter可继续研究，但不阻塞本轮三源训练；任何接入均生成新数据版本。
对象状态完整评估
指标暂缓；首轮正式效果评估只做VSI-Bench。上述代码属于论文2 Part A，不同步到论文1handoff。

## 2. 数据架构

```text
ADT / Hypersim / ScanNet++ V2 source-native records
  → source adapter + canonical validator
  → scene_states.jsonl + frame_states.jsonl + qa_manifest_exact_verified.jsonl
  → build_unified_three_source_manifest.py
  → unified manifest + exact-input registry + split/report hashes
  → PartAUnifiedDataset
  → deterministic source-balanced indices
  → PartACPUStateCollator
  → exact media + QA labels + actual-input-only K<=384 state targets
```

### 2.1 Canonical与exact-frame合同

Canonical坐标为米制右手系`x=right,y=up,z=back`，相机forward为`-z`。对象ID为
`source:scene:raw_object_id`；类别映射固定canonical vocabulary，源类别保留用于审计。缺失能力写
`null`并由capability/field mask关闭对应loss，禁止0填充伪GT。

ADT先取trajectory、calibration与direct GT共同支持的最大连续raw-frame clip，再在clip内执行GUIDE
动态16--32帧采样并映射回原MP4 raw IDs；禁止邻帧替代、外推和伪标签。全部60,207条QA保留，但
固定为`scene_associated_unlocalized`、`qa_visual_support_verified=false`、
`evidence_frame_indices=null`。Hypersim QA为单帧`frame_verified`。ScanNet++ V2已完成855景
canonical/exact validation。

ScanNet按D-61采用fail-closed QA-level state gate：92,145条合法VSI QA全保留用于QA loss，
但identity completeness是existence/category/center/extent/visibility五项state loss的总前置门。本次
actual input frames中任一可见instance无法经source-native lineage唯一连接为
`frame_info_inst_id → official_object_id → preprocessed_instance_id/3D bbox`，或exact-frame binding不完整，
整条QA的五项state loss全关。不得删掉未对齐对象后继续监督其余对象，否则`L_exist`
会产生假负类；individual field mask只在identity completeness总门通过后使用。

ScanNet support evidence只允许`frame_verified_direct_id`、`qa_only_identity_incomplete`、`invalid`
三值。顺序/同号配对、Hungarian、投影和VLM伪标签不得进入Gold，仅可用于QC/proxy报告。
当前1,201个VSI ScanNet scene的frame-info和3D文件coverage已成立，但direct identity lineage未闭合；
data Gate等待pilot的exact RGB/pose、official aggregation/segs/axisAlignment、label map和预处理provenance小文件，
当前不需全量depth或2D mask。

### 2.2 三源统一manifest与split

D-62 v2合同（代码完成、静态Review PASS、真实构建待执行服务器授权）：

- 基于`schema_version + seed + source_dataset + scene_id`的确定性scene-level分桶；
- split只允许`train|val`；seed42、source+scene稳定hash、约10% val与其余train；scene零交集；
- 完整`(source_dataset, scene_id, qa_id)`join与重复/错源/错scene拒绝；
- 每源exact QA manifest的resolved path、byte size和SHA256 registry；
- unified logical rows hash、文件SHA256、逐源/逐split scene与QA统计；
- 按`source × split`定义的source-balanced weight与确定性round-robin索引；
- 任何`split=smoke` fail closed。

train内部另生成固定、source-stratified engineering subset registry，保存scene/QA IDs、exact visual
inputs与content hash，不按question/loss/performance挑选。该subset在full train中正常出现且不额外
加权；它不是第三个split。工程事务的模型权重、optimizer、scheduler、RNG/sampler状态全部标记
`non-promotable`并丢弃。formal A0/A1-O从同一冻结初始化step0重训。

当前代码的权威路径已迁移到v2并对`split=smoke`/非三源fail closed；三源v2真实逐源/逐split
计数仍未在执行服务器生成。未来ScanNet接入不增量混入本版本，而是创建新registry/schema/
manifest、adapter并重新Preflight。

### 2.3 K=384对象选择

对象集合严格来自本次actual input frames中geometry-valid且可见的对象，不读取question。超过384时
按冻结的可见性证据排序，并以object ID稳定tie-break；manifest/collator显式记录
`selected_object_ids`和`truncated_object_ids`。empty GT产生合法空target。whole-scene overflow
统计不能替代actual-input per-QA overflow，禁止静默截断。

## 3. 模型与loss

```text
GUIDE/Qwen standard QA forward
  ├─ QA logits / QA loss
  └─ final hidden at authoritative visual-prefix positions
       → [B,Nvisual,D] + valid mask + exact frame spans
       → K=384 learned set slots
       → independent cross-attention side decoder
       → existence / category / normalized center / normalized extent /
          per-actual-view visibility
       → detached Hungarian assignment + D-58 state loss
```

A0不实例化state head；A1-O仅在训练时增加side head与
`total_loss = qa_loss + lambda_state * state_loss`。side branch不向语言序列插token、不回写QA
hidden/logits。A0与A1-O检测到MoPE即fail closed。

Hungarian pair cost对valid的existence BCE、category NLL、scene-normalized center/extent Smooth-L1
和actual-view visibility BCE等权平均；无效项移除并按参与项数归一。匹配后existence监督全部384
slots，其他分项只监督matched且valid对象；empty GT只计算全负existence。

## 4. 训练、checkpoint与日志

`scripts/parta/train_parta.py --arm a0|a1o`与`src/parta/runner.py`共用同一数据、optimizer、
schedule和预算。matched合同绑定manifest、source内容、初始化、seed、actual frame binding、
distributed strategy、world size、per-rank/effective global batch、gradient accumulation、workers及
其他非白名单训练语义；只允许state-head/state-loss形成arm差异。

训练checkpoint原子保存model/head、optimizer、scheduler、global step、epoch、sampler cursor、RNG、
完整resolved config和artifact digests。DDP/FSDP入口按`cuda:LOCAL_RANK`绑定；FSDP使用full model/
optimizer state collective与官方optimizer state转换。逐step JSONL记录QA loss、五项state loss、
总loss、lambda、shared/head梯度、有效GT/slots、matching、source、frames、吞吐和峰值显存。

A1-O-drop不重训。训练进程只导出head-free artifact；`audit_a1o_drop_load.py`在独立、未实例化
state head的GUIDE模型中strict加载并审计source final checkpoint、dropped/missing/unexpected keys、
GUIDE/VGGT digest与forward身份。

## 5. T0与统一GPU Gate

- T0-A已在真实GPU和五个固定fixture上`complete_passed`；它证明工程合同和梯度路径，不是论文效果。
- `run_t0_b.py`在source-stratified batches上分别测共享参数`g_QA/g_state`，检查expected-source
  registry、loss/mask/component、matching、exact frames、checkpoint内容恢复和provenance一致性。
- CPU mock只能产生`awaiting_gpu`，不能产生formal pass。

D-62 coverage matrix（代码完成、静态Review PASS、runtime receipts尚未生成）为：

```text
T0-A + 三源T0-B + fixed-train-subset overfit/matched real runner
     + independent 16/24/32 resource profile
     + checkpoint/resume + head-free val audit + validators/resource gates
     → pre-authorization → Gate@CONFIG: APPROVE → freeze → formal A0/A1-O from step 0
```

正式producer绑定仓库canonical path及预注册内容SHA。每阶段报告绑定私有新目录、run/command/
config/manifest/model/checkpoint/exact-frame身份；旧报告、mock和synthetic artifact不能冒充GPU证据。
ScanNet不再触发`awaiting_data`；registry必须精确等于冻结三源。工程subset产物不可promote到
正式训练。profiling至少要求一个非OOM、finite且峰值显存低于设备总显存90%的候选。coverage
matrix通过仍只得到`authorization_pending`；必须由外部
`Gate@CONFIG: APPROVE`生成freeze artifact，再以`--finalize-only`基于原phase artifacts授权，不能
自动冻结配置或重跑GPU阶段。

## 6. VSI-Bench评测

`run_matched_vsibench_eval.py`只运行A0与A1-O-drop，二者使用相同VSI-Bench task/generation配置。
评测入口显式注册`qwen3_vl_parta`插件；A0 checkpoint和A1-O-drop source checkpoint均绑定completed
training run的final path、SHA、role与global step。fresh receipts绑定plan、arm、artifact、task、
sample identity和raw-result SHA；比较器拒绝旧结果、双臂同一raw文件或歧义score schema。

checkpoint只用val按最低source-balanced val QA loss选择，tie取最早step。VSI-Bench在冻结
checkpoint/config后one-shot matched运行，不用于调参。主门是scene/video-level paired bootstrap
95% CI的`Δoverall`：下界`>0`为可靠GO，点估计正但CI跨0为inconclusive，点估计非正为NO-GO；
八个子项仅诊断。当前没有真实A0/A1-O-drop训练或评测结果。

## 7. 当前状态

### D-62 code complete / static Review PASS / runtime pending

- 三源v2 train/val schema与engineering subset registry代码；真实split计数/registry artifact待执行；
- 旧四源/三分split/guide_smoke gate已迁移为三源coverage matrix；
- engineering state non-promotion与formal step0重训断言；
- val-only checkpoint selector与paired-bootstrap decision report；
- exact T0-A/T0-B、validator、trusted worker/producer、formal startup与VSI one-shot绑定。

最终全量静态Review PASS（2026-08-10 orchestrator mailbox）且`git diff --check`通过；Review覆盖
`src/parta/{unified_data,gate_orchestration,checkpoint,runner,t0_b_runtime,checkpoint_selection,
vsibench_eval,worker_trust}.py`、对应Part A CLI与tests。该Review没有生成执行artifact；按D-63，
Python/py_compile/pytest/真实数据构建/CPU runner/GPU均未运行。

### D-62前已完成并Review通过（legacy实现证据）

- 三源unified manifest/split、dataset/CPU collator、source-balanced sampler与K384 audit；
- matched A0/A1-O runner、全状态checkpoint/resume、DDP/FSDP代码路径及独立head-free audit；
- T0-B runner与机器可判report；
- overfit/profile/GUIDE smoke统一编排、producer/freeze/finalize合同（旧编排；D-62要求迁移）；
- matched A0/A1-O-drop VSI-Bench eval与结果身份验证。

以上是D-62前的历史静态/CPU证据，不表示D-62改动已在当前执行服务器通过，也不表示GPU Gate
或正式训练通过。

### Historical / superseded prerequisites

- ScanNet lineage与production adapter可继续作为未来数据版本工作，但不再阻塞当前三源版本；
- v1四源manifest与独立smoke等待逻辑被D-62覆盖，保留仅作历史实现追溯。

### Awaiting execution authorization

- 已审代码commit同步与执行服务器revision/clean audit；
- `py_compile`、Part A pytest、真实三源v2 manifest/engineering subset构建；
- validator、dataloader/collator、K384、Gate producer/trust CPU验证。

### Awaiting GPU and Gate

- 三源T0-B；
- A1-O small-set overfit；
- 16/24/32帧显存与吞吐profiling；
- fixed-train-subset matched runner、FSDP/resume/head-free forward与coverage matrix；
- 正式A0/A1-O训练与A0/A1-O-drop VSI-Bench eval。

### Provisional / TBD

- 已冻结：`seed=42`、`K=384`、动态16--32帧、val约0.10且无smoke split；
- T0-B设计合同50--100 batches，开发runner可先20--50；finite/nonzero通过率至少95%；
- 候选`lambda_state=clip(0.1*g_QA/g_state,0.01,0.1)`；
- `lambda_state`、steps/epoch、batch/world size、LR、save/val频率、峰值显存与吞吐均须在profile后
  由`Gate@CONFIG`冻结。

正式训练Gate当前关闭，且未获得正式训练授权。
