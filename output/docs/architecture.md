# Part A A1-O Architecture

更新时间：2026-08-13（D-62 v2三源runtime数据合同与单卡T0-B正式GPU证据PASS）
状态：D-62 v2 canonical、train/val manifest、engineering registry、K=384审计、
三源validator与单卡T0-B均`complete_passed`。当前不再等待D-62 runtime artifacts或T0-B；
下一个权威Gate是真实四卡32帧worst-case resource profile与matched runner。
`Gate@CONFIG`仍未批准，正式训练HOLD。v1三分split/四源Gate仅为历史实现。

> **当前覆盖声明**：本轮正式registry固定ADT、Hypersim、ScanNet++ V2，共`375,498 QA /
> 1,355 scenes`：ADT `60,207/183`、Hypersim `176,774/317`、ScanNet++ V2 `138,517/855`。
> ScanNet++原始VSI清单为856 scenes / 138,701 QA；场景`00dd871005`的逐帧元信息为空
> （`frames_not_nonempty_list`、`frame_count=0`），其184 QA经用户批准显式排除，不参与训练。
> Gold、canonical、exact与交付均一致为855/138,517；审计证据：
> `/data2/wlx/logs/parta/d62_scannetpp_excluded_scene_audit.log`。frozen inventory已修正、复验并用于
> 正式manifest v2；三源runtime数据审计和T0-B已有PASS证据。

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
- legacy coverage matrix曾包含fixed-train-subset overfit；当前权威路线保留三源T0-B、
  matched runner、独立resource profile与checkpoint/resume/head-free/validator，并已取消
  fixed-subset overfit必跑要求（取代独立GUIDE smoke phase）；
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
`evidence_frame_indices=null`。Hypersim QA为单帧`frame_verified`。ScanNet++ V2已完成855景、
138,517 QA的canonical/exact validation；`00dd871005`及其184 QA因空逐帧元信息不进入正式数据。

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

D-62 v2合同（代码、静态Review、执行服务器CPU验证与真实runtime产物均PASS）：

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

当前执行配置把该registry显式设为**每源1 scene**；它仅是身份/复现登记，不要求fixed-subset
overfit或任何小样本训练，也不改变full-train sampler/权重。此配置不改变manifest v2 schema。

当前代码的权威路径已迁移到v2并对`split=smoke`/非三源fail closed。正式
manifest v2已生成`375,498`行：`train=345,642`、`val=29,856`，共`1,355 scenes`，
`train__val` scene交集为0。每源1 scene的engineering subset registry已生成且明确
`promotable_to_formal_training=false`。exact canonical input registry、文件哈希和逐源计数均已绑定。
未来ScanNet接入不增量混入本版本，而是创建新registry/
schema/manifest、adapter并重新Preflight。

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

- T0-A已在真实GPU和五个固定fixture上`complete_passed`，原运行revision为
  `eb2a0912bde0e8db97acc713f261f06b42b68427`。后续T0-B通过reviewed semantic-tree
  compatibility、checkpoint payload与loaded-model三方SHA256一致性复用该T0-A，未因无关commit
  变化重跑。T0-A证明工程合同和梯度路径，不是论文效果。
- `run_t0_b.py`在source-stratified batches上分别测共享参数`g_QA/g_state`，检查expected-source
  registry、loss/mask/component、matching、exact frames、checkpoint内容恢复和provenance一致性。
- CPU mock只能产生`awaiting_gpu`，不能产生formal pass。

正式单卡T0-B已在commit `e6aa19cb3a633ca19bf6e6ba4aa4b5241b4676ac`上完成，证据路径为
`/data2/wlx/output/parta/t0_b_e6aa19c_gpu5_retry2`：

- `status=complete_passed`、`formal_gpu_evidence=true`；
- 30/30 batches，ADT/Hypersim/ScanNet++ V2各10 batches；
- 13项hard checks全部PASS，包括finite loss、shared/head gradients、matching、exact frame和
  checkpoint/resume equivalence；
- 梯度校准valid fraction为`1.0`，得到候选
  `lambda_state=0.02150771327925621`。该值仍须由`Gate@CONFIG`冻结；
- `checkpoint-resume-probe.pt`已生成，作为恢复等价性证据保留。

**T0-B没有记录CUDA peak memory。** `/usr/bin/time -v`报告的
`Maximum resident set size=39,816,712 KiB`是主机RAM RSS，不是GPU显存。目前能做的
唯一显存结论是：该单卡运行在一张总显存`97,871 MiB`的H20上成功，启动时
占用约`5 MiB`。这不能推导真实peak VRAM，也不足以冻结四卡DDP/FSDP配置。

D-62原coverage matrix（作为历史设计记录）为：

```text
T0-A + 三源T0-B + fixed-train-subset overfit/matched real runner
     + independent 4×H20 32-frame worst-case resource profile
     + checkpoint/resume + head-free val audit + validators/resource gates
     → pre-authorization → Gate@CONFIG: APPROVE → freeze → formal A0/A1-O from step 0
```

用户在2026-08-12确认的当前执行路线是：保留每源1 scene engineering registry但不执行
fixed-subset overfit；resource profiling只把正式32帧worst-case作为必测点，不把16/24作为正式
候选。若32帧不可行，优先调per-GPU batch、gradient accumulation、gradient checkpointing和FSDP；
仍不可行才生成版本化低帧数exact binding并重跑数据审计。T0-B已PASS；当前
尚缺真实四卡32帧worst-case profile和matched distributed runner证据。`Gate@CONFIG`仍未批准。

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

### D-62 v2数据与单卡T0-B已正式PASS

- ScanNet++ V2已完成855 scenes / 138,517 QA的Gold、canonical、exact validation；原始清单中的
  `00dd871005`（184 QA）因逐帧元信息为空被显式排除；
- 三源canonical revalidation与统一validator均`complete_passed`：ADT `183/60,207`、
  Hypersim `317/176,774`、ScanNet++ V2 `855/138,517`，合计`1,355/375,498`；
- manifest v2已真实生成：`train=345,642`、`val=29,856`，scene零交叉；
- 每源1 scene engineering registry、exact-input registry和content hashes已生成；
- 真实dataset/collator/source-balanced sampler/K=384 CPU审计PASS；
- T0-A已PASS并通过审核的semantic compatibility复用；
- 单卡T0-B在H20上运行30 batches并`complete_passed`，三源各10 batches、13项检查
  全部PASS，checkpoint/resume equivalence PASS；
- 旧四源/三分split/guide_smoke gate已迁移为三源coverage matrix；
- engineering state non-promotion与formal step0重训断言；
- val-only checkpoint selector与paired-bootstrap decision report；
- exact T0-A/T0-B、validator、trusted worker/producer、formal startup与VSI one-shot绑定。

当前权威T0-B证据为
`/data2/wlx/output/parta/t0_b_e6aa19c_gpu5_retry2`，console为
`/data2/wlx/logs/parta/t0_b_e6aa19c_gpu5_retry2.console.log`。该运行只证明单卡训练数学链路、
数据链路和checkpoint恢复正确，不构成四卡显存/通信/吞吐证据。

### D-62前已完成并Review通过（legacy实现证据）

- 三源unified manifest/split、dataset/CPU collator、source-balanced sampler与K384 audit；
- matched A0/A1-O runner、全状态checkpoint/resume、DDP/FSDP代码路径及独立head-free audit；
- T0-B runner与机器可判report；
- overfit/profile/GUIDE smoke统一编排、producer/freeze/finalize合同（旧编排；D-62要求迁移）；
- matched A0/A1-O-drop VSI-Bench eval与结果身份验证。

以上旧版三源真实构建/审计已完成；它不等于D-62 v2正式runtime artifacts。D-62代码本身已在
当前执行服务器通过CPU代码测试，但GPU Gate与正式训练仍未通过。

### Historical / superseded prerequisites

- ScanNet lineage与production adapter可继续作为未来数据版本工作，但不再阻塞当前三源版本；
- v1四源manifest与独立smoke等待逻辑被D-62覆盖，保留仅作历史实现追溯。

### Next execution-server GPU work

- 在真实`4 × H20`上仅以A1-O运行32帧worst-case resource profile；同一profile必须分别运行
  一个DDP候选和一个FSDP候选，必须记录每rank CUDA peak allocated/reserved/total memory、
  吞吐和OOM/finite状态；两个候选都必须形成封闭证据，允许其中一个OOM；
- 两个候选固定`lambda_state=0.02150771327925621`，除strategy与各自output/run-dir外，LR、
  weight decay、max grad norm、per-rank batch=1、gradient accumulation、num workers、
  gradient checkpointing、dtype、模型与数据身份及effective global batch必须完全一致，并以
  normalized execution-contract hash绑定；
- OOM由每rank worker独立落盘并由父profile聚合；早期OOM或被torchrun终止的peer必须保留
  rank/stage/reason/nullable memory字段。OOM候选的吞吐不可测，必须为null，禁止伪造；
- 任一rank在backward/optimizer异常时必须先原子写本rank failure artifact后直接退出，禁止在非对称
  失败后进入新的NCCL collective；父profile必须使用硬超时，超时后terminate并在需要时kill，保存
  timeout artifact，禁止无限等待；
- DDP/FSDP各自必须在任何模型/VGGT/head/wrap/optimizer重内存操作前，于其run-dir原子生成独立
  fresh pre-execution matched-contract，绑定command/data/manifest/GUIDE/VGGT identity和完整
  normalized execution contract；OOM候选也不得缺失。profile producer必须重开两个preflight
  payload并验证除distributed strategy外完全一致；成功候选还须追加并核对runtime matched payload；
- 至少一个候选必须非OOM、finite且每rank peak allocated低于对应总显存90%。按
  `max throughput → min max-rank allocated → strategy lexical`确定性选择策略，并将所选策略、
  per-rank batch、gradient accumulation、gradient checkpointing和effective global batch冻结为
  matched A0/A1-O共同正式配置；
- 使用同一分布式runner验证A0/A1-O matched startup、checkpoint/resume和head-free forward；
- 汇总证据后请求`Gate@CONFIG`；APPROVE前不得开始正式训练。

#### Completed CPU/data steps (historical checklist; no longer pending)

- 先把代码中的frozen inventory从ScanNet++ `138,701/855`、三源`375,682/1,355`修正为
  `138,517/855`与`375,498/1,355`，同步修正对应测试；经静态Review后在执行服务器重新运行
  `py_compile`与Part A CPU测试。此前的213/213 PASS只对应旧inventory，不可冒充修正后的PASS；
- CPU复验PASS后，使用三个已完成canonical roots运行`build_unified_three_source_manifest.py`，生成train/val-only
  unified manifest v2与exact-input registry；
- `engineering-scenes-per-source=1`必须显式写入命令和报告；registry只用于身份/复现登记；
- 保存真实逐源逐split计数、engineering selected scene/QA IDs、exact inputs与content hashes；
- 随后运行真实三源dataset/collator/K384 CPU审计；再针对新工程Gate语义补Preflight并修改代码。

### Superseded pending list (historical; T0-B is now PASS)

- 三源T0-B：已于2026-08-13正式PASS；
- 正式32帧worst-case显存与吞吐profiling：仍待真实四卡执行；
- FSDP/resume/head-free forward与修订后的coverage matrix；
- 正式A0/A1-O训练与A0/A1-O-drop VSI-Bench eval。

### Provisional / TBD

- 已冻结：`seed=42`、`K=384`、动态16--32帧、val约0.10且无smoke split；
- T0-B已以30 batches完成正式单卡Gate；finite/nonzero有效率为100%；
- 实测候选`lambda_state=0.02150771327925621`；
- `lambda_state`、steps/epoch、batch/world size、LR、save/val频率、峰值显存与吞吐均须在profile后
  由`Gate@CONFIG`冻结。

正式训练Gate当前关闭，且未获得正式训练授权。

## 8. 当前至A2的权威决策树

```text
已完成
  三源canonical/exact与排除场景审计
  → manifest v2（375,498 QA / 1,355 scenes；train 345,642 / val 29,856；scene零交叉）
  → engineering registry + exact-input registry
  → dataset/collator/source-balanced sampler/K=384审计
  → canonical revalidation + three-source validator
  → T0-A PASS
  → 单卡T0-B PASS（30 batches；10/source；13 checks PASS）
    ↓
当前四卡GPU Gate
  4 × H20、A1-O 32帧worst-case DDP+FSDP resource profile
  → 真实每rank CUDA peak、吞吐、DDP/FSDP、checkpoint/resume和matched runner证据
  → 证据汇总 → Gate@CONFIG
    ├─ HOLD：补齐证据/配置
    └─ APPROVE
         ↓
  A0与A1-O从相同初始化step0、相同全量manifest独立正式训练
         ↓
  val-only同规则选checkpoint → A1-O head-free → matched one-shot VSI-Bench
         ↓
  Δoverall 95% CI下界>0：GO至A2 Relation Graph
  点估计>0但CI跨0：INCONCLUSIVE；点估计≤0：NO-GO
```
