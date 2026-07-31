# Part A A1-O Architecture

更新时间：2026-07-30（D-59）
状态：Engineering skeleton曾`REVIEW PASS`；ADT whole-MP4 sampler已被真实pilot supersede，目标合同待实现/Review；真实T0未通过

## 1. 范围

当前实现只覆盖论文2 Part A首批工程链：

- ADT v2/Hypersim到canonical三表的数据适配、校验和exact-frame绑定；
- matched A0/A1-O所需的Qwen visual tap、A1-O普通set-slot side head、D-58 loss；
- A1-O-drop head-free加载审计、T0-A/T0-B机器报告和provenance骨架。

不包含正式训练、A1-P、persistent slots、Relation Graph、Spatial Memory、MoPE或A2–A4。
依据`state/paper1_handoff_sync.md`，这些文件属于论文2 Part A专属实现，不同步到论文1
handoff。

## 2. 数据架构

```text
source-native ADT v2 / Hypersim tables
  │
  ├─ adapt_scene()
  ├─ adapt_frame()
  └─ adapt_qa()
        ↓
parta_canonical_v1
  ├─ scene_states.jsonl
  ├─ frame_states.jsonl
  └─ qa_manifest.jsonl
        ↓
validate_records()  [fail closed]
        ↓
ADT: trajectory ∩ calibration ∩ required direct-GT temporal support
        ↓
maximal contiguous GT-supported raw-frame training clip
        ↓
GUIDE deterministic 16–32 positions inside clip
        ↓
map to original MP4 raw frame IDs
        ↓
exact keys + indices + actual-visible object union + binding SHA256
        ↓
Part A loader / real T0 Runner
```

Canonical坐标为米制右手系`x=right,y=up,z=back`，相机forward为`-z`。ADT执行显式轴变换，
Hypersim为identity。对象ID必须为`source:scene:raw_object_id`；类别进入固定canonical
vocabulary，源类别被保留用于审计。缺失能力写`null`并关闭对应field/capability mask，禁止以0
伪装GT。

ADT exact-frame链按D-59先求共同GT支持的最大连续clip，再使用GUIDE采样器并映射回原MP4 raw
IDs。每个ID必须存在对应canonical frame state，frame key/index必须逐位置一致。对象target集合
只能是这些实际输入帧中有效可见对象的并集。禁止邻帧替代、外推或伪标签；clip内选中ID仍不
满足冻结阈值即整景失败。provenance保存whole-video total/FPS、clip首尾raw ID/device timestamp、
支持能力、采样参数、mapped IDs与hash；A0/A1-O共享全部字段。

现有`guide_exact_raw_mp4_v1`直接在whole MP4上取帧的实现已被真实pilot supersede：固定ADT
场景的32个采样帧中有6个落在trajectory span之外。该实现不得用于正式ADT manifest；目标
`guide_exact_over_gt_supported_clip_v1`尚未实现或Review，必须先完成183景coverage audit。

## 3. 模型架构

```text
Qwen standard QA forward
  ├─ QA logits
  └─ [only when return_visual_state_tap=True]
       final hidden at authoritative visual positions
       → padded [B,Nvisual,D] + valid mask
       → exact frame token counts / IDs / spans
       → K=384 learned set slots
       → independent cross-attention decoder
       → existence logits
       → canonical category logits
       → normalized world center
       → normalized extent
       → per-actual-view visibility logits
       → D-58 set loss
```

visual tap复用multimodal scatter的authoritative visual position mask，因此不包含question、
answer和padding。side branch不向语言序列插token，不修改QA hidden或logits。A0不attach
state head；A1-O只在训练时attach`parta_state_head`。两者检测到MoPE均失败。

## 4. D-58 assignment与loss

Hungarian pair cost对每个slot/object pair取所有valid分项的等权平均：

1. positive existence BCE；
2. canonical category NLL；
3. scene-normalized center Smooth-L1；
4. scene-normalized extent Smooth-L1；
5. actual-input-view valid visibility BCE。

无效项从pair cost移除并按实际参与项数归一；assignment在detached cost上执行，不参与反传。
匹配后：

- existence覆盖全部384 slots，matched/unmatched分别为正/负，正负组均值各权重0.5；
- category只监督matched且category-valid对象；
- center/extent只监督matched且field-valid对象，同时记录米制MAE；
- visibility只监督matched、实际输入帧和valid位置；
- empty GT只计算全负existence；其余四项精确关闭；
- GT对象数超过384直接失败，禁止静默截断。

## 5. Head-free评测与provenance

A1-O-drop不重训，也不实例化state head。加载器只允许过滤
`parta_state_head.*`，并记录loaded/dropped/missing/unexpected keys。真实T0必须证明：

- 标准QA模型从A1-O checkpoint恢复共享权重；
- “head存在但旁路”和“不实例化head”使用同一标准QA forward；
- 两者QA logits在D-58预注册dtype容差内等价。

每个run必须保存resolved config、manifest内容SHA256、初始化和checkpoint稳定指纹、
code/dirty状态、seed、exact-frame artifact与hash、训练预算及机器可读审计。产物原子写入，
running与complete状态不可覆盖；失败run不得标为完成。

## 6. Fail-closed不变量

- A0/A1-O不允许MoPE；A1-O输出不得回写QA。
- matched A0/A1-O必须共享初始化、manifest、exact frame IDs、数据顺序、optimizer、steps和预算。
- same visual/different question的frame IDs、mask、spans必须bitwise相同。
- 每个启用loss分项对共享可训练参数的fp32 grad norm必须finite且`>1e-12`。
- T0-B必须有50–100个source-balanced batches，至少95%的`g_QA`和`g_state`均finite且
  `>1e-12`。
- 固定fixtures缺一即失败，不允许自动替换。
- 真实pose/投影QC、四源manifest、零scene overlap、真实T0均是正式训练前Gate。

## 7. 已验证与未验证

已验证：除D-59新增采样目标外，两个独立ReviewAgent在修复闭环后均`REVIEW PASS`；
synthetic/CPU测试覆盖schema反例、旧GUIDE whole-MP4 raw-ID hard fail、Hungarian置换/
empty/all-masked、finite/backward、visual tap、
head-key过滤、T0数值合同和provenance；合计24项通过，`py_compile`及`git diff --check`通过。

未验证：183景coverage、D-59 sampler实现/Review、真实GUIDE mapped raw-ID frame-state生成、
真实pose/投影QC、五个固定fixture上的真实Qwen T0-A、真实head-free logits等价、
ADT+Hypersim及最终四源T0-B。不得将synthetic测试或whole-MP4 pilot失败描述为真实T0 PASS。
