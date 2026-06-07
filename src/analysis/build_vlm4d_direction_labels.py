"""build_vlm4d_direction_labels.py — VLM4D answer → 运动方向类别标签 + 诊断（验证③ / D-13）.

服务于 D-13（OPEN，MoPE 取层）。
================================================================================
目的
----
验证③（逐层线性 probe，定位 MoPE 物理/方向信息峰值层）需要一份"每视频一个运动
方向类别标签"的 --video-list 喂给 `src/analysis/mope_layer_features.py`。

VLM4D `real_mc.json` 的 `question_type` 字段恒为 "multiple-choice"（无运动子类，
已验证），不能当方向标签。本脚本改用 `answer`（选项**全文**，是方向/运动描述）
归一化成有限的方向类别集合当标签。

PENDING[D-13]: 本脚本的方向类别集合 + 归一化映射（见模块级 DIRECTION_RULES /
MOTION_RULES）是【初版】。我（Orchestrator）和 CodeAgent 都在 Delta 上，看不到执行
服务器上 real_mc.json 的真实 answer 词表，因此本脚本【一次运行同时产出：① probe 用
的 --video-list；② 一份详尽诊断】。用户在执行服务器跑一次、把诊断（尤其 UNMATCHED
去重 answer 列表）贴回来后，再据此扩充/定稿 DIRECTION_RULES，重跑定稿 --video-list。
D-13 取层决策本身由下游 probe 结果定，与本脚本的标签方案解除后无需改接口（仅可能
扩充关键词表）。

================================================================================
输入 → 输出
----------
输入：VLM4D 标注 JSON（**JSON 数组**，非 jsonl）。默认 env VLM4D_JSONL，其次 --jsonl，
      兜底 /data2/wlx/data/VLM4D/QA/real_mc.json。每条含
      id / question / choices(dict A-D) / answer(选项全文) / question_type / video(完整 HF URL)。
      视频根目录 env VLM4D_VIDEO_ROOT（兜底 /data2/wlx/data/VLM4D）；
      绝对路径 = root + _video_rel_path(video)。

输出（--out，默认 src/analysis/vlm4d_direction_video_list.jsonl）：probe 用的 --video-list。
      格式【与 mope_layer_features.py 的 --video-list 解析完全一致】：jsonl，每行
          {"video": "<绝对路径>", "direction": "<方向类别>", "id": "<原 id>"}
      → 下游 `mope_layer_features.py` 传 `--label-key direction`。
      仅写入：方向标签 != UNMATCHED、未被 --video-policy 冲突丢弃、且视频文件存在的样本。

诊断：打到 stdout（结构化分节），供用户贴回。

字段常量与 _video_rel_path / _answer_to_letter 的逻辑【照抄】自已验证的
`src/vendor/lmms-eval/lmms_eval/tasks/vlm4d/utils.py`（不 import：vendor 模块顶层依赖
datasets/pandas/yaml/loguru 等重包，登录节点不一定可用；本脚本保持纯标准库、可在
登录节点和执行服务器都跑）。real_mc.json 字段语义以 vlm4d/utils.py 为准。

约束
----
- 纯 Python + 标准库（json/os/re/argparse/collections）。不依赖 numpy/pandas。CPU-only。
- 不修改任何 vendor 代码、不跑模型。
- 归一化映射、类别集合、关键词表全部放模块级常量，集中可改。
- UNMATCHED 绝不静默并入 other：单列计数 + 完整去重列表，便于扩充映射。
"""

import argparse
import json
import os
import random
import re
import statistics
import sys
from collections import Counter, defaultdict, OrderedDict

# ---------------------------------------------------------------------------
# 字段名常量 — 照抄自 vlm4d/utils.py（VERIFIED 2026-06-06）。
#   每条: id / question / choices(dict A-D) / answer(选项全文) / question_type / video
# ---------------------------------------------------------------------------
FIELD_VIDEO = "video"            # 完整 HF URL（resolve/main/ 之后为本地相对路径）
FIELD_QUESTION = "question"      # 问题文本
FIELD_QUESTION_TYPE = "question_type"  # 恒为 "multiple-choice"，无语义子类
FIELD_CHOICES = "choices"        # dict {"A": .., "B": .., "C": .., "D": ..}
FIELD_ANSWER = "answer"          # ground-truth 选项全文（需反查 choices 得字母）
FIELD_ID = "id"

DEFAULT_JSONL = "/data2/wlx/data/VLM4D/QA/real_mc.json"
DEFAULT_VIDEO_ROOT = "/data2/wlx/data/VLM4D"
DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "vlm4d_direction_video_list.jsonl"
)

# 哨兵：未匹配方向（绝不并入 other）。
UNMATCHED = "UNMATCHED"


# ---------------------------------------------------------------------------
# _video_rel_path — 照抄 vlm4d/utils.py 逻辑（不依赖任何重包）。
# ---------------------------------------------------------------------------
def _video_rel_path(url):
    """从完整 HF URL 取本地相对路径。
    e.g. https://huggingface.co/datasets/shijiezhou/VLM4D/resolve/main/videos_real/davis/x.mp4
         -> videos_real/davis/x.mp4
    URL 不含 resolve/main/ 时兜底用 basename。
    """
    url = str(url)
    if "resolve/main/" in url:
        return url.split("resolve/main/")[-1]
    return os.path.basename(url)


# ---------------------------------------------------------------------------
# _answer_to_letter — 照抄 vlm4d/utils.py 逻辑（去 loguru 依赖，用本地 warn）。
# 本脚本不直接拿字母当标签，但保留它用于诊断（answer 是否可正常反查为字母）。
# ---------------------------------------------------------------------------
def _answer_to_letter(doc, warn_sink=None):
    """把 answer 全文映射为选项字母；反查不到返回原 answer 字符串（兜底）。"""
    choices = doc.get(FIELD_CHOICES)
    answer = str(doc.get(FIELD_ANSWER, "")).strip()
    if isinstance(choices, dict):
        for k, v in choices.items():
            if str(v).strip() == answer:
                return k
    if warn_sink is not None:
        warn_sink.append(answer)
    return answer


# ===========================================================================
# 方向归一化映射（模块级、集中、易改）—— PENDING[D-13] 初版。
# ---------------------------------------------------------------------------
# 设计：先把 answer 文本做轻量规整（小写、去标点、去冠词/常见停用词），再按
# DIRECTION_RULES 顺序用正则匹配；命中第一条规则即返回其类别。全不命中 → UNMATCHED。
#
# 初版方向类别（待诊断后调整）：
#   left, right, up, down, toward(接近/朝相机), away(远离), clockwise, counterclockwise, other
# 'other' 仅给"明确是方向/运动描述但不属于上述具体类别"的兜底关键词（如 stationary）。
# 未命中任何规则的 answer → UNMATCHED（不是 other），单独计数 + 去重列出。
#
# 规则顺序很重要：更具体/更长的短语放前面，避免被子串误匹配
#   （如 counterclockwise 必须在 clockwise 之前；moving away 在 away 之前等价但安全）。
# 每条规则用 \b 词边界，匹配规整后的文本。
# ===========================================================================
DIRECTION_CATEGORIES = [
    "left",
    "right",
    "up",
    "down",
    "toward",
    "away",
    "clockwise",
    "counterclockwise",
    "other",
]

# (category, [regex pattern strings])  —— 按顺序匹配，先到先得。
DIRECTION_RULES = [
    # 旋转方向：counterclockwise / anticlockwise 必须先于 clockwise 判定。
    ("counterclockwise", [
        r"\bcounter[\s\-]?clockwise\b",
        r"\banti[\s\-]?clockwise\b",
        r"\bccw\b",
    ]),
    ("clockwise", [
        r"\bclockwise\b",
        r"\bcw\b",
    ]),
    # 相机轴向：靠近 / 远离。
    ("toward", [
        r"\btoward(s)?\b",
        r"\bcloser\b",
        r"\bapproach(ing|es|ed)?\b",
        r"\bnearer\b",
        r"\bgetting closer\b",
        r"\bzoom(ing)? in\b",
        r"\binto the (camera|screen|frame)\b",
    ]),
    ("away", [
        r"\baway\b",
        r"\bfarther\b",
        r"\bfurther\b",
        r"\breced(e|ing|es)\b",
        r"\bretreat(ing|s)?\b",
        r"\bgetting (farther|further)\b",
        r"\bzoom(ing)? out\b",
        r"\bout of the (frame|screen)\b",
    ]),
    # 平面四向。leftward/rightward/upward/downward 也覆盖。
    ("left", [
        r"\bleft\b",
        r"\bleft[\s\-]?ward(s)?\b",
        r"\bto the left\b",
    ]),
    ("right", [
        r"\bright\b",
        r"\bright[\s\-]?ward(s)?\b",
        r"\bto the right\b",
    ]),
    ("up", [
        r"\bup\b",
        r"\bup[\s\-]?ward(s)?\b",
        r"\bupwards?\b",
        r"\brising\b",
        r"\bascend(ing|s)?\b",
    ]),
    ("down", [
        r"\bdown\b",
        r"\bdown[\s\-]?ward(s)?\b",
        r"\bdownwards?\b",
        r"\bfalling\b",
        r"\bdescend(ing|s)?\b",
    ]),
    # 兜底 'other'：明确是运动/方向描述但不属于上述具体方向类别。
    ("other", [
        r"\bstationary\b",
        r"\bnot moving\b",
        r"\bno motion\b",
        r"\bstill\b",
        r"\bin place\b",
        r"\brotat(e|ing|es|ion)\b",   # 旋转但未注明 cw/ccw
        r"\bspin(ning|s)?\b",
        r"\bmov(e|ing|es)\b",         # 泛"运动"无方向
    ]),
]

# 预编译。
_COMPILED_DIRECTION_RULES = [
    (cat, [re.compile(p) for p in pats]) for cat, pats in DIRECTION_RULES
]

# answer 规整：去掉这些冠词/停用词（作为整词），降低匹配噪声。
_STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "it", "they",
              "object", "objects", "camera", "moving", "motion"}
# 注意：'camera'/'moving' 放停用词是为减少噪声，但部分规则（如 "into the camera"、
# "zoom in"）依赖这些词——规整在【匹配前】对一份"轻规整"文本做，规则同时在
# 轻规整文本上跑；为安全，方向匹配用【未去停用词】的轻规整文本（仅小写+去标点+压空格），
# 去停用词文本仅用于诊断展示。见 _normalize_answer / classify_direction。


def _normalize_answer(text):
    """轻规整：小写、把标点换成空格、压缩空白。用于方向正则匹配。"""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)   # 标点 → 空格
    text = re.sub(r"\s+", " ", text).strip()
    return text


def classify_direction(answer_text):
    """answer 全文 → 方向类别（DIRECTION_CATEGORIES 之一）或 UNMATCHED。

    在轻规整文本（小写+去标点+压空格，保留所有词）上按 DIRECTION_RULES 顺序匹配，
    命中第一条规则即返回其类别。全不命中 → UNMATCHED（不是 other）。
    """
    norm = _normalize_answer(answer_text)
    if not norm:
        return UNMATCHED
    for cat, compiled in _COMPILED_DIRECTION_RULES:
        for pat in compiled:
            if pat.search(norm):
                return cat
    return UNMATCHED


# ===========================================================================
# 运动类别（粗分，仅用于诊断；不作最终标签）—— PENDING[D-13] 初版。
# 从 question 文本关键词分到 5 类。命中第一条即定；全不命中 → other。
# ===========================================================================
MOTION_CATEGORIES = ["translational", "rotational", "perspective", "continuity", "other"]

MOTION_RULES = [
    ("rotational", [
        r"\brotat", r"\bspin", r"\bclockwise\b", r"\bturn(ing|s|ed)?\b",
        r"\brevolv", r"\borbit",
    ]),
    ("perspective", [
        r"\bcamera\b", r"\bviewpoint\b", r"\bperspective\b", r"\bpan(ning|s)?\b",
        r"\btilt", r"\bzoom", r"\bpoint of view\b", r"\bobserver\b",
    ]),
    ("continuity", [
        r"\bcontinu", r"\bappear", r"\bdisappear", r"\bvisible\b",
        r"\boccluded?\b", r"\border\b", r"\bsequence\b", r"\bbefore\b", r"\bafter\b",
    ]),
    ("translational", [
        r"\bmove", r"\bmotion\b", r"\bdirection\b", r"\bleft\b", r"\bright\b",
        r"\bup(ward)?\b", r"\bdown(ward)?\b", r"\btoward", r"\baway\b",
        r"\btranslat", r"\bslid", r"\bshift",
    ]),
]
_COMPILED_MOTION_RULES = [
    (cat, [re.compile(p) for p in pats]) for cat, pats in MOTION_RULES
]


def classify_motion(question_text):
    """question 全文 → 运动粗类（MOTION_CATEGORIES 之一）。全不命中 → other。"""
    norm = _normalize_answer(question_text)
    for cat, compiled in _COMPILED_MOTION_RULES:
        for pat in compiled:
            if pat.search(norm):
                return cat
    return "other"


# ===========================================================================
# 加载 + 处理
# ===========================================================================
def load_records(jsonl_path):
    """读 VLM4D 标注 JSON 数组。返回 list[dict]。"""
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        # 兜底：若顶层是 {"data": [...]} 之类。
        for key in ("data", "questions", "annotations", "records"):
            if key in data and isinstance(data[key], list):
                return data[key]
        raise ValueError(
            f"{jsonl_path} 顶层是 dict 但找不到列表字段（data/questions/...）。"
        )
    if not isinstance(data, list):
        raise ValueError(f"{jsonl_path} 不是 JSON 数组（实际 {type(data)}）。")
    return data


def abs_video_path(video_url, video_root):
    return os.path.join(video_root, _video_rel_path(video_url))


def _hist_line(counter, total, width=40):
    """单类计数 → 简单文本直方图行（仅诊断用）。"""
    bar = "#" * int(round(width * (counter / total))) if total else ""
    return bar


# ===========================================================================
# main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="VLM4D answer → 运动方向类别标签（probe --video-list）+ 诊断（D-13）。"
    )
    parser.add_argument(
        "--jsonl",
        default=os.environ.get("VLM4D_JSONL", DEFAULT_JSONL),
        help="VLM4D 标注 JSON 数组路径（默认 env VLM4D_JSONL，兜底 %(default)s）。",
    )
    parser.add_argument(
        "--video-root",
        default=os.environ.get("VLM4D_VIDEO_ROOT", DEFAULT_VIDEO_ROOT),
        help="视频根目录（默认 env VLM4D_VIDEO_ROOT，兜底 %(default)s）。"
             "绝对路径 = root + resolve/main 之后的相对路径。",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help="输出 --video-list（jsonl）路径，默认 %(default)s。",
    )
    parser.add_argument(
        "--video-policy",
        choices=["consensus", "first", "per-question"],
        default="consensus",
        help="每视频多题方向冲突处理策略："
             "consensus(默认，所有题方向一致才保留)/first(取首题)/"
             "per-question(不去重，每题一行，仅对照用)。",
    )
    parser.add_argument(
        "--require-video-exists",
        dest="require_video_exists",
        action="store_true",
        default=True,
        help="仅写入视频文件存在(os.path.exists)的样本（默认开）。",
    )
    parser.add_argument(
        "--no-require-video-exists",
        dest="require_video_exists",
        action="store_false",
        help="关闭视频存在性过滤（调试用：在没有视频的机器上也产出 --video-list）。",
    )
    parser.add_argument(
        "--sample-seed", type=int, default=0,
        help="示例抽样的随机种子（默认 0；用于诊断里的随机样本展示）。",
    )
    parser.add_argument(
        "--n-examples", type=int, default=15,
        help="诊断里展示的样本示例条数（默认 15）。",
    )
    args = parser.parse_args()

    print("=" * 78)
    print("build_vlm4d_direction_labels.py — 诊断报告（D-13 验证③，初版方向映射）")
    print("=" * 78)
    print(f"--jsonl        : {args.jsonl}")
    print(f"--video-root   : {args.video_root}")
    print(f"--out          : {args.out}")
    print(f"--video-policy : {args.video_policy}")
    print(f"require-video-exists: {args.require_video_exists}")
    print()

    if not os.path.isfile(args.jsonl):
        print(f"[FATAL] 找不到标注文件：{args.jsonl}", file=sys.stderr)
        print("        请在执行服务器上运行，或用 --jsonl 指定正确路径。", file=sys.stderr)
        sys.exit(2)

    records = load_records(args.jsonl)
    n_total = len(records)
    print(f"[1] 总记录数（题目数）: {n_total}")

    # --- 逐题处理：方向标签 + 运动粗类 + 视频路径 ---------------------------
    answer_counter = Counter()
    qtype_counter = Counter()
    direction_counter = Counter()
    motion_counter = Counter()
    cross_dir_motion = defaultdict(Counter)   # direction -> motion -> count
    unmatched_answers = Counter()             # 未匹配 answer 去重 + 计数
    answer_to_letter_fail = []                # _answer_to_letter 反查失败的 answer

    per_question = []   # list of dict(id, video_abs, video_rel, direction, motion, question, choices, answer)
    video_to_dirs = defaultdict(list)         # video_abs -> [(direction, qrow_index)]

    for rec in records:
        qid = rec.get(FIELD_ID)
        answer = rec.get(FIELD_ANSWER, "")
        question = rec.get(FIELD_QUESTION, "")
        qtype = rec.get(FIELD_QUESTION_TYPE, "<MISSING>")
        video_url = rec.get(FIELD_VIDEO, "")

        answer_counter[str(answer)] += 1
        qtype_counter[str(qtype)] += 1

        # _answer_to_letter 诊断（answer 是否能正常反查 choices 字母）。
        _answer_to_letter(rec, warn_sink=answer_to_letter_fail)

        direction = classify_direction(answer)
        motion = classify_motion(question)
        direction_counter[direction] += 1
        motion_counter[motion] += 1
        cross_dir_motion[direction][motion] += 1
        if direction == UNMATCHED:
            unmatched_answers[str(answer).strip()] += 1

        video_rel = _video_rel_path(video_url)
        video_abs = abs_video_path(video_url, args.video_root)

        row = {
            "id": qid,
            "video_abs": video_abs,
            "video_rel": video_rel,
            "direction": direction,
            "motion": motion,
            "question": question,
            "choices": rec.get(FIELD_CHOICES),
            "answer": answer,
        }
        per_question.append(row)
        video_to_dirs[video_abs].append((direction, len(per_question) - 1))

    # --- 诊断输出 ----------------------------------------------------------
    n_videos = len(video_to_dirs)
    qpv = [len(v) for v in video_to_dirs.values()]
    print(f"[2] 去重视频数: {n_videos}")
    if qpv:
        print(f"    questions-per-video: min={min(qpv)} "
              f"median={statistics.median(qpv):.1f} max={max(qpv)}")
        qpv_hist = Counter(qpv)
        print("    questions-per-video 直方（题数:视频个数）:")
        for k in sorted(qpv_hist):
            print(f"      {k:>3} 题 : {qpv_hist[k]:>5}  {_hist_line(qpv_hist[k], n_videos)}")
    print()

    print("[3] question_type value_counts（确认是否恒 'multiple-choice'）:")
    for val, cnt in qtype_counter.most_common():
        print(f"    {val!r:30} : {cnt}")
    print()

    print(f"[4] answer 原始文本 value_counts（共 {len(answer_counter)} 个去重值，top 30）:")
    for val, cnt in answer_counter.most_common(30):
        disp = val if len(val) <= 60 else val[:57] + "..."
        print(f"    {cnt:>6}  {disp!r}")
    print()

    print("[5] 方向归一化后各类别计数（含 UNMATCHED）:")
    ordered_cats = DIRECTION_CATEGORIES + [UNMATCHED]
    for cat in ordered_cats:
        cnt = direction_counter.get(cat, 0)
        print(f"    {cat:18} : {cnt:>6}  {_hist_line(cnt, n_total)}")
    # 任何不在预期集合里的（理论上不会有）。
    for cat, cnt in direction_counter.items():
        if cat not in ordered_cats:
            print(f"    {cat:18} : {cnt:>6}  [UNEXPECTED]")
    print()
    print(f"[5b] UNMATCHED answer 去重列表（共 {len(unmatched_answers)} 个，按计数降序）:")
    print("     >>> 把这一段贴回，用于扩充 DIRECTION_RULES <<<")
    if not unmatched_answers:
        print("     (无 UNMATCHED — 全部 answer 已归类)")
    for val, cnt in unmatched_answers.most_common():
        disp = val if len(val) <= 80 else val[:77] + "..."
        print(f"     {cnt:>6}  {disp!r}")
    print()

    if answer_to_letter_fail:
        print(f"[5c] _answer_to_letter 反查失败 {len(answer_to_letter_fail)} 条"
              f"（answer 全文未在 choices 中找到对应 value）——诊断用，不影响方向标签:")
        for val, cnt in Counter(answer_to_letter_fail).most_common(15):
            disp = val if len(val) <= 60 else val[:57] + "..."
            print(f"     {cnt:>6}  {disp!r}")
        print()

    print("[6] 运动类别（粗分，仅诊断）计数:")
    for cat in MOTION_CATEGORIES:
        cnt = motion_counter.get(cat, 0)
        print(f"    {cat:16} : {cnt:>6}  {_hist_line(cnt, n_total)}")
    print()
    print("[6b] 方向类别 × 运动类别 交叉表（行=方向，列=运动）:")
    col_cats = MOTION_CATEGORIES
    header = "    " + f"{'direction':18}" + "".join(f"{c[:6]:>9}" for c in col_cats) + f"{'total':>9}"
    print(header)
    for cat in ordered_cats:
        if direction_counter.get(cat, 0) == 0:
            continue
        rowc = cross_dir_motion.get(cat, {})
        cells = "".join(f"{rowc.get(c, 0):>9}" for c in col_cats)
        print(f"    {cat:18}{cells}{sum(rowc.values()):>9}")
    print()

    # --- 应用 --video-policy ----------------------------------------------
    # 产出 (video_abs, direction, id) 候选；统计冲突丢弃 / UNMATCHED / 文件缺失。
    out_rows = []
    conflict_dropped = 0
    unmatched_dropped_videos = 0
    missing_video = 0
    missing_video_examples = []

    if args.video_policy == "per-question":
        # 不去重：每题一行（仅对照用）。
        candidates = []
        for row in per_question:
            candidates.append((row["video_abs"], row["direction"], row["id"]))
    else:
        candidates = []
        for video_abs, dir_list in video_to_dirs.items():
            if args.video_policy == "consensus":
                dirs = {d for d, _ in dir_list}
                # 去掉 UNMATCHED 不参与一致性判定？——保守：UNMATCHED 视为一个值，
                # 若该视频含 UNMATCHED 则其一致集合含 UNMATCHED，最终被 UNMATCHED 过滤。
                if len(dirs) == 1:
                    the_dir = next(iter(dirs))
                    qid = per_question[dir_list[0][1]]["id"]
                    candidates.append((video_abs, the_dir, qid))
                else:
                    conflict_dropped += 1
            elif args.video_policy == "first":
                the_dir, idx = dir_list[0]
                candidates.append((video_abs, the_dir, per_question[idx]["id"]))

    # 过滤 UNMATCHED + 视频存在性。
    for video_abs, direction, qid in candidates:
        if direction == UNMATCHED:
            unmatched_dropped_videos += 1
            continue
        if args.require_video_exists and not os.path.exists(video_abs):
            missing_video += 1
            if len(missing_video_examples) < 10:
                missing_video_examples.append(video_abs)
            continue
        out_rows.append({"video": video_abs, "direction": direction, "id": qid})

    # --- 写出 --video-list -------------------------------------------------
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- 最终统计 ----------------------------------------------------------
    print(f"[7] 应用 --video-policy='{args.video_policy}' 后:")
    print(f"    最终写出样本数            : {len(out_rows)}")
    print(f"    conflict_dropped (一致性) : {conflict_dropped}"
          + ("  (per-question 策略不去重，不适用)" if args.video_policy == "per-question" else ""))
    print(f"    UNMATCHED 丢弃            : {unmatched_dropped_videos}")
    print(f"    视频文件缺失丢弃          : {missing_video}"
          + ("" if args.require_video_exists else "  (存在性过滤已关闭)"))
    if missing_video_examples:
        print("    缺失视频示例（前 10）:")
        for p in missing_video_examples:
            print(f"      {p}")
    final_dir_counter = Counter(r["direction"] for r in out_rows)
    print("    最终各方向类别样本数（类别均衡度）:")
    for cat in DIRECTION_CATEGORIES:
        cnt = final_dir_counter.get(cat, 0)
        if cnt:
            print(f"      {cat:18} : {cnt:>6}  {_hist_line(cnt, max(len(out_rows),1))}")
    print()

    # --- 样本示例 ----------------------------------------------------------
    print(f"[8] 样本示例（前 {args.n_examples} 条 + 随机 {args.n_examples} 条）:")
    def _show(row):
        ch = row["choices"]
        ch_disp = ch if not isinstance(ch, dict) else ", ".join(f"{k}:{v}" for k, v in ch.items())
        print(f"    id={row['id']!r} dir={row['direction']} motion={row['motion']}")
        print(f"      Q: {str(row['question'])[:120]}")
        print(f"      choices: {str(ch_disp)[:160]}")
        print(f"      answer : {str(row['answer'])[:120]!r}")
        print(f"      video  : {row['video_rel']}")

    print("  -- 前 N 条 --")
    for row in per_question[: args.n_examples]:
        _show(row)
    print("  -- 随机 N 条 --")
    rng = random.Random(args.sample_seed)
    sample = rng.sample(per_question, min(args.n_examples, len(per_question)))
    for row in sample:
        _show(row)
    print()

    print("[9] 写出 --video-list:")
    print(f"    路径: {os.path.abspath(args.out)}")
    print(f"    行数: {len(out_rows)}")
    print(f"    字段: video / direction / id  → 下游 mope_layer_features.py 传 --label-key direction")
    print("=" * 78)


if __name__ == "__main__":
    main()
