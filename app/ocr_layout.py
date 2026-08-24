# -*- coding: utf-8 -*-
"""
OCR 版面重排（V3.1 · 针对真实化验单照片的三个识别痛点）：

  1. 【旋转】横版化验单竖着拍（真实样张里 3/8 份如此），文字框整体转了 90°，
     按像素顺序拼接得到的就是用户反馈里那种"结果列整块、项目列整块"的乱序
     文本。本模块提供逐方向的可比评分 text_weight()，server 层对 0/90/180/270
     四个方向各跑一遍 OCR，取得分最高者。
  2. 【左右双栏】报告单常见"1-8 项在左、9-15 项在右"的双栏排版。此前按 y
     聚成一行后左右两个指标被拼进同一行，而解析器是【一行一指标】——右栏
     整列丢失，正是"上传第二张还是 17 条"背后的漏识别来源之一。
     find_panel_split() 用横向投影空隙探测分栏线，reconstruct_lines() 把每个
     视觉行拆成左、右两条独立文本行，逐行配对不丢。
  3. 【PII】真实照片印有"姓名：×××"。原图本就不落库（规范 1.2），识别文本
     入库前由 redact_pii_text() 把姓名与手机号/证件号/银行卡/邮箱模式替换为
     [已脱敏]，与 serving.audit 的扫描口径一致（那边只查不改，这里是它的
     上游净化，不是绕过）。

全模块只依赖标准库，item 为 {text, x0, y0, x1, y1[, score]} 字典 ——
不 import numpy/opencv，才能在无依赖环境跑单测（tests/test_ocr_layout.py）。
"""
from __future__ import annotations

import re
from statistics import median

__all__ = [
    "text_weight",
    "cluster_rows",
    "find_panel_split",
    "reconstruct_lines",
    "redact_pii_text",
]

# ---------------------------------------------------------------- 方向评分
_CJK = re.compile(r"[\u4e00-\u9fa5]")
_ALNUM = re.compile(r"[A-Za-z0-9]")
_SYM = re.compile(r"[./:%↑↓<>≤≥~\-]")


def text_weight(text: str) -> float:
    """
    一段识别文本"像正常化验单内容"的权重。中文最重（转错方向时中文几乎
    全灭，是最灵敏的信号），字母数字次之，常见符号最轻。
    server 层的方向得分 = Σ(识别置信度 × text_weight)。
    """
    if not text:
        return 0.0
    return (
        2.0 * len(_CJK.findall(text))
        + 1.0 * len(_ALNUM.findall(text))
        + 0.3 * len(_SYM.findall(text))
    )


# ---------------------------------------------------------------- 行聚类
def _h(it: dict) -> float:
    return max(float(it["y1"]) - float(it["y0"]), 1.0)


def _yc(it: dict) -> float:
    return (float(it["y0"]) + float(it["y1"])) / 2.0


def _xc(it: dict) -> float:
    return (float(it["x0"]) + float(it["x1"])) / 2.0


def cluster_rows(items: list[dict]) -> list[list[dict]]:
    """
    按 y 中心把文字框聚成视觉行。锚点用行内均值、容差用两者较小高度的
    0.45 倍（与 V3 相同的防"行雪球"策略），额外下限 5px 抵御检测抖动。
    返回自上而下的行列表；行内不排序（交给 reconstruct_lines 按栏处理）。
    """
    rows: list[dict] = []
    for it in sorted(items, key=_yc):
        placed = None
        for r in rows:
            if abs(_yc(it) - r["anchor"]) <= max(min(_h(it), r["h"]) * 0.45, 5.0):
                placed = r
                break
        if placed:
            placed["items"].append(it)
            placed["anchor"] = sum(_yc(x) for x in placed["items"]) / len(placed["items"])
            placed["h"] = sum(_h(x) for x in placed["items"]) / len(placed["items"])
        else:
            rows.append({"anchor": _yc(it), "h": _h(it), "items": [it]})
    rows.sort(key=lambda r: r["anchor"])
    return [r["items"] for r in rows]


# ---------------------------------------------------------------- 双栏探测
def find_panel_split(items: list[dict], page_w: float) -> float | None:
    """
    探测左右双栏的分界 x。方法：把"体宽正常"的文字框（宽 < 0.55 页宽，
    排除横贯整页的医院名/标题行）的 x 区间做合并，找合并后区间之间的
    最大空隙。空隙要同时满足：
      · 中心落在页面 30%~72% 之间（分栏线不会贴边）
      · 宽度 ≥ max(3.5% 页宽, 1.2 × 中位行高)（表格列间距达不到这个量级）
      · 左右两侧各 ≥ 6 个文字框，且 ≥ 3 个视觉行同时有左右内容
    不满足则返回 None（单栏，保持整行输出）。
    """
    if not items or page_w <= 0:
        return None
    body = [it for it in items if (float(it["x1"]) - float(it["x0"])) < 0.55 * page_w]
    if len(body) < 12:
        return None

    ivs = sorted((float(it["x0"]), float(it["x1"])) for it in body)
    merged: list[list[float]] = []
    for a, b in ivs:
        if merged and a <= merged[-1][1] + 1.0:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    if len(merged) < 2:
        return None

    # 关键判据：真正的分栏线两侧【各自】都有带中文的项目名框（每栏都有自己的
    # "检验项目"列）；而"项目名↔数值"的列间距虽然也很宽，但它右侧直到下一栏
    # 之前只有数字/区间/单位 —— 用这一条把两者区分开，比单纯取最大空隙可靠。
    cjk = re.compile(r"[\u4e00-\u9fa5]")
    xs = sorted((_xc(it), bool(cjk.search(it["text"]))) for it in body)

    def _cjk_both_sides(cx: float) -> bool:
        left_cjk = sum(1 for x, has in xs if x < cx and has)
        right_cjk = sum(1 for x, has in xs if x >= cx and has)
        return left_cjk >= 2 and right_cjk >= 2

    med_h = median(_h(it) for it in body)
    best: tuple[float, float] | None = None  # (gap_width, split_x)
    for (l0, l1), (r0, _r1) in zip(merged, merged[1:]):
        gap = r0 - l1
        cx = (l1 + r0) / 2.0
        if gap <= 0 or not (0.30 * page_w <= cx <= 0.72 * page_w):
            continue
        if gap < max(0.035 * page_w, 1.2 * med_h):
            continue
        if not _cjk_both_sides(cx):
            continue
        if best is None or gap > best[0]:
            best = (gap, cx)
    if best is None:
        return None

    split_x = best[1]
    left = [it for it in body if _xc(it) < split_x]
    right = [it for it in body if _xc(it) >= split_x]
    if len(left) < 6 or len(right) < 6:
        return None
    both = sum(
        1
        for row in cluster_rows(body)
        if any(_xc(x) < split_x for x in row) and any(_xc(x) >= split_x for x in row)
    )
    if both < 3:
        return None
    return split_x


# ---------------------------------------------------------------- 重排主入口
def reconstruct_lines(items: list[dict], page_w: float) -> tuple[list[str], str]:
    """
    把 OCR 文字框重排成"一行一条完整记录"的文本行列表。
    返回 (lines, layout)，layout ∈ {"two_panel", "single"}。

    双栏时每个视觉行输出两行：先左栏、后右栏 —— 下游解析器一行一指标，
    这样右栏不再被拼进左栏行尾丢掉。
    """
    if not items:
        return [], "single"
    split = find_panel_split(items, page_w)
    layout = "two_panel" if split is not None else "single"
    lines: list[str] = []
    for row in cluster_rows(items):
        if split is None:
            cells = sorted(row, key=lambda x: float(x["x0"]))
            txt = "  ".join(c["text"] for c in cells).strip()
            if txt:
                lines.append(txt)
            continue
        left = sorted((c for c in row if _xc(c) < split), key=lambda x: float(x["x0"]))
        right = sorted((c for c in row if _xc(c) >= split), key=lambda x: float(x["x0"]))
        for cells in (left, right):
            txt = "  ".join(c["text"] for c in cells).strip()
            if txt:
                lines.append(txt)
    return lines, layout


# ---------------------------------------------------------------- PII 脱敏
#: 与 serving.audit.PII_PATTERNS 同口径的内容模式 + 化验单表头的"姓名"字段。
_PII_SUBS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"(姓\s*名\s*[:：]?\s*)[\u4e00-\u9fa5A-Za-z·]{1,8}"), r"\1[已脱敏]"),
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "[已脱敏]"),                # 手机号
    (re.compile(r"(?<!\d)(?:\d{17}[\dXx]|\d{15})(?!\d)"), "[已脱敏]"),    # 身份证
    (re.compile(r"(?<!\d)\d{16,19}(?!\d)"), "[已脱敏]"),                  # 银行卡
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[已脱敏]"),               # 邮箱
)


def redact_pii_text(text: str) -> tuple[str, int]:
    """OCR 文本入库前脱敏。返回 (脱敏后文本, 命中次数)。"""
    if not text:
        return text, 0
    n = 0
    for pat, rep in _PII_SUBS:
        text, k = pat.subn(rep, text)
        n += k
    return text, n
