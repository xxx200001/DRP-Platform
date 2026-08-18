"""
OCR 文本 -> 结构化长表 解析器（规范 4.1：OCR + 结构化引擎，第二环）。

上游：任意 OCR 引擎输出的化验单纯文本（本模块不做图像识别，只做文本结构化）。
下游：data/cleaning.LabDataCleaner —— 本模块产出的 DataFrame 直接喂给它，
      单位换算、生理极限拦截、三态标注全部由既有管线完成，此处绝不重复实现。

【职责边界 —— 解析器只做三件事】
  1. 把一行 OCR 文本拆成 (指标名, 数值, 单位, 打印参考区间, 高低标记)
  2. 用 lexicon.IndicatorLexicon 把指标名归一到标准码，并给出置信度
  3. 对每行给出结构置信度与问题清单；低于阈值的行进【人工复核队列】，
     默认不进数据帧 —— 规范 4.1"强校验"的落点：宁可让人多看一眼，
     不让一条可疑数据静默入模。

【解析器刻意"不做"的事（比做什么更重要）】
  - 不做单位换算、不做量级纠错 —— 那是 units.py 的职责，且它有 plausible_range
    做唯一性保证；解析器若抢着改数，等于两处各改一半，出错无法归因。
  - 不做小数点自动补齐。OCR 把 4.5 认成 45 时，45 对很多指标是合法值，
    自动补点就是凭空造数据。解析器能做的是【交叉证据降置信】：化验单上
    打印的参考区间与注册表区间量级严重不符时，标记 unit_or_scale_suspect
    并压低置信度，让这行走人工复核。
  - 不猜没写的字段。单位缺失就是缺失（置信度打折），由 units.py 决定
    是否走它的 magnitude_fix（那边有唯一性约束，是安全的）。

【行格式适配】国内化验单主流打印格式（含 OCR 常见畸变）：
    丙氨酸氨基转移酶 ALT      45    U/L     0-40      ↑
    葡萄糖(GLU)  6.8 mmol/L  参考值:3.9-6.1  H
    血小板计数 250 10^9/L 125-350
    白细胞计数6.5×10^9/L          （名值粘连、×号）
    肌酐 CREA 1.2 mg/dL 0.5-1.2   （非常用单位，交给 units.py 换算）
    乙肝表面抗原  阴性             （定性行：登记不入量化帧）
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field

import pandas as pd

from ..data.cleaning import COL_MEASURED_AT, COL_PATIENT_ID
from ..data.constants import COL_INDICATOR, COL_UNIT, COL_VALUE
from ..data.reference import ReferenceRegistry, to_halfwidth
from .lexicon import IndicatorLexicon, LexiconMatch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 词法层正则。全部在【半角化 + 空白规整】之后的文本上运行。
# ---------------------------------------------------------------------------

#: 打印参考区间：3.9-6.1 / 0~40 / 125—350，可带"参考值:"等前缀
_REF_RANGE = re.compile(
    r"(?:参考(?:值|区间|范围)?|正常(?:值|范围)?)?\s*[:：]?\s*"
    r"(?P<lo>\d+(?:\.\d+)?)\s*[-~—－～‐]{1,2}\s*(?P<hi>\d+(?:\.\d+)?)"
)
#: 单侧参考：<5 / ≤6.5 / >40
_REF_ONESIDE = re.compile(
    r"(?:参考(?:值|区间|范围)?|正常(?:值|范围)?)\s*[:：]?\s*[<>≤≥]\s*\d+(?:\.\d+)?"
)

#: 高低标记。箭头与"偏高/偏低"无歧义，允许出现在任意位置；
#: 裸 H/L 与单位尾字符（U/L、10^9/L 的 L）高度混淆，只认【行尾】且
#: 前一字符不是字母/数字/斜杠/脱字符的情形 —— 宁可漏掉一个中置 flag，
#: 不能把千百行的单位咬掉一角（那会让 unit 全体失配，走 units.py 兜底）。
_FLAG_ANYWHERE = re.compile(r"[↑↓]|偏高|偏低")
_FLAG_HL_END = re.compile(r"(?<![A-Za-z0-9/^])[HL]\s*$")

#: 数值 token（此时千分位逗号、OCR 句号小数点已在预处理阶段修复）
_NUMBER = re.compile(r"\d+(?:\.\d+)?")

#: 单位 token：数值后紧跟的计量单位。字符集覆盖 10^9/L、×10^12/L、mmHg、
#: kg/m2、μmol/L、%、mIU/L、/HP 等。最长 14 字符，防止把后续文字吞进来。
_UNIT_AFTER_VALUE = re.compile(
    r"^[\s]*(?P<unit>(?:[x×]?10\^?\d{1,2}/L)|(?:[A-Za-zμµ%][A-Za-zμµ%0-9^/·.\-]{0,13}))"
)

#: 定性结果关键词
_QUALITATIVE = re.compile(r"阴性|阳性|弱阳性|未见异常|未检出|正常$")

#: 页眉页脚/患者信息行关键词 —— 命中且行内无"数值+单位"结构时整行跳过
_META_LINE = re.compile(
    r"姓名|性别|年龄|床号|科别|科室|送检|采集|接收|审核|检验者|报告(?:时间|日期)?"
    r"|条码|样本(?:号|类型)|标本|门诊号|住院号|医院|页码|第\s*\d+\s*页|NO[.:]"
)

_MIN_CONFIDENCE_DEFAULT = 0.75


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class ParsedRow:
    """单行解析结果。indicator_code=None 的行永远不会进入量化数据帧。"""

    raw_line: str
    indicator_code: str | None
    matched_name: str | None       # 命中词典的那个片段（复核界面高亮用）
    value: float | None
    unit: str | None               # 原样保留，换算交给 units.py
    printed_ref: tuple[float, float] | None
    flag: str | None               # "↑"/"↓"/"H"/"L"/"偏高"/"偏低"
    confidence: float              # 名称置信 × 结构置信，∈[0,1]
    is_qualitative: bool = False
    issues: tuple[str, ...] = field(default=())
    lexicon: LexiconMatch | None = None

    @property
    def ingestible(self) -> bool:
        return (
            self.indicator_code is not None
            and self.value is not None
            and not self.is_qualitative
        )

    def to_log_dict(self) -> dict:
        return {
            "raw_line": self.raw_line,
            "indicator_code": self.indicator_code,
            "value": self.value,
            "unit": self.unit,
            "printed_ref": list(self.printed_ref) if self.printed_ref else None,
            "flag": self.flag,
            "confidence": round(self.confidence, 3),
            "is_qualitative": self.is_qualitative,
            "issues": list(self.issues),
        }


@dataclass
class ParseReport:
    """整页解析统计 + 人工复核队列。全链路日志（规范 4.2）要存它。"""

    n_lines: int = 0
    n_meta_skipped: int = 0
    n_ingested: int = 0
    n_review: int = 0
    n_qualitative: int = 0
    n_unmatched: int = 0
    review_queue: list[ParsedRow] = field(default_factory=list)
    unmatched: list[ParsedRow] = field(default_factory=list)
    qualitative: list[ParsedRow] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"OCR结构化: 输入{self.n_lines}行, 跳过元信息{self.n_meta_skipped}, "
            f"入帧{self.n_ingested}, 待复核{self.n_review}, "
            f"定性{self.n_qualitative}, 未识别{self.n_unmatched}"
        )

    def to_log_dict(self) -> dict:
        return {
            "n_lines": self.n_lines,
            "n_meta_skipped": self.n_meta_skipped,
            "n_ingested": self.n_ingested,
            "n_review": self.n_review,
            "n_qualitative": self.n_qualitative,
            "n_unmatched": self.n_unmatched,
            "review_queue": [r.to_log_dict() for r in self.review_queue],
            "unmatched_samples": [r.raw_line for r in self.unmatched[:50]],
        }


# ---------------------------------------------------------------------------
# 解析器主体
# ---------------------------------------------------------------------------
class LabReportParser:
    """
    化验单文本解析器。构造一次可复用；parse() 无副作用。

    min_confidence: 低于该置信度的行进 review_queue，不进数据帧。
    调低这个阈值前先想清楚：被它拦住的每一行，都是解析器自己都没把握的行。
    """

    def __init__(
        self,
        registry: ReferenceRegistry,
        min_confidence: float = _MIN_CONFIDENCE_DEFAULT,
    ):
        self.registry = registry
        self.lexicon = IndicatorLexicon(registry)
        self.min_confidence = min_confidence

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def parse(self, text: str) -> tuple[list[ParsedRow], ParseReport]:
        report = ParseReport()
        rows: list[ParsedRow] = []

        # 预处理：展开复合行（如 血压 152/96 mmHg 或多括号块）
        expanded_lines: list[str] = []
        for raw_line in text.splitlines():
            line_str = _preprocess(raw_line)
            if not line_str:
                continue

            # 1. 血压复合格式 (SBP/DBP 152/96 mmHg / 血压 152/96 mmHg 等)
            bp_m = re.search(
                r"(?:(?:SBP[/／]DBP|收缩压[/／]舒张压|血压|BP)[^\d]*)?(\d{2,3})\s*[/／]\s*(\d{2,3})\s*(?:mmHg)?",
                line_str,
                re.I,
            )
            if bp_m and any(kw in line_str.upper() for kw in ("SBP", "DBP", "血压", "BP", "MMHG")):
                sbp_val, dbp_val = bp_m.group(1), bp_m.group(2)
                expanded_lines.append(f"收缩压 SBP {sbp_val} mmHg 90-140")
                expanded_lines.append(f"舒张压 DBP {dbp_val} mmHg 60-90")
                continue

            # 2. 括号包含的多指标行: 如 (152/96 mmHg) (HGB 98 g/L)
            blocks = re.findall(r"[(（\[【][^)）\]】]+[)）\]】]", line_str)
            if len(blocks) > 1 and all(re.search(r"\d", b) for b in blocks):
                for b in blocks:
                    expanded_lines.append(b.strip(" ()[]（）【】"))
                continue

            expanded_lines.append(raw_line)

        for raw_line in expanded_lines:
            line = _preprocess(raw_line)
            if not line:
                continue
            report.n_lines += 1

            if _META_LINE.search(line) and not _has_value_structure(line):
                report.n_meta_skipped += 1
                continue

            row = self._parse_line(raw_line.strip(), line)
            rows.append(row)

            if row.is_qualitative:
                report.n_qualitative += 1
                report.qualitative.append(row)
            elif not row.ingestible:
                report.n_unmatched += 1
                report.unmatched.append(row)
            elif row.confidence < self.min_confidence:
                report.n_review += 1
                report.review_queue.append(row)
            else:
                report.n_ingested += 1

        logger.info(report.summary())
        return rows, report

    # ------------------------------------------------------------------
    def _parse_line(self, raw_line: str, line: str) -> ParsedRow:
        issues: list[str] = []

        # 1) 高低标记：先摘走，避免与单位/名称粘连
        flag = None
        m = _FLAG_ANYWHERE.search(line)
        if m:
            flag = m.group(0)
            line = (line[: m.start()] + " " + line[m.end():]).strip()
        else:
            m = _FLAG_HL_END.search(line)
            if m:
                flag = m.group(0).strip()
                line = line[: m.start()].strip()

        # 2) 打印参考区间：先于取值摘走，防止区间数字被误认为结果值。
        printed_ref: tuple[float, float] | None = None
        ref_m = _find_ref_range(line)
        if ref_m is not None:
            printed_ref = (float(ref_m.group("lo")), float(ref_m.group("hi")))
            line = (line[: ref_m.start()] + " " + line[ref_m.end():]).strip()
        else:
            one = _REF_ONESIDE.search(line)
            if one:
                line = (line[: one.start()] + " " + line[one.end():]).strip()

        # 3) 定性行判定：无数值且含定性关键词
        if not _NUMBER.search(line):
            if _QUALITATIVE.search(line):
                name_part = _QUALITATIVE.sub("", line).strip(" :：")
                return ParsedRow(
                    raw_line=raw_line, indicator_code=None, matched_name=name_part,
                    value=None, unit=None, printed_ref=None, flag=flag,
                    confidence=1.0, is_qualitative=True,
                )
            return ParsedRow(
                raw_line=raw_line, indicator_code=None, matched_name=None,
                value=None, unit=None, printed_ref=printed_ref, flag=flag,
                confidence=0.0, issues=("no_value",),
            )

        # 4) 智能数值与名称定位：优先匹配括号指标标记，避免被行首乱码/序号 888888 抢占数值
        paren = re.search(r"[(（\[【]([A-Za-zμµ0-9%/·\s\-+]+)[)）\]】]", line)
        if paren and any(c.isalpha() for c in paren.group(1)):
            after_paren = line[paren.end():]
            vm = _NUMBER.search(after_paren)
            if vm:
                value = float(vm.group(0))
                name_region = paren.group(0)
                rest = after_paren[vm.end():]
            else:
                vm = _NUMBER.search(line)
                value = float(vm.group(0))
                name_region = line[: vm.start()].strip(" :：·").rstrip("(（")
                rest = line[vm.end():]
        else:
            vm = _NUMBER.search(line)
            value = float(vm.group(0))
            name_region = line[: vm.start()].strip(" :：·").rstrip("(（")
            rest = line[vm.end():]

        unit = None
        um = _UNIT_AFTER_VALUE.match(rest)
        if um:
            unit = um.group("unit").strip()
            # 6.5×10^9/L 里的 ×/x 是数值科学计数的乘号，不是单位的一部分
            if re.match(r"^[x×]10", unit):
                unit = unit[1:]
        if unit is None and ("%" in name_region or "%" in line):
            unit = "%"

        if unit is None:
            issues.append("unit_missing")

        # 5) 名称 -> 词典。多层智能候选抽取
        match = self._match_name(name_region)
        if not match.matched:
            return ParsedRow(
                raw_line=raw_line, indicator_code=None,
                matched_name=name_region or None, value=value, unit=unit,
                printed_ref=printed_ref, flag=flag, confidence=0.0,
                issues=tuple(issues + ["indicator_unmatched"]), lexicon=match,
            )

        # 6) 结构置信度合成
        conf = match.confidence
        if unit is None:
            conf *= 0.90
        if printed_ref is not None:
            if match.code in ("NEUT", "LYMPH") and (unit == "%" or printed_ref[1] > 20):
                pass
            else:
                scale_issue = self._ref_scale_check(match.code, printed_ref)
                if scale_issue:
                    issues.append(scale_issue)
                    conf *= 0.70   # 量级疑点是最危险的疑点，压得最狠

        return ParsedRow(
            raw_line=raw_line, indicator_code=match.code,
            matched_name=name_region, value=value, unit=unit,
            printed_ref=printed_ref, flag=flag, confidence=conf,
            issues=tuple(issues), lexicon=match,
        )

    # ------------------------------------------------------------------
    def _match_name(self, name_region: str) -> LexiconMatch:
        candidates: list[str] = []
        if not name_region:
            return self.lexicon.lookup("")

        raw = name_region.strip()
        candidates.append(raw)

        # 1. 剥离前置/后置标记 [H], [L], [↑], [↓], [+], (H) 等
        cleaned = re.sub(r"^[\[\(\（【][HL↑↓+ -]+[\]\)\）】]\s*", "", raw)
        cleaned = re.sub(r"\s*[\[\(\（【][HL↑↓+ -]+[\]\)\）】]$", "", cleaned)
        if cleaned != raw:
            candidates.append(cleaned)

        # 2. 提取括号内/外内容: (UA/ URIC) -> UA/ URIC
        paren_match = re.search(r"[(（\[【]([^)）\]】]+)[)）\]】]", cleaned)
        if paren_match:
            inside = paren_match.group(1).strip()
            candidates.append(inside)
            outside = re.sub(r"[(（\[【][^)）\]】]*[)）\]】]", "", cleaned).strip()
            if outside:
                candidates.append(outside)

        # 3. 按 / 或 、 或 | 拆分复合指标: UA/URIC, Cr/SCR, HGB/HB, ALT/GPT
        for c in list(candidates):
            if "/" in c or "／" in c or "、" in c or "|" in c:
                parts = re.split(r"[/／、|]", c)
                for p in parts:
                    p_clean = p.strip()
                    if p_clean:
                        candidates.append(p_clean)
                        # 消除拉丁缩写内部的 OCR 空格: 'SC R' -> 'SCR'
                        p_nospaces = re.sub(r"\s+", "", p_clean)
                        if p_nospaces != p_clean:
                            candidates.append(p_nospaces)

        # 4. 拉丁尾与前缀
        for c in list(candidates):
            lt = re.search(r"[A-Za-zμµ][A-Za-zμµ0-9\-+]*\s*$", c)
            if lt:
                candidates.append(lt.group(0).strip())
                prefix = c[: lt.start()].strip()
                if prefix:
                    candidates.append(prefix)

        # 去重并查询词典
        seen = set()
        best: LexiconMatch | None = None
        for cand in candidates:
            c_clean = cand.strip(" :：·()（）[]【】")
            if not c_clean or c_clean in seen:
                continue
            seen.add(c_clean)
            m = self.lexicon.lookup(c_clean)
            if m.matched:
                if best is None or m.confidence > best.confidence:
                    best = m
                    if m.confidence >= 1.0:
                        break
        if best is not None:
            return best
        return self.lexicon.lookup(name_region)

    # ------------------------------------------------------------------
    def _ref_scale_check(self, code: str, printed_ref: tuple[float, float]) -> str | None:
        """
        交叉证据：化验单上打印的参考区间 vs 注册表参考区间的量级比对。
        量级差 >= 10 倍 -> 极可能是单位不同或 OCR 丢小数点，标记待复核。
        只降置信不改数：改数是 units.py 的事，它有唯一性保证。
        """
        meta = self.registry.get(code)
        if meta is None or not meta.intervals:
            return None
        iv = meta.match_interval(sex="ANY", age=None)
        if iv is None or iv.center is None:
            return None
        printed_center = (printed_ref[0] + printed_ref[1]) / 2.0
        if printed_center <= 0 or iv.center <= 0:
            return None
        ratio = printed_center / iv.center
        if ratio <= 0 or not math.isfinite(ratio):
            return None
        if abs(math.log10(ratio)) >= 1.0:
            return "unit_or_scale_suspect"
        return None


# ---------------------------------------------------------------------------
# 预处理与小工具
# ---------------------------------------------------------------------------
def _preprocess(raw: str) -> str:
    """半角化、OCR 标点修复、序号清理、空白规整。"""
    s = raw.strip()
    s = re.sub(r"^\s*\d+[\.、:：]\s*", "", s)       # 移除开头的行号序号 1. 2、等
    s = re.sub(r"^[\[\(\（【][HL↑↓+ -]+[\]\)\）】]\s*", "", s)  # 移除开头的 [H] 等标记
    s = to_halfwidth(s)
    s = s.replace("×", "x")
    s = re.sub(r"(?<=\d)[,，](?=\d{3}\b)", "", s)       # 千分位 250,000 -> 250000
    s = re.sub(r"(?<=\d)[。](?=\d)", ".", s)            # OCR 句号小数点 6。8 -> 6.8
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _has_value_structure(line: str) -> bool:
    """行内存在"数值 + 单位样"结构，用于避免误杀含元信息关键词的正文行。"""
    m = _NUMBER.search(line)
    if not m:
        return False
    return _UNIT_AFTER_VALUE.match(line[m.end():]) is not None


def _find_ref_range(line: str) -> re.Match | None:
    """
    找参考区间。约束：要么带"参考/正常"前缀，要么出现在第一个数值之后
    且区间前有空白间隔 —— 双数值行 "45 0-40" 里的 0-40 才会被认出来。
    """
    first_num = _NUMBER.search(line)
    for m in _REF_RANGE.finditer(line):
        prefixed = m.group(0).lstrip()[0] not in "0123456789"
        if prefixed:
            return m
        if first_num is not None and m.start() > first_num.end():
            return m
    return None


# ---------------------------------------------------------------------------
# 桥接：解析结果 -> 清洗管线标准长表
# ---------------------------------------------------------------------------
def rows_to_frame(
    rows: list[ParsedRow],
    patient_id: str,
    measured_at,
    min_confidence: float = _MIN_CONFIDENCE_DEFAULT,
) -> pd.DataFrame:
    """
    产出 LabDataCleaner.clean() 的标准输入长表。
    只收 ingestible 且置信度达标的行；单位原样透传，换算由 units.py 完成。
    """
    records = [
        {
            COL_PATIENT_ID: patient_id,
            COL_INDICATOR: r.indicator_code,
            COL_VALUE: r.value,
            COL_UNIT: r.unit,
            COL_MEASURED_AT: pd.Timestamp(measured_at),
        }
        for r in rows
        if r.ingestible and r.confidence >= min_confidence
    ]
    if not records:
        return pd.DataFrame(
            columns=[COL_PATIENT_ID, COL_INDICATOR, COL_VALUE, COL_UNIT, COL_MEASURED_AT]
        )
    return pd.DataFrame.from_records(records)


def parse_lab_text(
    text: str,
    registry: ReferenceRegistry,
    patient_id: str,
    measured_at,
    min_confidence: float = _MIN_CONFIDENCE_DEFAULT,
) -> tuple[pd.DataFrame, ParseReport, list[ParsedRow]]:
    """一步到位的便捷入口：OCR 文本 -> (标准长表, 解析报告, 全部行)。"""
    parser = LabReportParser(registry, min_confidence=min_confidence)
    rows, report = parser.parse(text)
    frame = rows_to_frame(rows, patient_id, measured_at, min_confidence=min_confidence)
    return frame, report, rows
