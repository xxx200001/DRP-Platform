"""
智能就医建议引擎（规范 6："根据异常指标精准推荐科室、检查项目"）。

【为什么是纯规则表，不是模型，更不是大模型】
科室/检查推荐是强合规敏感输出（规范 7）。它必须满足三个性质：
可枚举审查（法务能逐条看完所有可能输出）、可复现（同输入永远同输出）、
可追责（每条建议能指回触发它的具体指标）。规则表天然满足；任何生成式
方案都不满足。这不是技术保守，是这个功能的正确形态。

【与 compliance.py 的关系是"物理串联"而非"约定俗成"】
render_text() 的最后一行就是 assert_compliant() —— 词表若与合规模块的
禁用词冲突，会在单元测试阶段直接炸掉，而不是上线后被监管指出来。
所有措辞只从 compliance.ALLOWED_PHRASES 的家族里取："建议就诊""建议复查"
"推荐科室""检查项目"。永远不说"确诊""治疗""用药"。

【分级动作 —— 三档优先级的语义】
  URGENT  触发条件：任一指标命中注册表 critical（危急值）
          动作措辞：建议尽快前往正规医疗机构就诊
          危急值是检验医学的法定概念（危及生命，需立即处理），
          平台不能替医生处理它，但把它按"常规复查"输出等于隐瞒。
  SOON    触发条件：重度异常（|grade|==3）或整体风险分层为"极高危"
          动作措辞：建议近期就诊 + 针对性复查项目
  ROUTINE 触发条件：轻/中度异常，或分层"高危"但无异常指标
          动作措辞：建议复查随访

【异常分级的单一真源】
分级数学与 features/deviation.py 完全一致（区间边界 × grade_multiplier，
偏低侧用除法）。为避免 serving 依赖 features 的私有函数，这里独立实现
grade_value()，并由 tests/test_ingest.py 里的网格一致性测试钉死两者行为
相同 —— 谁改了其中一个而没改另一个，CI 立刻红。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum

import pandas as pd

from ..data.constants import (
    COL_AGE,
    COL_INDICATOR,
    COL_PATIENT_ID,
    COL_SEX,
    COL_VALUE,
    AbnormalGrade,
)
from ..data.reference import IndicatorMeta, ReferenceRegistry
from .compliance import assert_compliant, attach_disclaimer

logger = logging.getLogger(__name__)


class Priority(IntEnum):
    ROUTINE = 1
    SOON = 2
    URGENT = 3

    def label_cn(self) -> str:
        return {1: "常规随访", 2: "建议近期就诊", 3: "建议尽快就诊"}[int(self)]


# ---------------------------------------------------------------------------
# 规则表：指标组 -> (科室, 推荐检查项目)。
# 维护规则：新指标入 reference_intervals.yaml 时必须同步归组，
#          test_ingest.py 的覆盖测试会检查"每个注册指标都有归属组"。
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DeptRule:
    group: str
    department: str
    codes: tuple[str, ...]
    checkups: tuple[str, ...]


DEPARTMENT_RULES: tuple[DeptRule, ...] = (
    DeptRule(
        "肝功能", "消化内科（或肝病科）",
        ("ALT", "AST", "GGT", "ALP", "TBIL", "DBIL", "IBIL", "ALB", "GLB", "TP", "AGR", "TBA", "LAP"),
        ("肝功能全套复查", "腹部超声检查", "肝炎相关指标筛查"),
    ),
    DeptRule(
        "肾功能", "肾内科",
        ("CREA", "UREA", "UA", "UACR", "RBP"),
        ("肾功能复查", "尿常规检查", "泌尿系统超声检查"),
    ),
    DeptRule(
        "血糖代谢", "内分泌科",
        ("GLU", "HBA1C", "INS"),
        ("空腹血糖与糖化血红蛋白复查", "口服葡萄糖耐量试验(OGTT)检查"),
    ),
    DeptRule(
        "血脂", "心血管内科（或内分泌科）",
        ("TC", "TG", "HDLC", "LDLC"),
        ("血脂四项复查", "颈动脉超声检查"),
    ),
    DeptRule(
        "血常规", "血液内科",
        ("WBC", "NEUT", "LYMPH", "MONO", "BASO", "EOS", "PLT", "HGB", "RBC", "HCT", "MCV", "MCH", "MCHC", "RDW"),
        ("血常规复查", "外周血细胞形态学检查"),
    ),
    DeptRule(
        "电解质", "肾内科（或内分泌科）",
        ("K", "NA", "CL", "CA", "P"),
        ("电解质复查", "肾功能检查", "心电图检查"),
    ),
    DeptRule(
        "心肌酶谱", "心血管内科",
        ("LDH", "CK", "CKMB"),
        ("心肌酶谱复查", "心电图检查", "心脏超声检查"),
    ),
    DeptRule(
        "消化酶谱", "消化内科",
        ("AMY",),
        ("血淀粉酶复查", "腹部超声检查"),
    ),
    DeptRule(
        "免疫指标", "风湿免疫科",
        ("IGG", "IGA", "IGM", "C3", "C4"),
        ("免疫球蛋白全套复查", "补体C3/C4复查"),
    ),
    DeptRule(
        "炎症指标", "全科医学科（或感染科）",
        ("CRP",),
        ("C反应蛋白复查", "血常规检查"),
    ),
    DeptRule(
        "血压", "心血管内科",
        ("SBP", "DBP"),
        ("多次静息血压测量", "动态血压监测检查", "心电图检查"),
    ),
    DeptRule(
        "体重管理", "临床营养科（或内分泌科）",
        ("BMI",),
        ("体成分分析检查", "代谢相关指标复查"),
    ),
)

_CODE_TO_RULE: dict[str, DeptRule] = {
    code: rule for rule in DEPARTMENT_RULES for code in rule.codes
}


# ---------------------------------------------------------------------------
# 异常分级（与 features/deviation._grade_of 数学一致，测试钉死）
# ---------------------------------------------------------------------------
def grade_value(
    meta: IndicatorMeta, value: float, sex: str = "ANY", age: float | None = None
) -> AbnormalGrade:
    iv = meta.match_interval(sex=sex, age=age)
    if iv is None:
        return AbnormalGrade.NORMAL
    gm = meta.grade_multiplier
    if iv.upper is not None and value > iv.upper:
        if value >= iv.upper * gm.severe:
            return AbnormalGrade.SEVERE_HIGH
        if value >= iv.upper * gm.moderate:
            return AbnormalGrade.MODERATE_HIGH
        if value > iv.upper * gm.mild:
            return AbnormalGrade.MILD_HIGH
        return AbnormalGrade.NORMAL
    if iv.lower is not None and value < iv.lower:
        if gm.severe > 0 and value <= iv.lower / gm.severe:
            return AbnormalGrade.SEVERE_LOW
        if gm.moderate > 0 and value <= iv.lower / gm.moderate:
            return AbnormalGrade.MODERATE_LOW
        if gm.mild > 0 and value < iv.lower / gm.mild:
            return AbnormalGrade.MILD_LOW
    return AbnormalGrade.NORMAL


# ---------------------------------------------------------------------------
# 输出结构
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IndicatorFinding:
    """触发建议的单条指标事实。reason 只陈述事实，不作因果推断。"""

    code: str
    name_cn: str
    value: float
    unit: str
    grade: int
    is_critical: bool

    def reason_text(self, meta: IndicatorMeta, sex: str, age: float | None) -> str:
        iv = meta.match_interval(sex=sex, age=age)
        ref = ""
        if iv is not None:
            lo = "" if iv.lower is None else f"{iv.lower:g}"
            hi = "" if iv.upper is None else f"{iv.upper:g}"
            ref = f"，参考区间 {lo}-{hi}{self.unit}"
        return (
            f"{self.name_cn} 当前 {self.value:g}{self.unit}"
            f"（{AbnormalGrade.label_cn(self.grade)}{ref}）"
        )


@dataclass(frozen=True)
class ReferralItem:
    department: str
    group: str
    priority: Priority
    findings: tuple[IndicatorFinding, ...]
    checkups: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass
class ReferralAdvice:
    items: list[ReferralItem] = field(default_factory=list)
    general_note: str | None = None   # 分层驱动的整体建议（无异常指标时也可能有）
    rendered_text: str = ""

    def to_dict(self) -> dict:
        return {
            "items": [
                {
                    "department": it.department,
                    "group": it.group,
                    "priority": int(it.priority),
                    "priority_label": it.priority.label_cn(),
                    "findings": [f.code for f in it.findings],
                    "checkups": list(it.checkups),
                    "reasons": list(it.reasons),
                }
                for it in self.items
            ],
            "general_note": self.general_note,
            "rendered_text": self.rendered_text,
        }


# ---------------------------------------------------------------------------
# 引擎
# ---------------------------------------------------------------------------
class ReferralEngine:
    """
    输入：清洗后的长表（CleaningPipeline 输出，含 canonical 单位的 value）
          + 可选人口学（sex/age，用于年龄性别分层区间）
          + 可选整体风险分层名（service.RiskTierScheme 的输出）
    输出：ReferralAdvice。rendered_text 已通过合规断言并挂免责声明。
    """

    def __init__(self, registry: ReferenceRegistry, max_items: int = 5):
        self.registry = registry
        self.max_items = max_items

    # ------------------------------------------------------------------
    def advise(
        self,
        cleaned: pd.DataFrame,
        demographics: pd.DataFrame | None = None,
        risk_tier: str | None = None,
    ) -> ReferralAdvice:
        sex, age = extract_demographics(cleaned, demographics)
        findings = self._collect_findings(cleaned, sex, age)

        grouped: dict[str, list[IndicatorFinding]] = {}
        for f in findings:
            rule = _CODE_TO_RULE.get(f.code)
            if rule is None:
                logger.warning("指标 %s 未归属任何科室规则组，请补充 DEPARTMENT_RULES", f.code)
                continue
            grouped.setdefault(rule.group, []).append(f)

        items: list[ReferralItem] = []
        for group, fs in grouped.items():
            rule = next(r for r in DEPARTMENT_RULES if r.group == group)
            fs_sorted = sorted(fs, key=lambda x: (x.is_critical, abs(x.grade)), reverse=True)
            priority = _group_priority(fs_sorted, risk_tier)
            reasons = tuple(
                f.reason_text(self.registry.require(f.code), sex, age) for f in fs_sorted
            )
            items.append(
                ReferralItem(
                    department=rule.department, group=group, priority=priority,
                    findings=tuple(fs_sorted), checkups=rule.checkups, reasons=reasons,
                )
            )

        items.sort(
            key=lambda it: (int(it.priority), max(abs(f.grade) for f in it.findings)),
            reverse=True,
        )
        items = items[: self.max_items]

        general = _general_note(risk_tier, has_items=bool(items))
        advice = ReferralAdvice(items=items, general_note=general)
        advice.rendered_text = self.render_text(advice)
        return advice

    # ------------------------------------------------------------------
    def _collect_findings(
        self, cleaned: pd.DataFrame, sex: str, age: float | None
    ) -> list[IndicatorFinding]:
        """每个指标只取【最近一次】记录评估 —— 就医建议看的是当前状态。"""
        if cleaned.empty:
            return []
        df = cleaned
        if "measured_at" in df.columns:
            df = df.sort_values("measured_at").groupby(COL_INDICATOR, as_index=False).tail(1)

        out: list[IndicatorFinding] = []
        for _, row in df.iterrows():
            code = row[COL_INDICATOR]
            meta = self.registry.get(code)
            value = row[COL_VALUE]
            if meta is None or pd.isna(value):
                continue
            value = float(value)
            grade = grade_value(meta, value, sex=sex, age=age)
            critical = meta.is_critical(value)
            if grade == AbnormalGrade.NORMAL and not critical:
                continue
            out.append(
                IndicatorFinding(
                    code=code, name_cn=meta.name_cn, value=value,
                    unit=meta.canonical_unit, grade=int(grade), is_critical=critical,
                )
            )
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def render_text(advice: ReferralAdvice) -> str:
        """
        模板渲染。产出后立刻 assert_compliant —— 本方法是文本出口的唯一通道，
        规则表词汇一旦与禁用词冲突，测试期就会在这里爆炸。
        """
        lines: list[str] = ["【就医建议】以下内容基于本次检验指标自动生成，供您参考。"]
        if not advice.items and advice.general_note is None:
            lines.append("本次各项指标均在参考区间内，建议保持定期体检的习惯。")
        for it in advice.items:
            lines.append(f"◆ {it.priority.label_cn()}｜推荐科室：{it.department}")
            for r in it.reasons:
                lines.append(f"    - {r}")
            lines.append(f"    推荐检查项目：{'、'.join(it.checkups)}")
            if it.priority == Priority.URGENT:
                lines.append(
                    "    上述指标达到需要重点关注的程度，建议尽快前往正规医疗机构就诊，"
                    "由执业医师作出判断。"
                )
        if advice.general_note:
            lines.append(advice.general_note)

        text = attach_disclaimer("\n".join(lines))
        assert_compliant(text, source="referral.render_text")
        return text


# ---------------------------------------------------------------------------
# 内部小函数
# ---------------------------------------------------------------------------
def extract_demographics(
    cleaned: pd.DataFrame, demographics: pd.DataFrame | None
) -> tuple[str, float | None]:
    sex, age = "ANY", None
    if demographics is not None and not demographics.empty:
        row = demographics.iloc[0]
        if COL_SEX in demographics.columns and pd.notna(row.get(COL_SEX)):
            sex = str(row[COL_SEX]).upper()
        if COL_AGE in demographics.columns and pd.notna(row.get(COL_AGE)):
            age = float(row[COL_AGE])
        elif "birth_date" in demographics.columns and pd.notna(row.get("birth_date")):
            ref_time = None
            if "measured_at" in cleaned.columns and not cleaned.empty:
                ref_time = pd.to_datetime(cleaned["measured_at"]).max()
            ref_time = ref_time or pd.Timestamp.now()
            age = (ref_time - pd.Timestamp(row["birth_date"])).days / 365.25
    return sex, age


def _group_priority(fs: list[IndicatorFinding], risk_tier: str | None) -> Priority:
    if any(f.is_critical for f in fs):
        return Priority.URGENT
    if any(abs(f.grade) >= 3 for f in fs):
        return Priority.SOON
    if risk_tier == "极高危":
        return Priority.SOON
    return Priority.ROUTINE


def _general_note(risk_tier: str | None, has_items: bool) -> str | None:
    if risk_tier == "极高危":
        return (
            "本次综合评估的风险分层为「极高危」。无论上述单项指标情况如何，"
            "都建议您近期前往正规医疗机构就诊，由执业医师结合完整病史作出判断。"
        )
    if risk_tier == "高危" and not has_items:
        return (
            "本次综合评估的风险分层为「高危」，虽然各单项指标未见明显异常，"
            "仍建议您加强定期复查随访。"
        )
    return None


__all__ = [
    "DEPARTMENT_RULES",
    "DeptRule",
    "IndicatorFinding",
    "Priority",
    "ReferralAdvice",
    "ReferralEngine",
    "ReferralItem",
    "extract_demographics",
    "grade_value",
]
