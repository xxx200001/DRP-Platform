"""
病情趋势追踪与时序对比报告（规范 6）。

规范原话：
  - "病情趋势追踪：多次体检自动生成病情变化曲线、风险走势"
  - "时序对比报告：本次 vs 上次、近3次变化对比"

【本模块与 features/temporal.py 的分工】
temporal.py 产出的是【喂给模型的特征】—— 整窗口首尾对比、OLS 斜率，
服务于"预测准不准"。本模块产出的是【给用户看的报告】—— 相邻两次的
逐步对比、可直接渲染的曲线点位，服务于"用户看不看得懂"。两者的读者
不同，粒度也不同（相邻步进 vs 整窗口回归），因此没有直接复用
temporal._trend_of，而是基于同一个 RCV 数学（IndicatorMeta.rcv）
独立实现——真正共享的只有分级数学，通过导入 referral.grade_value
（已在批次 5 与 features.deviation 做过网格一致性测试）保证单一真源。

【风险走势为什么读 AuditLogger 而不是重新预测】
"风险走势"是历史每一次预测【当时】给出的概率，不是用今天的模型重新跑
一遍历史数据——后者会与用户当时看到的结果对不上，而且如果模型已经
迭代过（retrain.py），重算出来的曲线会包含"事后修正"，反而失去了
"模型当时是怎么判断的"这个复盘价值。审计日志的 append-only 特性
（audit.py）刚好保证了这条曲线不可被事后篡改。

【为什么"本次 vs 上次"只报告"真实变化"，不逐项罗列】
把 33 个指标的升降全列出来，噪声项（比如 CRP 从 3.2 波动到 4.1，
远在 RCV 内）会淹没真正有意义的变化，用户看不出重点，等同于没有
归纳。所以 compare_latest 对每个指标都计算 is_real_change，
render_text 只叙述 is_real_change=True 的条目；全体条目仍在
返回的结构化数据里，前端图表需要完整曲线时不受影响。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from ..data.constants import (
    COL_INDICATOR,
    COL_MEASURED_AT,
    COL_VALUE,
    AbnormalGrade,
)
from ..data.reference import IndicatorMeta, ReferenceRegistry
from .attribution import ChangeAttribution, RiskAttribution, explain_change
from .compliance import assert_compliant, attach_disclaimer
from .referral import extract_demographics, grade_value

logger = logging.getLogger(__name__)

# 相对变化在 RCV 内一律视为噪声，无论方向 —— 与 temporal.py 的判定标准一致，
# 确保"报告里说平稳"和"特征里判平稳"永远是同一个结论，不会自相矛盾。


# ---------------------------------------------------------------------------
# 单指标：本次 vs 上次
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IndicatorComparison:
    code: str
    name_cn: str
    unit: str
    prev_value: float
    curr_value: float
    prev_at: pd.Timestamp
    curr_at: pd.Timestamp
    delta: float
    delta_pct: float | None  # prev_value 为 0 时无意义，置 None
    rcv: float
    is_real_change: bool
    direction: str  # "上升" / "下降" / "平稳"
    prev_grade: int
    curr_grade: int

    @property
    def worsened(self) -> bool:
        """严重度是否恶化：分级绝对值变大，或从正常越界到异常。"""
        return abs(self.curr_grade) > abs(self.prev_grade)

    def to_dict(self) -> dict:
        return {
            "code": self.code, "name_cn": self.name_cn, "unit": self.unit,
            "prev_value": self.prev_value, "curr_value": self.curr_value,
            "prev_at": str(self.prev_at), "curr_at": str(self.curr_at),
            "delta": self.delta, "delta_pct": self.delta_pct, "rcv": self.rcv,
            "is_real_change": self.is_real_change, "direction": self.direction,
            "prev_grade": self.prev_grade, "curr_grade": self.curr_grade,
            "worsened": self.worsened,
        }

    def phrase(self) -> str:
        pct = f"（{self.delta_pct:+.1%}）" if self.delta_pct is not None else ""
        if not self.is_real_change:
            return f"{self.name_cn} 基本平稳，维持在 {self.curr_value:g}{self.unit} 附近"
        tag = "，程度较前次加重" if self.worsened else ""
        return (
            f"{self.name_cn} 由 {self.prev_value:g}{self.unit} "
            f"{self.direction}至 {self.curr_value:g}{self.unit}{pct}{tag}"
        )


def _direction_and_real(prev: float, curr: float, rcv: float) -> tuple[str, bool]:
    if abs(prev) < 1e-9:
        return ("平稳", False)
    rel = (curr - prev) / abs(prev)
    if abs(rel) <= rcv:
        return "平稳", False
    return ("上升" if rel > 0 else "下降"), True


def _compare_pair(
    meta: IndicatorMeta,
    prev_row: pd.Series,
    curr_row: pd.Series,
    sex: str,
    age: float | None,
) -> IndicatorComparison:
    prev_v, curr_v = float(prev_row[COL_VALUE]), float(curr_row[COL_VALUE])
    rcv = meta.rcv
    direction, is_real = _direction_and_real(prev_v, curr_v, rcv)
    delta_pct = (curr_v - prev_v) / abs(prev_v) if abs(prev_v) >= 1e-9 else None
    return IndicatorComparison(
        code=meta.code, name_cn=meta.name_cn, unit=meta.canonical_unit,
        prev_value=prev_v, curr_value=curr_v,
        prev_at=pd.Timestamp(prev_row[COL_MEASURED_AT]),
        curr_at=pd.Timestamp(curr_row[COL_MEASURED_AT]),
        delta=curr_v - prev_v, delta_pct=delta_pct, rcv=rcv, is_real_change=is_real,
        direction=direction,
        prev_grade=int(grade_value(meta, prev_v, sex, age)),
        curr_grade=int(grade_value(meta, curr_v, sex, age)),
    )


# ---------------------------------------------------------------------------
# 单指标：近 N 次曲线
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IndicatorSeries:
    code: str
    name_cn: str
    unit: str
    points: tuple[tuple[pd.Timestamp, float, int], ...]  # (时间, 数值, 分级) 按时间升序
    steps: tuple[IndicatorComparison, ...]  # 相邻两两对比，长度 = len(points)-1

    def to_dict(self) -> dict:
        return {
            "code": self.code, "name_cn": self.name_cn, "unit": self.unit,
            "points": [
                {"at": str(t), "value": v, "grade": g} for t, v, g in self.points
            ],
            "steps": [s.to_dict() for s in self.steps],
        }


class TrendEngine:
    """
    输入：清洗后的长表（同一患者、多次记录）+ 可选人口学。
    产出：指标级对比 / 曲线，以及（可选）风险走势与风险变化归因。
    """

    def __init__(self, registry: ReferenceRegistry):
        self.registry = registry

    # ------------------------------------------------------------------
    def compare_latest(
        self, cleaned: pd.DataFrame, demographics: pd.DataFrame | None = None
    ) -> list[IndicatorComparison]:
        """本次 vs 上次。只对【观测次数 >= 2】的指标产出条目。"""
        sex, age = extract_demographics(cleaned, demographics)
        out: list[IndicatorComparison] = []
        if cleaned.empty:
            return out
        for code, grp in cleaned.groupby(COL_INDICATOR):
            meta = self.registry.get(code)
            if meta is None or len(grp) < 2:
                continue
            grp = grp.sort_values(COL_MEASURED_AT)
            prev_row, curr_row = grp.iloc[-2], grp.iloc[-1]
            out.append(_compare_pair(meta, prev_row, curr_row, sex, age))
        out.sort(key=lambda c: (not c.is_real_change, -abs(c.curr_grade)))
        return out

    # ------------------------------------------------------------------
    def recent_series(
        self,
        cleaned: pd.DataFrame,
        n: int = 3,
        demographics: pd.DataFrame | None = None,
    ) -> list[IndicatorSeries]:
        """近 N 次变化对比。观测数不足 N 的指标，有多少给多少（不补齐、不报错）。"""
        sex, age = extract_demographics(cleaned, demographics)
        out: list[IndicatorSeries] = []
        if cleaned.empty:
            return out
        for code, grp in cleaned.groupby(COL_INDICATOR):
            meta = self.registry.get(code)
            if meta is None:
                continue
            grp = grp.sort_values(COL_MEASURED_AT).tail(n)
            points = tuple(
                (
                    pd.Timestamp(row[COL_MEASURED_AT]),
                    float(row[COL_VALUE]),
                    int(grade_value(meta, float(row[COL_VALUE]), sex, age)),
                )
                for _, row in grp.iterrows()
            )
            steps = tuple(
                _compare_pair(meta, grp.iloc[i], grp.iloc[i + 1], sex, age)
                for i in range(len(grp) - 1)
            )
            out.append(
                IndicatorSeries(
                    code=code, name_cn=meta.name_cn, unit=meta.canonical_unit,
                    points=points, steps=steps,
                )
            )
        return out


# ---------------------------------------------------------------------------
# 风险走势（读 AuditLogger 历史，不重新预测 —— 见模块顶部说明）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RiskTrajectoryPoint:
    at: pd.Timestamp
    probability: float
    risk_tier: str
    model_version: str
    trace_id: str

    def to_dict(self) -> dict:
        return {
            "at": str(self.at), "probability": self.probability,
            "risk_tier": self.risk_tier, "model_version": self.model_version,
            "trace_id": self.trace_id,
        }


@dataclass(frozen=True)
class RiskTrajectory:
    horizon: str
    points: tuple[RiskTrajectoryPoint, ...]  # created_at 升序

    @property
    def latest_change(self) -> tuple[float, float] | None:
        if len(self.points) < 2:
            return None
        return self.points[-2].probability, self.points[-1].probability

    def to_dict(self) -> dict:
        return {"horizon": self.horizon, "points": [p.to_dict() for p in self.points]}


def risk_trajectory_from_audit(
    audit,
    pseudo_id: str,
    horizon: str,
    since: str | None = None,
    until: str | None = None,
) -> RiskTrajectory:
    """audit: serving.audit.AuditLogger。类型不作强注解以避免循环导入。"""
    df = audit.history_for_patient(pseudo_id, since=since, until=until, horizon=horizon)
    if df.empty:
        return RiskTrajectory(horizon=horizon, points=())
    points = tuple(
        RiskTrajectoryPoint(
            at=pd.Timestamp(row["created_at"]), probability=float(row["probability"]),
            risk_tier=str(row["risk_tier"]), model_version=str(row.get("model_version", "")),
            trace_id=str(row["trace_id"]),
        )
        for _, row in df.iterrows()
    )
    return RiskTrajectory(horizon=horizon, points=points)


# ---------------------------------------------------------------------------
# 趋势干预建议与应对办法（规范 6 衍生：根据恶化/异常趋势动态给出生活与复查方案）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrendIntervention:
    system: str           # "肝胆代谢" / "糖代谢管理" / "心血管与血脂" / "肾脏与尿酸" / "常规健康管理"
    icon: str             # "🫀" / "🥗" / "🏃" / "💧" / "🩺"
    level: str            # "重点关注" / "积极改善" / "平稳维持"
    target_indicators: tuple[str, ...]
    diet_advice: tuple[str, ...]
    lifestyle_advice: tuple[str, ...]
    followup_cycle: str
    red_flags: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "system": self.system,
            "icon": self.icon,
            "level": self.level,
            "target_indicators": list(self.target_indicators),
            "diet_advice": list(self.diet_advice),
            "lifestyle_advice": list(self.lifestyle_advice),
            "followup_cycle": self.followup_cycle,
            "red_flags": list(self.red_flags),
        }


def build_interventions(comparisons: list[IndicatorComparison]) -> list[TrendIntervention]:
    """根据本次 vs 上次的恶化/异常指标，智能生成结构化改善办法与应对方案。"""
    out: list[TrendIntervention] = []

    # 按系统对异常/恶化指标进行归类
    abnormal_map = {c.code: c for c in comparisons if c.curr_grade != 0 or c.is_real_change}

    # 1. 肝胆系统
    liver_codes = [c for c in ("ALT", "AST", "GGT", "TBIL", "ALB") if c in abnormal_map]
    if liver_codes:
        worsened = any(abnormal_map[c].worsened for c in liver_codes)
        targets = tuple(f"{abnormal_map[c].name_cn} ({abnormal_map[c].curr_value:g}{abnormal_map[c].unit})" for c in liver_codes)
        out.append(
            TrendIntervention(
                system="肝胆代谢防护",
                icon="🧪",
                level="重点关注" if worsened else "积极改善",
                target_indicators=targets,
                diet_advice=(
                    "严格戒酒，避免酒精对肝实质细胞的持续刺激与代谢负担",
                    "减少高脂、油炸食品及高果糖浆摄入，控制每日烹调油用量在 25g 以内",
                    "适量摄入优质蛋白质（如清蒸深海鱼、脱脂奶、豆腐）与富含抗氧化物质的深色蔬菜",
                ),
                lifestyle_advice=(
                    "保持规律作息，避免熬夜与过度劳累，保证夜间 7~8 小时高质量睡眠",
                    "审慎使用未经医生评估的保健品、偏方或潜在肝毒性非处方药物",
                    "坚持中等强度有氧运动以辅助改善脂肪蓄积",
                ),
                followup_cycle="建议 2~4 周或 1~2 个月内复查肝功能全套及肝胆胰脾超声",
                red_flags=(
                    "若出现皮肤或巩膜发黄、持续右上腹胀痛、明显乏力厌油，请及时前往消化内科就诊",
                ),
            )
        )

    # 2. 血糖与糖代谢
    glu_codes = [c for c in ("GLU", "HBA1C", "INS") if c in abnormal_map]
    if glu_codes:
        worsened = any(abnormal_map[c].worsened for c in glu_codes)
        targets = tuple(f"{abnormal_map[c].name_cn} ({abnormal_map[c].curr_value:g}{abnormal_map[c].unit})" for c in glu_codes)
        out.append(
            TrendIntervention(
                system="糖代谢与血糖调控",
                icon="🩸",
                level="重点关注" if worsened else "积极改善",
                target_indicators=targets,
                diet_advice=(
                    "控制每餐碳水化合物总量，采用燕麦、糙米、杂豆等低 GI 粗杂粮替代 1/3~1/2 精制米面",
                    "进餐顺序建议先喝清汤、吃蔬菜、再吃蛋白质，最后吃主食，以平稳餐后血糖波动",
                    "杜绝含糖饮料、糕点与高糖零食，两餐之间如有饥饿感可适量食用低糖黄瓜、番茄",
                ),
                lifestyle_advice=(
                    "坚持餐后 30 分钟进行 15~20 分钟轻中度活动（如快走、做轻家务），避免餐后立即久坐",
                    "每周累计完成 150 分钟以上规律有氧运动与 2 次抗阻力量锻炼",
                    "建议配置家用血糖仪，定期监测并记录空腹与三餐后 2 小时血糖水平",
                ),
                followup_cycle="建议 1~3 个月内复查糖化血红蛋白 (HbA1c) 与空腹静脉血糖",
                red_flags=(
                    "若出现多饮、多尿、多食伴体重快速下降，或视物模糊、手足麻木，请及时就诊内分泌科",
                ),
            )
        )

    # 3. 血脂与心血管代谢
    lipid_codes = [c for c in ("TG", "TC", "LDLC", "HDLC", "SBP", "DBP", "BMI") if c in abnormal_map]
    if lipid_codes:
        worsened = any(abnormal_map[c].worsened for c in lipid_codes)
        targets = tuple(f"{abnormal_map[c].name_cn} ({abnormal_map[c].curr_value:g}{abnormal_map[c].unit})" for c in lipid_codes)
        out.append(
            TrendIntervention(
                system="心血管与脂代谢管理",
                icon="🫀",
                level="重点关注" if worsened else "积极改善",
                target_indicators=targets,
                diet_advice=(
                    "践行低盐低脂的地中海饮食结构，每日食盐摄入控制在 5 克以内（约一平啤酒瓶盖）",
                    "限制动物内脏、肥肉、黄油、椰子油及含反式脂肪酸的加工起酥食品",
                    "增加富含 Omega-3 脂肪酸的深海鱼类（三文鱼、鲭鱼）与燕麦等可溶性膳食纤维",
                ),
                lifestyle_advice=(
                    "超重或腰围超标者建议以每月减重 1~2 公斤为目标循序渐进改善体成分",
                    "戒烟并远离二手烟，限制饮酒，避免剧烈情绪波动与剧烈温差刺激",
                    "每天早晨起床后及晚间就寝前各测量一次静息血压并做好日记",
                ),
                followup_cycle="建议 1~3 个月复查血脂四项、同型半胱氨酸及动态血压",
                red_flags=(
                    "若突发持续胸闷、心前区压榨性疼痛、呼吸困难或头晕肢麻，请立即前往医院急诊就诊",
                ),
            )
        )

    # 4. 肾功能与尿酸代谢
    renal_codes = [c for c in ("UA", "CREA", "UREA", "UACR") if c in abnormal_map]
    if renal_codes:
        worsened = any(abnormal_map[c].worsened for c in renal_codes)
        targets = tuple(f"{abnormal_map[c].name_cn} ({abnormal_map[c].curr_value:g}{abnormal_map[c].unit})" for c in renal_codes)
        out.append(
            TrendIntervention(
                system="肾脏机能与尿酸排泄",
                icon="💧",
                level="重点关注" if worsened else "积极改善",
                target_indicators=targets,
                diet_advice=(
                    "严格限制高嘌呤食物摄入（动物内脏、浓肉汤/火锅汤、海鲜贝类及啤酒）",
                    "保证充足饮水，每日饮水量保持在 2000~2500 mL，分次饮用以促进尿酸溶解排泄",
                    "多摄入新鲜碱性蔬菜，适量饮用苏打水或低脂奶制品",
                ),
                lifestyle_advice=(
                    "避免暴饮暴食、剧烈无氧运动脱水或关节局部受凉等急性痛风诱发因素",
                    "避免非必要使用肾毒性非甾体抗炎药，注意保护肾小球滤过功能",
                ),
                followup_cycle="建议 1~2 个月复查血尿酸、血肌酐、尿素氮与晨尿常规",
                red_flags=(
                    "若出现单侧大脚趾或踝关节急性红肿热痛剧烈发作，或眼睑下肢浮肿、泡沫尿，请及时就诊肾内科或风湿免疫科",
                ),
            )
        )

    # 5. 兜底通用健康维持方案
    if not out:
        out.append(
            TrendIntervention(
                system="整体健康维持与预防保健",
                icon="🌿",
                level="平稳维持",
                target_indicators=("各项监测指标总体处于平稳参考区间",),
                diet_advice=(
                    "保持食物多样化，荤素搭配，多食新鲜蔬果与全谷物粗杂粮",
                    "饮食清淡少油少盐，规律进餐，避免暴饮暴食",
                ),
                lifestyle_advice=(
                    "每周坚持 150 分钟以上规律运动，保持良好心肺耐力",
                    "保持心态平衡，劳逸结合，养成良好的睡眠生物钟",
                ),
                followup_cycle="建议每 6~12 个月进行一次常规健康体检与指标跟踪",
                red_flags=(
                    "日常如遇身体明显不适，请随时前往医疗机构进行咨询与检查",
                ),
            )
        )

    return out


# ---------------------------------------------------------------------------
# 综合报告
# ---------------------------------------------------------------------------
@dataclass
class TrendReport:
    comparisons: list[IndicatorComparison] = field(default_factory=list)
    series: list[IndicatorSeries] = field(default_factory=list)
    risk_trajectories: dict[str, RiskTrajectory] = field(default_factory=dict)
    change_attribution: ChangeAttribution | None = None
    interventions: list[TrendIntervention] = field(default_factory=list)
    rendered_text: str = ""

    def to_dict(self) -> dict:
        return {
            "comparisons": [c.to_dict() for c in self.comparisons],
            "series": [s.to_dict() for s in self.series],
            "risk_trajectories": {k: v.to_dict() for k, v in self.risk_trajectories.items()},
            "change_attribution": self.change_attribution.to_dict()
            if self.change_attribution else None,
            "interventions": [it.to_dict() for it in self.interventions],
            "rendered_text": self.rendered_text,
        }


def build_trend_report(
    engine: TrendEngine,
    cleaned: pd.DataFrame,
    demographics: pd.DataFrame | None = None,
    recent_n: int = 3,
    audit=None,
    pseudo_id: str | None = None,
    horizons: tuple[str, ...] = ("1y", "3y", "5y"),
    prev_attribution: RiskAttribution | None = None,
    curr_attribution: RiskAttribution | None = None,
    risk_change_top_n: int = 5,
) -> TrendReport:
    """
    组装完整趋势报告。audit + pseudo_id 同时提供时才拉取风险走势
    （规范 1.2：pseudo_id 是假名化标识，不在此函数内做任何还原）。
    prev_attribution / curr_attribution 同时提供时，额外产出风险变化归因
    （复用 attribution.explain_change，不重新实现 SHAP 差分）。
    """
    comparisons = engine.compare_latest(cleaned, demographics)
    series = engine.recent_series(cleaned, n=recent_n, demographics=demographics)

    trajectories: dict[str, RiskTrajectory] = {}
    if audit is not None and pseudo_id:
        for h in horizons:
            traj = risk_trajectory_from_audit(audit, pseudo_id, h)
            if traj.points:
                trajectories[h] = traj

    change_attr = None
    if prev_attribution is not None and curr_attribution is not None:
        change_attr = explain_change(prev_attribution, curr_attribution, top_n=risk_change_top_n)

    interventions = build_interventions(comparisons)

    report = TrendReport(
        comparisons=comparisons,
        series=series,
        risk_trajectories=trajectories,
        change_attribution=change_attr,
        interventions=interventions,
    )
    report.rendered_text = render_trend_text(report)
    return report


def render_trend_text(report: TrendReport) -> str:
    """模板渲染 + 合规断言。只叙述真实变化与风险走势，详细干预见下方专用交互卡片。"""
    lines: list[str] = ["【病情趋势时序报告】"]

    real = [c for c in report.comparisons if c.is_real_change]
    if real:
        lines.append("本次与上次相比，以下指标发生了超出检测波动范围的真实变化：")
        for c in real:
            lines.append(f"  · {c.phrase()}")
    elif report.comparisons:
        lines.append("本次与上次相比，各项指标变化均在个体正常波动范围内，整体保持平稳。")
    else:
        lines.append("暂无可供对比的历史记录，积累两次及以上体检数据后即可查看变化趋势。")

    for h, traj in sorted(report.risk_trajectories.items()):
        change = traj.latest_change
        if change is None:
            continue
        prev_p, curr_p = change
        arrow = "上升" if curr_p > prev_p else ("下降" if curr_p < prev_p else "持平")
        lines.append(
            f"{h} 风险走势：由 {prev_p:.1%} {arrow}至 {curr_p:.1%}"
            f"（当前分层「{traj.points[-1].risk_tier}」）。"
        )

    if report.change_attribution is not None:
        ca = report.change_attribution
        lines.append("本次风险变化的主要驱动因素：")
        for f in ca.factors[:5]:
            lines.append(f"  · {f.phrase()}")

    text = attach_disclaimer("\n".join(lines))
    assert_compliant(text, source="trend.render_trend_text")
    return text


__all__ = [
    "IndicatorComparison",
    "IndicatorSeries",
    "RiskTrajectory",
    "RiskTrajectoryPoint",
    "TrendEngine",
    "TrendIntervention",
    "TrendReport",
    "build_interventions",
    "build_trend_report",
    "render_trend_text",
    "risk_trajectory_from_audit",
]
