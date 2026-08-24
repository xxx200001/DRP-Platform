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
    # V3.3：逐指标的"数值化"明细句（值 + 超限倍数 + 较上次趋势），
    # 与聚合统计 —— 前端"最需要关注的 N 个问题"从这里取真实数据组织文案，
    # 不再使用模板套话。
    details: tuple[str, ...] = ()
    stats: dict = field(default_factory=dict)

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
            "details": list(self.details),
            "stats": dict(self.stats),
        }


# --- V3.3 数值化明细的构件 -------------------------------------------------
#: 系统归组（干预建议口径）。ALP/DBIL 归肝胆——真实肝功能单常见项，
#: 此前缺失会导致"ALP 103↑"这类异常在干预卡上凭空消失。
_IV_SYSTEMS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("肝胆代谢防护", "🧪", ("ALT", "AST", "GGT", "ALP", "TBIL", "DBIL", "ALB")),
    ("糖代谢与血糖调控", "🩸", ("GLU", "HBA1C", "INS")),
    ("心血管与脂代谢管理", "🫀", ("TG", "TC", "LDLC", "HDLC", "SBP", "DBP", "BMI")),
    ("肾脏机能与尿酸排泄", "💧", ("UA", "CREA", "UREA", "UACR")),
)


def _over_text(value: float, ref_low, ref_high, grade: int) -> str:
    """把偏离量说成人话：超上限 1.8 倍 / 低于下限 25%。算不出就返回空。"""
    try:
        if grade > 0 and ref_high not in (None, 0):
            k = value / float(ref_high)
            if k > 1.005:
                txt = f"{k:.1f}"
                return "略超上限" if txt == "1.0" else f"达上限的 {txt} 倍"
        if grade < 0 and ref_low not in (None, 0):
            pct = (float(ref_low) - value) / abs(float(ref_low)) * 100
            if pct > 0.5:
                return f"低于下限约 {pct:.0f}%"
    except (TypeError, ZeroDivisionError, ValueError):
        pass
    return ""


def _entry_detail(e: dict, comp: IndicatorComparison | None) -> str:
    """单指标明细句：名称 值单位 · 偏离程度 · 较上次趋势。全部来自真实数据。"""
    bits = [f"{e['name_cn']} {e['value']:g}{e.get('unit') or ''}"]
    over = _over_text(e["value"], e.get("ref_low"), e.get("ref_high"), e.get("grade", 0))
    if over:
        bits.append(over)
    if comp is not None:
        if comp.is_real_change:
            pct = f"{comp.delta_pct:+.0%}" if comp.delta_pct is not None else ""
            bits.append(f"较上次{comp.direction}{pct}" + ("·加重" if comp.worsened else ""))
        else:
            bits.append("较上次平稳")
    else:
        bits.append("首次记录")
    return " · ".join(bits)


def build_interventions(
    comparisons: list[IndicatorComparison],
    snapshot: list[dict] | None = None,
) -> list[TrendIntervention]:
    """
    根据【最新一次各指标真实取值】+【本次 vs 上次真实变化】动态生成干预方案。

    V3.3 修复的两个"假动态"问题：
      1. 旧版只看 comparisons（要求同指标 ≥2 次观测）——患者首份报告里的
         ALT 112↑ 会被完全忽略，干预卡退化成兜底套话。现以 snapshot
         （每指标最新值+分级+参考界）为主判据，comparisons 只补充趋势信息。
      2. 建议条目与随访周期不再是整段固定文案：逐条按实际异常的指标拼装，
         并把真实数值/超限倍数写进句子里，不同化验单必然得到不同输出。
    snapshot 为空时退回旧行为（由 comparisons 推断），保持向后兼容。
    """
    comp_map = {c.code: c for c in comparisons}

    # 统一成 {code: entry} 的"最新异常"视图
    entries: dict[str, dict] = {}
    if snapshot:
        for e in snapshot:
            if e.get("grade") and e.get("value") is not None:
                entries[e["code"]] = dict(e)
    else:  # 兼容旧调用：从 comparisons 还原
        for c in comparisons:
            if c.curr_grade != 0 or c.is_real_change:
                entries[c.code] = {
                    "code": c.code, "name_cn": c.name_cn, "unit": c.unit,
                    "value": c.curr_value, "grade": c.curr_grade,
                    "ref_low": None, "ref_high": None,
                }

    out: list[TrendIntervention] = []

    for system, icon, codes in _IV_SYSTEMS:
        hit = [entries[c] for c in codes if c in entries]
        if not hit:
            continue
        hit.sort(key=lambda e: -abs(e.get("grade", 0)))
        comp_of = {e["code"]: comp_map.get(e["code"]) for e in hit}
        worsened_n = sum(1 for e in hit
                         if comp_of[e["code"]] is not None and comp_of[e["code"]].worsened)
        max_grade = max(abs(e.get("grade", 0)) for e in hit)
        level = "重点关注" if (worsened_n or max_grade >= 3) else "积极改善"
        targets = tuple(f"{e['name_cn']} ({e['value']:g}{e.get('unit') or ''})" for e in hit)
        details = tuple(_entry_detail(e, comp_of[e["code"]]) for e in hit)
        worst = hit[0]
        stats = {
            "n_abnormal": len(hit),
            "n_worsened": int(worsened_n),
            "max_grade": int(max_grade),
            "worst": {
                "name_cn": worst["name_cn"], "value": worst["value"],
                "unit": worst.get("unit") or "",
                "over": _over_text(worst["value"], worst.get("ref_low"),
                                   worst.get("ref_high"), worst.get("grade", 0)),
            },
        }
        has = lambda *cs: any(c in entries for c in cs)  # noqa: E731
        v = lambda c: entries[c]["value"]  # noqa: E731

        diet: list[str] = []
        life: list[str] = []
        flags: list[str] = []
        cycle = ""

        if system == "肝胆代谢防护":
            if has("GGT"):
                diet.append(
                    f"γ-谷氨酰转移酶 {v('GGT'):g} 对酒精与油腻负担最敏感——严格戒酒"
                    "（含啤酒、红酒与含酒精饮料），给肝细胞减负")
            else:
                diet.append("严格戒酒，避免酒精对肝实质细胞的持续刺激与代谢负担")
            if has("ALT", "AST"):
                worst_tx = entries.get("ALT") or entries.get("AST")
                diet.append(
                    f"{worst_tx['name_cn']}已达 {worst_tx['value']:g}{worst_tx.get('unit') or ''}，"
                    "减少油炸食品与高果糖浆（奶茶/含糖饮料）摄入，每日烹调油控制在 25g 以内")
            if has("ALP", "TBIL", "DBIL"):
                diet.append("碱性磷酸酶/胆红素相关指标偏离时饮食宜清淡规律，避免暴食油腻，减轻胆汁排泄负担")
            diet.append("适量优质蛋白（清蒸鱼、脱脂奶、豆腐）与深色蔬菜，为肝细胞修复提供底物")
            life.append("保持规律作息，23 点前入睡、保证 7~8 小时睡眠，避免熬夜加重肝脏代谢负担")
            life.append("审慎对待未经医生评估的保健品与偏方，避免额外肝毒性负担")
            if has("ALT", "AST", "GGT"):
                life.append("坚持每周 ≥150 分钟中等强度有氧运动，帮助减少肝内脂肪蓄积")
            cycle = (f"建议 {'2~4 周' if level == '重点关注' else '1~2 个月'}内复查肝功能全套"
                     f"（重点跟踪 {worst['name_cn']}）及肝胆胰脾超声")
            flags.append("若出现皮肤或巩膜发黄、尿色深如浓茶、持续右上腹胀痛或明显乏力厌油，请及时就诊消化内科")

        elif system == "糖代谢与血糖调控":
            if has("GLU"):
                diet.append(
                    f"本次空腹血糖 {v('GLU'):g} mmol/L——控制每餐碳水总量，"
                    "以燕麦、糙米、杂豆等低 GI 粗杂粮替代 1/3~1/2 精制米面")
            if has("HBA1C"):
                diet.append(
                    f"糖化血红蛋白 {v('HBA1C'):g}% 反映近 3 个月平均血糖——"
                    "杜绝含糖饮料与高糖零食，比单日控糖更重要的是长期稳定")
            diet.append("进餐顺序：先清汤和蔬菜、再蛋白质、最后主食，可平稳餐后血糖波动")
            life.append("餐后 30 分钟进行 15~20 分钟轻中度活动（快走、轻家务），避免餐后立即久坐")
            life.append("每周累计 ≥150 分钟规律有氧运动 + 2 次抗阻力量锻炼，提升肌肉对葡萄糖的摄取")
            life.append("可配置家用血糖仪，记录空腹与餐后 2 小时血糖，复诊时供医生参考")
            cycle = f"建议 {'1 个月' if level == '重点关注' else '1~3 个月'}内复查空腹静脉血糖与糖化血红蛋白 (HbA1c)"
            flags.append("若出现多饮、多尿、多食伴体重快速下降，或视物模糊、手足麻木，请及时就诊内分泌科")

        elif system == "心血管与脂代谢管理":
            if has("TG"):
                diet.append(
                    f"甘油三酯 {v('TG'):g} mmol/L 对糖和酒精最敏感——严格限制甜食、"
                    "含糖饮料与酒精，它们在肝脏会直接转化为甘油三酯")
            if has("LDLC", "TC"):
                e2 = entries.get("LDLC") or entries.get("TC")
                diet.append(
                    f"{e2['name_cn']} {e2['value']:g} mmol/L——限制动物内脏、肥肉、黄油"
                    "及含反式脂肪的加工起酥食品，换用橄榄油/茶籽油")
            diet.append("每周 2~3 次深海鱼（三文鱼、鲭鱼）补充 Omega-3，主食中加入燕麦等可溶性膳食纤维")
            if has("SBP", "DBP"):
                e3 = entries.get("SBP") or entries.get("DBP")
                diet.append(f"{e3['name_cn']}达 {e3['value']:g} mmHg——每日食盐控制在 5 克以内（约一平啤酒瓶盖）")
                life.append("每天早晚各测一次静息血压并记录，就诊时带上血压日记")
            if has("HDLC") and entries["HDLC"].get("grade", 0) < 0:
                life.append(
                    f"高密度脂蛋白 {v('HDLC'):g} mmol/L 偏低——规律有氧运动是提升"
                    "「好胆固醇」最有效的非药物手段，每周 ≥150 分钟")
            if has("BMI"):
                life.append(
                    f"BMI {v('BMI'):g}——以每月减重 1~2 公斤为节奏循序渐进，"
                    "优先减少腰腹脂肪（配合抗阻训练防止肌肉流失）")
            life.append("戒烟并远离二手烟，避免剧烈情绪波动与骤冷骤热刺激")
            cycle = f"建议 {'1 个月' if level == '重点关注' else '1~3 个月'}内复查血脂四项（重点跟踪 {worst['name_cn']}）"
            if has("SBP", "DBP"):
                cycle += "，并做动态血压监测"
            flags.append("若突发持续胸闷、心前区压榨性疼痛、呼吸困难或一侧肢体麻木无力，请立即拨打急救电话就近就诊")

        elif system == "肾脏机能与尿酸排泄":
            if has("UA"):
                diet.append(
                    f"血尿酸 {v('UA'):g} μmol/L——严格限制高嘌呤食物"
                    "（动物内脏、浓肉汤/火锅汤底、贝类海鲜及啤酒）")
                life.append("避免暴饮暴食、剧烈无氧运动后脱水、关节受凉等急性痛风诱发因素")
            diet.append("每日饮水 2000~2500 mL 并分次均匀摄入，促进代谢废物经肾排出")
            if has("CREA", "UREA"):
                e4 = entries.get("CREA") or entries.get("UREA")
                diet.append(f"{e4['name_cn']} {e4['value']:g}{e4.get('unit') or ''}——蛋白质摄入以适量优质蛋白为主，避免长期高蛋白饮食加重肾小球滤过负担")
            life.append("慎重对待具有潜在肾毒性的止痛类产品，使用前先咨询医生")
            cycle = f"建议 {'1 个月' if level == '重点关注' else '1~2 个月'}内复查肾功能（{ '、'.join(x['name_cn'] for x in hit[:3]) }）与晨尿常规"
            flags.append("若出现关节急性红肿热痛、眼睑或下肢浮肿、尿中泡沫经久不散，请及时就诊肾内科")

        out.append(
            TrendIntervention(
                system=system, icon=icon, level=level,
                target_indicators=targets,
                diet_advice=tuple(diet[:4]),
                lifestyle_advice=tuple(life[:4]),
                followup_cycle=cycle,
                red_flags=tuple(flags),
                details=details,
                stats=stats,
            )
        )

    # 排序：重点关注在前；同级先看"是否在恶化"，再看最大分级、异常项数 ——
    # 与 llm_advisor 的系统排序保持同一口径。
    out.sort(key=lambda iv: (0 if iv.level == "重点关注" else 1,
                             -iv.stats.get("n_worsened", 0),
                             -iv.stats.get("max_grade", 0),
                             -iv.stats.get("n_abnormal", 0)))

    # 兜底：确实全部平稳时的通用维持方案（唯一允许的"无个体参数"卡片）
    if not out:
        n_ok = len(snapshot) if snapshot else len(comparisons)
        out.append(
            TrendIntervention(
                system="整体健康维持与预防保健",
                icon="🌿",
                level="平稳维持",
                target_indicators=(f"本次 {n_ok} 项监测指标均处于参考区间内",),
                diet_advice=(
                    "保持食物多样化，荤素搭配，多食新鲜蔬果与全谷物粗杂粮",
                    "饮食清淡少油少盐，规律进餐，避免暴饮暴食",
                ),
                lifestyle_advice=(
                    "每周坚持 150 分钟以上规律运动，保持良好心肺耐力",
                    "保持心态平衡，劳逸结合，养成良好的睡眠生物钟",
                ),
                followup_cycle="建议每 6~12 个月进行一次常规健康体检与指标跟踪",
                red_flags=("日常如遇身体明显不适，请随时前往医疗机构进行咨询与检查",),
                details=(f"{n_ok} 项指标全部在参考区间内",),
                stats={"n_abnormal": 0, "n_worsened": 0, "max_grade": 0},
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
    # V3.3：每个指标【最新一次】取值 + 分级 + 参考界（含只有一次观测的指标）。
    # 干预建议与 AI 分析都从这里取"当前到底哪几项异常、偏离多少"，
    # 不再依赖需要 ≥2 次观测的 comparisons —— 首份报告也能得到针对性输出。
    latest_snapshot: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "comparisons": [c.to_dict() for c in self.comparisons],
            "series": [s.to_dict() for s in self.series],
            "risk_trajectories": {k: v.to_dict() for k, v in self.risk_trajectories.items()},
            "change_attribution": self.change_attribution.to_dict()
            if self.change_attribution else None,
            "interventions": [it.to_dict() for it in self.interventions],
            "rendered_text": self.rendered_text,
            "latest_snapshot": list(self.latest_snapshot),
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

    # V3.3：最新快照（含仅一次观测的指标）。参考界按该患者性别年龄匹配，
    # 供干预建议与 AI 分析计算"超上限几倍"这类真实数值化表述。
    sex, age = extract_demographics(cleaned, demographics)
    latest_snapshot: list[dict] = []
    for s in series:
        if not s.points:
            continue
        at, value, grade = s.points[-1]
        meta = engine.registry.get(s.code)
        iv = meta.match_interval(sex, age) if meta is not None else None
        latest_snapshot.append({
            "code": s.code, "name_cn": s.name_cn, "unit": s.unit,
            "value": float(value), "grade": int(grade), "at": str(at)[:10],
            "ref_low": iv.lower if iv is not None else None,
            "ref_high": iv.upper if iv is not None else None,
            "n_points": len(s.points),
        })
    latest_snapshot.sort(key=lambda e: -abs(e["grade"]))

    trajectories: dict[str, RiskTrajectory] = {}
    if audit is not None and pseudo_id:
        for h in horizons:
            traj = risk_trajectory_from_audit(audit, pseudo_id, h)
            if traj.points:
                trajectories[h] = traj

    change_attr = None
    if prev_attribution is not None and curr_attribution is not None:
        change_attr = explain_change(prev_attribution, curr_attribution, top_n=risk_change_top_n)

    interventions = build_interventions(comparisons, snapshot=latest_snapshot)

    report = TrendReport(
        comparisons=comparisons,
        series=series,
        risk_trajectories=trajectories,
        change_attribution=change_attr,
        interventions=interventions,
        latest_snapshot=latest_snapshot,
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
