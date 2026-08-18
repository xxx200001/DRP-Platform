"""
预测服务统一入口（规范 4.2 / 4.3 / 6 / 7）。

本模块存在的唯一理由：让"跳过归因、跳过日志、跳过合规检查"在物理上做不到。

规范 3.2 说"可解释性强制开启"，规范 4.2 说"每一次预测永久留存"。
如果这些是散落在各处、靠调用方自觉去调的函数，那么在赶工期的那个周五下午，
一定会有人为了让接口快 20ms 而跳过其中一步，并且没有人会发现。
所以平台只暴露 RiskPredictionService.predict() 这一个出口，它内部把
    对齐 -> 预测 -> 分层 -> 归因 -> 合规 -> 落日志
串成一条不可分割的路径，中间任何一步失败都不会有结果流出去。

【三个关键设计决策】

1. 风险分层切点是【固定的绝对概率】，来自验证集，不随线上人群漂移
   动态分位数切点（"今天预测值排前 5% 的算极高危"）会产生一个致命的产品问题：
   用户这次的所有指标一个没变，风险等级却从"中危"变成了"高危"，
   仅仅因为今天来做检查的人整体更健康。用户会认为平台在瞎报，
   而且你无法向他解释 —— 因为解释本身（"因为别人变健康了"）比错误更荒谬。
   切点必须在验证集上定一次，随模型一起持久化，重训才允许变。

2. 日志写失败 = 预测失败（strict_audit 默认 True）
   "结果照发、日志掉了"看似更可用，实则制造了一次永远查不清的预测。
   医疗场景里，无法追溯的输出比没有输出更危险。批量离线场景可以显式关掉。

3. 漂移检查是【批量语义】，不进单次预测路径
   单条样本算 PSI 毫无意义（一个样本的"分布"是什么？）。
   服务只把最近一次批量检查的漂移等级写进每条记录，
   让事故复盘时能看到"这次预测发生在漂移告警期间"。
   真正的检查由 check_drift() 定时批量跑。

【与大模型的边界】（规范 3.1）
本服务产出的所有文案都由模板生成，不经过大模型。大模型只允许在【上游】
做报告解析结构化、或在【下游】把本模块产出的结构化归因改写成更好读的话术；
无论哪种，改写后的文本都必须再过一次 compliance.assert_compliant()。
概率数值永远来自 LGBM/Cox-PH，绝不允许大模型生成 —— 这是规范 3.1 的红线。
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .attribution import AttributionEngine, RiskAttribution
from .audit import AuditLogger, PredictionRecord
from .compliance import ComplianceError, assert_compliant, attach_disclaimer, safe_fallback
from .drift import DriftMonitor, DriftReport

logger = logging.getLogger(__name__)

DEFAULT_TIER_NAMES: tuple[str, ...] = ("低危", "中危", "高危", "极高危")


# ---------------------------------------------------------------------------
# 风险分层（规范 6 四级分层）
# ---------------------------------------------------------------------------
@dataclass
class RiskTierScheme:
    """固定绝对概率切点。随模型一起持久化，重训才允许变（见模块说明第 1 条）。"""

    cutpoints: tuple[float, ...] = (0.05, 0.15, 0.40)
    names: tuple[str, ...] = DEFAULT_TIER_NAMES
    source: str = ""  # 切点从哪份验证结果定出来的，便于审计

    def __post_init__(self):
        self.cutpoints = tuple(float(c) for c in self.cutpoints)
        self.names = tuple(self.names)
        if len(self.names) != len(self.cutpoints) + 1:
            raise ValueError(
                f"层名数({len(self.names)})必须比切点数({len(self.cutpoints)})多 1"
            )
        if list(self.cutpoints) != sorted(self.cutpoints):
            raise ValueError(f"切点必须单调递增: {self.cutpoints}")

    @classmethod
    def from_probabilities(
        cls,
        probabilities,
        quantiles: tuple[float, ...] = (0.50, 0.80, 0.95),
        names: tuple[str, ...] = DEFAULT_TIER_NAMES,
        source: str = "",
    ) -> "RiskTierScheme":
        """
        从验证集（建议用 K 折 OOF：样本量最大、最稳）的预测概率定切点。
        定完就固化 —— 这是它与"线上动态分位数"的全部区别。
        """
        p = np.asarray(probabilities, dtype=float).ravel()
        if np.isnan(p).any():
            raise ValueError("定切点的概率数组含 NaN")
        return cls(
            cutpoints=tuple(float(x) for x in np.quantile(p, quantiles)),
            names=names,
            source=source or "validation_oof",
        )

    def assign(self, probability: float) -> str:
        """
        区间约定：左闭右开，切点值归入【更高】一层。

        即 cutpoints=(0.05, 0.15, 0.40) 对应
        低危[0,0.05) 中危[0.05,0.15) 高危[0.15,0.40) 极高危[0.40,1]。
        边界个体往高层归，是因为在慢病早筛里多提示一级的代价是一次复查，
        少提示一级的代价可能是一次漏诊（规范 5"敏感度严控漏诊"的同一取向）。
        本约定与 validation.metrics.risk_stratification_table 严格一致，
        两边不一致会导致"验证时算出来的分层发生率"与"线上实际分层"对不上。
        """
        return self.names[int(np.searchsorted(np.asarray(self.cutpoints), probability, side="right"))]

    def assign_many(self, probabilities) -> list[str]:
        idx = np.searchsorted(np.asarray(self.cutpoints), np.asarray(probabilities, dtype=float), side="right")
        return [self.names[i] for i in idx]

    def to_dict(self) -> dict:
        return {"cutpoints": list(self.cutpoints), "names": list(self.names), "source": self.source}

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "RiskTierScheme":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(cutpoints=tuple(d["cutpoints"]), names=tuple(d["names"]), source=d.get("source", ""))


# ---------------------------------------------------------------------------
@dataclass
class PredictionResult:
    """一次预测对外的完整产物。"""

    trace_id: str
    probability: float
    risk_tier: str
    horizon: str = ""
    attribution: RiskAttribution | None = None
    narrative: str = ""
    degraded: bool = False
    drift_level: str = ""
    model_version: str = ""

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "attribution"}
        d["attribution"] = self.attribution.to_dict() if self.attribution else None
        return d


@dataclass
class ServiceConfig:
    model_version: str = ""
    horizon: str = ""
    strict_audit: bool = True  # 日志写失败即预测失败
    top_n_factors: int = 10  # 规范 3.3 Top10
    explain: bool = True  # 规范 3.2：默认强开，关闭需显式声明
    attach_disclaimer_: bool = True  # 规范 7


class RiskPredictionService:
    """
    用法::

        svc = RiskPredictionService(
            model=model, tier_scheme=scheme, audit=audit_logger,
            display_names={"ALT": "谷丙转氨酶"},
            config=ServiceConfig(model_version="liver_v3", horizon="3y"),
        )
        results = svc.predict(X_new, patient_ids=["P001", "P002"])

    model 需实现 predict_risk(X) 与 shap_values(X)，并带 manifest ——
    即 models.LGBMRiskModel 的接口。HorizonBank 请按时程各建一个 service，
    因为归因、分层切点、漂移基线都是逐时程独立的。
    """

    def __init__(
        self,
        model,
        tier_scheme: RiskTierScheme,
        audit: AuditLogger | None = None,
        drift_monitor: DriftMonitor | None = None,
        display_names: dict[str, str] | None = None,
        config: ServiceConfig | None = None,
    ):
        self.model = model
        self.tier_scheme = tier_scheme
        self.audit = audit
        self.drift_monitor = drift_monitor
        self.config = config or ServiceConfig()
        self.engine = (
            AttributionEngine(model, display_names=display_names, top_n=self.config.top_n_factors)
            if self.config.explain
            else None
        )
        if not self.config.explain:
            logger.warning(
                "可解释性被显式关闭：规范 3.2 要求每次预测输出 SHAP 贡献度。"
                "此配置仅限压测/离线批量，禁止用于面向用户的线上服务。"
            )
        if audit is None and self.config.strict_audit:
            raise ValueError(
                "strict_audit=True 但未提供 AuditLogger。规范 4.2 要求每次预测永久留存；"
                "离线批量若确实不需要日志，请显式设 strict_audit=False。"
            )
        self._last_drift_level: str = ""

    # ------------------------------------------------------------------
    def check_drift(self, X_batch: pd.DataFrame) -> DriftReport | None:
        """批量漂移检查（规范 3.2）。定时任务调用，结果会带进后续预测记录。"""
        if self.drift_monitor is None:
            return None
        rep = self.drift_monitor.check(X_batch, model_version=self.config.model_version)
        self._last_drift_level = rep.level
        return rep

    # ------------------------------------------------------------------
    def _narrative(self, tier: str, attr: RiskAttribution | None) -> tuple[str, bool]:
        """
        模板化文案。不经大模型，因此天然合规；但仍然过一遍出口检测 ——
        模板里混入指标中文名（外部配置注入）时也可能带进违规词。
        """
        parts = [f"本次评估的风险分层为「{tier}」。"]
        if attr is not None and attr.factors:
            if attr.diffuse:
                parts.append("本次风险由多项指标共同构成，无单一主导因素。")
            raising = [f for f in attr.raising if not f.is_missing][:3]
            if raising:
                parts.append("推高风险的主要因素：" + "、".join(f.display for f in raising) + "。")
            lowering = [f for f in attr.lowering if not f.is_missing][:2]
            if lowering:
                parts.append("起到保护作用的因素：" + "、".join(f.display for f in lowering) + "。")
            missing = [f for f in attr.factors if f.is_missing][:3]
            if missing:
                parts.append(
                    "以下项目本次未检查，补齐后评估会更准确："
                    + "、".join(f.display for f in missing) + "。"
                )
        text = "".join(parts)
        try:
            assert_compliant(text, source="service_narrative")
        except ComplianceError as e:
            logger.error("模板文案触发合规拦截，已降级到兜底文案: %s", e)
            return safe_fallback(tier), True
        return (attach_disclaimer(text) if self.config.attach_disclaimer_ else text), False

    # ------------------------------------------------------------------
    def predict(
        self,
        X: pd.DataFrame,
        patient_ids: list[str] | pd.Series | None = None,
        raw_refs: list[str] | None = None,
        ocr_results: list[dict] | None = None,
        structured: list[dict] | None = None,
    ) -> list[PredictionResult]:
        """
        平台唯一预测出口。

        归因与概率来自【同一次】特征对齐：分两次调用会在列顺序不一致时
        产出"概率来自 A 列序、解释来自 B 列序"的错配 —— 模型不报错、
        概率看着正常、解释全是错的，属于最难查的一类线上事故。
        """
        if len(X) == 0:
            return []
        n = len(X)
        pids = list(patient_ids) if patient_ids is not None else [""] * n
        if len(pids) != n:
            raise ValueError(f"patient_ids 数量({len(pids)})与样本数({n})不符")

        probs = np.asarray(self.model.predict_risk(X), dtype=float).ravel()
        if np.isnan(probs).any():
            raise RuntimeError(
                f"模型输出含 {int(np.isnan(probs).sum())} 个 NaN 概率。"
                "这通常是特征表列错位或上游填充异常，绝不能把 NaN 当低风险发出去。"
            )
        tiers = self.tier_scheme.assign_many(probs)
        attrs = self.engine.explain(X, probabilities=probs) if self.engine else [None] * n

        meta = getattr(self.model, "meta_", {}) or {}
        results: list[PredictionResult] = []
        for i in range(n):
            narrative, degraded = self._narrative(tiers[i], attrs[i])
            rec = PredictionRecord(
                pseudo_id=self.audit.pseudonymize(pids[i]) if (self.audit and pids[i]) else "",
                horizon=self.config.horizon,
                raw_ref=(raw_refs[i] if raw_refs else ""),
                ocr_result=(ocr_results[i] if ocr_results else {}),
                structured=(structured[i] if structured else {}),
                features={c: X.iloc[i][c] for c in X.columns},
                model_version=self.config.model_version or str(meta.get("trained_at", "")),
                feature_hash=str(meta.get("feature_hash", "")),
                backend=str(meta.get("backend", "")),
                calibrated=bool(meta.get("calibrated", True)),
                probability=float(probs[i]),
                risk_tier=tiers[i],
                attribution=attrs[i].to_dict() if attrs[i] else {},
                drift_level=self._last_drift_level,
                degraded=degraded,
            )
            if self.audit is not None:
                try:
                    self.audit.log(rec)
                except Exception:
                    if self.config.strict_audit:
                        logger.exception("全链路日志写入失败，按规范 4.2 中断本次预测")
                        raise
                    logger.error("全链路日志写入失败（strict_audit=False，继续返回结果）")

            results.append(
                PredictionResult(
                    trace_id=rec.trace_id,
                    probability=float(probs[i]),
                    risk_tier=tiers[i],
                    horizon=self.config.horizon,
                    attribution=attrs[i],
                    narrative=narrative,
                    degraded=degraded,
                    drift_level=self._last_drift_level,
                    model_version=rec.model_version,
                )
            )
        logger.info(
            "完成 %d 条预测 [model=%s horizon=%s drift=%s 降级=%d]",
            n, self.config.model_version, self.config.horizon,
            self._last_drift_level or "未检查", sum(r.degraded for r in results),
        )
        return results
