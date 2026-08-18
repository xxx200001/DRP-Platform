"""确定性灰度路由与带结局线上 A/B 方向性对照。"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..models.registry import ModelRegistry, RegistryError
from ..validation import average_precision, roc_auc
from .audit import AuditLogger


@dataclass(frozen=True)
class RoutingDecision:
    version: str
    arm: str
    bucket: float

    def to_dict(self) -> dict:
        return asdict(self)


class TrafficRouter:
    """患者级粘性路由；同一患者在相同盐和流量配置下始终进入同一臂。"""

    def __init__(self, registry: ModelRegistry, routing_salt: str):
        if not routing_salt:
            raise ValueError("灰度路由盐不能为空")
        self.registry = registry
        self.routing_salt = routing_salt.encode("utf-8")

    def bucket(self, patient_id: str) -> float:
        digest = hmac.new(
            self.routing_salt, str(patient_id).encode("utf-8"), hashlib.sha256
        ).digest()
        return int.from_bytes(digest[:8], "big") / float(1 << 64) * 100.0

    def decide(self, patient_id: str) -> RoutingDecision:
        active = self.registry.get_active()
        if active is None:
            raise RegistryError("注册表中没有 ACTIVE 模型，无法提供预测")
        b = self.bucket(patient_id)
        canary = self.registry.get_canary()
        if canary is not None and b < canary.traffic_pct:
            return RoutingDecision(canary.version, "canary", b)
        return RoutingDecision(active.version, "active", b)


@dataclass
class ABComparison:
    champion: str
    challenger: str
    horizon: str | None = None
    n_champion: int = 0
    n_challenger: int = 0
    champion_auc: float | None = None
    challenger_auc: float | None = None
    champion_auc_pr: float | None = None
    challenger_auc_pr: float | None = None
    verdict: str = "insufficient_data"

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        scope = f"，时程 {self.horizon}" if self.horizon else ""
        guard = "A/B 仅用于线上方向性监控，不替代时间拆分、K折和外部集三层验证门禁。"
        if self.verdict == "insufficient_data":
            return (
                f"线上 A/B 数据不足{scope}：champion={self.champion} n={self.n_champion}，"
                f"challenger={self.challenger} n={self.n_challenger}。{guard}"
            )
        delta = float(self.challenger_auc) - float(self.champion_auc)
        label = {
            "challenger_better": "challenger 方向性更优",
            "champion_better": "champion 方向性更优",
            "inconclusive": "差异不足以形成方向性结论",
        }[self.verdict]
        return (
            f"{label}{scope}：champion AUC={self.champion_auc:.4f} (n={self.n_champion})，"
            f"challenger AUC={self.challenger_auc:.4f} (n={self.n_challenger})，"
            f"Δ={delta:+.4f}。{guard}"
        )


def _safe_metrics(
    frame: pd.DataFrame, min_samples: int
) -> tuple[float | None, float | None]:
    if len(frame) < min_samples or frame["outcome_event"].nunique() < 2:
        return None, None
    y = frame["outcome_event"].astype(int).to_numpy()
    p = frame["probability"].astype(float).to_numpy()
    return float(roc_auc(y, p)), float(average_precision(y, p))


def build_ab_comparison(
    audit: AuditLogger,
    champion: str,
    challenger: str,
    days: list[str] | None = None,
    horizon: str | None = None,
    min_per_arm: int = 30,
    practical_auc_delta: float = 0.01,
) -> ABComparison:
    frames = [audit.load_day(d) for d in (days or [])]
    frames = [f for f in frames if not f.empty]
    result = ABComparison(champion=champion, challenger=challenger, horizon=horizon)
    if not frames:
        return result
    data = AuditLogger.dedup_latest(pd.concat(frames, ignore_index=True))
    required = {"model_version", "outcome_event", "probability"}
    if not required.issubset(data.columns):
        return result
    data = data[data["outcome_event"].notna()].copy()
    if horizon is not None and "horizon" in data:
        data = data[data["horizon"] == horizon]
    c = data[data["model_version"] == champion]
    g = data[data["model_version"] == challenger]
    result.n_champion, result.n_challenger = len(c), len(g)
    ca, cap = _safe_metrics(c, min_per_arm)
    ga, gap = _safe_metrics(g, min_per_arm)
    result.champion_auc, result.champion_auc_pr = ca, cap
    result.challenger_auc, result.challenger_auc_pr = ga, gap
    if len(c) < min_per_arm or len(g) < min_per_arm or ca is None or ga is None:
        return result
    delta = ga - ca
    if delta >= practical_auc_delta:
        result.verdict = "challenger_better"
    elif delta <= -practical_auc_delta:
        result.verdict = "champion_better"
    else:
        result.verdict = "inconclusive"
    return result
