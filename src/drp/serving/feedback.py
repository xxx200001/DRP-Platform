"""授权随访回流、错误样本复盘队列与重训触发辅助。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .audit import AuditLogger


@dataclass(frozen=True)
class FollowUpFeedback:
    trace_id: str
    event_occurred: bool
    days_since_prediction: float
    consented: bool

    def __post_init__(self):
        if not self.trace_id.strip():
            raise ValueError("trace_id 不能为空")
        if not self.consented:
            raise ValueError("用户未授权保存随访反馈，已拒绝回流")
        if not pd.notna(self.days_since_prediction) or float(self.days_since_prediction) < 0:
            raise ValueError("距预测天数必须是非负数")


@dataclass
class ReviewCase:
    trace_id: str
    category: str
    risk_tier: str
    probability: float
    outcome_event: int
    outcome_days: float | None
    model_version: str
    horizon: str
    created_at: str
    priority_rank: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReviewQueue:
    cases: list[ReviewCase] = field(default_factory=list)
    n_total_labeled: int = 0
    n_false_negative: int = 0
    n_false_positive: int = 0
    n_confirmed_positive: int = 0

    def to_dict(self) -> dict:
        return {
            "cases": [c.to_dict() for c in self.cases],
            "n_total_labeled": self.n_total_labeled,
            "n_false_negative": self.n_false_negative,
            "n_false_positive": self.n_false_positive,
            "n_confirmed_positive": self.n_confirmed_positive,
        }

    def summary(self) -> str:
        return (
            f"已回流 {self.n_total_labeled} 条带结局样本；"
            f"优先复盘漏诊 {self.n_false_negative} 条、过度预警 {self.n_false_positive} 条，"
            f"并抽取判对高风险 {self.n_confirmed_positive} 条用于对照。"
        )


class FeedbackOrchestrator:
    def __init__(self, audit: AuditLogger):
        self.audit = audit

    def ingest_followup(self, feedback: FollowUpFeedback, day: str | None = None) -> bool:
        return self.audit.attach_outcome(
            feedback.trace_id,
            event=int(feedback.event_occurred),
            days=float(feedback.days_since_prediction),
            day=day,
        )

    def build_review_queue(
        self, days: list[str] | None = None, limit: int = 200
    ) -> ReviewQueue:
        selected_days = days or [datetime.now(timezone.utc).date().isoformat()]
        frames = [self.audit.load_day(d) for d in selected_days]
        frames = [f for f in frames if not f.empty]
        if not frames:
            return ReviewQueue()
        data = AuditLogger.dedup_latest(pd.concat(frames, ignore_index=True))
        if "outcome_event" not in data:
            return ReviewQueue()
        labeled = data[data["outcome_event"].notna()].copy()
        cases: list[tuple[int, float, ReviewCase]] = []
        counts = {"false_negative": 0, "false_positive": 0, "confirmed_positive": 0}
        for _, row in labeled.iterrows():
            event = int(row["outcome_event"])
            tier = str(row.get("risk_tier", ""))
            probability = float(row.get("probability", 0.0))
            if event == 1 and tier in {"低危", "中危"}:
                category, order = "false_negative", 0
            elif event == 0 and tier in {"高危", "极高危"}:
                category, order = "false_positive", 1
            elif event == 1 and tier in {"高危", "极高危"}:
                category, order = "confirmed_positive", 2
            else:
                continue
            counts[category] += 1
            case = ReviewCase(
                trace_id=str(row.get("trace_id", "")), category=category,
                risk_tier=tier, probability=probability, outcome_event=event,
                outcome_days=(float(row["outcome_days"]) if pd.notna(row.get("outcome_days")) else None),
                model_version=str(row.get("model_version", "")),
                horizon=str(row.get("horizon", "")), created_at=str(row.get("created_at", "")),
            )
            severity = probability if category == "false_positive" else -probability
            cases.append((order, severity, case))
        cases.sort(key=lambda x: (x[0], x[1], x[2].created_at))
        selected = [x[2] for x in cases[:limit]]
        for i, case in enumerate(selected):
            case.priority_rank = i
        return ReviewQueue(
            cases=selected, n_total_labeled=len(labeled),
            n_false_negative=counts["false_negative"],
            n_false_positive=counts["false_positive"],
            n_confirmed_positive=counts["confirmed_positive"],
        )


class RetrainJob:
    """只负责把已授权标签导出并触发调用方训练函数，不绕过验证/注册流程。"""

    def __init__(self, audit: AuditLogger, min_labeled: int = 300):
        if min_labeled < 1:
            raise ValueError("min_labeled 必须为正整数")
        self.audit = audit
        self.min_labeled = int(min_labeled)

    def training_frame(self, days: list[str] | None = None) -> pd.DataFrame:
        return self.audit.to_training_frame(days)

    def ready(self, days: list[str] | None = None) -> bool:
        return len(self.training_frame(days)) >= self.min_labeled

    def export(self, path: str | Path, days: list[str] | None = None) -> Path:
        frame = self.training_frame(days)
        if len(frame) < self.min_labeled:
            raise RuntimeError(
                f"带结局回流样本仅 {len(frame)} 条，低于重训阈值 {self.min_labeled}"
            )
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False)
        return target
