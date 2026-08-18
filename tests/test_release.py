from __future__ import annotations

from pathlib import Path

from drp.models import ModelRegistry
from drp.serving import AuditLogger, FeedbackOrchestrator, FollowUpFeedback, PredictionRecord, TrafficRouter
from drp.validation import ValidationReport


def _report(version: str, horizon: str) -> ValidationReport:
    return ValidationReport(model_id=version, horizon=horizon)


def test_registry_promote_canary_and_rollback(tmp_path: Path):
    reg = ModelRegistry(tmp_path / "registry")
    for version in ("v1", "v2"):
        bank = tmp_path / "artifacts" / version
        bank.mkdir(parents=True)
        reg.register(version, bank, {"3y": _report(version, "3y")})
    assert reg.promote("v1").status == "ACTIVE"
    assert reg.set_canary("v2", 10).status == "CANARY"
    router = TrafficRouter(reg, "test-salt")
    assert router.decide("patient-1").version in {"v1", "v2"}
    assert reg.promote("v2").status == "ACTIVE"
    assert reg.rollback().version == "v1"


def test_followup_requires_consent():
    try:
        FollowUpFeedback("trace", True, 30, False)
    except ValueError as exc:
        assert "授权" in str(exc)
    else:
        raise AssertionError("未授权反馈必须被拒绝")


def test_cross_day_followup_stays_queryable_in_original_shard(tmp_path: Path):
    audit = AuditLogger(tmp_path / "audit", salt="test-salt")
    old_day = "2026-01-02"
    record = PredictionRecord(
        pseudo_id="pseudo", horizon="3y", model_version="v1",
        feature_hash="hash", features={"ALT": 1.0}, probability=0.1,
        risk_tier="低危",
    )
    # 测试专用：直接把初始预测放入历史分片。
    with audit._path_for(old_day).open("w", encoding="utf-8") as handle:
        handle.write(record.to_json() + "\n")
    feedback = FeedbackOrchestrator(audit)
    assert feedback.ingest_followup(FollowUpFeedback(record.trace_id, True, 90, True), old_day)
    latest = audit.find(record.trace_id, day=old_day)
    assert latest is not None and latest.outcome_event == 1
    assert len(list(audit.iter_records(old_day))) == 2
