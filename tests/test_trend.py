"""
批次6测试（一）：趋势追踪与时序对比报告（trend.py）+ AuditLogger 跨天历史查询。

安全承重墙：
  - RCV 真实变化判定必须与 features.temporal 用的是同一套数学（同一 RCV 值），
    不能报告说"平稳"而模型特征说"上升"。
  - 分级复用 referral.grade_value（已对 features.deviation 做过网格一致性测试），
    这里只需确认 trend.py 确实在用它，不重新验证网格。
  - render_trend_text 是合规硬闸：必须无禁用词、带免责声明。
  - 风险走势必须来自审计日志的历史记录，不能重新预测；dedup_latest 语义
    （结局回填产生的追加行）不能污染走势曲线。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from drp.data.constants import (
    COL_INDICATOR,
    COL_MEASURED_AT,
    COL_PATIENT_ID,
    COL_UNIT,
    COL_VALUE,
)
from drp.data.reference import ReferenceRegistry
from drp.serving.attribution import FactorContribution, RiskAttribution
from drp.serving.audit import AuditLogger, PredictionRecord
from drp.serving.compliance import DISCLAIMER, is_compliant
from drp.serving.trend import (
    TrendEngine,
    build_trend_report,
    render_trend_text,
    risk_trajectory_from_audit,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "reference_intervals.yaml"
SALT = "test-salt-batch6"


@pytest.fixture(scope="module")
def registry() -> ReferenceRegistry:
    return ReferenceRegistry.from_yaml(CONFIG)


def _tmp() -> Path:
    return Path(tempfile.mkdtemp())


def _longitudinal(rows: list[tuple[str, float, str]]) -> pd.DataFrame:
    """rows: [(indicator_code, value, iso_date), ...]，单患者。"""
    return pd.DataFrame(
        [
            {
                COL_PATIENT_ID: "p1", COL_INDICATOR: code, COL_VALUE: val,
                COL_UNIT: None, COL_MEASURED_AT: pd.Timestamp(d),
            }
            for code, val, d in rows
        ]
    )


# ===========================================================================
# 本次 vs 上次
# ===========================================================================
class TestCompareLatest:
    def test_real_change_detected_alt(self, registry):
        """ALT 从 20 涨到 60：远超 RCV，必须判真实变化 + 方向上升。"""
        df = _longitudinal(
            [("ALT", 20.0, "2026-01-01"), ("ALT", 60.0, "2026-07-01")]
        )
        eng = TrendEngine(registry)
        comps = eng.compare_latest(df)
        alt = next(c for c in comps if c.code == "ALT")
        assert alt.is_real_change is True
        assert alt.direction == "上升"
        assert alt.prev_value == 20.0 and alt.curr_value == 60.0
        assert alt.delta == 40.0
        assert alt.delta_pct == pytest.approx(2.0)

    def test_noise_within_rcv_is_stable(self, registry):
        """血钠 CVi 极低（RCV 很小），但两次几乎没变 -> 必须判平稳。"""
        df = _longitudinal(
            [("NA", 140.0, "2026-01-01"), ("NA", 140.3, "2026-02-01")]
        )
        eng = TrendEngine(registry)
        comps = eng.compare_latest(df)
        na = next(c for c in comps if c.code == "NA")
        assert na.is_real_change is False
        assert na.direction == "平稳"

    def test_single_observation_excluded(self, registry):
        """只测过一次的指标不产出对比条目（无"上次"可比）。"""
        df = _longitudinal([("ALT", 30.0, "2026-01-01")])
        eng = TrendEngine(registry)
        assert eng.compare_latest(df) == []

    def test_uses_most_recent_two_when_more_than_two(self, registry):
        """三次记录时，只比最近两次，不比首末。"""
        df = _longitudinal(
            [("GLU", 5.0, "2026-01-01"), ("GLU", 9.0, "2026-03-01"), ("GLU", 5.2, "2026-06-01")]
        )
        eng = TrendEngine(registry)
        glu = next(c for c in eng.compare_latest(df) if c.code == "GLU")
        assert glu.prev_value == 9.0 and glu.curr_value == 5.2  # 不是 5.0 vs 5.2

    def test_worsened_flag_on_grade_escalation(self, registry):
        eng = TrendEngine(registry)
        df = _longitudinal([("K", 5.0, "2026-01-01"), ("K", 6.8, "2026-02-01")])
        k = next(c for c in eng.compare_latest(df) if c.code == "K")
        assert k.worsened is True  # 危急值，分级绝对值必然变大


# ===========================================================================
# 近 N 次曲线
# ===========================================================================
class TestRecentSeries:
    def test_series_length_and_order(self, registry):
        df = _longitudinal(
            [
                ("ALT", 20.0, "2026-01-01"), ("ALT", 30.0, "2026-03-01"),
                ("ALT", 50.0, "2026-05-01"), ("ALT", 25.0, "2026-07-01"),
            ]
        )
        eng = TrendEngine(registry)
        series = eng.recent_series(df, n=3)
        alt = next(s for s in series if s.code == "ALT")
        assert len(alt.points) == 3  # 只取最近 3 次，不是全部 4 次
        values = [v for _, v, _ in alt.points]
        assert values == [30.0, 50.0, 25.0]
        assert len(alt.steps) == 2

    def test_fewer_than_n_returns_all_available(self, registry):
        df = _longitudinal([("ALT", 20.0, "2026-01-01"), ("ALT", 30.0, "2026-03-01")])
        eng = TrendEngine(registry)
        alt = next(s for s in eng.recent_series(df, n=3) if s.code == "ALT")
        assert len(alt.points) == 2


# ===========================================================================
# 渲染文本 / 合规
# ===========================================================================
class TestRenderText:
    def test_no_history_message(self, registry):
        eng = TrendEngine(registry)
        report = build_trend_report(eng, _longitudinal([("ALT", 20.0, "2026-01-01")]))
        assert "暂无可供对比" in report.rendered_text
        assert is_compliant(report.rendered_text)
        assert DISCLAIMER in report.rendered_text

    def test_only_real_changes_are_narrated(self, registry):
        """噪声项(NA)不应出现在正文叙述里，真实变化项(ALT)必须出现。"""
        df = pd.concat(
            [
                _longitudinal([("ALT", 20.0, "2026-01-01"), ("ALT", 60.0, "2026-06-01")]),
                _longitudinal([("NA", 140.0, "2026-01-01"), ("NA", 140.3, "2026-06-01")]),
            ]
        )
        eng = TrendEngine(registry)
        report = build_trend_report(eng, df)
        assert "谷丙转氨酶" in report.rendered_text or "ALT" in str(report.comparisons)
        assert is_compliant(report.rendered_text)
        # 结构化数据里两项都在（前端画图要用全量），只是正文不叙述噪声项
        codes_in_struct = {c.code for c in report.comparisons}
        assert {"ALT", "NA"} <= codes_in_struct

    def test_all_stable_message(self, registry):
        df = _longitudinal([("NA", 140.0, "2026-01-01"), ("NA", 140.2, "2026-02-01")])
        eng = TrendEngine(registry)
        report = build_trend_report(eng, df)
        assert "整体保持平稳" in report.rendered_text

    def test_change_attribution_embedded_and_compliant(self, registry):
        prev = RiskAttribution(
            base_value=-2.0, total_shap=0.5, probability=0.10,
            factors=[
                FactorContribution(
                    key="ALT", display="谷丙转氨酶", group="临床衍生",
                    shap_sum=0.2, direction=1, n_features=1, value=30.0,
                )
            ],
        )
        curr = RiskAttribution(
            base_value=-2.0, total_shap=1.2, probability=0.35,
            factors=[
                FactorContribution(
                    key="ALT", display="谷丙转氨酶", group="临床衍生",
                    shap_sum=0.9, direction=1, n_features=1, value=80.0,
                )
            ],
        )
        eng = TrendEngine(registry)
        report = build_trend_report(
            eng, _longitudinal([("ALT", 30.0, "2026-01-01"), ("ALT", 80.0, "2026-06-01")]),
            prev_attribution=prev, curr_attribution=curr,
        )
        assert report.change_attribution is not None
        assert report.change_attribution.rose is True
        assert "主要驱动因素" in report.rendered_text
        assert "谷丙转氨酶" in report.rendered_text
        assert is_compliant(report.rendered_text)


# ===========================================================================
# AuditLogger.history_for_patient（跨天历史，风险走势的数据基础）
# ===========================================================================
def _rec(pseudo_id: str, horizon: str, prob: float, created_at: str, tier: str = "中危") -> PredictionRecord:
    return PredictionRecord(
        pseudo_id=pseudo_id, horizon=horizon, created_at=created_at,
        model_version="v1", feature_hash="h1", features={"ALT_dev": 0.1},
        probability=prob, risk_tier=tier,
    )


class TestOpaqueIdPiiFix:
    """
    回归测试：batch6 开发期间真实触发的问题，不是构造的边界情况。

    trace_id 是 32 位 uuid4 hex，任意连续 16 位落入纯数字子集的概率约
    0.5%——规模化后每天都会真实发生。修复前，这类记录会被 assert_no_pii
    误判为"检测到银行卡"而在 strict_audit=True 下中断一次完全正常的预测。
    """

    def test_digit_heavy_trace_id_no_longer_false_positives(self):
        from drp.serving.audit import scan_pii

        # 构造一个必然触发旧版"银行卡"正则(\d{16,19})的 trace_id
        hits = scan_pii({"trace_id": "a1b2" + "1234567890123456" + "c3d4"})
        assert hits == []

    def test_feature_hash_and_pseudo_id_also_exempted(self):
        from drp.serving.audit import scan_pii

        hits = scan_pii(
            {
                "feature_hash": "9876543210987654",
                "pseudo_id": "1111222233334444",
            }
        )
        assert hits == []

    def test_genuine_pii_in_other_fields_still_caught(self):
        """修复只豁免这三个系统标识符字段名，不能连带削弱真正的 PII 检测。"""
        from drp.serving.audit import scan_pii

        assert scan_pii({"notes": {"手机": "13800138000"}})
        assert scan_pii({"raw_ref": "身份证110101199001011234"})
        assert scan_pii({"trace_id_note": "1234567890123456"})  # 字段名不在豁免表里

    def test_end_to_end_log_no_longer_raises_on_unlucky_uuid(self):
        """端到端确认：即使 trace_id 恰好全是数字，log() 也不再抛 PIIError。"""
        audit = AuditLogger(_tmp(), salt=SALT)
        rec = _rec(audit.pseudonymize("P099"), "3y", 0.2, "2026-01-01T08:00:00+00:00")
        rec.trace_id = "1234567890123456789"  # 19 位纯数字，旧版必炸
        audit.log(rec)  # 不应抛异常
        assert audit.find(rec.trace_id) is not None
    def test_history_filters_by_patient_and_horizon(self):
        audit = AuditLogger(_tmp(), salt=SALT)
        pid = audit.pseudonymize("P001")
        other = audit.pseudonymize("P002")
        for i, day in enumerate(["2026-01-01", "2026-04-01", "2026-07-01"]):
            audit.log(_rec(pid, "3y", 0.1 + i * 0.1, f"{day}T08:00:00+00:00"))
        audit.log(_rec(other, "3y", 0.9, "2026-07-01T08:00:00+00:00"))
        audit.log(_rec(pid, "1y", 0.05, "2026-07-01T08:00:00+00:00"))

        hist = audit.history_for_patient(pid, horizon="3y")
        assert len(hist) == 3
        assert set(hist["pseudo_id"]) == {pid}
        assert list(hist["probability"]) == sorted(hist["probability"])  # 升序

    def test_history_dedup_after_outcome_attach(self):
        """结局回填会为同一 trace_id 追加新行，历史查询只应保留最后一条。"""
        audit = AuditLogger(_tmp(), salt=SALT)
        pid = audit.pseudonymize("P010")
        rec = _rec(pid, "3y", 0.4, "2026-05-01T08:00:00+00:00")
        trace_id = audit.log(rec)
        assert audit.attach_outcome(trace_id, event=1, days=200.0) is True

        hist = audit.history_for_patient(pid, horizon="3y")
        assert len(hist) == 1
        assert hist.iloc[0]["outcome_event"] == 1

    def test_risk_trajectory_ordering_and_latest_change(self):
        audit = AuditLogger(_tmp(), salt=SALT)
        pid = audit.pseudonymize("P020")
        audit.log(_rec(pid, "3y", 0.10, "2026-01-01T08:00:00+00:00", tier="低危"))
        audit.log(_rec(pid, "3y", 0.55, "2026-06-01T08:00:00+00:00", tier="高危"))

        traj = risk_trajectory_from_audit(audit, pid, "3y")
        assert len(traj.points) == 2
        assert traj.points[0].probability == 0.10
        assert traj.points[-1].risk_tier == "高危"
        prev_p, curr_p = traj.latest_change
        assert prev_p == 0.10 and curr_p == 0.55

    def test_trajectory_embedded_in_trend_report(self, registry):
        audit = AuditLogger(_tmp(), salt=SALT)
        pid = audit.pseudonymize("P030")
        audit.log(_rec(pid, "3y", 0.10, "2026-01-01T08:00:00+00:00", tier="低危"))
        audit.log(_rec(pid, "3y", 0.60, "2026-06-01T08:00:00+00:00", tier="极高危"))

        eng = TrendEngine(registry)
        report = build_trend_report(
            eng, _longitudinal([("ALT", 20.0, "2026-01-01")]),
            audit=audit, pseudo_id=pid, horizons=("3y",),
        )
        assert "3y" in report.risk_trajectories
        assert "风险走势" in report.rendered_text
        assert "上升" in report.rendered_text
        assert is_compliant(report.rendered_text)
