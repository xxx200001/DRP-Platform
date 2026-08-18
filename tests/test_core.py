"""
核心行为单元测试。

这些测试保护的是【精度不可回退】的底线，必须全部进 CI 且不允许 skip。
任何一条挂掉都意味着线上精度会出问题，而不只是代码风格问题。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drp.data import LabDataCleaner, MeasureStatus, ReferenceRegistry, UnitValidator  # noqa: E402
from drp.data.units import ValidationCode  # noqa: E402
from drp.features.temporal import (  # noqa: E402
    PersistencePattern,
    TrendLabel,
    _ols_slope,
    _persistence_of,
    _trend_of,
)
from drp.validation import (  # noqa: E402
    LeakageError,
    as_of_filter,
    assert_no_future_records,
    assert_no_patient_overlap,
    patient_level_split,
    time_based_split,
)

CONFIGS = ROOT / "configs"


@pytest.fixture(scope="module")
def registry() -> ReferenceRegistry:
    return ReferenceRegistry.from_yaml(CONFIGS / "reference_intervals.yaml")


@pytest.fixture(scope="module")
def validator(registry) -> UnitValidator:
    return UnitValidator(registry)


# ===========================================================================
# 参考区间
# ===========================================================================
class TestReferenceIntervals:
    def test_sex_specific_interval_selected(self, registry):
        """男女参考区间不同的指标必须按性别取对应区间 —— 这是偏离度特征的根基。"""
        alt = registry.require("ALT")
        m = alt.match_interval("M", 40)
        f = alt.match_interval("F", 40)
        assert m.upper == 50 and f.upper == 40
        # 同一个值 45，在男性是正常，在女性是异常
        assert m.contains(45) and not f.contains(45)

    def test_age_specific_interval_narrowest_wins(self, registry):
        """同时命中多条区间时取年龄跨度最窄的（最具体的）。"""
        alp = registry.require("ALP")
        young = alp.match_interval("F", 30)
        old = alp.match_interval("F", 60)
        assert young.upper == 100
        assert old.upper == 135

    def test_unknown_sex_falls_back_to_union(self, registry):
        """性别未知时取男女并集，宁可放宽也不误判异常。"""
        hgb = registry.require("HGB")
        u = hgb.match_interval("U", 40)
        assert u.lower == 115 and u.upper == 175

    def test_rcv_differs_by_indicator(self, registry):
        """
        RCV 必须因指标而异。这条测试的意义：防止有人图省事把趋势判定
        改成固定百分比阈值 —— 那会同时毁掉高变异和低变异指标的趋势特征。
        """
        na_rcv = registry.require("NA").rcv
        crp_rcv = registry.require("CRP").rcv
        assert na_rcv < 0.05, "血钠 CVi 极低，RCV 应该很小"
        assert crp_rcv > 0.8, "CRP CVi 极高，RCV 应该很大"
        assert crp_rcv > na_rcv * 10

    def test_alias_index_no_conflict(self, registry):
        """别名词典必须一对一，冲突会导致 OCR 结构化把两个指标混淆。"""
        assert registry.resolve_alias("谷丙转氨酶") == "ALT"
        assert registry.resolve_alias("GPT") == "ALT"
        assert registry.resolve_alias("ＡＬＴ") == "ALT"  # 全角
        assert registry.resolve_alias(" alt ") == "ALT"
        assert registry.resolve_alias("不存在的指标") is None


# ===========================================================================
# 单位与生理极限校验（规范 4.1）
# ===========================================================================
class TestUnitValidation:
    def test_unit_conversion_creatinine(self, validator):
        """肌酐 mg/dL -> μmol/L。不换算会让模型读到差 88 倍的值。"""
        r = validator.validate("CREA", 1.1, "mg/dL")
        assert r.is_valid
        assert r.code is ValidationCode.UNIT_CONVERTED
        assert abs(r.value - 97.24) < 0.1

    def test_unit_conversion_platelet(self, validator):
        r = validator.validate("PLT", 245000, "/uL")
        assert r.is_valid and abs(r.value - 245.0) < 0.01

    def test_out_of_plausible_rejected(self, validator):
        """超生理极限必须硬拒绝，绝不允许进入特征层。"""
        r = validator.validate("ALT", 99999, "U/L")
        assert not r.is_valid
        assert r.value is None
        assert r.code is ValidationCode.OUT_OF_PLAUSIBLE

    def test_ambiguous_magnitude_is_rejected_not_guessed(self, validator):
        """
        多个量级候选都合理时必须拒绝，不能猜。

        血糖 108 无单位：按 mg/dL 是 6.0，按 10.8 或 1.08 也都在生理范围。
        猜错的代价（把正常人判成危急低血糖，或反过来）远大于丢一条数据。
        """
        r = validator.validate("GLU", 108, None)
        assert not r.is_valid

    def test_critical_value_flagged_but_valid(self, validator):
        """危急值是【有效数据】，只是要额外走告警通道，不能因此丢弃。"""
        r = validator.validate("K", 6.8, "mmol/L")
        assert r.is_valid
        assert r.is_critical
        assert r.value == 6.8

    def test_symbol_prefixed_value_parsed(self, validator):
        r = validator.validate("CRP", "<0.5", "mg/L")
        assert r.is_valid and r.value == 0.5

    def test_unknown_indicator_reported(self, validator):
        r = validator.validate("某某神秘指标", 3.14, None)
        assert not r.is_valid
        assert r.code is ValidationCode.UNKNOWN_INDICATOR

    def test_magnitude_fix_disabled_for_narrow_range(self, validator):
        """
        血钾这类生理范围极窄的指标禁用量级纠错。
        3.5-5.3 的范围内，任何量级纠错都是危险的猜测。
        """
        r = validator.validate("K", 42.0, None)
        assert not r.is_valid


# ===========================================================================
# 三态缺失值（规范 1.2）
# ===========================================================================
class TestThreeStateStatus:
    def test_status_distinguishes_normal_abnormal(self, registry):
        demo = pd.DataFrame(
            [{"patient_id": "P1", "sex": "F", "birth_date": pd.Timestamp("1980-01-01")}]
        )
        records = pd.DataFrame(
            [
                {"patient_id": "P1", "indicator_code": "ALT", "value": 20,
                 "unit": "U/L", "measured_at": pd.Timestamp("2023-01-01")},
                {"patient_id": "P1", "indicator_code": "ALT", "value": 60,
                 "unit": "U/L", "measured_at": pd.Timestamp("2023-06-01")},
            ]
        )
        clean, _ = LabDataCleaner(registry).clean(records, demographics=demo)
        assert clean.iloc[0]["status"] == MeasureStatus.NORMAL
        # 女性 ALT 上限 40，60 超标
        assert clean.iloc[1]["status"] == MeasureStatus.ABNORMAL

    def test_no_mean_imputation_anywhere(self, registry):
        """
        回归测试：确保清洗层不会偷偷填充缺失值。
        某个指标完全没有记录时，输出里就应该没有它，而不是一行均值。
        """
        demo = pd.DataFrame([{"patient_id": "P1", "sex": "M", "birth_date": pd.Timestamp("1980-01-01")}])
        records = pd.DataFrame(
            [{"patient_id": "P1", "indicator_code": "ALT", "value": 25,
              "unit": "U/L", "measured_at": pd.Timestamp("2023-01-01")}]
        )
        clean, _ = LabDataCleaner(registry).clean(records, demographics=demo)
        assert set(clean["indicator_code"]) == {"ALT"}
        assert len(clean) == 1


# ===========================================================================
# 时序特征（规范 2.3）
# ===========================================================================
class TestTemporalFeatures:
    def test_slope_positive_for_rising_series(self):
        t = np.array([np.datetime64("2020-01-01") + np.timedelta64(30 * i, "D") for i in range(5)])
        v = np.array([20.0, 25.0, 30.0, 35.0, 40.0])
        slope, r2 = _ols_slope(t, v, log_transform=False)
        assert slope > 4.0  # 约每月 +5
        assert r2 > 0.99

    def test_log_transform_makes_growth_comparable(self):
        """
        对数化后，20→40 和 200→400 应该得到相近斜率。
        这正是右偏指标必须 log 化的理由：临床上两者都是"翻倍"，同等严重。
        """
        t = np.array([np.datetime64("2020-01-01") + np.timedelta64(30 * i, "D") for i in range(3)])
        s_low, _ = _ols_slope(t, np.array([20.0, 28.0, 40.0]), log_transform=True)
        s_high, _ = _ols_slope(t, np.array([200.0, 280.0, 400.0]), log_transform=True)
        assert abs(s_low - s_high) < 0.01

    def test_trend_uses_rcv_not_fixed_threshold(self, registry):
        """
        同样 10% 的变化：血钠算真实上升，CRP 算噪声平稳。
        这是本平台趋势判定正确性的核心测试。
        """
        t = np.array([np.datetime64("2020-01-01"), np.datetime64("2020-07-01")])
        na = registry.require("NA")
        crp = registry.require("CRP")

        assert _trend_of(np.array([140.0, 154.0]), t, na) is TrendLabel.RISING
        assert _trend_of(np.array([5.0, 5.5]), t, crp) is TrendLabel.STABLE

    def test_persistence_transient_vs_persistent(self):
        """一过性 vs 持续性必须分开（规范 2.4）—— 两者临床含义完全不同。"""
        assert _persistence_of(np.array([False, False, False])) is PersistencePattern.NEVER_ABNORMAL
        assert _persistence_of(np.array([False, True, True])) is PersistencePattern.PERSISTENT
        assert _persistence_of(np.array([False, True, False])) is PersistencePattern.TRANSIENT
        assert _persistence_of(np.array([True, False, True, False])) is PersistencePattern.RECURRENT


# ===========================================================================
# 防泄露（规范 5）—— 最重要的一组测试
# ===========================================================================
class TestLeakagePrevention:
    @staticmethod
    def _make():
        cohort = pd.DataFrame(
            {
                "patient_id": ["P1", "P2"],
                "index_date": [pd.Timestamp("2022-01-01"), pd.Timestamp("2022-06-01")],
            }
        )
        records = pd.DataFrame(
            {
                "patient_id": ["P1", "P1", "P2", "P2"],
                "indicator_code": ["ALT"] * 4,
                "value": [20.0, 30.0, 25.0, 35.0],
                "measured_at": [
                    pd.Timestamp("2021-06-01"),  # P1 索引前，保留
                    pd.Timestamp("2022-03-01"),  # P1 索引后，必须剔除
                    pd.Timestamp("2021-01-01"),  # P2 索引前，保留
                    pd.Timestamp("2022-08-01"),  # P2 索引后，必须剔除
                ],
            }
        )
        return cohort, records

    def test_as_of_filter_removes_future_records(self):
        cohort, records = self._make()
        out = as_of_filter(records, cohort)
        assert len(out) == 2
        assert set(out["value"]) == {20.0, 25.0}
        # 辅助列必须清理干净，否则会带着标签期信息流进特征层
        assert "index_date" not in out.columns

    def test_blanking_period_removes_pre_index_window(self):
        """
        空白期必须真的生效。这是防"确诊前密集检查"泄露的唯一手段，
        少了它，模型会学成"最近查得勤=要出事"，在无症状早筛场景直接失效。
        """
        cohort = pd.DataFrame(
            {"patient_id": ["P1"], "index_date": [pd.Timestamp("2022-01-01")]}
        )
        records = pd.DataFrame(
            {
                "patient_id": ["P1", "P1"],
                "indicator_code": ["ALT", "ALT"],
                "value": [20.0, 99.0],
                "measured_at": [pd.Timestamp("2021-06-01"), pd.Timestamp("2021-12-20")],
            }
        )
        out = as_of_filter(records, cohort, blanking_days=30)
        assert len(out) == 1
        assert out.iloc[0]["value"] == 20.0

    def test_assert_no_future_records_raises(self):
        cohort, records = self._make()
        with pytest.raises(LeakageError, match="未来数据泄露"):
            assert_no_future_records(records, cohort)

    def test_patient_split_never_splits_a_patient(self):
        cohort = pd.DataFrame(
            {
                "patient_id": [f"P{i//3}" for i in range(300)],  # 每人 3 条样本
                "index_date": pd.date_range("2020-01-01", periods=300, freq="D"),
                "label": ([1] * 30 + [0] * 270),
            }
        )
        sp = patient_level_split(cohort, test_size=0.3, stratify_col="label", seed=7)
        assert_no_patient_overlap(cohort, sp.train_idx, sp.test_idx)

    def test_random_split_would_leak(self):
        """
        反面测试：证明朴素随机切分确实会泄露。
        保留这条是为了让任何想"图省事用 train_test_split"的人看到后果。
        """
        cohort = pd.DataFrame(
            {
                "patient_id": [f"P{i//3}" for i in range(300)],
                "index_date": pd.date_range("2020-01-01", periods=300, freq="D"),
            }
        )
        rng = np.random.default_rng(0)
        idx = rng.permutation(300)
        with pytest.raises(LeakageError, match="患者级泄露"):
            assert_no_patient_overlap(cohort, idx[:210], idx[210:])

    def test_time_split_moves_boundary_patients_to_train(self):
        """跨界患者必须整体归训练集，同时满足时间约束和患者约束。"""
        cohort = pd.DataFrame(
            {
                "patient_id": ["P1", "P1", "P2", "P2"],
                "index_date": [
                    pd.Timestamp("2020-01-01"), pd.Timestamp("2023-01-01"),
                    pd.Timestamp("2020-02-01"), pd.Timestamp("2020-03-01"),
                ],
            }
        )
        sp = time_based_split(cohort, cutoff="2021-01-01")
        assert_no_patient_overlap(cohort, sp.train_idx, sp.test_idx)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
