"""
验证与评估层单元测试（批次3）。

这里保护的是【数字本身可信】的底线。模型层的 bug 会让 AUC 掉下去，
一眼能看见；评估层的 bug 会让 AUC 涨上去，没人会去查 —— 后者才是
规范 5 反复强调的"实验室高分、线上拉胯"的真正来源。

因此本文件的重点不是"函数能跑"，而是：
    1. 指标算得对（与 sklearn / 朴素实现逐位对拍，含大量并列值）
    2. 该炸的必须炸（NaN 标签、单一类别、患者跨折、样本量不足）
    3. 门禁真的拦得住（低 AUC、未校准概率、缺外部集、分层反转）

运行方式（与 test_models.py 相同）：
    pytest tests/test_validation.py -q
    python tests/test_validation.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ---- pytest 兼容层：无 pytest 时提供最小 raises / importorskip ----
try:
    import pytest

    _HAVE_PYTEST = True
except ImportError:  # pragma: no cover
    _HAVE_PYTEST = False

    class _Skip(Exception):
        pass

    class pytest:  # type: ignore[no-redef]
        @staticmethod
        @contextmanager
        def raises(exc):
            try:
                yield
            except exc:
                return
            raise AssertionError(f"预期抛出 {exc.__name__} 但没有抛出")

        @staticmethod
        def importorskip(name):
            try:
                return __import__(name)
            except ImportError:
                raise _Skip(name)


from drp.data.constants import COL_INDEX_DATE, COL_PATIENT_ID  # noqa: E402
from drp.validation import (  # noqa: E402
    LeakageError,
    MetricsError,
    ReleaseBlocked,
    ValidationGate,
    apply_gate,
    assert_fold_integrity,
    assert_release_ready,
    average_precision,
    bootstrap_ci,
    brier_scores,
    calibration_table,
    concordance_index,
    cross_validate,
    evaluate_binary,
    evaluate_survival,
    expected_calibration_error,
    patient_stratified_kfold,
    risk_stratification_table,
    roc_auc,
    rolling_origin_folds,
    run_three_layer_validation,
    stratification_violations,
    threshold_at_alert_rate,
    threshold_at_sensitivity,
)

RNG = np.random.default_rng(20260815)


# ===========================================================================
# 测试夹具
# ===========================================================================
def _cohort(n_pat=1200, seed=1, shift=0.0, noise=0.0, start="2019-01-01"):
    """带患者复诊、时间跨度、可控信号强度的合成队列。"""
    r = np.random.default_rng(seed)
    rows = []
    for i in range(n_pat):
        pid = f"P{seed}_{i}"
        t0 = int(r.integers(0, 1300))
        for _ in range(int(r.integers(1, 4))):
            rows.append((pid, pd.Timestamp(start) + pd.Timedelta(days=t0 + int(r.integers(0, 150)))))
    co = (
        pd.DataFrame(rows, columns=[COL_PATIENT_ID, COL_INDEX_DATE])
        .sort_values(COL_INDEX_DATE)
        .reset_index(drop=True)
    )
    n = len(co)
    X = pd.DataFrame({f"f{j}": r.normal(0, 1, n) for j in range(6)})
    lin = 1.6 * X.f0 + 1.1 * X.f1 - 0.8 * X.f2 + 0.6 * X.f3 * X.f4 + shift
    if noise:
        lin = lin + r.normal(0, noise, n)
    co["y"] = (r.random(n) < 1 / (1 + np.exp(-(lin - 2.2)))).astype(float)
    return co, X


def _fit_predict_factory(damage: float = 0.0):
    """真实的 fit_predict 回调：GBDT + 保序回归校准（模仿 LGBMRiskModel 的行为）。"""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import train_test_split

    def fit_predict(Xtr, ytr, Xte):
        Xa, Xc, ya, yc = train_test_split(
            Xtr, ytr, test_size=0.2, random_state=0, stratify=ytr
        )
        m = HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.08, random_state=0
        ).fit(Xa, ya)
        iso = IsotonicRegression(out_of_bounds="clip").fit(m.predict_proba(Xc)[:, 1], yc)
        p = iso.predict(m.predict_proba(Xte)[:, 1])
        if damage:  # 人为破坏校准，用于验证门禁能拦住
            p = np.clip(p * (1 + damage), 1e-6, 1 - 1e-6)
        return p

    return fit_predict


# ===========================================================================
# 1. 排序指标：与参考实现逐位对拍
# ===========================================================================
def test_auc_matches_sklearn_with_heavy_ties():
    from sklearn.metrics import roc_auc_score

    for seed in range(6):
        r = np.random.default_rng(seed)
        n = int(r.integers(300, 1500))
        y = (r.random(n) < 0.08).astype(float)
        if y.sum() < 5:
            continue
        # 刻意四舍五入到 1 位小数，制造大量并列分数
        s = np.round(r.normal(y * 1.2, 1.0), 1)
        assert abs(roc_auc(y, s) - roc_auc_score(y, s)) < 1e-12


def test_average_precision_matches_sklearn():
    from sklearn.metrics import average_precision_score

    for seed in range(6):
        r = np.random.default_rng(100 + seed)
        n = 800
        y = (r.random(n) < 0.1).astype(float)
        s = np.round(r.normal(y * 1.0, 1.0), 1)
        assert abs(average_precision(y, s) - average_precision_score(y, s)) < 1e-12


def test_metrics_reject_bad_inputs():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    with pytest.raises(MetricsError):  # 预测含 NaN 必须当场炸，不能静默出 nan
        roc_auc(y, np.array([0.1, np.nan, 0.3, 0.4]))
    with pytest.raises(MetricsError):  # 标签含 NaN = 删失样本没剔干净
        roc_auc(np.array([0.0, np.nan, 1.0, 1.0]), np.array([0.1, 0.2, 0.3, 0.4]))
    with pytest.raises(MetricsError):  # 非 0/1 标签
        roc_auc(np.array([0.0, 2.0, 1.0, 1.0]), np.array([0.1, 0.2, 0.3, 0.4]))
    with pytest.raises(MetricsError):  # 长度不一致
        roc_auc(y, np.array([0.1, 0.2]))


def test_evaluate_binary_rejects_single_class():
    y = np.zeros(100)
    with pytest.raises(MetricsError):
        evaluate_binary(y, np.linspace(0, 1, 100), n_boot=0)


# ===========================================================================
# 2. 阈值：敏感度必须真的达标（规范"严控漏诊"）
# ===========================================================================
def test_threshold_at_sensitivity_actually_reaches_target():
    r = np.random.default_rng(3)
    y = (r.random(4000) < 0.06).astype(float)
    p = np.clip(r.random(4000) * 0.3 + y * 0.25, 0, 1)
    for target in (0.99, 0.95, 0.90, 0.80, 0.50):
        op = threshold_at_sensitivity(y, p, target)
        assert op.sensitivity >= target - 1e-9, (target, op.sensitivity)
        # 重算一遍确认阈值语义是 p >= threshold
        pred = p >= op.threshold
        assert abs(pred[y == 1].mean() - op.sensitivity) < 1e-12
        assert op.n_missed == int((y == 1).sum() - pred[y == 1].sum())


def test_threshold_at_sensitivity_picks_highest_qualifying_threshold():
    """同等敏感度下必须选最高阈值 = 最少报警。选低了就是白白多报警。"""
    y = np.array([1.0, 0, 0, 1, 0, 0, 1, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    op = threshold_at_sensitivity(y, p, 0.66)
    assert op.sensitivity >= 0.66
    assert abs(op.threshold - 0.6) < 1e-12
    assert op.alert_rate == 0.4


def test_threshold_at_alert_rate_respects_budget():
    r = np.random.default_rng(4)
    y = (r.random(3000) < 0.1).astype(float)
    p = np.clip(r.random(3000) * 0.4 + y * 0.3, 0, 1)
    op = threshold_at_alert_rate(y, p, 0.10)
    assert 0.09 <= op.alert_rate <= 0.115
    assert op.n_flagged == int((p >= op.threshold).sum())


def test_threshold_rejects_invalid_target():
    y = np.array([0.0, 1.0, 1.0, 0.0])
    p = np.array([0.1, 0.9, 0.8, 0.2])
    with pytest.raises(MetricsError):
        threshold_at_sensitivity(y, p, 1.5)
    with pytest.raises(MetricsError):
        threshold_at_alert_rate(y, p, 0.0)


# ===========================================================================
# 3. 校准：概率数值必须可信（规范 6 要直接展示给用户）
# ===========================================================================
def test_perfect_calibration_has_low_ece_and_unit_oe():
    r = np.random.default_rng(5)
    p = r.random(30000) * 0.5
    y = (r.random(30000) < p).astype(float)  # 构造完全校准的数据
    assert expected_calibration_error(y, p, n_bins=10) < 0.02
    m = evaluate_binary(y, p, n_boot=0)
    assert 0.95 < m.o_e_ratio < 1.05


def test_miscalibration_is_detected_with_direction():
    r = np.random.default_rng(6)
    p_true = r.random(20000) * 0.4
    y = (r.random(20000) < p_true).astype(float)
    over = np.clip(p_true * 1.8, 0, 1)  # 系统性高估风险
    m = evaluate_binary(y, over, n_boot=0)
    assert m.ece > 0.05
    assert m.o_e_ratio < 0.85  # 实测 < 预测 = 高估
    # 排序完全没变，AUC 一模一样 —— 这正是"AUC 高但概率错"的典型形态
    assert abs(roc_auc(y, over) - roc_auc(y, p_true)) < 1e-12


def test_brier_skill_negative_when_probabilities_useless():
    r = np.random.default_rng(7)
    y = (r.random(5000) < 0.05).astype(float)
    p = np.full(5000, 0.5)  # 无脑报 0.5：Brier 看着不大，BSS 必须为负
    brier, bss = brier_scores(y, p)
    assert brier < 0.3 and bss < 0


def test_calibration_table_quantile_bins_are_populated():
    r = np.random.default_rng(8)
    p = np.clip(r.beta(1.2, 20, 5000), 0, 1)  # 高度左偏，等宽分箱会空掉大半
    y = (r.random(5000) < p).astype(float)
    tbl = calibration_table(y, p, n_bins=10, strategy="quantile")
    assert len(tbl) >= 9
    assert (tbl["n"] > 100).all()


# ===========================================================================
# 4. 风险分层（规范 6 四级分层的验收依据）
# ===========================================================================
def test_stratification_monotonic_for_good_model_and_inverted_for_bad():
    r = np.random.default_rng(9)
    p = np.clip(r.beta(2, 8, 20000), 0, 1)
    y = (r.random(20000) < p).astype(float)

    good = risk_stratification_table(y, p)
    assert stratification_violations(good) == []
    assert good["obs_rate"].is_monotonic_increasing
    assert good["cum_capture"].iloc[0] == 1.0  # 全部层累计 = 100% 阳性
    assert good["lift"].iloc[-1] > good["lift"].iloc[0]

    bad = risk_stratification_table(y, 1.0 - p)  # 把风险方向反过来
    assert stratification_violations(bad), "分层反转必须被检出"


def test_stratification_with_fixed_cutpoints():
    """上线后必须用固定概率切点，不能每天按当日人群分位数漂移。"""
    y = np.array([0.0] * 90 + [1.0] * 10)
    p = np.concatenate([np.full(90, 0.02), np.full(10, 0.7)])
    tbl = risk_stratification_table(y, p, cutpoints=(0.05, 0.3, 0.6))
    assert tbl.loc[0, "n"] == 90 and tbl.loc[3, "n"] == 10
    assert tbl.loc[3, "obs_rate"] == 1.0


# ===========================================================================
# 5. 置信区间：患者级整簇 bootstrap
# ===========================================================================
def test_cluster_bootstrap_ci_is_wider_than_row_bootstrap():
    """
    同患者的多次体检不是独立观测。行级 bootstrap 会把重复样本当独立信息，
    把 CI 做窄 —— 这是"虚高"的另一种形态（虚高的是可信度）。
    """
    r = np.random.default_rng(11)
    n_pat = 200
    y_pat = (r.random(n_pat) < 0.2).astype(float)
    s_pat = r.normal(y_pat * 1.0, 1.0)
    reps = 6
    y = np.repeat(y_pat, reps)
    s = np.repeat(s_pat, reps)
    groups = np.repeat([f"P{i}" for i in range(n_pat)], reps)

    lo_row, hi_row = bootstrap_ci(y, s, roc_auc, n_boot=300, groups=None, seed=1)
    lo_cl, hi_cl = bootstrap_ci(y, s, roc_auc, n_boot=300, groups=groups, seed=1)
    assert (hi_cl - lo_cl) > 1.5 * (hi_row - lo_row)


# ===========================================================================
# 6. C-index（规范 5 生存模型指标）
# ===========================================================================
def _naive_cindex(t, e, r):
    c = comp = 0.0
    n = len(t)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if e[i] == 1 and (t[i] < t[j] or (t[i] == t[j] and e[j] == 0)):
                comp += 1
                c += 1.0 if r[i] > r[j] else (0.5 if r[i] == r[j] else 0.0)
    return c / comp


def test_concordance_index_matches_naive_implementation():
    for seed in range(5):
        r = np.random.default_rng(200 + seed)
        n = 250
        t = r.integers(1, 30, n).astype(float)  # 大量时间并列
        e = (r.random(n) < 0.4).astype(float)
        risk = np.round(r.normal(0, 1, n), 1)  # 大量风险并列
        assert abs(concordance_index(t, e, risk) - _naive_cindex(t, e, risk)) < 1e-12


def test_concordance_index_perfect_and_reversed():
    t = np.array([10.0, 20, 30, 40, 50])
    e = np.array([1.0, 1, 1, 1, 0])
    perfect = np.array([5.0, 4, 3, 2, 1])  # 风险越高活得越短
    assert concordance_index(t, e, perfect) == 1.0
    assert concordance_index(t, e, -perfect) == 0.0
    assert abs(concordance_index(t, e, np.zeros(5)) - 0.5) < 1e-12  # 全并列 = 0.5


def test_concordance_index_rejects_bad_inputs():
    t = np.array([1.0, 2.0, 3.0])
    with pytest.raises(MetricsError):
        concordance_index(t, np.array([0.0, 1.0, 2.0]), np.array([1.0, 2, 3]))
    with pytest.raises(MetricsError):
        concordance_index(np.array([0.0, 2.0, 3.0]), np.array([1.0, 1, 0]), np.array([1.0, 2, 3]))
    with pytest.raises(MetricsError):  # 全删失 -> 无可比对
        concordance_index(t, np.zeros(3), np.array([1.0, 2, 3]))


def test_evaluate_survival_reports_ci():
    r = np.random.default_rng(12)
    n = 800
    risk = r.normal(0, 1, n)
    t = np.clip(r.exponential(np.exp(-risk) * 500), 1, 2000)
    e = (r.random(n) < 0.6).astype(float)
    sm = evaluate_survival(t, e, risk, label="cox", n_boot=30)
    assert sm.c_index > 0.6
    assert sm.c_index_lo < sm.c_index < sm.c_index_hi
    assert "C-index" in sm.summary()


# ===========================================================================
# 7. 交叉验证：折必须是患者级的
# ===========================================================================
def test_kfold_is_patient_level_and_partitions_all_rows():
    co, _ = _cohort(n_pat=400, seed=21)
    folds = patient_stratified_kfold(co, n_splits=5, stratify_col="y", seed=1)
    assert len(folds) == 5
    assert_fold_integrity(co, folds, label_col="y", min_test_positives=1)
    for f in folds:
        tr = set(co.iloc[f.train_idx][COL_PATIENT_ID])
        te = set(co.iloc[f.test_idx][COL_PATIENT_ID])
        assert not (tr & te), "同一患者不得跨折"


def test_kfold_stratification_balances_positive_rate():
    co, _ = _cohort(n_pat=800, seed=22)
    rates = [
        co.iloc[f.test_idx]["y"].mean()
        for f in patient_stratified_kfold(co, n_splits=5, stratify_col="y", seed=2)
    ]
    assert max(rates) - min(rates) < 0.05, f"各折阳性率差异过大: {rates}"


def test_assert_fold_integrity_catches_uncovered_rows():
    co, _ = _cohort(n_pat=300, seed=23)
    folds = patient_stratified_kfold(co, n_splits=4, stratify_col="y", seed=3)
    folds = folds[:3]  # 人为丢掉一折 -> 有样本从未被验证
    with pytest.raises(LeakageError):
        assert_fold_integrity(co, folds, require_partition=True)


def test_rolling_origin_folds_never_train_on_the_future():
    co, _ = _cohort(n_pat=900, seed=24)
    folds = rolling_origin_folds(co, n_splits=4, gap_days=30, min_train_frac=0.5)
    assert len(folds) >= 2
    for f in folds:
        tr_max = pd.to_datetime(co.iloc[f.train_idx][COL_INDEX_DATE]).max()
        te_min = pd.to_datetime(co.iloc[f.test_idx][COL_INDEX_DATE]).min()
        assert te_min > tr_max, "滚动验证的测试集必须严格晚于训练集"
        assert (te_min - tr_max).days >= 1
        tr = set(co.iloc[f.train_idx][COL_PATIENT_ID])
        te = set(co.iloc[f.test_idx][COL_PATIENT_ID])
        assert not (tr & te)
    # 训练集随折数递增（扩展窗口）
    sizes = [f.train_idx.size for f in folds]
    assert sizes == sorted(sizes)


def test_cross_validate_rejects_censored_labels():
    co, X = _cohort(n_pat=300, seed=25)
    y = co["y"].to_numpy().copy()
    y[:10] = np.nan  # 删失样本没剔干净
    folds = patient_stratified_kfold(co, n_splits=3, stratify_col="y", seed=4)
    with pytest.raises(ValueError):
        cross_validate(co, X, y, _fit_predict_factory(), folds)


def test_cross_validate_oof_and_headline_guard():
    co, X = _cohort(n_pat=700, seed=26)
    folds = patient_stratified_kfold(co, n_splits=4, stratify_col="y", seed=5)
    rep = cross_validate(
        co, X, co["y"].to_numpy(), _fit_predict_factory(), folds, label="t", n_boot=30
    )
    assert len(rep.folds) == 4
    assert rep.oof is not None and rep.oof.n == len(co)  # OOF 覆盖全量
    assert rep.min_auc <= rep.mean_auc <= rep.max_auc
    assert rep.std_auc >= 0
    # 刻意的设计：K 折均值不允许当作对外口径
    with pytest.raises(ValueError):
        _ = rep.headline_auc
    assert "禁止作为对外精度口径" in rep.summary()


def test_cross_validate_rejects_scarce_fold_positives():
    co, X = _cohort(n_pat=200, seed=27)
    folds = patient_stratified_kfold(co, n_splits=5, stratify_col="y", seed=6)
    with pytest.raises(ValueError):
        cross_validate(co, X, co["y"].to_numpy(), _fit_predict_factory(), folds,
                       min_fold_positives=10_000)


# ===========================================================================
# 8. 三层协议与上线门禁（规范 5 + 规范 9）
# ===========================================================================
def _run(co, X, external=None, gate=None, **kw):
    return run_three_layer_validation(
        co, X, co["y"].to_numpy(), _fit_predict_factory(kw.pop("damage", 0.0)),
        external=external, gate=gate, model_id="t", horizon="3y",
        n_splits=3, n_boot=40, **kw,
    )


def test_three_layer_passes_with_strong_model():
    co, X = _cohort(n_pat=2500, seed=31)
    co_ex, X_ex = _cohort(n_pat=800, seed=32, shift=-0.2)
    rep = _run(co, X, external=(co_ex, X_ex, co_ex["y"].to_numpy()))
    assert rep.headline is not None and rep.headline_auc > 0.82
    assert not rep.blocked, [g.line() for g in rep.gates if not g.passed]
    assert rep.status in ("PASS", "CONDITIONAL")
    assert_release_ready(rep)  # 不抛异常
    s = rep.summary()
    assert "时间拆分" in s and "外部集" in s and "上线门禁" in s


def test_gate_blocks_weak_model():
    co, X = _cohort(n_pat=2500, seed=33, noise=3.5)  # 强噪声 -> 判别力不足
    co_ex, X_ex = _cohort(n_pat=800, seed=34, noise=3.5)
    rep = _run(co, X, external=(co_ex, X_ex, co_ex["y"].to_numpy()))
    assert rep.blocked and rep.status == "BLOCKED"
    assert any("AUC" in g.name for g in rep.failures())
    with pytest.raises(ReleaseBlocked):
        assert_release_ready(rep)


def test_gate_blocks_missing_external_dataset():
    co, X = _cohort(n_pat=2000, seed=35)
    rep = _run(co, X, external=None)
    assert rep.blocked
    assert any("外部" in g.name for g in rep.failures())


def test_external_waiver_is_recorded_not_silent():
    co, X = _cohort(n_pat=2000, seed=36)
    # 用宽松阈值隔离出"豁免"这一件事本身，避免被其它门禁项干扰
    lenient = ValidationGate(
        min_auc_roc=0.60, min_auc_ci_lower=0.50, min_pr_lift=1.2,
        min_specificity_at_target=0.0, max_ece=0.5, max_oe_deviation=0.9,
    )
    rep = _run(co, X, external=None, gate=lenient, allow_missing_external=True)
    assert not rep.blocked
    assert rep.external_waived is True
    assert rep.status == "CONDITIONAL"  # 豁免必须留痕，不能显示为完全通过
    assert rep.to_dict()["external_waived"] is True


def test_gate_blocks_broken_calibration():
    """排序不变、只把概率整体放大 —— AUC 完全不动，但概率数值全错。"""
    co, X = _cohort(n_pat=2500, seed=37)
    rep = _run(co, X, external=None, allow_missing_external=True, damage=1.4)
    names = [g.name for g in rep.failures()]
    assert any(n in ("校准误差 ECE", "实测/预测比 O:E") for n in names), names


def test_gate_config_is_recorded_for_audit():
    """门禁阈值必须随报告归档：事后要能查出当时用的是什么标准。"""
    co, X = _cohort(n_pat=2000, seed=38)
    gate = ValidationGate(min_auc_roc=0.99, require_external=False)
    rep = _run(co, X, external=None, gate=gate, allow_missing_external=True)
    assert rep.gate_config["min_auc_roc"] == 0.99
    assert rep.blocked
    g = next(g for g in rep.gates if g.name.startswith("AUC-ROC"))
    assert g.threshold == 0.99


def test_report_json_is_serializable_and_archivable(tmp_path=None):
    co, X = _cohort(n_pat=2000, seed=39)
    rep = _run(co, X, external=None, allow_missing_external=True)
    out = Path(tmp_path or tempfile.mkdtemp()) / "reports" / "val.json"
    rep.save_json(out)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["model_id"] == "t" and loaded["horizon"] == "3y"
    assert loaded["status"] in ("PASS", "CONDITIONAL", "BLOCKED")
    assert loaded["layers"]["time_split"]["metrics"]["auc_roc"] > 0.5
    assert len(loaded["gates"]) >= 8
    assert loaded["layers"]["cross_validation"]["cv"]["n_folds"] == 3


def test_apply_gate_is_reusable_for_review():
    """判定与执行分离：同一份结果可以用更严的标准复审，不必重跑模型。"""
    co, X = _cohort(n_pat=2000, seed=40)
    rep = _run(co, X, external=None, allow_missing_external=True)
    strict = apply_gate(rep, ValidationGate(min_auc_roc=0.999, require_external=False))
    assert any(not g.passed for g in strict)


def test_uncalibrated_scores_block_release():
    """把 raw margin 直接送进评估：校准类指标跳过，门禁必须拦。"""
    co, X = _cohort(n_pat=1500, seed=41)
    y = co["y"].to_numpy()
    m = evaluate_binary(y, np.linspace(-3, 4, len(y))[np.argsort(np.argsort(y + RNG.normal(0, 1, len(y))))],
                        label="raw", n_boot=0)
    assert m.calibrated_input is False
    assert np.isnan(m.ece) and np.isnan(m.brier)
    assert "跳过" in m.summary()


def test_summary_always_prints_accuracy_baseline():
    """规范"禁止只看准确率"的落地检查：准确率必须与无脑基线并排出现。"""
    co, X = _cohort(n_pat=800, seed=42)
    y = co["y"].to_numpy()
    p = np.clip(RNG.random(len(y)) * 0.2 + y * 0.3, 0, 1)
    s = evaluate_binary(y, p, n_boot=0).summary()
    assert "全判阴性也能拿到" in s


# ===========================================================================
# 无 pytest 环境的直跑入口
# ===========================================================================
if __name__ == "__main__":
    import logging
    import traceback

    logging.basicConfig(level=logging.ERROR)
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    n_ok = n_fail = n_skip = 0
    for name, fn in fns:
        try:
            kwargs = {}
            if "tmp_path" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
                kwargs["tmp_path"] = tempfile.mkdtemp()
            fn(**kwargs)
            print(f"PASS  {name}")
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            if not _HAVE_PYTEST and type(e).__name__ == "_Skip":
                print(f"SKIP  {name} (缺 {e})")
                n_skip += 1
            else:
                print(f"FAIL  {name}: {e}")
                traceback.print_exc()
                n_fail += 1
    print(f"\n{n_ok} passed, {n_fail} failed, {n_skip} skipped")
    sys.exit(1 if n_fail else 0)
