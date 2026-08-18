"""
模型层单元测试（批次1）。

与 test_core.py 一样，这里保护的是【精度不可回退】的底线：
标签删失规则、focal 梯度正确性、训练/推理一致性、时程单调性、
持久化 roundtrip。任何一条挂掉都意味着线上会出静默的精度事故。

说明：本文件兼容两种运行方式 ——
    pytest tests/test_models.py -q          # CI 标准方式
    python tests/test_models.py             # 无 pytest 的环境直跑
Cox-PH / focal-loss-on-LightGBM 用例在缺 lifelines / lightgbm 的
环境自动 SKIP（CI 镜像必须装齐，本地开发机可以缺）。
"""

from __future__ import annotations

import sys
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


from drp.data.constants import COL_INDEX_DATE, COL_PATIENT_ID, FEATURE_GROUP_DEVIATION  # noqa: E402
from drp.features.base import FeatureManifest, FeatureSpec  # noqa: E402
from drp.models import (  # noqa: E402
    COL_EVENT,
    COL_TIME_TO_EVENT,
    FocalBinary,
    HorizonBank,
    LGBMConfig,
    LGBMRiskModel,
    build_horizon_label,
    compute_scale_pos_weight,
    undersample_majority,
    usable_mask,
)

RNG = np.random.default_rng(20260815)


# ===========================================================================
# 1. 结局标签（labels.py 三条铁律）
# ===========================================================================
def _cohort(rows):
    df = pd.DataFrame(rows, columns=[COL_EVENT, COL_TIME_TO_EVENT])
    df[COL_PATIENT_ID] = [f"P{i}" for i in range(len(df))]
    return df


def test_label_rules():
    co = _cohort([
        (1, 200),    # 铁律1: H内发生 -> 1
        (1, 400),    # 铁律2: 发生但晚于H(365) -> 0
        (0, 400),    # 铁律2: 随访满H未发生 -> 0
        (0, 200),    # 铁律3: 删失 -> NaN
    ])
    y, st = build_horizon_label(co, 365)
    assert y.tolist()[:3] == [1.0, 0.0, 0.0]
    assert np.isnan(y.iloc[3])
    assert (st.n_pos, st.n_neg, st.n_censored) == (1, 2, 1)
    assert usable_mask(y).sum() == 3


def test_label_horizon_independent_censoring():
    """同一删失样本：1年时程是合法阴性，5年时程必须剔除（docstring 错误B）。"""
    co = _cohort([(0, 800)])
    y1, _ = build_horizon_label(co, 365)
    y5, _ = build_horizon_label(co, 1825)
    assert y1.iloc[0] == 0.0
    assert np.isnan(y5.iloc[0])


def test_label_rejects_nonpositive_time():
    with pytest.raises(ValueError):
        build_horizon_label(_cohort([(1, 0)]), 365)


def test_label_rejects_bad_event():
    with pytest.raises(ValueError):
        build_horizon_label(_cohort([(2, 100)]), 365)


# ===========================================================================
# 2. 不均衡处理（imbalance.py）
# ===========================================================================
def test_scale_pos_weight_and_cap():
    y = np.array([1] * 10 + [0] * 90, dtype=float)
    assert abs(compute_scale_pos_weight(y) - 9.0) < 1e-9
    y2 = np.array([1] * 2 + [0] * 998, dtype=float)
    assert compute_scale_pos_weight(y2, cap=100.0) == 100.0


def test_undersample_keeps_all_positives():
    y = np.array([1] * 20 + [0] * 500, dtype=float)
    keep = undersample_majority(y, max_neg_pos_ratio=5.0, seed=1)
    yk = y[keep]
    assert (yk == 1).sum() == 20
    assert (yk == 0).sum() == 100
    # 幂等：比例已达标时原样返回
    keep2 = undersample_majority(yk, max_neg_pos_ratio=5.0, seed=1)
    assert len(keep2) == len(yk)


def test_focal_gradient_matches_finite_difference():
    """解析梯度 vs 损失的中心差分 —— focal 实现正确性的硬校验。"""
    focal = FocalBinary(alpha=0.3, gamma=2.0)
    z = RNG.normal(0, 2, size=300)
    y = (RNG.random(300) < 0.3).astype(float)
    eps = 1e-5
    fd = (focal.loss(z + eps, y) - focal.loss(z - eps, y)) / (2 * eps)
    g = focal.grad(z, y)
    rel = np.abs(g - fd) / (np.abs(fd) + 1e-8)
    assert float(rel.max()) < 1e-3, f"focal 梯度与数值梯度不符, max rel err={rel.max():.2e}"
    h = focal.hess(z, y)
    assert (h > 0).all(), "Hessian 必须为正，否则 LightGBM 牛顿步不稳定"


def test_label_nan_rejected_by_weight_calc():
    with pytest.raises(ValueError):
        compute_scale_pos_weight(np.array([1.0, 0.0, np.nan]))


# ===========================================================================
# 3. LGBMRiskModel（sklearn 回退后端全流程；lightgbm 专属路径在装库的 CI 上跑）
# ===========================================================================
def _make_clf_data(n=1600, seed=7):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"f{i}": rng.normal(0, 1, n) for i in range(6)})
    logit = 1.6 * X["f0"] - 1.2 * X["f1"] + 0.8 * X["f2"] - 1.5
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    # 医疗常态：带缺失（三态信号由特征层处理，这里只验证 NaN 不炸）
    for c in ("f3", "f4"):
        X.loc[rng.random(n) < 0.12, c] = np.nan
    specs = [
        FeatureSpec(name=f"f{i}", group=FEATURE_GROUP_DEVIATION, dtype="numeric",
                    monotone=(1 if i == 0 else 0))
        for i in range(6)
    ]
    mf = FeatureManifest()
    mf.extend(specs)
    return X, pd.Series(y), mf


def _fast_cfg(**kw) -> LGBMConfig:
    base = dict(backend="sklearn", n_estimators=300, learning_rate=0.1,
                min_child_samples=20, early_stopping_rounds=60,
                imbalance="scale_pos_weight", calibration="isotonic")
    base.update(kw)
    return LGBMConfig(**base)


def test_lgbm_model_end_to_end(tmp_path=None):
    X, y, mf = _make_clf_data()
    cut = int(len(X) * 0.8)  # 前80%当训练（按行序模拟时间序）
    Xtr, ytr, Xva, yva = X.iloc[:cut], y.iloc[:cut], X.iloc[cut:], y.iloc[cut:]

    model = LGBMRiskModel(_fast_cfg())
    model.fit(Xtr, ytr, mf, eval_set=(Xva, yva))
    p = model.predict_risk(Xva)

    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(yva, p)
    assert auc > 0.85, f"可分数据上 AUC 仅 {auc:.3f}，训练流程有问题"
    assert p.min() >= 0.0 and p.max() <= 1.0
    assert model.calibrator_ is not None, "提供了 eval_set 时校准器必须生效"

    # ---- 持久化 roundtrip ----
    d = Path(tmp_path) if tmp_path else Path("/tmp/drp_test_model")
    model.save(d / "m1")
    loaded = LGBMRiskModel.load(d / "m1")
    np.testing.assert_allclose(loaded.predict_risk(Xva), p, rtol=0, atol=1e-12)


def test_serving_alignment_shuffled_columns():
    """列顺序打乱 + 混入清单外杂列，预测必须与原顺序完全一致（防列错位事故）。"""
    X, y, mf = _make_clf_data()
    cut = int(len(X) * 0.8)
    model = LGBMRiskModel(_fast_cfg())
    model.fit(X.iloc[:cut], y.iloc[:cut], mf, eval_set=(X.iloc[cut:], y.iloc[cut:]))

    Xq = X.iloc[cut:].copy()
    p_ref = model.predict_risk(Xq)
    shuffled = Xq[list(reversed(Xq.columns))].copy()
    shuffled["junk_col"] = 123.0
    np.testing.assert_allclose(model.predict_risk(shuffled), p_ref, atol=1e-12)


def test_monotone_constraint_respected():
    """f0 声明 monotone=+1 后，f0 增大时预测风险不得下降（含校准后）。"""
    X, y, mf = _make_clf_data()
    cut = int(len(X) * 0.8)
    model = LGBMRiskModel(_fast_cfg())
    model.fit(X.iloc[:cut], y.iloc[:cut], mf, eval_set=(X.iloc[cut:], y.iloc[cut:]))

    grid = pd.DataFrame(0.0, index=range(41), columns=X.columns)
    grid["f0"] = np.linspace(-3, 3, 41)
    p = model.predict_risk(grid)
    assert (np.diff(p) >= -1e-9).all(), "单调约束被违反"


def test_focal_requires_calibration_and_lightgbm():
    with pytest.raises(ValueError):
        LGBMConfig(imbalance="focal", calibration="none").validate()
    X, y, mf = _make_clf_data(n=400)
    m = LGBMRiskModel(_fast_cfg(imbalance="focal"))
    with pytest.raises(RuntimeError):  # sklearn 回退后端不支持 focal
        m.fit(X.iloc[:300], y.iloc[:300], mf, eval_set=(X.iloc[300:], y.iloc[300:]))


def test_missing_eval_set_disables_calibration_with_warning():
    X, y, mf = _make_clf_data(n=600)
    model = LGBMRiskModel(_fast_cfg())
    model.fit(X, y, mf, eval_set=None)
    assert model.calibrator_ is None


# ===========================================================================
# 4. HorizonBank（1/3/5 年编排）
# ===========================================================================
def _make_survival_data(n=1800, seed=11):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"f{i}": rng.normal(0, 1, n) for i in range(5)})
    risk_score = 1.2 * X["f0"] - 0.9 * X["f1"]
    true_t = rng.exponential(scale=900 * np.exp(-0.8 * risk_score))
    censor_t = rng.uniform(300, 2400, size=n)
    t_obs = np.minimum(true_t, censor_t)
    event = (true_t <= censor_t).astype(int)
    cohort = pd.DataFrame({
        COL_PATIENT_ID: [f"P{i:05d}" for i in range(n)],
        COL_INDEX_DATE: pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 600, n), "D"),
        COL_EVENT: event,
        COL_TIME_TO_EVENT: np.maximum(t_obs, 1.0),
    })
    specs = [FeatureSpec(name=f"f{i}", group=FEATURE_GROUP_DEVIATION, dtype="numeric")
             for i in range(5)]
    mf = FeatureManifest()
    mf.extend(specs)
    return X, cohort, mf


def test_horizon_bank_end_to_end(tmp_path=None):
    X, cohort, mf = _make_survival_data()
    cut = int(len(X) * 0.75)
    bank = HorizonBank(base_config=_fast_cfg())
    bank.fit(
        X.iloc[:cut], cohort.iloc[:cut], mf,
        X_valid=X.iloc[cut:], cohort_valid=cohort.iloc[cut:],
    )
    risks = bank.predict_risk(X.iloc[cut:])
    assert list(risks.columns) == ["risk_1y", "risk_3y", "risk_5y"]
    arr = risks.to_numpy()
    assert (arr >= -1e-12).all() and (arr <= 1 + 1e-12).all()
    assert (np.diff(arr, axis=1) >= -1e-9).all(), "累计风险必须随时程单调不减"

    # 排序有效性：高风险特征的样本，1年风险应排得更靠前
    from sklearn.metrics import roc_auc_score

    y1, _ = build_horizon_label(cohort.iloc[cut:], 365)
    m = usable_mask(y1).to_numpy()
    auc = roc_auc_score(y1.loc[m], risks.loc[m, "risk_1y"])
    assert auc > 0.75, f"1年时程 AUC 仅 {auc:.3f}"

    d = Path(tmp_path) if tmp_path else Path("/tmp/drp_test_bank")
    bank.save(d / "bank")
    loaded = HorizonBank.load(d / "bank")
    np.testing.assert_allclose(
        loaded.predict_risk(X.iloc[cut:]).to_numpy(), arr, atol=1e-12
    )


def test_horizon_bank_rejects_scarce_positives():
    X, cohort, mf = _make_survival_data(n=200, seed=3)
    cohort = cohort.copy()
    cohort[COL_EVENT] = 0
    cohort.loc[cohort.index[:5], COL_EVENT] = 1  # 阳性远少于 30
    cohort[COL_TIME_TO_EVENT] = 2000.0
    cohort.loc[cohort.index[:5], COL_TIME_TO_EVENT] = 100.0
    with pytest.raises(ValueError):
        HorizonBank(base_config=_fast_cfg()).fit(X, cohort, mf)


# ===========================================================================
# 5. Cox-PH（缺 lifelines 时自动 SKIP；CI 镜像必须安装）
# ===========================================================================
def test_cox_ph_end_to_end():
    pytest.importorskip("lifelines")
    from drp.models import CoxConfig, CoxPHModel

    X, cohort, mf = _make_survival_data(n=1200, seed=5)
    cut = int(len(X) * 0.8)
    cox = CoxPHModel(CoxConfig(penalizer=0.05))
    cox.fit(X.iloc[:cut], cohort.iloc[:cut], train_idx=np.arange(cut))
    assert cox.concordance_ > 0.65
    risks = cox.predict_risk_at(X.iloc[cut:], days=[365, 1095, 1825])
    arr = risks.to_numpy()
    assert (np.diff(arr, axis=1) >= -1e-9).all()
    hr = cox.hazard_ratios()
    assert {"HR", "HR_ci_low", "HR_ci_high", "p"} <= set(hr.columns)


def test_lightgbm_native_paths():
    """lightgbm 专属：focal 训练 + 内置 TreeSHAP。仅在装了 lightgbm 的环境跑。"""
    pytest.importorskip("lightgbm")
    X, y, mf = _make_clf_data(n=1200)
    cut = int(len(X) * 0.8)
    model = LGBMRiskModel(LGBMConfig(
        backend="lightgbm", n_estimators=400, learning_rate=0.1,
        min_child_samples=20, early_stopping_rounds=60,
        imbalance="focal", calibration="isotonic",
    ))
    model.fit(X.iloc[:cut], y.iloc[:cut], mf, eval_set=(X.iloc[cut:], y.iloc[cut:]))
    p = model.predict_risk(X.iloc[cut:])
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(y.iloc[cut:], p) > 0.85
    contrib, base = model.shap_values(X.iloc[cut:cut + 20])
    assert contrib.shape == (20, len(mf))
    # SHAP 可加性：贡献之和 + 基准 = raw score
    raw = model._predict_uncalibrated(X.iloc[cut:cut + 20])
    from drp.models.imbalance import _sigmoid

    np.testing.assert_allclose(_sigmoid(contrib.sum(axis=1).to_numpy() + base), raw, atol=1e-6)


# ===========================================================================
# 无 pytest 环境的直跑入口
# ===========================================================================
if __name__ == "__main__":
    import logging
    import tempfile
    import traceback

    logging.basicConfig(level=logging.WARNING)
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
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
