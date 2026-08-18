"""
端到端演示：模型训练 -> 三层验证 -> 上线门禁（规范 5 / 规范 9）。

运行:
    cd drp && PYTHONPATH=src python examples/demo_validation.py

这个脚本演示的是【训练脚本应该长什么样】：模型怎么训不是重点，
重点是任何一个要上线的模型都必须走完同一条验证路径，并把报告归档。

刻意在合成数据里埋了三个真实世界的坑，用来看门禁是否拦得住：
    - 患者多次复诊（不做患者级切分就会 AUC 虚高）
    - 外部集来自"另一家医院"（分布偏移 + 噪声更大）
    - 随访删失（1/3/5 年逐时程剔除，而不是一刀切删人）
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drp.data.constants import (  # noqa: E402
    COL_INDEX_DATE,
    COL_PATIENT_ID,
    FEATURE_GROUP_DEVIATION,
)
from drp.features.base import FeatureManifest, FeatureSpec  # noqa: E402
from drp.models import (  # noqa: E402
    COL_EVENT,
    COL_TIME_TO_EVENT,
    LGBMConfig,
    LGBMRiskModel,
    build_horizon_label,
    usable_mask,
)
from drp.validation import (  # noqa: E402
    apply_gate,
    ReleaseBlocked,
    ValidationGate,
    assert_release_ready,
    evaluate_survival,
    run_three_layer_validation,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("demo")

FEATURES = ["ALT_dev", "AST_dev", "PLT_dev", "ALB_dev", "age_z", "bmi_z"]


# ---------------------------------------------------------------------------
def make_center(n_patients: int, seed: int, noise: float = 0.0, shift: float = 0.0):
    """合成一个"中心"的数据：队列（含随访结局）+ 特征表。"""
    r = np.random.default_rng(seed)
    rows = []
    for i in range(n_patients):
        pid = f"C{seed}-{i:05d}"
        first = int(r.integers(0, 1300))
        for _ in range(int(r.integers(1, 4))):  # 同一患者多次体检
            rows.append((pid, pd.Timestamp("2019-01-01") + pd.Timedelta(days=first + int(r.integers(0, 160)))))

    cohort = (
        pd.DataFrame(rows, columns=[COL_PATIENT_ID, COL_INDEX_DATE])
        .sort_values(COL_INDEX_DATE)
        .reset_index(drop=True)
    )
    n = len(cohort)
    X = pd.DataFrame({name: r.normal(0, 1, n) for name in FEATURES})

    # 线性风险 + 一个交互项（让树模型有活干）。系数与基线风险经过标定，
    # 使 3 年阳性率落在 3%~15%、理论 AUC 落在 0.82~0.88 —— 即规范描述的
    # 真实慢病场景。合成数据一旦阳性率失真（比如做成 70%），门禁里的
    # AUC-PR 提升、报警率、分层 lift 全部失去参考意义。
    lp = 0.85 * (
        1.5 * X["ALT_dev"]
        + 1.1 * X["AST_dev"]
        - 0.9 * X["PLT_dev"]
        + 0.7 * X["age_z"] * X["bmi_z"]
    ) + shift + r.normal(0, noise, n)
    # 由风险生成发病时间；随访时长独立生成 -> 天然产生删失
    time_to_event = np.clip(r.exponential(np.exp(-lp) * 50000), 30, 1e5)
    followup = r.uniform(200, 2200, n)
    cohort[COL_EVENT] = (time_to_event <= followup).astype(int)
    cohort[COL_TIME_TO_EVENT] = np.minimum(time_to_event, followup).round()
    return cohort, X


def make_manifest() -> FeatureManifest:
    """真实项目里这份清单由 FeaturePipeline 产出，这里手写以保持示例自足。"""
    mf = FeatureManifest()
    for name in FEATURES:
        mf.add(
            FeatureSpec(
                name=name,
                group=FEATURE_GROUP_DEVIATION,
                dtype="numeric",
                indicator=name.split("_")[0],
                description=f"{name} 偏离度",
                monotone=0,
            )
        )
    return mf


# ---------------------------------------------------------------------------
def main() -> None:
    log.info("=" * 72)
    log.info("步骤 1/4  合成两家中心的数据（本院 + 外部院）")
    cohort, X = make_center(6000, seed=11, noise=0.5)
    ext_cohort, ext_X = make_center(2000, seed=22, noise=0.9, shift=-0.25)

    horizon_name, horizon_days = "3y", 1095
    y, stats = build_horizon_label(cohort, horizon_days, horizon_name=horizon_name)
    y_ext, stats_ext = build_horizon_label(ext_cohort, horizon_days, horizon_name=f"{horizon_name}-ext")
    print("\n本院 " + stats.summary())
    print("外部 " + stats_ext.summary() + "\n")

    # 留一份未剔除删失的外部集，供步骤 4 的生存口径复核使用
    ext_cohort_full, ext_X_full = ext_cohort.copy(), ext_X.copy()

    # 逐时程剔除删失样本（labels.py 铁律 3），三张表必须同步对齐
    m, m_ext = usable_mask(y).to_numpy(), usable_mask(y_ext).to_numpy()
    cohort, X, y = cohort[m].reset_index(drop=True), X[m].reset_index(drop=True), y[m].to_numpy()
    ext_cohort = ext_cohort[m_ext].reset_index(drop=True)
    ext_X, y_ext = ext_X[m_ext].reset_index(drop=True), y_ext[m_ext].to_numpy()

    manifest = make_manifest()

    # -----------------------------------------------------------------
    log.info("=" * 72)
    log.info("步骤 2/4  定义 fit_predict 回调（三层共用同一套训练逻辑）")

    def fit_predict(X_tr: pd.DataFrame, y_tr: np.ndarray, X_te: pd.DataFrame) -> np.ndarray:
        """
        注意：校准集必须从【训练集内部】再切一刀，绝不能用外层的测试集 ——
        用测试集拟合校准器就是 leakage.py 的泄露 4，且这种泄露只让概率变好看、
        不影响 AUC，极难在事后发现。
        """
        n = len(X_tr)
        cut = int(n * 0.8)
        cfg = LGBMConfig(backend="auto", n_estimators=300, learning_rate=0.06,
                         min_child_samples=30, calibration="isotonic")
        model = LGBMRiskModel(cfg)
        model.fit(
            X_tr.iloc[:cut], y_tr[:cut], manifest,
            eval_set=(X_tr.iloc[cut:], y_tr[cut:]),
        )
        return model.predict_risk(X_te)

    # -----------------------------------------------------------------
    log.info("=" * 72)
    log.info("步骤 3/4  三层验证 + 上线门禁")
    report = run_three_layer_validation(
        cohort, X, y, fit_predict,
        external=(ext_cohort, ext_X, y_ext),
        gate=ValidationGate(),          # 默认即规范 9 的承诺下限 AUC>=0.82
        model_id="demo_liver_fibrosis",
        horizon=horizon_name,
        n_splits=4,
        n_boot=300,
        gap_days=30,
    )
    print("\n" + report.summary() + "\n")

    out = Path(__file__).resolve().parents[1] / "artifacts" / f"validation_{horizon_name}.json"
    report.save_json(out)
    print(f"验证报告已归档: {out}")

    try:
        assert_release_ready(report)
        print("\n✅ 门禁通过，允许进入发布流程。")
    except ReleaseBlocked as e:
        print(f"\n⛔ 门禁拦截：\n{e}")

    # 判定与执行分离：同一份结果可以随时用另一套标准复审，不必重跑模型。
    # 这在"某病种要求更高门槛"或"事后复盘当时标准是否过松"时非常有用。
    strict = apply_gate(report, ValidationGate(min_auc_roc=0.95, require_external=True))
    n_bad = sum(1 for g in strict if not g.passed)
    print(f"\n[复审] 若把承诺提到 AUC≥0.95，同一份结果会有 {n_bad} 项不达标：")
    for g in strict:
        if not g.passed:
            print(g.line())

    # -----------------------------------------------------------------
    log.info("=" * 72)
    log.info("步骤 4/4  生存口径复核：C-index（规范 5）")
    # 同一个风险分也可以用生存口径评。这里刻意在【外部集】上做，且用
    # 未剔除删失的全量外部数据 —— 绝不能在训练数据上算指标，那种数字
    # 只会虚高，正是规范 5 要杜绝的东西。
    risk_ext = fit_predict(X, y, ext_X_full)
    sm = evaluate_survival(
        ext_cohort_full[COL_TIME_TO_EVENT], ext_cohort_full[COL_EVENT], risk_ext,
        label="外部集全量（含删失样本）",
        groups=ext_cohort_full[COL_PATIENT_ID], n_boot=100,
    )
    print("\n" + sm.summary())
    print(
        f"\n说明：二分类口径为算 3 年标签丢掉了 {int((~m_ext).sum())} 条删失样本"
        f"（{(~m_ext).mean():.1%}），C-index 则把它们全用上了 —— 删失者提供的"
        '"至少活过 T 天"同样是信息。两个口径互为交叉校验：若显著背离，'
        "通常意味着删失与风险相关（informative censoring），"
        "此时应改用 Cox-PH（survival.py）或 IPCW 加权。"
    )


if __name__ == "__main__":
    main()
