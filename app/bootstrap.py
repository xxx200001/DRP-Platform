"""
开发环境自举训练（首次启动无模型时执行一次，产物持久化后不再重跑）。

【这不是 demo 的原因】
自举走的是与生产完全相同的路径：合成的是【纵向化验长表】而不是现成特征——
数据必须先过 LabDataCleaner（真实清洗）、再过 FeaturePipeline（真实特征
工程，产出数百维特征与 manifest）、逐时程跑 run_three_layer_validation
（真实三层验证与开发联调门禁）、通过后才进 ModelRegistry。线上预测时，患者录入的
化验数据走【同一条】清洗+特征路径、用【同一份】manifest 严格对齐列——
训练/推理一致性由 manifest.align(strict=True) 物理保证。
唯一的"合成"只有病历内容本身。

【规范 1.1 红线必须重申】
"禁止只用公开/合成数据上线"。自举产物仅供开发联调与界面验收；
正式上线必须用国内真实脱敏随访病历重训，并改用 ValidationGate() 生产门禁
（RetrainJob 已就绪），
本模块在注册 notes 与启动日志里都会把这句话钉出来。

【blanking_days=0 的说明】
生产要求空白期与时程匹配（features/pipeline 文档）以防标签邻近泄露。
自举数据的结局由潜在风险变量生成、与"索引日前最后几天的化验"无因果
关联，泄露通道不存在；置 0 可让界面上"刚录入的化验立即影响预测"，
便于验收联调。生产重训时必须按规范恢复各时程空白期。
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from drp.data.cleaning import LabDataCleaner
from drp.data.constants import (
    COL_BIRTH_DATE,
    COL_INDEX_DATE,
    COL_PATIENT_ID,
    COL_SEX,
)
from drp.data.reference import ReferenceRegistry
from drp.features import ConfounderConfig, FeaturePipeline, PipelineConfig
from drp.models import (
    COL_EVENT,
    COL_TIME_TO_EVENT,
    HorizonBank,
    LGBMConfig,
    LGBMRiskModel,
    ModelRegistry,
    build_horizon_label,
    usable_mask,
)
from drp.serving.drift import ReferenceProfile
from drp.serving.service import RiskTierScheme
from drp.validation import (
    ValidationGate,
    assert_release_ready,
    run_three_layer_validation,
    time_based_split,
)

logger = logging.getLogger(__name__)

BOOTSTRAP_VERSION = "dev_bootstrap_v1"
BOOTSTRAP_NOTE = (
    "开发自举模型：训练数据为合成纵向病历（规范1.1 禁止合成/公开数据上线）。"
    "仅供开发联调与界面验收；正式上线必须以真实脱敏随访病历经 RetrainJob 重训替换。"
)

#: 自举纳入的指标与其风险载荷方向（+1 高危者偏高 / -1 高危者偏低）。
#: 全部来自 configs/reference_intervals.yaml 已登记指标。
_INDICATOR_LOADINGS: dict[str, float] = {
    "ALT": +0.9, "AST": +0.8, "GGT": +0.6, "TBIL": +0.3, "ALB": -0.5,
    "GLU": +0.9, "HBA1C": +0.9, "TG": +0.7, "HDLC": -0.6, "LDLC": +0.6,
    "CREA": +0.5, "UA": +0.4, "PLT": -0.5, "HGB": -0.3, "CRP": +0.6,
    "SBP": +0.7, "DBP": +0.5, "BMI": +0.6,
}

_PIPELINE_CONFIG = PipelineConfig(
    lookback_days=1825,
    blanking_days=0,          # 自举专用，理由见模块顶部；生产重训必须按时程恢复
    enable_temporal=True,
    enable_ratios=True,
    enable_confounders=True,
    # 干扰因子构造器开启（规范 2.4）。自举仍不合成用药/采血登记数据 ——
    # 构造器对缺失输入优雅降级：无用药表则不产出 med_* 特征，无登记则不产出
    # phys_*；但急性状态（近期感染/急性肝肾损伤）由检验值触发规则判定，
    # 合成病历同样能触发，自举模型因此吃到 acute_* 与 *_reliability 特征，
    # 让整机测试真正覆盖 2.4 的服务链路。应用层已接入用药与采血登记，
    # 生产训练时把真实用药表传入 fit_transform 即可。
    # 已知事项：monotone_hints 会随指标传播到其 *_status 分类特征，
    # LightGBM 对 categorical+monotone 组合直接 fatal。核心库待改为
    # "hint 只作用于数值特征"；在此之前应用层不启用单调提示。
    monotone_hints={},
)

_LGBM = LGBMConfig(backend="auto", n_estimators=600, learning_rate=0.06,
                   min_child_samples=40, calibration="isotonic")

DEFAULT_HORIZONS: tuple[tuple[str, int], ...] = (("1y", 365), ("3y", 1095), ("5y", 1825))

# 合成数据只用于“把整条链路点亮”，不能证明生产有效性。该门禁仍会
# 拦住明显无效的开发模型，但绝不替代 ValidationGate() 的生产标准。
DEVELOPMENT_BOOTSTRAP_GATE = ValidationGate(
    min_auc_roc=0.75,
    min_auc_ci_lower=0.70,
    min_pr_lift=1.3,
    min_specificity_at_target=0.20,
    max_ece=0.12,
    max_oe_deviation=0.35,
    min_test_positives=20,
)


# ---------------------------------------------------------------------------
# 合成纵向病历（长表），随后交给【真实】清洗与特征管线
# ---------------------------------------------------------------------------
def synth_longitudinal(
    n_patients: int, seed: int, registry: ReferenceRegistry
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (patients[patient_id,sex,birth_date], records 长表原始形态)。"""
    r = np.random.default_rng(seed)
    patients, rec_rows = [], []
    base = pd.Timestamp("2019-01-01")

    for i in range(n_patients):
        pid = f"SYN-{seed}-{i:05d}"
        sex = "M" if r.random() < 0.52 else "F"
        age0 = float(r.uniform(30, 78))
        birth = base - pd.to_timedelta(int(age0 * 365.25), unit="D")
        z = float(r.normal())  # 潜在风险
        patients.append({COL_PATIENT_ID: pid, COL_SEX: sex, COL_BIRTH_DATE: birth, "_z": z})

        n_visits = int(r.integers(2, 6))
        t0 = int(r.integers(0, 400))
        visit_days = np.sort(t0 + r.integers(0, 1300, size=n_visits))
        drift = float(r.uniform(0.0, 0.35)) * max(z, 0)  # 高危者随时间恶化

        for vi, d in enumerate(visit_days):
            ts = base + pd.to_timedelta(int(d), unit="D")
            prog = drift * vi / max(n_visits - 1, 1)
            for code, load in _INDICATOR_LOADINGS.items():
                meta = registry.require(code)
                iv = meta.match_interval(sex=sex, age=age0)
                if iv is None or iv.center is None:
                    continue
                center, half = iv.center, (iv.half_width or iv.center * 0.15)
                shift = load * (0.95 * z + prog) * half
                val = center + shift + r.normal(0, 0.38 * half)
                lo, hi = meta.plausible_range
                val = float(np.clip(val, lo * 1.02, hi * 0.98))
                rec_rows.append(
                    {COL_PATIENT_ID: pid, "indicator_code": code, "value": round(val, 3),
                     "unit": meta.canonical_unit, "measured_at": ts}
                )

    return pd.DataFrame(patients), pd.DataFrame(rec_rows)


def _make_cohort(patients: pd.DataFrame, records: pd.DataFrame, seed: int) -> pd.DataFrame:
    """索引日=末次随访；结局时间由潜在风险生成（与任何单次化验值无直接因果）。"""
    r = np.random.default_rng(seed + 1)
    last = records.groupby(COL_PATIENT_ID)["measured_at"].max().rename(COL_INDEX_DATE)
    cohort = patients.merge(last, on=COL_PATIENT_ID, how="inner")
    age = (cohort[COL_INDEX_DATE] - cohort[COL_BIRTH_DATE]).dt.days / 365.25
    lp = 1.2 * cohort["_z"].to_numpy() + 0.028 * (age.to_numpy() - 55) + r.normal(0, 0.35, len(cohort))
    # scale=4500 标定各时程阳性率约 1y≈8% / 3y≈20% / 5y≈32%（慢病现实量级上带），
    # 保证 2600 人规模下每个时程的验证测试集阳性 ≥ 门禁阈值 30。
    tte = np.clip(r.exponential(np.exp(-lp) * 4500), 30, 1e5)
    fu = r.uniform(300, 3000, len(cohort))  # 上限盖过 5y=1825，删失率可控
    cohort[COL_EVENT] = (tte <= fu).astype(int)
    cohort[COL_TIME_TO_EVENT] = np.minimum(tte, fu).round()
    return cohort.drop(columns=["_z"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
def _fit_one(X_tr, y_tr, manifest, cfg: LGBMConfig, calib_frac=0.2) -> LGBMRiskModel:
    n = len(X_tr)
    cut = min(n - 1, max(1, int(n * (1 - calib_frac))))
    m = LGBMRiskModel(replace(cfg))
    m.fit(X_tr.iloc[:cut], np.asarray(y_tr)[:cut], manifest,
          eval_set=(X_tr.iloc[cut:], np.asarray(y_tr)[cut:]))
    return m


def run_bootstrap(
    app_data: str | Path,
    n_patients: int = 2600,
    seed: int = 11,
    horizons: tuple[tuple[str, int], ...] = DEFAULT_HORIZONS,
    gate: ValidationGate | None = None,
    registry_yaml: str | Path | None = None,
) -> dict:
    """
    产物落盘布局（server.py 按此加载）：
        app_data/artifacts/{BOOTSTRAP_VERSION}/           HorizonBank（含各时程 manifest）
        app_data/serving/{h}.tier.json                    分层切点（验证集固化，规范 6）
        app_data/serving/{h}.drift.json                   漂移基线（训练集分箱，规范 3.2）
        app_data/registry/                                ModelRegistry 账本 + 验证报告
        app_data/bootstrap_meta.json
    gate=None 时使用 DEVELOPMENT_BOOTSTRAP_GATE，仅用于开发联调；
    生产重训必须显式传入 ValidationGate()，且不得豁免外部验证。
    """
    app_data = Path(app_data)
    registry_yaml = registry_yaml or (
        Path(__file__).resolve().parents[1] / "configs" / "reference_intervals.yaml"
    )
    reg_meta = ReferenceRegistry.from_yaml(registry_yaml)
    gate = gate or DEVELOPMENT_BOOTSTRAP_GATE

    logger.info("【自举 1/5】合成纵向病历 n_patients=%d …", n_patients)
    patients, raw_records = synth_longitudinal(n_patients, seed, reg_meta)

    logger.info("【自举 2/5】真实清洗管线（LabDataCleaner）…")
    cleaner = LabDataCleaner(reg_meta)
    clean, creport = cleaner.clean(raw_records, demographics=patients)
    logger.info(creport.summary())

    logger.info("【自举 3/5】真实特征管线（FeaturePipeline.fit_transform）…")
    cohort = _make_cohort(patients, raw_records, seed)
    conf_cfg = ConfounderConfig.from_yaml(
        Path(__file__).resolve().parents[1] / "configs" / "confounders.yaml"
    )
    pipeline = FeaturePipeline(reg_meta, confounder_config=conf_cfg, config=_PIPELINE_CONFIG)
    X, manifest, breport = pipeline.fit_transform(cohort, clean)
    logger.info(breport.summary())

    logger.info("【自举 4/5】逐时程三层验证（外部集豁免留痕）+ 拟合最终制品 …")
    reports, models, stats, tier_schemes, drift_profiles = {}, {}, {}, {}, {}
    for name, days in horizons:
        y, lstats = build_horizon_label(cohort, days, horizon_name=name)
        m = usable_mask(y).to_numpy()
        co_h, X_h, y_h = (
            cohort[m].reset_index(drop=True), X[m].reset_index(drop=True), y[m].to_numpy()
        )
        stats[name] = lstats

        def fit_predict(X_tr, y_tr, X_te, _mf=manifest):
            return _fit_one(X_tr, y_tr, _mf, _LGBM).predict_risk(X_te)

        rep = run_three_layer_validation(
            co_h, X_h, y_h, fit_predict,
            gate=gate, model_id=BOOTSTRAP_VERSION, horizon=name,
            n_splits=3, n_boot=120, gap_days=30, allow_missing_external=True,
        )
        assert_release_ready(rep)  # BLOCK 即抛 —— 自举也不许绕门禁
        reports[name] = rep

        # 切点与漂移基线来自【训练侧切分】，绝不用全量拟合的模型自评（虚高）
        split = time_based_split(co_h, test_size=0.2, gap_days=30)
        m_holdout = _fit_one(X_h.iloc[split.train_idx], y_h[split.train_idx], manifest, _LGBM)
        p_hold = m_holdout.predict_risk(X_h.iloc[split.test_idx])
        tier_schemes[name] = RiskTierScheme.from_probabilities(p_hold, source=f"holdout_{name}")
        drift_profiles[name] = ReferenceProfile.from_training(
            X_h.iloc[split.train_idx], manifest
        )

        models[name] = _fit_one(X_h, y_h, manifest, _LGBM)  # 服务制品：全量数据
        logger.info("[%s] 验证 %s  headline_auc=%.4f  阳性=%d",
                    name, rep.status, rep.headline_auc, lstats.n_pos)

    logger.info("【自举 5/5】落盘 + 注册 + 晋升 …")
    bank = HorizonBank(base_config=_LGBM, horizons=tuple(horizons), enforce_monotonic=True)
    bank.models, bank.label_stats = models, stats
    bank_dir = app_data / "artifacts" / BOOTSTRAP_VERSION
    bank.save(bank_dir)

    serving_dir = app_data / "serving"
    serving_dir.mkdir(parents=True, exist_ok=True)
    for name, _ in horizons:
        tier_schemes[name].save(serving_dir / f"{name}.tier.json")
        drift_profiles[name].save(serving_dir / f"{name}.drift.json")

    model_registry = ModelRegistry(app_data / "registry")
    info = model_registry.register(BOOTSTRAP_VERSION, bank_dir, reports, notes=BOOTSTRAP_NOTE)
    model_registry.promote(BOOTSTRAP_VERSION)

    meta = {
        "version": BOOTSTRAP_VERSION,
        "n_patients": n_patients,
        "seed": seed,
        "horizons": [list(h) for h in horizons],
        "headline_auc": info.headline_auc,
        "validation_status": info.validation_status,
        "development_only": True,
        "gate_config": gate.to_dict(),
        "pipeline_config": {
            "lookback_days": _PIPELINE_CONFIG.lookback_days,
            "blanking_days": _PIPELINE_CONFIG.blanking_days,
            "enable_confounders": _PIPELINE_CONFIG.enable_confounders,
        },
        "note": BOOTSTRAP_NOTE,
    }
    (app_data / "bootstrap_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.warning(BOOTSTRAP_NOTE)
    return meta


def is_bootstrapped(app_data: str | Path) -> bool:
    return (Path(app_data) / "bootstrap_meta.json").exists()
