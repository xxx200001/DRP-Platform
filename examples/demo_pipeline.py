"""
端到端演示：从原始脏数据 -> 清洗 -> 特征工程 -> 防泄露切分。

运行:
    cd drp && PYTHONPATH=src python examples/demo_pipeline.py

这个脚本刻意在模拟数据里埋了各种真实世界的脏数据，用来验证清洗层确实拦得住：
    - 肌酐用 mg/dL 单位（需换算）
    - 血小板用 /uL 单位（数值差 1000 倍）
    - 超生理极限的 ALT = 99999
    - 未登记的指标名 "某某神秘指标"
    - 同一时刻重复上报
    - "<0.01" 这类带符号的数值
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drp.data import DuplicatePolicy, LabDataCleaner, ReferenceRegistry  # noqa: E402
from drp.features import ConfounderConfig, FeaturePipeline, PipelineConfig  # noqa: E402
from drp.validation import (  # noqa: E402
    assert_split_integrity,
    patient_level_split,
    time_based_split,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("demo")

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
RNG = np.random.default_rng(20240815)


# ---------------------------------------------------------------------------
# 模拟数据生成
# ---------------------------------------------------------------------------
def make_fake_data(n_patients: int = 600):
    """生成带纵向随访的模拟体检数据。真实项目请替换为脱敏随访病历导入。"""
    patients, records, meds = [], [], []

    panel = {
        "ALT": (25, 12), "AST": (24, 9), "GGT": (30, 18), "TBIL": (12, 4),
        "ALB": (44, 3), "TP": (73, 5), "CREA": (75, 15), "UREA": (5.2, 1.3),
        "UA": (330, 80), "GLU": (5.3, 0.9), "HBA1C": (5.5, 0.6),
        "TC": (4.6, 0.9), "TG": (1.4, 0.8), "HDLC": (1.35, 0.3), "LDLC": (2.8, 0.7),
        "WBC": (6.2, 1.5), "NEUT": (3.6, 1.1), "LYMPH": (1.9, 0.5),
        "PLT": (230, 55), "HGB": (145, 15), "K": (4.2, 0.35), "NA": (141, 2.5),
        "CRP": (2.0, 3.0), "SBP": (124, 14), "DBP": (78, 9),
    }

    for i in range(n_patients):
        pid = f"P{i:05d}"
        sex = "M" if RNG.random() < 0.52 else "F"
        birth = pd.Timestamp("1955-01-01") + pd.Timedelta(days=int(RNG.integers(0, 16000)))
        # 潜在风险因子：驱动指标漂移与结局标签，模拟真实的疾病进展过程
        latent = RNG.normal(0, 1)

        patients.append(
            {
                "patient_id": pid,
                "sex": sex,
                "birth_date": birth,
                "height_cm": RNG.normal(172 if sex == "M" else 160, 7),
                "weight_kg": RNG.normal(72 if sex == "M" else 58, 11) + latent * 4,
                "waist_cm": RNG.normal(86 if sex == "M" else 76, 9) + latent * 5,
                "smoking_status": int(RNG.integers(0, 3)),
                "smoking_pack_years": max(0, RNG.normal(8, 10)),
                "drinking_status": int(RNG.integers(0, 4)),
                "exercise_freq_per_week": int(RNG.integers(0, 6)),
                "hx_hypertension": RNG.choice([1, 0, None], p=[0.25, 0.65, 0.10]),
                "hx_diabetes": RNG.choice([1, 0, None], p=[0.12, 0.78, 0.10]),
                "hx_fatty_liver": RNG.choice(["是", "否", None], p=[0.30, 0.60, 0.10]),
                "fh_diabetes": RNG.choice([1, 0, None], p=[0.20, 0.70, 0.10]),
                "_latent": latent,
            }
        )

        # 2~6 次随访，间隔 6~18 个月
        n_visits = int(RNG.integers(2, 7))
        t = pd.Timestamp("2018-01-01") + pd.Timedelta(days=int(RNG.integers(0, 400)))
        for v in range(n_visits):
            for code, (mu, sd) in panel.items():
                if RNG.random() < 0.12:  # 12% 该项没查 -> 天然的三态 MISSING
                    continue
                drift = latent * 0.30 * v * sd  # 高风险个体指标随时间恶化
                val = RNG.normal(mu, sd) + drift
                # 按指标量级截断到合理下限，避免模拟数据本身制造大量脏值
                floor = max(mu * 0.15, 0.05)
                records.append(
                    {
                        "patient_id": pid,
                        "indicator_code": code,
                        "value": round(float(max(val, floor)), 2),
                        "unit": None,
                        "measured_at": t,
                    }
                )
            t += pd.Timedelta(days=int(RNG.integers(180, 540)))

        if RNG.random() < 0.22:
            meds.append(
                {
                    "patient_id": pid,
                    "medication_name": RNG.choice(
                        ["阿托伐他汀钙片", "二甲双胍缓释片", "缬沙坦胶囊", "苯溴马隆片"]
                    ),
                    "start_date": pd.Timestamp("2019-06-01"),
                    "end_date": None,
                }
            )

    return pd.DataFrame(patients), pd.DataFrame(records), pd.DataFrame(meds)


def inject_dirty_rows(records: pd.DataFrame) -> pd.DataFrame:
    """注入真实世界脏数据，用于验证清洗层的拦截能力。"""
    dirty = [
        # 单位换算：肌酐 mg/dL
        {"patient_id": "P00000", "indicator_code": "CREA", "value": 1.1,
         "unit": "mg/dL", "measured_at": pd.Timestamp("2020-03-01")},
        # 单位换算：血小板 /uL，差 1000 倍
        {"patient_id": "P00001", "indicator_code": "PLT", "value": 245000,
         "unit": "/uL", "measured_at": pd.Timestamp("2020-03-01")},
        # 超生理极限，必须拒绝
        {"patient_id": "P00002", "indicator_code": "ALT", "value": 99999,
         "unit": "U/L", "measured_at": pd.Timestamp("2020-03-01")},
        # 未登记指标名
        {"patient_id": "P00003", "indicator_code": "某某神秘指标", "value": 3.14,
         "unit": None, "measured_at": pd.Timestamp("2020-03-01")},
        # 带小于号的数值
        {"patient_id": "P00004", "indicator_code": "CRP", "value": "<0.5",
         "unit": "mg/L", "measured_at": pd.Timestamp("2020-03-01")},
        # 别名 + 全角
        {"patient_id": "P00005", "indicator_code": "谷丙转氨酶", "value": 33,
         "unit": "U/L", "measured_at": pd.Timestamp("2020-03-01")},
        # 同时刻重复上报
        {"patient_id": "P00006", "indicator_code": "GLU", "value": 5.4,
         "unit": "mmol/L", "measured_at": pd.Timestamp("2020-03-01")},
        {"patient_id": "P00006", "indicator_code": "GLU", "value": 5.6,
         "unit": "mmol/L", "measured_at": pd.Timestamp("2020-03-01")},
        # 血糖 108 且没写单位 —— 这是【真正无法自动判定】的情况：
        # 按 mg/dL 换算是 6.0，按 10.8 或 1.08 读也都在生理范围内。
        # 系统必须拒绝而不是猜，这条用来验证"多候选即放弃"的保守策略。
        {"patient_id": "P00007", "indicator_code": "GLU", "value": 108,
         "unit": None, "measured_at": pd.Timestamp("2020-03-01")},
    ]
    return pd.concat([records, pd.DataFrame(dirty)], ignore_index=True)


def make_cohort(patients: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    """
    构造队列：每个患者取【最后一次随访】作为索引日期，模拟"用历史预测未来"。
    标签由潜在风险因子生成，与特征存在真实关联（否则 AUC 会是 0.5）。
    """
    last = records.groupby("patient_id")["measured_at"].max().rename("index_date")
    cohort = patients.merge(last, on="patient_id", how="inner")

    age = (cohort["index_date"] - cohort["birth_date"]).dt.days / 365.25
    logit = -2.2 + 0.9 * cohort["_latent"] + 0.035 * (age - 55)
    prob = 1 / (1 + np.exp(-logit))
    cohort["label_3y"] = (RNG.random(len(cohort)) < prob).astype(int)
    return cohort.drop(columns=["_latent"])


# ---------------------------------------------------------------------------
def main() -> None:
    log.info("=" * 70)
    log.info("步骤 1/5  生成模拟数据并注入脏数据")
    patients, records, meds = make_fake_data()
    records = inject_dirty_rows(records)
    log.info("原始记录 %d 条 / 患者 %d 人 / 用药记录 %d 条",
             len(records), len(patients), len(meds))

    log.info("=" * 70)
    log.info("步骤 2/5  加载参考区间与干扰因子配置")
    registry = ReferenceRegistry.from_yaml(CONFIG_DIR / "reference_intervals.yaml")
    conf_cfg = ConfounderConfig.from_yaml(CONFIG_DIR / "confounders.yaml")
    log.info("已登记指标 %d 个，药物类别 %d 类", len(registry), len(conf_cfg.med_classes))

    log.info("=" * 70)
    log.info("步骤 3/5  数据清洗（单位换算 / 生理极限拦截 / 去重 / 三态标注）")
    cleaner = LabDataCleaner(registry, duplicate_policy=DuplicatePolicy.MEDIAN)
    clean, report = cleaner.clean(records, demographics=patients)
    print("\n" + report.summary() + "\n")
    if report.rejected_samples:
        print("被拒绝的样本（前 5 条）:")
        for r in report.rejected_samples[:5]:
            print(f"  - {r['raw_name']}: {r['detail']}")
        print()

    log.info("=" * 70)
    log.info("步骤 4/5  特征工程")
    cohort = make_cohort(patients, records)
    pipeline = FeaturePipeline(
        registry,
        confounder_config=conf_cfg,
        config=PipelineConfig(
            lookback_days=1825,
            blanking_days=90,          # 3 年预测 -> 90 天空白期
            enable_temporal=True,
            enable_ratios=True,
            enable_confounders=True,
            monotone_hints={"HBA1C": 1, "GLU": 1, "LDLC": 1, "SBP": 1},
        ),
    )
    X, manifest, build_report = pipeline.fit_transform(cohort, clean, medications=meds)
    print("\n" + build_report.summary() + "\n")

    from collections import Counter
    print("特征分组统计:")
    for g, c in sorted(Counter(s.group for s in manifest.specs).items()):
        print(f"  {g:18s} {c:4d} 个")

    print("\n临床衍生特征样例（前 5 行）:")
    ratio_cols = [c for c in manifest.by_group("clinical_ratio")][:6]
    print(X[ratio_cols].head().round(3).to_string())

    print("\n时序特征样例 —— ALT:")
    alt_cols = [c for c in X.columns if c.startswith("ALT_") and
                c.split("_", 1)[1] in ("n_obs", "slope", "trend", "persistence", "abnormal_ratio")]
    print(X[alt_cols].head().round(3).to_string())

    log.info("=" * 70)
    log.info("步骤 5/5  防泄露切分")
    sp_time = time_based_split(cohort, test_size=0.25, gap_days=30)
    assert_split_integrity(cohort, sp_time, label_col="label_3y", min_test_positives=5)
    print(f"\n时间拆分: {sp_time!r}  {sp_time.detail}")

    sp_pat = patient_level_split(cohort, test_size=0.25, stratify_col="label_3y", seed=42)
    assert_split_integrity(cohort, sp_pat, label_col="label_3y", min_test_positives=5)
    print(f"患者级切分: {sp_pat!r}  {sp_pat.detail}")

    pos_rate = cohort["label_3y"].mean()
    print(f"\n阳性率: {pos_rate:.2%}（真实慢病数据通常 3%~15%，需按规范 3.2 做不均衡处理）")

    out_dir = Path(__file__).resolve().parents[1] / "artifacts"
    out_dir.mkdir(exist_ok=True)
    manifest.save(out_dir / "feature_manifest.json")
    print(f"\n特征清单已保存: {out_dir / 'feature_manifest.json'}")
    print("下一步: examples/demo_validation.py —— 基于这份清单训练模型并跑三层验证。")


if __name__ == "__main__":
    main()
