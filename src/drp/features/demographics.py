"""
人口学与既往史特征（规范 2.1）。

规范要求：年龄、性别、BMI、吸烟史、饮酒史、家族史、既往病史。

这些是最基础的特征，但有两个细节做错了会显著掉点：

【细节 1】年龄不能只给一个连续值
年龄与慢病风险的关系是非线性的，且在不同年龄段斜率差异很大。
树模型能自己学分段，但需要样本量。同时给出：
  - age 连续值
  - age_group 分箱（临床常用切点，不是等宽分箱）
  - age_squared （用于 Cox 模型这类线性模型，树模型可忽略）
能在中等样本量下明显加速收敛。

【细节 2】既往史必须区分"无"和"未采集"
问卷里"糖尿病史：否"和"该字段没填"是完全不同的。前者是有效信息，
后者是缺失。全部当成 0 会让模型对数据不全的用户系统性低估风险——
而数据不全的用户往往是新用户，恰恰是平台最需要准确预测的人群。
本模块对既往史统一用三态：1=有 / 0=无 / NaN=未采集。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..data.constants import COL_INDEX_DATE, FEATURE_GROUP_DEMO
from .base import BaseFeatureBuilder, FeatureSpec
from .deviation import _compute_age

logger = logging.getLogger(__name__)

# 临床常用年龄切点，不是等宽分箱
AGE_BINS = [0, 18, 30, 40, 45, 50, 55, 60, 65, 70, 75, 80, 200]

# 既往史字段：列名 -> 中文名。三态处理：1有/0无/NaN未采集
HISTORY_FIELDS = {
    "hx_hypertension": "高血压史",
    "hx_diabetes": "糖尿病史",
    "hx_hyperlipidemia": "血脂异常史",
    "hx_cad": "冠心病史",
    "hx_stroke": "脑卒中史",
    "hx_ckd": "慢性肾病史",
    "hx_hbv": "乙肝病毒感染史",
    "hx_fatty_liver": "脂肪肝史",
    "hx_cancer": "肿瘤史",
    "hx_gout": "痛风史",
}

FAMILY_HISTORY_FIELDS = {
    "fh_diabetes": "糖尿病家族史",
    "fh_hypertension": "高血压家族史",
    "fh_cad": "冠心病家族史",
    "fh_stroke": "脑卒中家族史",
    "fh_cancer": "肿瘤家族史",
    "fh_ckd": "肾病家族史",
}

# 生活方式字段：列名 -> (中文名, 特征类型, 取值说明)
LIFESTYLE_FIELDS = {
    "smoking_status": ("吸烟状态", "categorical", "0从不/1已戒/2现吸"),
    "smoking_pack_years": ("吸烟包年数", "numeric", "每日包数×吸烟年数"),
    "drinking_status": ("饮酒状态", "categorical", "0从不/1偶尔/2经常/3每日"),
    "drinking_g_per_week": ("每周酒精摄入量(g)", "numeric", ""),
    "exercise_freq_per_week": ("每周运动次数", "numeric", ""),
    "sleep_hours": ("日均睡眠时长(小时)", "numeric", ""),
}


class DemographicFeatureBuilder(BaseFeatureBuilder):
    name = "demographic"

    def build(
        self,
        cohort: pd.DataFrame,
        records: pd.DataFrame,
        **kwargs,
    ) -> tuple[pd.DataFrame, list[FeatureSpec]]:
        cohort = cohort.reset_index(drop=True)
        n = len(cohort)
        feats: dict[str, np.ndarray] = {}
        specs: list[FeatureSpec] = []

        # ---------------- 年龄 ----------------
        age = _compute_age(cohort)
        feats["age"] = age
        specs.append(
            FeatureSpec(
                name="age",
                group=FEATURE_GROUP_DEMO,
                dtype="numeric",
                description="索引日期时的年龄(岁)",
                monotone=1,  # 慢病风险随年龄单调递增，这是极少数可以放心设约束的特征
                allow_missing=False,
            )
        )

        bins = np.digitize(np.nan_to_num(age, nan=-1), AGE_BINS) - 1
        bins = np.where(np.isnan(age), np.nan, bins.astype(float))
        feats["age_group"] = bins
        specs.append(
            FeatureSpec(
                name="age_group",
                group=FEATURE_GROUP_DEMO,
                dtype="categorical",
                description=f"年龄分箱(临床切点 {AGE_BINS})",
            )
        )

        # ---------------- 性别 ----------------
        if "sex" in cohort.columns:
            sex_raw = cohort["sex"].fillna("U").astype(str).str.upper()
            feats["sex_is_male"] = np.where(
                sex_raw == "M", 1.0, np.where(sex_raw == "F", 0.0, np.nan)
            )
            specs.append(
                FeatureSpec(
                    name="sex_is_male",
                    group=FEATURE_GROUP_DEMO,
                    dtype="binary",
                    description="性别 1男/0女/NaN未知",
                    allow_missing=False,
                )
            )

        # ---------------- 体格 ----------------
        for col, (name_cn, mono) in {
            "height_cm": ("身高(cm)", 0),
            "weight_kg": ("体重(kg)", 0),
            "waist_cm": ("腰围(cm)", 1),
        }.items():
            if col in cohort.columns:
                feats[col] = pd.to_numeric(cohort[col], errors="coerce").to_numpy(dtype=float)
                specs.append(
                    FeatureSpec(
                        name=col,
                        group=FEATURE_GROUP_DEMO,
                        dtype="numeric",
                        description=name_cn,
                        monotone=mono,
                    )
                )

        # BMI：优先用已有列，否则从身高体重算
        bmi = self._resolve_bmi(cohort, n)
        if bmi is not None:
            feats["bmi"] = bmi
            specs.append(
                FeatureSpec(
                    name="bmi",
                    group=FEATURE_GROUP_DEMO,
                    dtype="numeric",
                    description="体质指数 kg/m²",
                )
            )
            # 中国成人标准分级，与国际 WHO 标准切点不同，必须用中国标准
            feats["bmi_category_cn"] = np.select(
                [bmi < 18.5, bmi < 24.0, bmi < 28.0, bmi >= 28.0],
                [0.0, 1.0, 2.0, 3.0],
                default=np.nan,
            )
            specs.append(
                FeatureSpec(
                    name="bmi_category_cn",
                    group=FEATURE_GROUP_DEMO,
                    dtype="categorical",
                    description="BMI分级(中国标准) 0消瘦/1正常/2超重/3肥胖",
                    monotone=1,
                )
            )

        # 腰高比：比 BMI 更能反映中心性肥胖，对代谢性疾病预测力更强
        if "waist_cm" in cohort.columns and "height_cm" in cohort.columns:
            w = pd.to_numeric(cohort["waist_cm"], errors="coerce").to_numpy(dtype=float)
            h = pd.to_numeric(cohort["height_cm"], errors="coerce").to_numpy(dtype=float)
            whtr = np.divide(w, h, out=np.full(n, np.nan), where=(h > 1e-6))
            feats["waist_height_ratio"] = whtr
            specs.append(
                FeatureSpec(
                    name="waist_height_ratio",
                    group=FEATURE_GROUP_DEMO,
                    dtype="numeric",
                    description="腰高比(>0.5提示中心性肥胖)，代谢风险预测优于BMI",
                    monotone=1,
                )
            )

        # ---------------- 既往史 / 家族史（三态） ----------------
        for col, name_cn in {**HISTORY_FIELDS, **FAMILY_HISTORY_FIELDS}.items():
            if col not in cohort.columns:
                continue
            feats[col] = _to_tristate(cohort[col])
            specs.append(
                FeatureSpec(
                    name=col,
                    group=FEATURE_GROUP_DEMO,
                    dtype="numeric",  # 刻意不用 binary：NaN 表示未采集，是有意义的第三态
                    description=f"{name_cn} 1有/0无/NaN未采集",
                    monotone=1,
                )
            )

        hx_cols = [c for c in HISTORY_FIELDS if c in cohort.columns]
        if hx_cols:
            hx = np.column_stack([_to_tristate(cohort[c]) for c in hx_cols])
            feats["hx_count"] = np.nansum(hx, axis=1)
            feats["hx_answered_ratio"] = np.mean(~np.isnan(hx), axis=1)
            specs.extend(
                [
                    FeatureSpec(
                        name="hx_count",
                        group=FEATURE_GROUP_DEMO,
                        dtype="numeric",
                        description="既往病史阳性项数(共病负担)",
                        monotone=1,
                    ),
                    FeatureSpec(
                        name="hx_answered_ratio",
                        group=FEATURE_GROUP_DEMO,
                        dtype="numeric",
                        description="既往史字段填答率。低填答率本身是预测不确定性的信号",
                    ),
                ]
            )

        # ---------------- 生活方式 ----------------
        for col, (name_cn, dtype, note) in LIFESTYLE_FIELDS.items():
            if col not in cohort.columns:
                continue
            feats[col] = pd.to_numeric(cohort[col], errors="coerce").to_numpy(dtype=float)
            mono = 1 if col in ("smoking_pack_years", "drinking_g_per_week") else 0
            specs.append(
                FeatureSpec(
                    name=col,
                    group=FEATURE_GROUP_DEMO,
                    dtype=dtype,  # type: ignore[arg-type]
                    description=f"{name_cn} {note}".strip(),
                    monotone=mono,
                )
            )

        # ---------------- 索引日期的季节/年份 ----------------
        # 检验值有季节性（如维生素D、血压冬高夏低），年份捕捉检测方法变迁
        idx = pd.to_datetime(cohort[COL_INDEX_DATE], errors="coerce")
        feats["index_month"] = idx.dt.month.astype(float).to_numpy()
        specs.append(
            FeatureSpec(
                name="index_month",
                group=FEATURE_GROUP_DEMO,
                dtype="categorical",
                description="索引日期月份(捕捉指标季节性波动)",
            )
        )

        out = pd.DataFrame(feats, index=cohort.index)
        self._check_alignment(cohort, out)
        return out, specs

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_bmi(cohort: pd.DataFrame, n: int) -> np.ndarray | None:
        if "bmi" in cohort.columns:
            return pd.to_numeric(cohort["bmi"], errors="coerce").to_numpy(dtype=float)
        if "height_cm" in cohort.columns and "weight_kg" in cohort.columns:
            h = pd.to_numeric(cohort["height_cm"], errors="coerce").to_numpy(dtype=float) / 100.0
            w = pd.to_numeric(cohort["weight_kg"], errors="coerce").to_numpy(dtype=float)
            bmi = np.divide(w, h**2, out=np.full(n, np.nan), where=(h > 0.5))
            # 生理极限拦截，防止身高体重录入错误产生离谱 BMI
            bmi[(bmi < 8) | (bmi > 90)] = np.nan
            return bmi
        return None


def _to_tristate(s: pd.Series) -> np.ndarray:
    """
    三态转换：1=有 / 0=无 / NaN=未采集。

    接受多种输入写法：1/0、True/False、"是"/"否"、"有"/"无"、"Y"/"N"。
    无法识别的值一律转 NaN —— 宁可当作未采集，也不要猜成 0。
    """
    mapping = {
        "1": 1.0, "是": 1.0, "有": 1.0, "y": 1.0, "yes": 1.0, "true": 1.0, "阳性": 1.0,
        "0": 0.0, "否": 0.0, "无": 0.0, "n": 0.0, "no": 0.0, "false": 0.0, "阴性": 0.0,
    }
    out = np.full(len(s), np.nan)
    for i, v in enumerate(s.to_numpy()):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if isinstance(v, (bool, np.bool_)):
            out[i] = float(v)
            continue
        if isinstance(v, (int, float, np.integer, np.floating)):
            out[i] = 1.0 if float(v) > 0 else 0.0
            continue
        key = str(v).strip().lower()
        if key in mapping:
            out[i] = mapping[key]
    return out
