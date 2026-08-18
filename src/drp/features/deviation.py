"""
偏离度 + 异常分级 + 三态状态特征（规范 2.1 后半 / 2.2）。

规范要求：
  - 性别、年龄分层参考区间归一化
  - 指标偏离度特征：当前值与对应年龄性别参考区间的偏离百分比
  - 异常标记特征：轻度/中度/重度异常分级

为什么偏离度比原始值更有预测力（这是本模块存在的全部理由）：

  同样是肌酐 95 μmol/L：
    - 30 岁男性 → 参考区间 57-97，属于正常上限，几乎无风险
    - 65 岁女性 → 参考区间 41-73，超上限 30%，已提示明显肾功能损害

  直接把原始值 95 丢给模型，模型必须自己从数据里学会"肌酐要结合年龄性别看"。
  它能学到，但需要海量样本，而且学到的是统计关联而非临床机制，在样本稀疏
  的老年女性亚组上极不可靠。

  把偏离度算好了喂进去，等于把几十年积累的临床参考区间知识直接注入模型，
  在中小样本量下（3 万条级别）能带来显著且稳定的 AUC 提升。

本模块为每个指标产出 5 类特征：
  {CODE}_value       原始值（换算后，保留 NaN）
  {CODE}_status      三态：0未检查/1正常/2异常
  {CODE}_z_ref       区间标准化偏离，|z|<=1 即在区间内
  {CODE}_pct_dev     超出最近边界的百分比，区间内为 0（带符号）
  {CODE}_grade       异常分级 -3..3
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..data.constants import (
    COL_BIRTH_DATE,
    COL_INDEX_DATE,
    COL_INDICATOR,
    COL_MEASURED_AT,
    COL_PATIENT_ID,
    COL_SEX,
    COL_VALUE,
    FEATURE_GROUP_DEVIATION,
    FEATURE_GROUP_RAW,
    FEATURE_GROUP_STATUS,
    AbnormalGrade,
    MeasureStatus,
)
from ..data.reference import IndicatorMeta, ReferenceRegistry
from .base import BaseFeatureBuilder, FeatureSpec

logger = logging.getLogger(__name__)


class DeviationFeatureBuilder(BaseFeatureBuilder):
    name = "deviation"

    def __init__(
        self,
        registry: ReferenceRegistry,
        indicators: list[str] | None = None,
        emit_raw_value: bool = True,
        monotone_hints: dict[str, int] | None = None,
    ):
        """
        monotone_hints : {指标码: 方向}。方向作用在 pct_dev/grade 上。
            例：{"HBA1C": 1} 表示糖化越高风险越高（单调递增约束）。
            只对有明确临床共识的指标设置，滥用会削弱模型表达能力。
        """
        self.registry = registry
        self.indicators = [c.upper() for c in (indicators or registry.codes)]
        self.emit_raw_value = emit_raw_value
        self.monotone_hints = monotone_hints or {}

    # ------------------------------------------------------------------
    def build(
        self,
        cohort: pd.DataFrame,
        records: pd.DataFrame,
        **kwargs,
    ) -> tuple[pd.DataFrame, list[FeatureSpec]]:
        cohort = cohort.reset_index(drop=True)
        latest = self._latest_per_indicator(cohort, records)

        ages = _compute_age(cohort)
        sexes = cohort[COL_SEX].fillna("U").astype(str).str.upper().to_numpy() if COL_SEX in cohort.columns else np.full(len(cohort), "U")

        feats: dict[str, np.ndarray] = {}
        specs: list[FeatureSpec] = []

        for code in self.indicators:
            meta = self.registry.get(code)
            if meta is None:
                continue
            values = latest.get(code)
            if values is None:
                values = np.full(len(cohort), np.nan)

            z, pct, grade, status = self._compute(meta, values, sexes, ages)

            if self.emit_raw_value:
                feats[f"{code}_value"] = values
                specs.append(
                    FeatureSpec(
                        name=f"{code}_value",
                        group=FEATURE_GROUP_RAW,
                        dtype="numeric",
                        indicator=code,
                        description=f"{meta.name_cn} 最近一次数值({meta.canonical_unit})",
                    )
                )

            feats[f"{code}_status"] = status
            specs.append(
                FeatureSpec(
                    name=f"{code}_status",
                    group=FEATURE_GROUP_STATUS,
                    dtype="categorical",
                    indicator=code,
                    description=f"{meta.name_cn} 三态: 0未检查/1正常/2异常",
                )
            )

            feats[f"{code}_z_ref"] = z
            specs.append(
                FeatureSpec(
                    name=f"{code}_z_ref",
                    group=FEATURE_GROUP_DEVIATION,
                    dtype="numeric",
                    indicator=code,
                    description=f"{meta.name_cn} 参考区间标准化偏离(|z|<=1为区间内)",
                    monotone=self.monotone_hints.get(code, 0),
                )
            )

            feats[f"{code}_pct_dev"] = pct
            specs.append(
                FeatureSpec(
                    name=f"{code}_pct_dev",
                    group=FEATURE_GROUP_DEVIATION,
                    dtype="numeric",
                    indicator=code,
                    description=f"{meta.name_cn} 超出参考区间边界的百分比(区间内为0)",
                    monotone=self.monotone_hints.get(code, 0),
                )
            )

            feats[f"{code}_grade"] = grade
            specs.append(
                FeatureSpec(
                    name=f"{code}_grade",
                    group=FEATURE_GROUP_DEVIATION,
                    dtype="numeric",
                    indicator=code,
                    description=f"{meta.name_cn} 异常分级 -3重度低..0正常..3重度高",
                    monotone=self.monotone_hints.get(code, 0),
                )
            )

        out = pd.DataFrame(feats, index=cohort.index)
        self._check_alignment(cohort, out)
        return out, specs

    # ------------------------------------------------------------------
    def _latest_per_indicator(
        self, cohort: pd.DataFrame, records: pd.DataFrame
    ) -> dict[str, np.ndarray]:
        """
        取每个患者每个指标【最近一次】的值，按 cohort 顺序对齐。

        注意 records 必须已经过 as_of_filter —— 本函数不做时间过滤，
        它信任上游。这是刻意的设计：时间过滤只在一个地方做，
        分散做必然有人漏掉某条路径。
        """
        if records.empty:
            return {}

        rec = records.copy()
        rec[COL_MEASURED_AT] = pd.to_datetime(rec[COL_MEASURED_AT])
        rec = rec.sort_values(COL_MEASURED_AT)

        # 若 cohort 中同一患者有多个索引日期（多次预测），需按 index_date 分别取最近值
        multi_index_date = (
            COL_INDEX_DATE in cohort.columns
            and cohort.groupby(COL_PATIENT_ID, observed=True)[COL_INDEX_DATE].nunique().max() > 1
        )
        if multi_index_date:
            return self._latest_asof(cohort, rec)

        last = rec.groupby([COL_PATIENT_ID, COL_INDICATOR], observed=True)[COL_VALUE].last()
        wide = last.unstack(COL_INDICATOR)
        wide = wide.reindex(cohort[COL_PATIENT_ID].to_numpy())
        return {c: wide[c].to_numpy(dtype=float) for c in wide.columns}

    def _latest_asof(self, cohort: pd.DataFrame, rec: pd.DataFrame) -> dict[str, np.ndarray]:
        """同一患者多个索引日期时，用 merge_asof 做逐样本时间对齐。"""
        out: dict[str, np.ndarray] = {}
        left = cohort[[COL_PATIENT_ID, COL_INDEX_DATE]].copy()
        left[COL_INDEX_DATE] = pd.to_datetime(left[COL_INDEX_DATE])
        left["_row"] = np.arange(len(left))
        left = left.sort_values(COL_INDEX_DATE)

        for code, grp in rec.groupby(COL_INDICATOR, observed=True):
            right = grp[[COL_PATIENT_ID, COL_MEASURED_AT, COL_VALUE]].sort_values(COL_MEASURED_AT)
            merged = pd.merge_asof(
                left,
                right,
                left_on=COL_INDEX_DATE,
                right_on=COL_MEASURED_AT,
                by=COL_PATIENT_ID,
                direction="backward",
            )
            arr = np.full(len(cohort), np.nan)
            arr[merged["_row"].to_numpy()] = merged[COL_VALUE].to_numpy(dtype=float)
            out[str(code)] = arr
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _compute(
        meta: IndicatorMeta,
        values: np.ndarray,
        sexes: np.ndarray,
        ages: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(values)
        z = np.full(n, np.nan)
        pct = np.full(n, np.nan)
        grade = np.full(n, np.nan)
        status = np.full(n, MeasureStatus.MISSING, dtype=float)

        # 参考区间只取决于 (sex, age分箱)，缓存避免对每一行重复匹配
        cache: dict[tuple, object] = {}

        for i in range(n):
            v = values[i]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue

            age = ages[i]
            key = (sexes[i], -1 if np.isnan(age) else int(age))
            iv = cache.get(key)
            if key not in cache:
                iv = meta.match_interval(sexes[i], None if np.isnan(age) else float(age))
                cache[key] = iv
            if iv is None:
                status[i] = MeasureStatus.NORMAL
                continue

            in_range = iv.contains(v)
            status[i] = MeasureStatus.NORMAL if in_range else MeasureStatus.ABNORMAL

            # ---- z_ref: 以区间半宽为单位的标准化偏离 ----
            center, half = iv.center, iv.half_width
            if center is not None and half:
                z[i] = (v - center) / half
            elif iv.upper is not None and iv.upper != 0:
                z[i] = v / iv.upper  # 单侧上限区间的退化处理
            elif iv.lower is not None and iv.lower != 0:
                z[i] = v / iv.lower

            # ---- pct_dev: 超出最近边界的相对百分比 ----
            if in_range:
                pct[i] = 0.0
            elif iv.upper is not None and v > iv.upper:
                pct[i] = (v - iv.upper) / iv.upper if iv.upper != 0 else np.inf
            elif iv.lower is not None and v < iv.lower:
                pct[i] = -((iv.lower - v) / iv.lower) if iv.lower != 0 else -np.inf
            else:
                pct[i] = 0.0

            grade[i] = float(_grade_of(meta, v, iv))

        pct = np.clip(pct, -50.0, 50.0)  # 防止边界为极小值时产生 inf 毁掉分裂点
        return z, pct, grade, status


def grade_of(meta: IndicatorMeta, value: float, interval) -> AbnormalGrade:
    """
    公开包装：展示层（应用后端 / 前端）需要与特征层【同一套】分级口径。

    存在的理由是防止口径分叉：前端若自己按"超出上限就叫异常"画色块，
    页面上的红色和模型里的 grade 特征会指向不同的临床严重度，
    用户看到的解释与模型依据从此对不上。所有分级只允许从这里出。
    """
    return _grade_of(meta, value, interval)


def _grade_of(meta: IndicatorMeta, value: float, iv) -> AbnormalGrade:
    """
    异常分级。分级边界 = 参考区间边界 × grade_multiplier。

    偏高侧用上限做乘法，偏低侧用下限做除法 —— 因为"低于下限一半"和
    "高于上限两倍"在临床上才是对称的严重程度，直接对称加减是错的。
    """
    gm = meta.grade_multiplier
    if iv.upper is not None and value > iv.upper:
        if value >= iv.upper * gm.severe:
            return AbnormalGrade.SEVERE_HIGH
        if value >= iv.upper * gm.moderate:
            return AbnormalGrade.MODERATE_HIGH
        if value > iv.upper * gm.mild:
            return AbnormalGrade.MILD_HIGH
        return AbnormalGrade.NORMAL
    if iv.lower is not None and value < iv.lower:
        if gm.severe > 0 and value <= iv.lower / gm.severe:
            return AbnormalGrade.SEVERE_LOW
        if gm.moderate > 0 and value <= iv.lower / gm.moderate:
            return AbnormalGrade.MODERATE_LOW
        if gm.mild > 0 and value < iv.lower / gm.mild:
            return AbnormalGrade.MILD_LOW
        return AbnormalGrade.NORMAL
    return AbnormalGrade.NORMAL


def _compute_age(cohort: pd.DataFrame) -> np.ndarray:
    """优先用 birth_date + index_date 精确计算，退化到已有 age 列。"""
    from ..data.constants import COL_AGE

    if COL_BIRTH_DATE in cohort.columns and COL_INDEX_DATE in cohort.columns:
        birth = pd.to_datetime(cohort[COL_BIRTH_DATE], errors="coerce")
        idx = pd.to_datetime(cohort[COL_INDEX_DATE], errors="coerce")
        age = (idx - birth).dt.days / 365.25
        return age.to_numpy(dtype=float)
    if COL_AGE in cohort.columns:
        return pd.to_numeric(cohort[COL_AGE], errors="coerce").to_numpy(dtype=float)
    logger.warning(
        "队列表既无 birth_date+index_date 也无 age，年龄分层参考区间将退化为通用区间，"
        "偏离度特征精度会明显下降。"
    )
    return np.full(len(cohort), np.nan)
