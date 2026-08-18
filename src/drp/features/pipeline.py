"""
特征工程总管线。

把所有构造器串成一条不可绕过的流水线，并在关键位置埋硬断言。
这是**唯一**允许的特征构造入口 —— 训练脚本、推理服务、回测脚本
全部走这里，禁止任何模块自己拼特征。

为什么要强制单一入口：
训练时用 A 套逻辑、推理时用 B 套逻辑（training-serving skew）是线上精度
掉点的头号原因，而且极难排查——两边分别测都正常，合起来就是不准。
唯一可靠的解法是物理上只存在一份实现。

执行顺序（有依赖，不可调换）：
    1. as-of 时间对齐 + 未来数据断言   ← 泄露防线，必须最先
    2. 人口学特征
    3. 偏离度特征（产出 {CODE}_value，后续依赖）
    4. 临床衍生比值（依赖上一步的 _value 列）
    5. 时序特征
    6. 干扰因子
    7. 特征清单合并 + 常量列剔除 + 一致性校验
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.constants import COL_INDEX_DATE, COL_PATIENT_ID
from ..data.reference import ReferenceRegistry
from ..validation.leakage import as_of_filter, assert_no_future_records
from .base import FeatureManifest, FeatureSpec
from .confounders import ConfounderConfig, ConfounderFeatureBuilder
from .demographics import DemographicFeatureBuilder
from .deviation import DeviationFeatureBuilder
from .ratios import ClinicalRatioBuilder
from .temporal import TemporalFeatureBuilder

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """特征管线配置。随模型一起持久化，保证推理时行为完全一致。"""

    indicators: list[str] | None = None
    lookback_days: int = 1825
    blanking_days: int = 30
    """空白期：索引日期前 N 天的数据不用于特征构造。

    这是防"确诊前密集检查"泄露的关键参数，取值必须与预测时长匹配：
        1 年预测   -> 30 天
        3 年预测   -> 90 天
        5 年预测   -> 180 天
    设成 0 会让模型学到"最近查得勤=要出事"，在无症状早筛场景完全失效。
    """
    max_points: int = 20
    min_points_for_slope: int = 3
    enable_temporal: bool = True
    enable_ratios: bool = True
    enable_confounders: bool = True
    drop_constant_features: bool = True
    max_missing_rate: float = 0.98
    """缺失率超过此值的特征直接剔除。

    注意阈值设得很高（98%）是刻意的：医疗数据里高缺失特征往往信息量很大
    （做了某项罕见检查本身就是强信号）。按常见的 50% 阈值砍会砍掉真正有用的东西。
    """
    monotone_hints: dict[str, int] = field(default_factory=dict)


@dataclass
class BuildReport:
    n_samples: int = 0
    n_features: int = 0
    n_dropped_constant: int = 0
    n_dropped_missing: int = 0
    dropped_names: list[str] = field(default_factory=list)
    missing_rate_by_group: dict[str, float] = field(default_factory=dict)
    elapsed_sec: float = 0.0

    def summary(self) -> str:
        lines = [
            f"特征构造完成: {self.n_samples} 样本 × {self.n_features} 特征 "
            f"(耗时 {self.elapsed_sec:.1f}s)",
            f"  剔除常量列 {self.n_dropped_constant} | 剔除超高缺失列 {self.n_dropped_missing}",
        ]
        if self.missing_rate_by_group:
            parts = [f"{g}={r:.1%}" for g, r in sorted(self.missing_rate_by_group.items())]
            lines.append("  各组缺失率: " + " | ".join(parts))
        return "\n".join(lines)


class FeaturePipeline:
    def __init__(
        self,
        registry: ReferenceRegistry,
        confounder_config: ConfounderConfig | None = None,
        config: PipelineConfig | None = None,
    ):
        self.registry = registry
        self.config = config or PipelineConfig()
        self.confounder_config = confounder_config
        self.manifest: FeatureManifest | None = None

        inds = self.config.indicators
        self.demo_builder = DemographicFeatureBuilder()
        self.dev_builder = DeviationFeatureBuilder(
            registry, indicators=inds, monotone_hints=self.config.monotone_hints
        )
        self.ratio_builder = ClinicalRatioBuilder(registry) if self.config.enable_ratios else None
        self.temporal_builder = (
            TemporalFeatureBuilder(
                registry,
                indicators=inds,
                lookback_days=self.config.lookback_days,
                max_points=self.config.max_points,
                min_points_for_slope=self.config.min_points_for_slope,
            )
            if self.config.enable_temporal
            else None
        )
        self.conf_builder = (
            ConfounderFeatureBuilder(confounder_config, indicators=inds)
            if (self.config.enable_confounders and confounder_config is not None)
            else None
        )

    # ------------------------------------------------------------------
    def fit_transform(
        self,
        cohort: pd.DataFrame,
        records: pd.DataFrame,
        medications: pd.DataFrame | None = None,
        state_flags: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, FeatureManifest, BuildReport]:
        """
        训练阶段调用。会生成并保存 FeatureManifest。

        注意本管线【没有任何需要 fit 的统计量】—— 不做标准化、不做编码、
        不做特征选择。这是刻意的设计：所有特征都是逐样本独立计算的确定性函数，
        因此从根本上不可能发生预处理泄露（leakage.py 里的泄露 4）。

        如果后续要加 target encoding 之类需要 fit 的变换，必须包在
        FitGuard 里并只用训练集 fit。
        """
        X, manifest, report = self._build(cohort, records, medications, state_flags)

        if self.config.drop_constant_features:
            X, manifest, report = self._prune(X, manifest, report)

        self.manifest = manifest
        logger.info(report.summary())
        return X, manifest, report

    def transform(
        self,
        cohort: pd.DataFrame,
        records: pd.DataFrame,
        medications: pd.DataFrame | None = None,
        state_flags: pd.DataFrame | None = None,
        manifest: FeatureManifest | None = None,
    ) -> pd.DataFrame:
        """
        推理阶段调用。必须传入训练时保存的 manifest，输出严格对齐其列顺序。
        """
        mf = manifest or self.manifest
        if mf is None:
            raise RuntimeError(
                "transform 需要 FeatureManifest。请传入训练时保存的清单，"
                "否则无法保证训练/推理特征一致，这是线上事故的高发原因。"
            )
        X, _, _ = self._build(cohort, records, medications, state_flags)
        return mf.align(X, strict=True)

    # ------------------------------------------------------------------
    def _build(
        self,
        cohort: pd.DataFrame,
        records: pd.DataFrame,
        medications: pd.DataFrame | None,
        state_flags: pd.DataFrame | None,
    ) -> tuple[pd.DataFrame, FeatureManifest, BuildReport]:
        t0 = time.perf_counter()
        self._validate_cohort(cohort)
        cohort = cohort.reset_index(drop=True)

        # ---- 1. 防泄露：as-of 对齐 ----
        aligned = as_of_filter(
            records,
            cohort,
            lookback_days=self.config.lookback_days,
            blanking_days=self.config.blanking_days,
        )
        assert_no_future_records(aligned, cohort, blanking_days=self.config.blanking_days)

        blocks: list[pd.DataFrame] = []
        manifest = FeatureManifest()

        # ---- 2. 人口学 ----
        demo, specs = self.demo_builder.build(cohort, aligned)
        blocks.append(demo)
        manifest.extend(specs)

        # ---- 3. 偏离度（产出 _value 列，后续依赖） ----
        dev, specs = self.dev_builder.build(cohort, aligned)
        blocks.append(dev)
        manifest.extend(specs)

        # ---- 4. 临床衍生比值 ----
        if self.ratio_builder is not None:
            ratios, specs = self.ratio_builder.build(cohort, aligned, value_frame=dev)
            blocks.append(ratios)
            manifest.extend(specs)

        # ---- 5. 时序 ----
        if self.temporal_builder is not None:
            temp, specs = self.temporal_builder.build(cohort, aligned)
            blocks.append(temp)
            manifest.extend(specs)

        # ---- 6. 干扰因子 ----
        if self.conf_builder is not None:
            conf, specs = self.conf_builder.build(
                cohort, aligned, medications=medications, state_flags=state_flags
            )
            blocks.append(conf)
            manifest.extend(specs)

        X = pd.concat(blocks, axis=1)

        # 时序模块会同时产出 {CODE}_last，与偏离度的 {CODE}_value 高度重复但不完全相同
        # （前者是窗口内最后一次，后者是 as-of 最近一次），保留两者由模型自行取舍。
        dup = X.columns[X.columns.duplicated()].tolist()
        if dup:
            raise ValueError(f"特征列名冲突: {dup}。请检查各构造器的命名前缀。")

        report = BuildReport(
            n_samples=len(X),
            n_features=X.shape[1],
            elapsed_sec=time.perf_counter() - t0,
        )
        report.missing_rate_by_group = self._missing_by_group(X, manifest)
        return X, manifest, report

    # ------------------------------------------------------------------
    def _prune(
        self, X: pd.DataFrame, manifest: FeatureManifest, report: BuildReport
    ) -> tuple[pd.DataFrame, FeatureManifest, BuildReport]:
        """剔除常量列与超高缺失列。剔除动作全部记入 report，便于复盘。"""
        drop: list[str] = []

        miss = X.isna().mean()
        high_missing = miss[miss > self.config.max_missing_rate].index.tolist()
        drop.extend(high_missing)
        report.n_dropped_missing = len(high_missing)

        remaining = [c for c in X.columns if c not in set(drop)]
        nunique = X[remaining].nunique(dropna=True)
        constant = nunique[nunique <= 1].index.tolist()
        drop.extend(constant)
        report.n_dropped_constant = len(constant)

        if drop:
            drop_set = set(drop)
            X = X.drop(columns=drop)
            manifest = FeatureManifest(specs=[s for s in manifest.specs if s.name not in drop_set])
            report.dropped_names = drop
            report.n_features = X.shape[1]
            logger.info("剔除 %d 个无效特征（常量/超高缺失）", len(drop))

        return X, manifest, report

    @staticmethod
    def _missing_by_group(X: pd.DataFrame, manifest: FeatureManifest) -> dict[str, float]:
        gmap = manifest.group_map()
        out: dict[str, list[float]] = {}
        rates = X.isna().mean()
        for name, rate in rates.items():
            g = gmap.get(str(name), "unknown")
            out.setdefault(g, []).append(float(rate))
        return {g: float(np.mean(v)) for g, v in out.items()}

    @staticmethod
    def _validate_cohort(cohort: pd.DataFrame) -> None:
        required = [COL_PATIENT_ID, COL_INDEX_DATE]
        missing = [c for c in required if c not in cohort.columns]
        if missing:
            raise ValueError(f"队列表缺少必需列: {missing}")
        if cohort[COL_INDEX_DATE].isna().any():
            raise ValueError("队列表存在空的 index_date，无法做时间对齐")
        if cohort.duplicated(subset=[COL_PATIENT_ID, COL_INDEX_DATE]).any():
            raise ValueError(
                "队列表存在重复的 (patient_id, index_date)。"
                "同一患者同一索引日期只能有一个样本，否则会造成重复计数和评估偏差。"
            )
