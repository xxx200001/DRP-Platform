"""
特征基础设施：特征清单（FeatureManifest）与特征构造器基类。

为什么要有 Manifest 这一层，而不是直接返回一个 DataFrame：

  1. SHAP 归因聚合（规范 3.2 / 6）
     模型有几百个特征，直接把 Top10 原始特征名甩给用户毫无意义
     （"ALT_slope_per_month 贡献 0.03" 用户看不懂）。必须按指标和语义分组
     聚合后再展示："肝功能指标持续上升趋势" 才是可用的归因。

  2. 数据漂移监控（规范 3.2）
     不同特征组的漂移含义完全不同：raw_value 组漂移可能是人群变化，
     status 组漂移是检查项目结构变化，temporal 组漂移是随访频率变化。
     必须分组监控，混在一起看只能看到"漂了"却不知道为什么。

  3. 训练/推理特征一致性
     线上推理时特征顺序、缺失特征处理必须和训练完全一致。Manifest 随模型
     一起序列化，推理时严格按它对齐 —— 这是特征错位事故的唯一防线。

  4. 单调性约束
     部分指标与风险的关系有明确临床方向（如 HbA1c 越高糖尿病风险越高）。
     在 Manifest 里声明后可直接传给 LightGBM 的 monotone_constraints，
     显著提升模型在稀疏区域的外推可靠性和可解释性。
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd

logger = logging.getLogger(__name__)

FeatureDType = Literal["numeric", "categorical", "binary"]


@dataclass(frozen=True)
class FeatureSpec:
    """单个特征的元数据。"""

    name: str
    group: str  # 见 constants.FEATURE_GROUP_*
    dtype: FeatureDType
    indicator: str | None = None  # 派生自哪个指标，用于 SHAP 按指标聚合
    description: str = ""
    monotone: int = 0  # -1 / 0 / 1，LightGBM 单调性约束方向
    allow_missing: bool = True  # False 表示该特征缺失即为数据异常

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeatureManifest:
    """特征清单。与模型一起持久化，推理时用于严格对齐。"""

    specs: list[FeatureSpec] = field(default_factory=list)

    def add(self, spec: FeatureSpec) -> None:
        if any(s.name == spec.name for s in self.specs):
            raise ValueError(f"特征名重复: {spec.name}")
        self.specs.append(spec)

    def extend(self, specs: Iterable[FeatureSpec]) -> None:
        for s in specs:
            self.add(s)

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.specs]

    def by_group(self, group: str) -> list[str]:
        return [s.name for s in self.specs if s.group == group]

    def by_indicator(self, indicator: str) -> list[str]:
        return [s.name for s in self.specs if s.indicator == indicator]

    def categorical_names(self) -> list[str]:
        return [s.name for s in self.specs if s.dtype == "categorical"]

    def monotone_constraints(self) -> list[int]:
        """按 names 顺序返回，直接传给 LightGBM 的 monotone_constraints 参数。"""
        return [s.monotone for s in self.specs]

    def group_map(self) -> dict[str, str]:
        return {s.name: s.group for s in self.specs}

    def indicator_map(self) -> dict[str, str | None]:
        return {s.name: s.indicator for s in self.specs}

    # ---------------- 训练/推理一致性 ----------------
    def align(self, df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
        """
        把任意特征表对齐到本清单的列顺序。线上推理必须调用。

        strict=True 时，出现清单外的多余列会告警；缺失列按 NaN 补齐
        （因为 LightGBM 能处理 NaN，而列错位会直接毁掉预测）。
        """
        missing = [n for n in self.names if n not in df.columns]
        extra = [c for c in df.columns if c not in self.names]

        if missing:
            msg = f"推理特征表缺少 {len(missing)} 个特征，将补 NaN: {missing[:10]}"
            if strict:
                logger.warning(msg)
            for n in missing:
                df[n] = pd.NA
        if extra and strict:
            logger.warning("推理特征表含 %d 个清单外特征，已忽略: %s", len(extra), extra[:10])

        out = df[self.names].copy()
        for s in self.specs:
            if s.dtype in ("numeric", "binary"):
                out[s.name] = pd.to_numeric(out[s.name], errors="coerce")
        return out

    # ---------------- 持久化 ----------------
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in self.specs], f, ensure_ascii=False, indent=2)
        logger.info("特征清单已保存: %s (%d 个特征)", p, len(self.specs))

    @classmethod
    def load(cls, path: str | Path) -> "FeatureManifest":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls(specs=[FeatureSpec(**d) for d in json.load(f)])

    def __len__(self) -> int:
        return len(self.specs)

    def __repr__(self) -> str:
        from collections import Counter

        c = Counter(s.group for s in self.specs)
        return f"<FeatureManifest {len(self.specs)} 特征 {dict(c)}>"


class BaseFeatureBuilder(ABC):
    """
    特征构造器基类。

    契约（所有子类必须遵守，CI 会检查）：
      - build() 是【纯函数】：不修改入参，不依赖全局状态
      - build() 只能看到 as-of 过滤后的数据，禁止访问索引日期之后的信息
      - build() 必须同时返回特征表和对应的 FeatureSpec 列表
      - 返回的特征表索引必须与 cohort 完全对齐（同长度、同顺序）
      - 缺失值一律留 NaN，禁止任何形式的填充
    """

    name: str = "base"

    @abstractmethod
    def build(
        self,
        cohort: pd.DataFrame,
        records: pd.DataFrame,
        **kwargs,
    ) -> tuple[pd.DataFrame, list[FeatureSpec]]:
        """
        cohort  : 一行一个预测样本，含 patient_id / index_date / sex / birth_date
        records : as-of 过滤后的长表检验记录
        """
        raise NotImplementedError

    def _check_alignment(self, cohort: pd.DataFrame, feats: pd.DataFrame) -> None:
        if len(feats) != len(cohort):
            raise ValueError(
                f"{self.name}: 特征表行数 {len(feats)} 与队列 {len(cohort)} 不一致。"
                "特征构造器必须保证一一对齐，否则会造成静默的样本错位。"
            )
