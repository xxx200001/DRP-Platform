"""
时程模型库 HorizonBank（规范 3.3：输出 1年/3年/5年 病情进展风险概率）。

一个病种 = 一个 HorizonBank = 三个独立的 LGBMRiskModel（1y/3y/5y）。
本模块负责把 labels.py（逐时程标签+删失剔除）、lgbm.py（单时程模型）
串成一条不可绕过的编排线，训练脚本与推理服务只跟它打交道。

【为什么是"三个二分类"而不是"一个多分类/一个回归"】
  三个时程的删失剔除集合不同（labels.py 错误 B），样本集合本来就
  不一样，只能各训各的。多分类会强迫三者共用样本集，等于把 5 年
  时程的删失剔除强加给 1 年时程，白扔数据。

【时程单调性修正（enforce_monotonic，默认开）】
  三个独立模型的预测可能出现 risk_1y > risk_3y 这种物理上不可能的
  交叉（累计风险随时间只增不减）。对同一用户展示 "1年风险12%、
  3年风险9%"，任何医生一眼就知道模型有问题。修正方式为按时程做
  逐样本 cummax —— 这是保序回归在三点上的退化形式，只提不降、
  幅度最小。修正幅度会记录日志：若大面积触发（>5% 样本、平均修正
  >2 个百分点），说明某个时程模型本身有问题，必须回头查，
  而不是靠修正糊过去。

【每时程独立的特征表（X 可传 dict）】
  pipeline.PipelineConfig.blanking_days 的文档写明：空白期必须与
  预测时长匹配（1年→30天，3年→90天，5年→180天）。严格做法是每个
  时程用各自 blanking 跑一遍特征管线，因此 fit/predict 的 X 参数
  既接受单个 DataFrame（三时程共用，快速路线），也接受
  {"1y": X1, "3y": X3, "5y": X5}（严格路线，正式模型必须用）。
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.base import FeatureManifest
from .labels import (
    COL_EVENT,
    COL_TIME_TO_EVENT,
    DEFAULT_HORIZONS,
    LabelStats,
    build_horizon_label,
    usable_mask,
)
from .lgbm import LGBMConfig, LGBMRiskModel

logger = logging.getLogger(__name__)

_BANK_META = "bank_meta.json"

XLike = pd.DataFrame | dict[str, pd.DataFrame]


def _x_for(X: XLike, name: str) -> pd.DataFrame:
    if isinstance(X, dict):
        if name not in X:
            raise KeyError(f"X 字典缺少时程 '{name}' 的特征表，现有: {list(X)}")
        return X[name]
    return X


class HorizonBank:
    """
    用法::

        bank = HorizonBank(base_config=LGBMConfig())
        bank.fit(X_tr, cohort_tr, manifest, X_valid=X_va, cohort_valid=co_va)
        risks = bank.predict_risk(X_new)     # DataFrame[risk_1y, risk_3y, risk_5y]
        bank.save("artifacts/bank_liver")
    """

    def __init__(
        self,
        base_config: LGBMConfig | None = None,
        horizons: tuple[tuple[str, int], ...] = DEFAULT_HORIZONS,
        enforce_monotonic: bool = True,
        per_horizon_config: dict[str, LGBMConfig] | None = None,
    ):
        self.base_config = base_config or LGBMConfig()
        self.horizons = tuple(horizons)
        self.enforce_monotonic = enforce_monotonic
        self.per_horizon_config = per_horizon_config or {}
        self.models: dict[str, LGBMRiskModel] = {}
        self.label_stats: dict[str, LabelStats] = {}

    # ------------------------------------------------------------------
    def fit(
        self,
        X: XLike,
        cohort: pd.DataFrame,
        manifest: FeatureManifest,
        X_valid: XLike | None = None,
        cohort_valid: pd.DataFrame | None = None,
        event_col: str = COL_EVENT,
        time_col: str = COL_TIME_TO_EVENT,
    ) -> "HorizonBank":
        """
        cohort / cohort_valid 需含随访结局列（event, time_to_event_days）。
        X 与 cohort 逐行对齐；验证集必须来自防泄露切分（时间在后、患者不重叠），
        这一点由调用方用 leakage 工具保证并 assert —— bank 不重复切分，
        因为它看不到全量队列，重复切分只会制造第二套口径。
        """
        if (X_valid is None) != (cohort_valid is None):
            raise ValueError("X_valid 与 cohort_valid 必须同时提供或同时省略")

        for name, days in self.horizons:
            Xh = _x_for(X, name)
            if len(Xh) != len(cohort):
                raise ValueError(f"[{name}] 特征表({len(Xh)})与队列({len(cohort)})行数不一致")

            y, stats = build_horizon_label(cohort, days, event_col, time_col, horizon_name=name)
            self.label_stats[name] = stats
            m = usable_mask(y).to_numpy()
            if stats.n_pos < 30:
                raise ValueError(
                    f"[{name}] 训练集可用阳性仅 {stats.n_pos} 例（<30）。"
                    "该时程无法训练出可信模型，请补充数据或暂缓该时程上线。"
                )

            eval_set = None
            if X_valid is not None:
                Xvh = _x_for(X_valid, name)
                yv, vstats = build_horizon_label(
                    cohort_valid, days, event_col, time_col, horizon_name=f"{name}-valid"
                )
                mv = usable_mask(yv).to_numpy()
                if vstats.n_pos < 10:
                    logger.warning(
                        "[%s] 验证集可用阳性仅 %d 例，早停/校准会很不稳定。", name, vstats.n_pos
                    )
                eval_set = (Xvh.loc[mv], yv.loc[mv])

            cfg = self.per_horizon_config.get(name, replace(self.base_config))
            logger.info("===== 训练时程 %s (%d 天) =====", name, days)
            model = LGBMRiskModel(cfg)
            model.fit(Xh.loc[m], y.loc[m], manifest, eval_set=eval_set)
            self.models[name] = model
        return self

    # ------------------------------------------------------------------
    def predict_risk(self, X: XLike, calibrated: bool = True) -> pd.DataFrame:
        """输出 DataFrame[risk_1y, risk_3y, risk_5y]（列序按时程从短到长）。"""
        if not self.models:
            raise RuntimeError("HorizonBank 尚未训练/加载。")
        cols: dict[str, np.ndarray] = {}
        index = None
        for name, _days in self.horizons:
            Xh = _x_for(X, name)
            index = Xh.index if index is None else index
            cols[f"risk_{name}"] = self.models[name].predict_risk(Xh, calibrated=calibrated)
        out = pd.DataFrame(cols, index=index)
        if self.enforce_monotonic:
            out = self._enforce_monotonic(out)
        return out

    def _enforce_monotonic(self, risks: pd.DataFrame) -> pd.DataFrame:
        arr = risks.to_numpy(dtype=float)
        fixed = np.maximum.accumulate(arr, axis=1)
        delta = fixed - arr
        touched = (delta > 1e-12).any(axis=1)
        if touched.any():
            frac = float(touched.mean())
            mean_fix = float(delta[touched].max(axis=1).mean())
            msg = (
                f"时程单调性修正: {touched.sum()}/{len(risks)} 样本 "
                f"({frac:.1%}) 出现交叉，平均修正幅度 {mean_fix:.4f}"
            )
            if frac > 0.05 and mean_fix > 0.02:
                logger.warning("%s —— 触发面过大，请排查各时程模型质量，勿依赖修正兜底！", msg)
            else:
                logger.info(msg)
        return pd.DataFrame(fixed, index=risks.index, columns=risks.columns)

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        if not self.models:
            raise RuntimeError("HorizonBank 尚未训练，无法保存。")
        d = Path(path)
        d.mkdir(parents=True, exist_ok=True)
        for name, model in self.models.items():
            model.save(d / name)
        meta = {
            "horizons": [[n, days] for n, days in self.horizons],
            "enforce_monotonic": self.enforce_monotonic,
            "label_stats": {
                n: {
                    "n_total": s.n_total, "n_pos": s.n_pos,
                    "n_neg": s.n_neg, "n_censored": s.n_censored,
                }
                for n, s in self.label_stats.items()
            },
        }
        with (d / _BANK_META).open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info("HorizonBank 已保存: %s (%d 个时程)", d, len(self.models))
        return d

    @classmethod
    def load(cls, path: str | Path) -> "HorizonBank":
        d = Path(path)
        with (d / _BANK_META).open("r", encoding="utf-8") as f:
            meta = json.load(f)
        bank = cls(
            horizons=tuple((n, int(days)) for n, days in meta["horizons"]),
            enforce_monotonic=bool(meta.get("enforce_monotonic", True)),
        )
        for name, _ in bank.horizons:
            bank.models[name] = LGBMRiskModel.load(d / name)
        logger.info("HorizonBank 已加载: %s (%d 个时程)", d, len(bank.models))
        return bank
