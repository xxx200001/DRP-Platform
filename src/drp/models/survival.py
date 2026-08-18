"""
Cox-PH 生存分析模型（规范 3.1 模型栈第二成员）。

【为什么在 LightGBM 之外还必须有 Cox-PH —— 两者不是二选一，是分工】

1. 删失样本全利用
   labels.py 的时程二分类必须剔除删失样本（5 年时程删失率往往 30%+）。
   Cox-PH 通过偏似然把删失样本"活到几时"的部分信息也用上，
   在随访不满的新平台冷启动期，这是不可放弃的样本量。

2. 一个模型出全时间轴
   LightGBM 时程库要 1/3/5 年三个模型；Cox 一次拟合给出任意 t 的
   S(t)，1-S(t) 即累计风险，天然满足时程单调性（风险随 t 只增不减）。

3. 医学审查的通用语言
   风险比（HR）与 95% CI 是临床论文与院内评审的标准表达。
   平台要过医疗合作方的评审，必须能拿出这张表。

分工结论（写进技术方案的口径）：
   LightGBM 时程库 = 精度主力（非线性、交互、NaN 原生）；
   Cox-PH = 删失利用 + 全时间轴 + 医学可审查的基准与交叉校验。
   两者 C-index/AUC 差距过大（>0.05）本身就是重要诊断信号 ——
   通常意味着强非线性效应或数据问题，验证协议（批次2）会自动比对。

【工程取舍，改代码前必读】

a) 依赖 lifelines，不自研求解器。
   Cox 偏似然的 Newton 迭代、Efron 结平处理、稳健方差，每一处都是
   统计正确性深坑。医疗产品里手搓生存模型属于不可接受的风险，
   lifelines 是该领域事实标准。环境未安装时 fit 阶段明确报错并给出
   安装命令 —— 绝不静默降级成别的模型（规范 3.1 禁止乱换）。

b) Cox 不吃 NaN，必须显式处理缺失 —— 但只允许"中位数 + 已有三态列"。
   中位数（只在训练集上算，FitGuard 声明）填充数值列；"查没查过"
   的信号不丢，因为特征表里本来就有 _status 三态列一起进模型。
   这是全库唯一允许填充的位置，且填充值随模型持久化，推理期复用。

c) 特征必须做子集/正则。
   数百个共线特征直接进 Cox 会病态（方差爆炸、系数不可解释）。
   默认 L2 penalizer=0.1，并自动剔除高缺失(>30%)与近零方差特征；
   正式使用时建议显式传 feature_subset（如各指标的 _value/_z_ref
   加人口学），而不是全量特征硬灌。
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..validation.leakage import FitGuard
from .labels import COL_EVENT, COL_TIME_TO_EVENT, check_survival_columns

logger = logging.getLogger(__name__)

_DUR = "__duration__"
_EVT = "__event__"


def _try_import_lifelines():
    try:
        from lifelines import CoxPHFitter  # noqa: PLC0415

        return CoxPHFitter
    except ImportError:
        return None


@dataclass
class CoxConfig:
    penalizer: float = 0.1
    l1_ratio: float = 0.0
    feature_subset: list[str] | None = None
    max_missing_rate: float = 0.30
    """特征缺失率超过该值即剔除。Cox 需要填充，缺失越多填充失真越大，
    宁可不要这个特征 —— 树模型侧（NaN 原生）仍会用它，信息不丢。"""
    min_variance: float = 1e-10
    robust: bool = False  # 稳健(三明治)方差；同患者多样本时建议开

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CoxFitReport:
    n_samples: int = 0
    n_events: int = 0
    n_features_in: int = 0
    n_features_used: int = 0
    dropped_missing: list[str] = field(default_factory=list)
    dropped_constant: list[str] = field(default_factory=list)
    concordance: float = float("nan")

    def summary(self) -> str:
        return (
            f"Cox-PH 拟合完成: n={self.n_samples} 事件={self.n_events} "
            f"特征 {self.n_features_used}/{self.n_features_in} "
            f"(剔除高缺失 {len(self.dropped_missing)}、近常量 {len(self.dropped_constant)}) "
            f"训练集 C-index={self.concordance:.4f}"
        )


class CoxPHModel:
    """
    用法::

        cox = CoxPHModel(CoxConfig(feature_subset=[...]))
        cox.fit(X_tr, cohort_tr, train_idx=split.train_idx)   # FitGuard 声明
        risk = cox.predict_risk_at(X_new, days=[365, 1095, 1825])
        cox.hazard_ratios()        # HR + 95% CI，给医学评审
    """

    def __init__(self, config: CoxConfig | None = None):
        self.config = config or CoxConfig()
        self.fitter_ = None
        self.features_: list[str] | None = None
        self.medians_: pd.Series | None = None
        self.report_: CoxFitReport | None = None

    # ------------------------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame,
        cohort: pd.DataFrame,
        event_col: str = COL_EVENT,
        time_col: str = COL_TIME_TO_EVENT,
        train_idx: np.ndarray | None = None,
    ) -> "CoxPHModel":
        """
        X 与 cohort 必须逐行对齐（FeaturePipeline 的输出天然满足）。
        train_idx：切分后的训练索引，仅用于 FitGuard 泄露声明；
        传 None 时按"X 已是纯训练集"处理并记告警。
        """
        CoxPHFitter = _try_import_lifelines()
        if CoxPHFitter is None:
            raise RuntimeError(
                "Cox-PH 需要 lifelines：pip install lifelines。"
                "规范 3.1 模型栈固定，未安装时不做任何静默替代。"
            )
        check_survival_columns(cohort, event_col, time_col)
        if len(X) != len(cohort):
            raise ValueError(f"特征表({len(X)})与队列表({len(cohort)})行数不一致")

        rep = CoxFitReport(n_samples=len(X))

        # ---- 特征选择 ----
        cand = self.config.feature_subset or [
            c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])
        ]
        missing_cols = [c for c in cand if c not in X.columns]
        if missing_cols:
            raise ValueError(f"feature_subset 中不存在的列: {missing_cols[:5]}")
        rep.n_features_in = len(cand)

        sub = X[cand]
        miss = sub.isna().mean()
        rep.dropped_missing = miss[miss > self.config.max_missing_rate].index.tolist()
        keep = [c for c in cand if c not in set(rep.dropped_missing)]
        var = sub[keep].var(skipna=True)
        rep.dropped_constant = var[var.fillna(0) < self.config.min_variance].index.tolist()
        keep = [c for c in keep if c not in set(rep.dropped_constant)]
        if not keep:
            raise ValueError("特征筛选后为空。请检查 feature_subset 或放宽 max_missing_rate。")
        self.features_ = keep
        rep.n_features_used = len(keep)

        # ---- 缺失填充：只在训练集上算中位数（泄露 4 防线） ----
        with FitGuard("cox_median_impute") as g:
            g.declare_fit_data(
                np.asarray(train_idx) if train_idx is not None else np.arange(len(X))
            )
            if train_idx is None:
                logger.warning(
                    "CoxPHModel.fit 未传 train_idx：默认整个 X 就是训练集。"
                    "若 X 实为全量数据，中位数填充已构成预处理泄露！"
                )
            self.medians_ = X[keep].median(skipna=True)

        df = X[keep].fillna(self.medians_).copy()
        df[_DUR] = pd.to_numeric(cohort[time_col], errors="coerce").to_numpy(dtype=float)
        df[_EVT] = pd.to_numeric(cohort[event_col], errors="coerce").to_numpy(dtype=float)
        rep.n_events = int(df[_EVT].sum())
        if rep.n_events < 30:
            logger.warning(
                "事件数仅 %d（经验底线：每特征 ≥10 事件）。当前特征数 %d，"
                "Cox 系数将极不稳定，请缩小 feature_subset 或加大 penalizer。",
                rep.n_events, len(keep),
            )

        self.fitter_ = CoxPHFitter(
            penalizer=self.config.penalizer, l1_ratio=self.config.l1_ratio
        )
        self.fitter_.fit(
            df, duration_col=_DUR, event_col=_EVT, robust=self.config.robust
        )
        rep.concordance = float(self.fitter_.concordance_index_)
        self.report_ = rep
        logger.info(rep.summary())
        return self

    # ------------------------------------------------------------------
    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.fitter_ is None or self.features_ is None:
            raise RuntimeError("Cox 模型尚未训练/加载。")
        out = pd.DataFrame(index=X.index)
        for c in self.features_:
            col = X[c] if c in X.columns else pd.Series(np.nan, index=X.index)
            out[c] = pd.to_numeric(col, errors="coerce")
        return out.fillna(self.medians_)

    def predict_risk_at(
        self, X: pd.DataFrame, days: list[int] = [365, 1095, 1825]
    ) -> pd.DataFrame:
        """各时间点累计风险 1 - S(t)。列名 risk_{d}d，天然随 t 单调不减。"""
        Xp = self._prepare(X)
        surv = self.fitter_.predict_survival_function(Xp, times=days)
        risk = (1.0 - surv.T).clip(0.0, 1.0)
        risk.columns = [f"risk_{int(d)}d" for d in days]
        risk.index = X.index
        return risk

    def predict_partial_hazard(self, X: pd.DataFrame) -> pd.Series:
        """相对风险（exp(Xβ)），用于排序/与 LightGBM 交叉校验 C-index。"""
        ph = self.fitter_.predict_partial_hazard(self._prepare(X))
        return pd.Series(np.asarray(ph).ravel(), index=X.index, name="partial_hazard")

    def hazard_ratios(self) -> pd.DataFrame:
        """HR + 95% CI + p 值表（医学评审口径），按 |ln HR| 排序。"""
        if self.fitter_ is None:
            raise RuntimeError("Cox 模型尚未训练/加载。")
        s = self.fitter_.summary
        out = pd.DataFrame(
            {
                "HR": s["exp(coef)"],
                "HR_ci_low": s["exp(coef) lower 95%"],
                "HR_ci_high": s["exp(coef) upper 95%"],
                "p": s["p"],
            }
        )
        return out.reindex(out["HR"].apply(lambda h: abs(np.log(h))).sort_values(ascending=False).index)

    @property
    def concordance_(self) -> float:
        return float(self.fitter_.concordance_index_) if self.fitter_ is not None else float("nan")

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        if self.fitter_ is None:
            raise RuntimeError("模型尚未训练，无法保存。")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "config": self.config.to_dict(),
                "fitter": self.fitter_,
                "features": self.features_,
                "medians": self.medians_,
                "report": self.report_,
            },
            p,
        )
        logger.info("Cox 模型已保存: %s", p)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "CoxPHModel":
        blob = joblib.load(path)
        m = cls(CoxConfig(**blob["config"]))
        m.fitter_ = blob["fitter"]
        m.features_ = blob["features"]
        m.medians_ = blob["medians"]
        m.report_ = blob.get("report")
        return m
