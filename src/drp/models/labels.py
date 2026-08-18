"""
结局标签构建（规范 1.1 / 3.3 的地基）。

规范要求数据"必须带时间维度标签：1年/3年/5年发病、进展、并发症结局"，
模型层输出"1年/3年/5年 病情进展风险概率"。本模块负责把随访原始信息
（是否发生结局 event + 随访/发生时长 time_to_event_days）转换成
每个预测时程的二分类标签。

【为什么这一步必须单独成模块，且规则不容许工程师自由发挥】

随访数据天然存在"删失"（censoring）：一个人随访到第 2 年就失联了、
没发病 —— 你并不知道他第 3 年会不会发病。删失处理错了，后面模型再好
标签都是错的，这是比任何超参更致命的正确性问题。

三条铁律（build_horizon_label 的全部逻辑）：

    1. event=1 且 time <= H          -> y = 1   （H 年内确实发生了）
    2. time > H （无论 event）        -> y = 0   （随访满 H 年且期内未发生；
                                                  哪怕之后发生了，对"H年内风险"
                                                  这个问题答案也是 0）
    3. event=0 且 time <= H          -> y = NaN （删失：H 年内没发生，但也
                                                  没随访满 H 年，真实答案未知，
                                                  必须从该时程的训练/评估中剔除）

最常见的两个错误做法，以及为什么绝对禁止：

  【错误 A】把删失样本当 0
    删失的人里其实有一部分后来发病了，全记 0 等于系统性把阳性标成阴性。
    后果：模型整体低估风险，且低估幅度随删失率上升 —— 而删失率高的
    恰恰是随访时间短的新样本，正是线上最像的人群。线下 AUC 看不出问题
    （排序关系还在），但概率标定整体偏低，风险分层直接失真。

  【错误 B】把删失样本直接从队列里删掉（而不是逐时程剔除）
    1 年时程里，随访 2 年的删失样本是完全合法的阴性样本（time>365）。
    按 5 年口径一刀切删人，会白扔大量短时程的有效样本。
    正确做法是【每个时程各自算一遍标签、各自取 usable 掩码】，
    这正是本模块按时程逐个输出的原因。

进阶说明（写给算法负责人）：逐时程剔除删失在删失与风险无关
（non-informative censoring）时是无偏的。若怀疑删失与风险相关
（例如高危者更容易脱落随访），标准解法是 IPCW 加权或直接用生存模型
（survival.py 的 Cox-PH 天然处理删失，不丢任何样本）——
这也是规范 3.1 把 Cox-PH 列进模型栈的原因之一。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 随访结局标准列名。与 data/constants.py 的长表列名同级，全平台统一。
COL_EVENT = "event"
COL_TIME_TO_EVENT = "time_to_event_days"

#: 平台标准预测时程（规范 3.3：1年/3年/5年）
DEFAULT_HORIZONS: tuple[tuple[str, int], ...] = (
    ("1y", 365),
    ("3y", 1095),
    ("5y", 1825),
)


@dataclass
class LabelStats:
    """单个时程的标签统计。全链路日志（规范 4.2）与训练报告都要记录。"""

    horizon_name: str
    horizon_days: int
    n_total: int = 0
    n_pos: int = 0
    n_neg: int = 0
    n_censored: int = 0

    @property
    def n_usable(self) -> int:
        return self.n_pos + self.n_neg

    @property
    def pos_rate(self) -> float:
        return self.n_pos / self.n_usable if self.n_usable else float("nan")

    @property
    def censor_rate(self) -> float:
        return self.n_censored / self.n_total if self.n_total else float("nan")

    def summary(self) -> str:
        return (
            f"[{self.horizon_name}/{self.horizon_days}d] "
            f"可用 {self.n_usable}/{self.n_total} "
            f"(阳性 {self.n_pos}, 阳性率 {self.pos_rate:.2%}, "
            f"删失剔除 {self.n_censored}, 删失率 {self.censor_rate:.1%})"
        )


def check_survival_columns(
    cohort: pd.DataFrame,
    event_col: str = COL_EVENT,
    time_col: str = COL_TIME_TO_EVENT,
) -> None:
    """
    随访结局列的强校验。训练入口必须先调用，坏数据在这里就地拦截，
    不允许带病进入标签构建（规范 4.1"上游错一个数据，下游模型全错"）。
    """
    missing = [c for c in (event_col, time_col) if c not in cohort.columns]
    if missing:
        raise ValueError(
            f"队列表缺少随访结局列: {missing}。"
            f"需要 {event_col}(0/1) 与 {time_col}(索引日期到结局/末次随访的天数)。"
        )

    ev = pd.to_numeric(cohort[event_col], errors="coerce")
    if ev.isna().any():
        raise ValueError(f"{event_col} 列存在无法解析的值（应为 0/1）")
    bad_ev = ~ev.isin([0, 1])
    if bad_ev.any():
        raise ValueError(
            f"{event_col} 列存在 0/1 之外的值: {sorted(ev[bad_ev].unique().tolist())[:5]}"
        )

    tt = pd.to_numeric(cohort[time_col], errors="coerce")
    if tt.isna().any():
        raise ValueError(f"{time_col} 列存在缺失/无法解析的值")
    if (tt <= 0).any():
        n = int((tt <= 0).sum())
        raise ValueError(
            f"{time_col} 存在 {n} 条非正值。随访时长必须 > 0；"
            "time<=0 通常意味着索引日期晚于结局日期 —— 这是队列构建阶段的"
            "时间对齐 bug，必须回上游修，禁止在标签层悄悄丢掉。"
        )


def build_horizon_label(
    cohort: pd.DataFrame,
    horizon_days: int,
    event_col: str = COL_EVENT,
    time_col: str = COL_TIME_TO_EVENT,
    horizon_name: str | None = None,
) -> tuple[pd.Series, LabelStats]:
    """
    构建单一时程的二分类标签。

    返回
    ----
    y : float Series，与 cohort 等长同序。1.0 / 0.0 / NaN（删失，需剔除）。
        刻意用 float+NaN 而不是直接删行 —— 保持与特征表逐行对齐，
        由调用方用 usable_mask() 显式取子集，杜绝静默错位。
    stats : LabelStats
    """
    check_survival_columns(cohort, event_col, time_col)

    ev = pd.to_numeric(cohort[event_col], errors="coerce").to_numpy()
    tt = pd.to_numeric(cohort[time_col], errors="coerce").to_numpy(dtype=float)

    y = np.full(len(cohort), np.nan, dtype=float)
    y[(ev == 1) & (tt <= horizon_days)] = 1.0  # 铁律 1
    y[tt > horizon_days] = 0.0                 # 铁律 2（覆盖 event 任意取值）
    # 铁律 3：event=0 且 tt<=H 保持 NaN

    stats = LabelStats(
        horizon_name=horizon_name or f"{horizon_days}d",
        horizon_days=horizon_days,
        n_total=len(y),
        n_pos=int(np.nansum(y == 1.0)),
        n_neg=int(np.nansum(y == 0.0)),
        n_censored=int(np.isnan(y).sum()),
    )
    logger.info("标签构建 %s", stats.summary())

    if stats.n_usable and stats.censor_rate > 0.5:
        logger.warning(
            "[%s] 删失率 %.1f%% 过高：过半样本随访不满 %d 天。"
            "该时程的可用样本严重缩水，建议改用 Cox-PH（survival.py）"
            "以免浪费删失样本的部分信息。",
            stats.horizon_name,
            stats.censor_rate * 100,
            horizon_days,
        )
    return pd.Series(y, index=cohort.index, name=f"y_{stats.horizon_name}"), stats


def usable_mask(y: pd.Series) -> pd.Series:
    """该时程可参与训练/评估的样本掩码（非删失）。"""
    return y.notna()


@dataclass
class HorizonLabelSet:
    """全部时程的标签集合，供 HorizonBank 与验证协议使用。"""

    labels: dict[str, pd.Series] = field(default_factory=dict)
    stats: dict[str, LabelStats] = field(default_factory=dict)
    horizons: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return "\n".join(self.stats[k].summary() for k in self.horizons)


def build_all_horizon_labels(
    cohort: pd.DataFrame,
    horizons: tuple[tuple[str, int], ...] = DEFAULT_HORIZONS,
    event_col: str = COL_EVENT,
    time_col: str = COL_TIME_TO_EVENT,
) -> HorizonLabelSet:
    """一次性构建所有时程标签。各时程的删失剔除互相独立（见模块 docstring 错误 B）。"""
    out = HorizonLabelSet()
    for name, days in horizons:
        y, st = build_horizon_label(cohort, days, event_col, time_col, horizon_name=name)
        out.labels[name] = y
        out.stats[name] = st
        out.horizons[name] = days
    return out
