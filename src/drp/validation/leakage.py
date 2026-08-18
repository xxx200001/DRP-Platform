"""
数据泄露守卫（规范 5：杜绝虚高指标）。

规范原话："禁止简单随机划分训练/测试集（会数据泄露、假高精度）"。
这个模块把"禁止"变成代码层面的硬约束，而不是靠人自觉。

医疗时序预测里最常见、最致命的四类泄露，本模块逐条拦截：

【泄露 1】未来特征
    用索引日期之后的检验结果构造特征。最典型：直接对全表做 groupby.mean()。
    症状：线下 AUC 0.95，线上 0.6。
    拦截：assert_no_future_records / as_of_filter

【泄露 2】同患者跨集
    同一患者的多次体检被随机分到训练集和测试集。模型记住了这个人，不是学到规律。
    症状：AUC 虚高 0.05~0.15，且随患者复诊次数增加而加剧。
    拦截：patient_level_split / assert_no_patient_overlap

【泄露 3】时间穿越
    用未来时间段的数据训练、过去时间段测试。真实业务永远是"用过去预测未来"。
    症状：线下正常，模型上线后随时间推移持续掉点（因为学到的是过时的分布）。
    拦截：time_based_split

【泄露 4】预处理泄露
    在切分前对全量数据做 fit（标准化、编码、特征选择、采样）。
    症状：AUC 虚高 0.01~0.05，很隐蔽，最难发现。
    拦截：FitGuard 上下文管理器 + 强制 fit_on_train_only 约定

用法示例见 examples/ 目录。所有训练脚本必须在切分后立刻调用
`assert_split_integrity`，这是 CI 里的强制检查项。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.constants import COL_INDEX_DATE, COL_MEASURED_AT, COL_PATIENT_ID

logger = logging.getLogger(__name__)


class LeakageError(AssertionError):
    """泄露检测失败。这是训练流程的硬中断，不允许 catch 后继续。"""


# ---------------------------------------------------------------------------
# 泄露 1：未来特征
# ---------------------------------------------------------------------------
def as_of_filter(
    records: pd.DataFrame,
    cohort: pd.DataFrame,
    lookback_days: int | None = None,
    blanking_days: int = 0,
) -> pd.DataFrame:
    """
    as-of 时间对齐：只保留每个患者【索引日期之前】的检验记录。

    参数
    ----
    blanking_days : 空白期。索引日期前 N 天的数据也一并剔除。

        为什么需要空白期 —— 这是很多人忽略但极其重要的一点：
        患者在确诊前夕往往已经出现症状并被密集检查，这段时间的检验数据
        携带了"即将确诊"的强信号。模型学到的是"最近查得勤=要出事"，
        而不是真正的早期风险规律。这种模型在真正需要它的场景（无症状人群
        早筛）上完全失效。

        建议值：1年预测用 30 天，3/5年预测用 90 天。

    lookback_days : 回溯窗口。None 表示不限制，用全部历史。
    """
    if COL_INDEX_DATE not in cohort.columns:
        raise ValueError(f"队列表缺少 {COL_INDEX_DATE} 列，无法做 as-of 对齐")

    idx = cohort[[COL_PATIENT_ID, COL_INDEX_DATE]].drop_duplicates()
    idx[COL_INDEX_DATE] = pd.to_datetime(idx[COL_INDEX_DATE])

    merged = records.merge(idx, on=COL_PATIENT_ID, how="inner")
    merged[COL_MEASURED_AT] = pd.to_datetime(merged[COL_MEASURED_AT])

    cutoff = merged[COL_INDEX_DATE] - pd.to_timedelta(blanking_days, unit="D")
    mask = merged[COL_MEASURED_AT] <= cutoff

    if lookback_days is not None:
        floor = merged[COL_INDEX_DATE] - pd.to_timedelta(lookback_days, unit="D")
        mask &= merged[COL_MEASURED_AT] >= floor

    n_dropped = int((~mask).sum())
    if n_dropped:
        logger.info(
            "as-of 对齐: 剔除 %d 条索引日期之后/窗口之外的记录 (blanking=%d天, lookback=%s)",
            n_dropped,
            blanking_days,
            lookback_days,
        )
    # 必须丢掉 index_date 辅助列：留着会在下游再次 merge 时产生
    # index_date_x / index_date_y 后缀冲突，且这是隐性携带标签期信息的列，
    # 不应该流入特征构造环节。
    out = merged[mask].drop(columns=[COL_INDEX_DATE])
    return out.reset_index(drop=True)


def assert_no_future_records(
    records: pd.DataFrame,
    cohort: pd.DataFrame,
    blanking_days: int = 0,
) -> None:
    """硬断言：特征表里不能含有索引日期之后的记录。"""
    idx = cohort[[COL_PATIENT_ID, COL_INDEX_DATE]].drop_duplicates()
    idx[COL_INDEX_DATE] = pd.to_datetime(idx[COL_INDEX_DATE])
    # 记录表若已携带 index_date（来自上游 merge），先丢掉再重新关联，
    # 避免 merge 产生 _x/_y 后缀导致断言静默失效。
    rec = records.drop(columns=[COL_INDEX_DATE], errors="ignore")
    m = rec.merge(idx, on=COL_PATIENT_ID, how="inner")
    m[COL_MEASURED_AT] = pd.to_datetime(m[COL_MEASURED_AT])
    cutoff = m[COL_INDEX_DATE] - pd.to_timedelta(blanking_days, unit="D")
    bad = m[COL_MEASURED_AT] > cutoff
    if bad.any():
        sample = m[bad].head(5)[[COL_PATIENT_ID, COL_MEASURED_AT, COL_INDEX_DATE]]
        raise LeakageError(
            f"检测到未来数据泄露: {int(bad.sum())} 条记录晚于索引日期(含{blanking_days}天空白期)。\n"
            f"样例:\n{sample.to_string(index=False)}\n"
            "请在特征构造前调用 as_of_filter()。"
        )


# ---------------------------------------------------------------------------
# 泄露 2 & 3：切分
# ---------------------------------------------------------------------------
@dataclass
class SplitResult:
    train_idx: np.ndarray
    test_idx: np.ndarray
    strategy: str
    detail: str = ""

    def __repr__(self) -> str:
        return f"<Split {self.strategy}: train={len(self.train_idx)} test={len(self.test_idx)}>"


def patient_level_split(
    cohort: pd.DataFrame,
    test_size: float = 0.2,
    stratify_col: str | None = None,
    seed: int = 42,
) -> SplitResult:
    """
    患者级随机切分。同一患者的所有样本必然落在同一侧。

    stratify_col 通常传标签列，保证正负样本比例一致 —— 医疗高危样本本来就少
    （规范 3.2），不分层会导致测试集正样本数太少，AUC 置信区间大到没意义。
    """
    rng = np.random.default_rng(seed)
    pat = cohort[[COL_PATIENT_ID]].copy()
    if stratify_col is not None:
        # 患者级标签取 max：只要该患者出现过阳性结局就算阳性患者
        pat[stratify_col] = cohort[stratify_col].to_numpy()
        pat = pat.groupby(COL_PATIENT_ID, as_index=False, observed=True)[stratify_col].max()
    else:
        pat = pat.drop_duplicates()

    test_patients: set = set()
    if stratify_col is not None:
        for _, grp in pat.groupby(stratify_col, observed=True):
            ids = grp[COL_PATIENT_ID].to_numpy()
            rng.shuffle(ids)
            k = max(1, int(round(len(ids) * test_size)))
            test_patients.update(ids[:k].tolist())
    else:
        ids = pat[COL_PATIENT_ID].to_numpy()
        rng.shuffle(ids)
        k = max(1, int(round(len(ids) * test_size)))
        test_patients.update(ids[:k].tolist())

    is_test = cohort[COL_PATIENT_ID].isin(test_patients).to_numpy()
    return SplitResult(
        train_idx=np.where(~is_test)[0],
        test_idx=np.where(is_test)[0],
        strategy="patient_level",
        detail=f"测试集患者数={len(test_patients)}/{pat[COL_PATIENT_ID].nunique()}",
    )


def time_based_split(
    cohort: pd.DataFrame,
    cutoff: str | pd.Timestamp | None = None,
    test_size: float = 0.2,
    gap_days: int = 0,
) -> SplitResult:
    """
    时间拆分验证（规范 5 第一条）：旧数据训练，新数据测试。

    这是【最接近线上真实场景】的验证方式，也是唯一能暴露"概念漂移"的方式。
    平台对外声称的 AUC 必须以这个口径为准，而不是 K 折的平均值。

    gap_days : 训练集末尾与测试集开头之间的间隔，进一步防止边界泄露。

    注意：本函数会自动把跨界患者整体划到训练侧，同时满足时间约束和患者约束。
    """
    dates = pd.to_datetime(cohort[COL_INDEX_DATE])
    if cutoff is None:
        cutoff = dates.quantile(1 - test_size)
    cutoff = pd.Timestamp(cutoff)

    test_mask = dates > cutoff + pd.to_timedelta(gap_days, unit="D")
    train_mask = dates <= cutoff

    # 跨界患者：既有 cutoff 前样本又有 cutoff 后样本 -> 全部归训练集，
    # 避免同一个人同时出现在两侧（泄露 2）
    test_patients = set(cohort.loc[test_mask, COL_PATIENT_ID])
    train_patients = set(cohort.loc[train_mask, COL_PATIENT_ID])
    overlap = test_patients & train_patients
    if overlap:
        logger.info("时间切分发现 %d 名跨界患者，其测试侧样本已剔除以避免患者级泄露", len(overlap))
        test_mask &= ~cohort[COL_PATIENT_ID].isin(overlap)

    return SplitResult(
        train_idx=np.where(train_mask.to_numpy())[0],
        test_idx=np.where(test_mask.to_numpy())[0],
        strategy="time_based",
        detail=f"cutoff={cutoff.date()} gap={gap_days}天 跨界患者剔除={len(overlap)}",
    )


def assert_no_patient_overlap(
    cohort: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray
) -> None:
    tr = set(cohort.iloc[train_idx][COL_PATIENT_ID])
    te = set(cohort.iloc[test_idx][COL_PATIENT_ID])
    inter = tr & te
    if inter:
        raise LeakageError(
            f"检测到患者级泄露: {len(inter)} 名患者同时出现在训练集和测试集。"
            f"样例: {list(inter)[:5]}。请改用 patient_level_split()。"
        )


def assert_split_integrity(
    cohort: pd.DataFrame,
    split: SplitResult,
    label_col: str | None = None,
    min_test_positives: int = 30,
) -> None:
    """
    切分后的强制体检。训练脚本必须调用，CI 里也要跑。
    """
    assert_no_patient_overlap(cohort, split.train_idx, split.test_idx)

    if len(split.test_idx) == 0 or len(split.train_idx) == 0:
        raise LeakageError(f"切分产生空集合: {split!r}")

    if label_col is not None:
        n_pos = int(cohort.iloc[split.test_idx][label_col].sum())
        if n_pos < min_test_positives:
            raise LeakageError(
                f"测试集正样本仅 {n_pos} 例（阈值 {min_test_positives}）。"
                "样本量不足时 AUC 的置信区间会宽到没有参考价值，"
                "此时报出的任何精度数字都不可信。请扩大测试集或补充数据。"
            )

    if split.strategy == "time_based":
        tr_max = pd.to_datetime(cohort.iloc[split.train_idx][COL_INDEX_DATE]).max()
        te_min = pd.to_datetime(cohort.iloc[split.test_idx][COL_INDEX_DATE]).min()
        if te_min <= tr_max:
            raise LeakageError(
                f"时间切分越界: 测试集最早索引日期 {te_min} 不晚于训练集最晚 {tr_max}"
            )

    logger.info("切分完整性检查通过: %r (%s)", split, split.detail)


# ---------------------------------------------------------------------------
# 泄露 4：预处理泄露
# ---------------------------------------------------------------------------
class FitGuard:
    """
    预处理泄露守卫。任何 fit 操作必须在这个上下文里声明数据来源。

    用法::

        with FitGuard("standard_scaler") as g:
            g.declare_fit_data(train_idx)      # 声明只用训练集
            scaler.fit(X[train_idx])
        X_all = scaler.transform(X)            # transform 可以用全量

    如果 declare_fit_data 传入的索引与测试集有交集，直接抛 LeakageError。
    """

    _current_test_idx: np.ndarray | None = None

    @classmethod
    def register_test_indices(cls, test_idx: np.ndarray) -> None:
        """训练脚本切分完成后立刻调用一次，之后全局生效。"""
        cls._current_test_idx = np.asarray(test_idx)

    def __init__(self, name: str):
        self.name = name
        self._declared = False

    def __enter__(self) -> "FitGuard":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None and not self._declared:
            raise LeakageError(
                f"FitGuard('{self.name}') 退出时未调用 declare_fit_data()。"
                "所有 fit 操作必须显式声明数据来源，否则无法保证没有预处理泄露。"
            )

    def declare_fit_data(self, idx: np.ndarray) -> None:
        self._declared = True
        if FitGuard._current_test_idx is None:
            logger.warning(
                "FitGuard('%s'): 尚未注册测试集索引，跳过校验。"
                "请在切分后调用 FitGuard.register_test_indices()。",
                self.name,
            )
            return
        inter = np.intersect1d(np.asarray(idx), FitGuard._current_test_idx)
        if inter.size:
            raise LeakageError(
                f"预处理泄露: '{self.name}' 的 fit 数据包含 {inter.size} 条测试集样本。"
                f"样例索引: {inter[:5].tolist()}"
            )
