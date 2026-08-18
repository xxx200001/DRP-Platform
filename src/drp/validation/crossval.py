"""
K 折交叉验证与时序滚动验证（规范 5 第二条）。

规范原话："必须三层验证：时间拆分验证 / K折交叉验证 / 外部独立数据集验证"。

【三层验证各自回答什么问题 —— 这决定了它们不能互相替代】

    时间拆分   → "上线之后到底行不行？"      唯一能暴露概念漂移的口径
    K 折       → "这个数字稳不稳、是不是运气好？"  唯一能给出方差的口径
    外部数据集 → "换一家医院还灵不灵？"        唯一能暴露过拟合到本院的口径

最常见的误用是拿 K 折的平均 AUC 对外汇报。K 折允许用未来数据训练、预测过去，
这在真实业务里永远不可能发生 —— 它算出来的数字系统性偏高（慢病数据上通常
高 0.02~0.05），而且概念漂移越严重偏得越多。所以本模块的 CVReport 刻意
【不提供】可直接对外的 AUC 字段，访问 headline_auc 会直接抛异常并说明原因。
对外数字只能来自 protocol.py 的时间拆分层。

【K 折在本平台的正确用途】
    1. 给时间拆分的那个单点数字配一个方差参考：折间标准差 > 0.04 时，
       时间拆分算出来的 0.85 和 0.81 其实没有区别，别拿它调超参。
    2. 超参搜索（此时必须用 patient_stratified_kfold，且搜索空间要小）。
    3. OOF 预测：K 折的测试集恰好不重不漏铺满全量样本，把各折预测拼起来
       能得到样本量最大、置信区间最窄的一份评估 —— 用来定风险分层切点
       比用单个测试集稳得多。

【本模块的两种折都必须是患者级的】
    直接用 sklearn 的 KFold/StratifiedKFold 会把同一患者的多次体检拆到
    训练和测试两侧（leakage.py 泄露 2），AUC 虚高 0.05~0.15。
    所以这里不提供任何行级切分入口。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.constants import COL_INDEX_DATE, COL_PATIENT_ID
from .leakage import LeakageError, SplitResult, assert_no_patient_overlap
from .metrics import MIN_TRUSTWORTHY_POSITIVES, BinaryMetrics, evaluate_binary

logger = logging.getLogger(__name__)

#: 折间 AUC 标准差告警线。超过这个值说明结果高度依赖切分运气。
FOLD_STD_WARN = 0.04


# ---------------------------------------------------------------------------
# 折的构造
# ---------------------------------------------------------------------------
def patient_stratified_kfold(
    cohort: pd.DataFrame,
    n_splits: int = 5,
    stratify_col: str | None = None,
    seed: int = 42,
) -> list[SplitResult]:
    """
    患者级分层 K 折。同一患者的所有样本必然整体落在同一折。

    stratify_col 传标签列时，在【患者级标签】（该患者是否出现过阳性结局）
    上做分层轮转分配，保证每折阳性率接近。医疗高危样本本来就少（规范 3.2），
    不分层时某一折的阳性数可能只有个位数，那一折的 AUC 纯属噪声，
    却会照样进平均值、把整体结论带偏。
    """
    if n_splits < 2:
        raise ValueError(f"n_splits 必须 >= 2，收到 {n_splits}")
    if COL_PATIENT_ID not in cohort.columns:
        raise ValueError(f"队列表缺少 {COL_PATIENT_ID} 列")

    rng = np.random.default_rng(seed)
    pat = cohort[[COL_PATIENT_ID]].copy()
    if stratify_col is not None:
        if stratify_col not in cohort.columns:
            raise ValueError(f"分层列 {stratify_col} 不存在")
        pat[stratify_col] = cohort[stratify_col].to_numpy()
        pat = pat.groupby(COL_PATIENT_ID, as_index=False, observed=True)[stratify_col].max()
        strata = [g[COL_PATIENT_ID].to_numpy() for _, g in pat.groupby(stratify_col, observed=True)]
    else:
        pat = pat.drop_duplicates()
        strata = [pat[COL_PATIENT_ID].to_numpy()]

    n_pat = int(pat[COL_PATIENT_ID].nunique())
    if n_pat < n_splits:
        raise ValueError(f"患者数({n_pat})少于折数({n_splits})，无法切分")

    assign: dict = {}
    for ids in strata:
        ids = ids.copy()
        rng.shuffle(ids)
        # 每层内轮转分配：既保证各折人数均衡，也保证各折阳性率均衡
        for k, pid in enumerate(ids):
            assign[pid] = k % n_splits

    fold_of_row = cohort[COL_PATIENT_ID].map(assign).to_numpy()
    folds: list[SplitResult] = []
    for k in range(n_splits):
        is_test = fold_of_row == k
        if not is_test.any():
            raise LeakageError(f"第 {k} 折为空，请减少 n_splits 或检查患者分布")
        folds.append(
            SplitResult(
                train_idx=np.where(~is_test)[0],
                test_idx=np.where(is_test)[0],
                strategy="patient_kfold",
                detail=f"fold {k + 1}/{n_splits} 测试患者数="
                f"{cohort.loc[is_test, COL_PATIENT_ID].nunique()}",
            )
        )
    logger.info("患者级 %d 折切分完成：共 %d 名患者，分层列=%s", n_splits, n_pat, stratify_col)
    return folds


def rolling_origin_folds(
    cohort: pd.DataFrame,
    n_splits: int = 4,
    gap_days: int = 30,
    min_train_frac: float = 0.5,
) -> list[SplitResult]:
    """
    时序滚动验证（rolling origin，扩展窗口）。

    这是 K 折在时序场景下的正确形态：每一折都只用【更早】的数据训练、
    预测【更晚】的一段时间。它同时给出方差和真实时序约束，代价是各折
    训练集大小不同（早期折样本少、指标偏低），因此折间方差天然大于
    普通 K 折 —— 解读时看趋势而不是看绝对值：
    若 AUC 随时间推移持续下降，说明存在概念漂移，模型必须定期重训
    （对应规范 3.2 的数据漂移监控与规范 6 的模型自动迭代）。

    min_train_frac : 第一折训练集至少覆盖多少比例的时间跨度样本。
    gap_days       : 训练末尾与测试开头的间隔，防边界泄露（同 time_based_split）。
    """
    if n_splits < 2:
        raise ValueError(f"n_splits 必须 >= 2，收到 {n_splits}")
    if COL_INDEX_DATE not in cohort.columns:
        raise ValueError(f"队列表缺少 {COL_INDEX_DATE} 列，无法做时序切分")

    dates = pd.to_datetime(cohort[COL_INDEX_DATE])
    qs = np.linspace(min_train_frac, 1.0, n_splits + 1)
    cutoffs = [pd.Timestamp(dates.quantile(q)) for q in qs]

    folds: list[SplitResult] = []
    for k in range(n_splits):
        lo, hi = cutoffs[k], cutoffs[k + 1]
        train_mask = dates <= lo
        test_mask = (dates > lo + pd.to_timedelta(gap_days, unit="D")) & (dates <= hi)

        # 跨界患者整体归训练侧（同 time_based_split 的处理）
        overlap = set(cohort.loc[test_mask, COL_PATIENT_ID]) & set(
            cohort.loc[train_mask, COL_PATIENT_ID]
        )
        if overlap:
            test_mask &= ~cohort[COL_PATIENT_ID].isin(overlap)

        tr = np.where(train_mask.to_numpy())[0]
        te = np.where(test_mask.to_numpy())[0]
        if te.size == 0 or tr.size == 0:
            logger.warning(
                "滚动验证第 %d 折为空（训练 %d / 测试 %d），已跳过。"
                "通常是 gap_days 过大或时间跨度太短。", k + 1, tr.size, te.size,
            )
            continue
        folds.append(
            SplitResult(
                train_idx=tr,
                test_idx=te,
                strategy="rolling_origin",
                detail=f"fold {k + 1}/{n_splits} 训练≤{lo.date()} 测试({lo.date()},{hi.date()}] "
                f"gap={gap_days}天 跨界患者剔除={len(overlap)}",
            )
        )
    if len(folds) < 2:
        raise ValueError(
            f"有效滚动折仅 {len(folds)} 个，无法给出方差参考。"
            "请缩小 gap_days、减少 n_splits，或确认队列时间跨度是否足够。"
        )
    return folds


def assert_fold_integrity(
    cohort: pd.DataFrame,
    folds: list[SplitResult],
    label_col: str | None = None,
    min_test_positives: int = 10,
    require_partition: bool = True,
) -> None:
    """
    折的强制体检。CI 必跑。

    require_partition：K 折要求各折测试集不重不漏铺满全量（滚动验证不满足，
    调用时传 False）。这条能抓住"折分配写错导致部分样本从未被验证过"这类
    静默 bug —— 它不会报错，只会让报告里的数字悄悄基于一个子集。
    """
    for k, f in enumerate(folds):
        assert_no_patient_overlap(cohort, f.train_idx, f.test_idx)
        if f.train_idx.size == 0 or f.test_idx.size == 0:
            raise LeakageError(f"第 {k + 1} 折存在空集合: {f!r}")
        if label_col is not None:
            n_pos = int(cohort.iloc[f.test_idx][label_col].sum())
            if n_pos < min_test_positives:
                raise LeakageError(
                    f"第 {k + 1} 折测试集阳性仅 {n_pos} 例（阈值 {min_test_positives}）。"
                    "该折的 AUC 是噪声，会污染整体均值。请减少折数或补充数据。"
                )
    if require_partition:
        all_test = np.concatenate([f.test_idx for f in folds])
        if all_test.size != np.unique(all_test).size:
            raise LeakageError("各折测试集存在重叠，不构成划分")
        if all_test.size != len(cohort):
            raise LeakageError(
                f"各折测试集合计 {all_test.size} 行，与队列 {len(cohort)} 行不符："
                "有样本从未被验证过，报告数字只覆盖了子集。"
            )


# ---------------------------------------------------------------------------
# 交叉验证执行
# ---------------------------------------------------------------------------
@dataclass
class CVReport:
    """交叉验证结果。刻意不提供可直接对外的 AUC —— 见模块说明。"""

    strategy: str = ""
    label: str = ""
    folds: list[BinaryMetrics] = field(default_factory=list)
    fold_details: list[str] = field(default_factory=list)
    oof: BinaryMetrics | None = None

    # ------------------------------------------------------------------
    def _vals(self, attr: str) -> np.ndarray:
        return np.asarray([getattr(m, attr) for m in self.folds], dtype=float)

    @property
    def mean_auc(self) -> float:
        return float(np.nanmean(self._vals("auc_roc"))) if self.folds else float("nan")

    @property
    def std_auc(self) -> float:
        return float(np.nanstd(self._vals("auc_roc"), ddof=1)) if len(self.folds) > 1 else 0.0

    @property
    def min_auc(self) -> float:
        return float(np.nanmin(self._vals("auc_roc"))) if self.folds else float("nan")

    @property
    def max_auc(self) -> float:
        return float(np.nanmax(self._vals("auc_roc"))) if self.folds else float("nan")

    @property
    def mean_auc_pr(self) -> float:
        return float(np.nanmean(self._vals("auc_pr"))) if self.folds else float("nan")

    @property
    def headline_auc(self) -> float:
        raise ValueError(
            "K 折的平均 AUC 不能作为对外口径（规范 5）。K 折允许用未来数据训练、"
            "预测过去，真实业务永远不会这样，它给出的数字系统性偏高。"
            "对外数字请用 protocol.py 时间拆分层的 AUC；本报告只用于看方差与稳定性。"
        )

    @property
    def unstable(self) -> bool:
        return self.std_auc > FOLD_STD_WARN

    def fold_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "fold": i + 1,
                    "n": m.n,
                    "n_pos": m.n_pos,
                    "pos_rate": m.pos_rate,
                    "auc_roc": m.auc_roc,
                    "auc_pr": m.auc_pr,
                    "ece": m.ece,
                }
                for i, m in enumerate(self.folds)
            ]
        )

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "label": self.label,
            "n_folds": len(self.folds),
            "mean_auc": self.mean_auc,
            "std_auc": self.std_auc,
            "min_auc": self.min_auc,
            "max_auc": self.max_auc,
            "mean_auc_pr": self.mean_auc_pr,
            "unstable": self.unstable,
            "folds": [m.to_dict() for m in self.folds],
            "fold_details": self.fold_details,
            "oof": self.oof.to_dict() if self.oof else None,
        }

    def summary(self) -> str:
        lines = [
            f"===== 交叉验证: {self.label or '(未命名)'} [{self.strategy}] =====",
            f"AUC-ROC  均值 {self.mean_auc:.4f}  标准差 {self.std_auc:.4f}  "
            f"范围 [{self.min_auc:.4f}, {self.max_auc:.4f}]",
            f"AUC-PR   均值 {self.mean_auc_pr:.4f}",
        ]
        tbl = self.fold_table().copy()
        for c in ("pos_rate",):
            tbl[c] = tbl[c].map(lambda v: f"{v:.2%}")
        for c in ("auc_roc", "auc_pr", "ece"):
            tbl[c] = tbl[c].map(lambda v: f"{v:.4f}")
        lines.append(tbl.to_string(index=False))
        if self.unstable:
            lines.append(
                f"⚠ 折间标准差 {self.std_auc:.4f} > {FOLD_STD_WARN}：结果高度依赖切分运气。"
                "此时用单次实验的 AUC 差异来选超参/选特征等同于掷骰子，"
                "请先补样本量或减少折数。"
            )
        if self.oof is not None:
            lines.append("")
            lines.append("--- OOF（各折预测拼接，样本量最大、CI 最窄，用于定分层切点）---")
            lines.append(self.oof.summary())
        lines.append(
            "注：以上数字仅用于评估稳定性，禁止作为对外精度口径（规范 5）。"
        )
        return "\n".join(lines)


def cross_validate(
    cohort: pd.DataFrame,
    X: pd.DataFrame,
    y,
    fit_predict,
    folds: list[SplitResult],
    label: str = "",
    strategy: str | None = None,
    groups_col: str = COL_PATIENT_ID,
    n_boot: int = 200,
    min_fold_positives: int = 10,
    compute_oof: bool = True,
    seed: int = 42,
) -> CVReport:
    """
    执行交叉验证。

    fit_predict : 可调用对象，签名 ``(X_train, y_train, X_test) -> 概率数组``。
        故意做成回调而不是接收模型实例：模型层的 fit 需要 manifest、eval_set、
        per-horizon 配置等，编排逻辑属于训练脚本；本模块只负责"切得对、算得对"。
        回调内部必须完成【全部】需要 fit 的步骤（含采样、校准），
        否则就是 leakage.py 的泄露 4（预处理泄露）。

    每一折都会重新调用 fit_predict，绝不复用上一折的模型。
    """
    y_arr = np.asarray(y, dtype=float).ravel()
    if np.isnan(y_arr).any():
        raise ValueError(
            f"标签含 {int(np.isnan(y_arr).sum())} 个 NaN（删失样本）。"
            "请先用 models.usable_mask 逐时程剔除，再把对齐后的 cohort/X/y 传进来"
            "（labels.py 铁律 3）。"
        )
    if not (len(cohort) == len(X) == y_arr.size):
        raise ValueError(
            f"cohort({len(cohort)}) / X({len(X)}) / y({y_arr.size}) 行数不一致"
        )

    groups_all = cohort[groups_col].to_numpy() if groups_col in cohort.columns else None
    rep = CVReport(strategy=strategy or (folds[0].strategy if folds else ""), label=label)

    oof_pred = np.full(y_arr.size, np.nan)
    for k, f in enumerate(folds):
        assert_no_patient_overlap(cohort, f.train_idx, f.test_idx)
        y_tr, y_te = y_arr[f.train_idx], y_arr[f.test_idx]
        n_pos_te = int(y_te.sum())
        if n_pos_te < min_fold_positives:
            raise ValueError(
                f"[{label}] 第 {k + 1} 折测试集阳性仅 {n_pos_te} 例"
                f"（阈值 {min_fold_positives}），该折指标是噪声，拒绝执行。"
            )

        logger.info(
            "[%s] 第 %d/%d 折: 训练 %d (阳性 %.2f%%) / 测试 %d (阳性 %.2f%%) | %s",
            label or "cv", k + 1, len(folds), f.train_idx.size, 100 * y_tr.mean(),
            f.test_idx.size, 100 * y_te.mean(), f.detail,
        )
        p_te = np.asarray(
            fit_predict(X.iloc[f.train_idx], y_tr, X.iloc[f.test_idx]), dtype=float
        ).ravel()
        if p_te.size != f.test_idx.size:
            raise ValueError(
                f"第 {k + 1} 折 fit_predict 返回 {p_te.size} 个预测，"
                f"与测试集 {f.test_idx.size} 行不符"
            )
        oof_pred[f.test_idx] = p_te

        m = evaluate_binary(
            y_te,
            p_te,
            label=f"{label} fold{k + 1}",
            groups=groups_all[f.test_idx] if groups_all is not None else None,
            n_boot=n_boot,
            seed=seed + k,
        )
        rep.folds.append(m)
        rep.fold_details.append(f.detail)

    if rep.unstable:
        logger.warning(
            "[%s] 折间 AUC 标准差 %.4f 超过告警线 %.2f：结论不稳定，"
            "请勿据此调参或对外汇报。", label or "cv", rep.std_auc, FOLD_STD_WARN,
        )

    if compute_oof:
        covered = ~np.isnan(oof_pred)
        n_pos_oof = int(y_arr[covered].sum())
        if covered.sum() >= 2 and n_pos_oof >= MIN_TRUSTWORTHY_POSITIVES:
            rep.oof = evaluate_binary(
                y_arr[covered],
                oof_pred[covered],
                label=f"{label} OOF",
                groups=groups_all[covered] if groups_all is not None else None,
                n_boot=n_boot,
                seed=seed + 999,
            )
        else:
            logger.info(
                "[%s] OOF 覆盖 %d 行、阳性 %d 例，样本不足以给出可信 OOF 指标，已跳过。",
                label or "cv", int(covered.sum()), n_pos_oof,
            )
    return rep
