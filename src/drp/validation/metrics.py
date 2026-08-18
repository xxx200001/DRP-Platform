"""
核心评估指标（规范 5）。

规范原话：
  - "核心评估指标（只看真实有用的）：AUC-ROC、AUC-PR、C-index（生存模型）、
     敏感度（严控漏诊）、特异度"
  - "禁止只看准确率"

本模块是平台唯一的指标口径。所有对外汇报的精度数字（包括规范 9 承诺的
"线上真实 AUC ≥ 0.82~0.88"）必须由这里产出，禁止各人各写一段
`roc_auc_score` 自己算 —— 口径不统一时，"AUC 0.86"这个数字本身就没有意义。

【设计决策逐条说明】

1. 每个指标都必须带置信区间，且 bootstrap 按【患者】重采样
   30 例阳性算出来的 AUC=0.86，其 95% CI 大约是 ±0.08 —— 也就是说真实值
   可能是 0.78。裸报一个 0.86 是误导，不是精度。
   更隐蔽的一点：同一患者的多次体检不是独立观测，按【行】做 bootstrap 会
   系统性低估方差，把置信区间做窄 —— 这是规范 5 要杜绝的"虚高"的另一种形态，
   只不过虚高的是可信度而不是均值。所以 groups 默认传 patient_id。

2. AUC-PR 必须与阳性率基线一起报，否则数字无法解读
   随机模型的 AUC-PR 等于阳性率。慢病阳性率 3%~15%（规范 3.2），
   AUC-PR=0.15 在阳性率 3% 时是 5 倍提升（很好），在阳性率 30% 时是灾难。
   同一个数字两种结论，所以本模块强制输出 auc_pr_baseline 与 pr_lift。
   为什么在不均衡场景 AUC-PR 比 AUC-ROC 更贴近产品体验：ROC 的 FPR 分母是
   海量阴性，多报几百个假阳性对 FPR 影响微乎其微，但对用户就是几百次
   无谓恐慌 + 几百份无谓检查。precision 直接回答"报警 100 次里几次是真的"。

3. 阈值绝不允许用 0.5
   校准良好的模型在 5% 阳性率下几乎不会输出 >0.5 的概率，用 0.5 卡阈值
   等于全判阴性 —— 敏感度 0，特异度 100%，"准确率 95%"。
   规范要求"敏感度（严控漏诊）"，所以本模块的阈值一律【按目标敏感度反解】
   （threshold_at_sensitivity），并强制同时报出该阈值下的特异度与报警率
   （alert_rate）：敏感度是买来的，代价必须同时摆在桌面上。

4. 校准误差（ECE / O:E）是独立验收项，不是可选项
   规范 6 要给用户展示 1/3/5 年概率【数值】、做四级风险分层。排序对但数值
   偏移的模型，AUC 再高，展示出去的每一个百分比都是错的。
   Brier 分数在低阳性率下天然很小（全预测 0.03 就能拿到 0.029），
   所以必须同时报 Brier skill score（相对"全预测阳性率"基线的提升），
   否则又是一个自欺指标。

5. 风险分层的【实际发生率单调性】是硬验收项
   规范 6 要做低/中/高/极高四级分层。分层表是唯一能让临床医生一眼判断
   模型好坏的东西：如果"高危"组的实际发生率没有高于"中危"组，这个分层
   就是错的，此时 AUC 多少都不能上线。risk_stratification_table 同时输出
   cum_capture（该层及以上捕获了多少阳性），这是产品定资源投放的直接依据。

6. 关于"准确率"
   本模块不提供 accuracy 字段。summary() 里会打印它，但强制并排打印
   "全判阴性基线"的准确率作为对照 —— 让任何人一眼看到这个指标为什么
   不能单独看。这是规范"禁止只看准确率"在代码层面的落地方式：
   不是不让算，是不让它单独出现。
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: 样本量下限。低于此值的评估结果只能看，不能作为上线依据（与
#: leakage.assert_split_integrity 的 min_test_positives 保持同一口径）。
MIN_TRUSTWORTHY_POSITIVES = 30

#: 平台标准风险分层分位点（规范 6 四级分层：低 / 中 / 高 / 极高）。
#: 50/80/95 对应人群占比 50% / 30% / 15% / 5%，按筛查资源投放的常规梯度设计。
DEFAULT_TIER_QUANTILES: tuple[float, ...] = (0.50, 0.80, 0.95)
DEFAULT_TIER_NAMES: tuple[str, ...] = ("低危", "中危", "高危", "极高危")


class MetricsError(ValueError):
    """指标输入非法。坏输入必须当场炸，不允许算出一个 nan 混进报告。"""


# ---------------------------------------------------------------------------
# 输入校验与基础工具
# ---------------------------------------------------------------------------
def _check_inputs(y, score) -> tuple[np.ndarray, np.ndarray]:
    y_arr = np.asarray(y, dtype=float).ravel()
    s_arr = np.asarray(score, dtype=float).ravel()

    if y_arr.size != s_arr.size:
        raise MetricsError(f"y 与 score 长度不一致: {y_arr.size} vs {s_arr.size}")
    if y_arr.size == 0:
        raise MetricsError("空样本无法评估")
    if np.isnan(y_arr).any():
        raise MetricsError(
            "y 含 NaN。删失样本必须在标签构建阶段用 usable_mask 剔除"
            "（labels.py 铁律 3），不允许带进评估。"
        )
    bad = ~np.isin(y_arr, (0.0, 1.0))
    if bad.any():
        raise MetricsError(f"y 必须是 0/1，发现 {int(bad.sum())} 个非法值")
    if np.isnan(s_arr).any():
        raise MetricsError(
            f"预测分数含 {int(np.isnan(s_arr).sum())} 个 NaN。"
            "NaN 会让 AUC 静默变成 nan 或被当成最小值排序，属于必须当场暴露的事故。"
        )
    return y_arr, s_arr


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """并列取平均秩。等价于 scipy.stats.rankdata(x, method='average')，全向量化。"""
    n = x.size
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    is_new = np.empty(n, dtype=bool)
    is_new[0] = True
    if n > 1:
        is_new[1:] = xs[1:] != xs[:-1]
    grp = np.cumsum(is_new) - 1
    start = np.flatnonzero(is_new)
    counts = np.diff(np.append(start, n))
    avg = start + (counts + 1) / 2.0  # 1-based 平均秩
    ranks = np.empty(n, dtype=float)
    ranks[order] = avg[grp]
    return ranks


def roc_auc(y, score) -> float:
    """
    AUC-ROC。用秩和公式实现（Mann-Whitney U），并列分数按 0.5 计。

    自己实现而不是直接调 sklearn 的原因：bootstrap 要跑几百次，秩和公式
    只需一次排序；更重要的是并列处理口径必须写死在平台内部，
    不能随第三方库版本变化 —— 上线数字的可复现性优先。
    """
    y_arr, s_arr = _check_inputs(y, score)
    n_pos = float(y_arr.sum())
    n_neg = float(y_arr.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _average_ranks(s_arr)
    return float((r[y_arr == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


@dataclass
class _Curve:
    """按分数降序、并列合并后的累计计数。ROC 与 PR 共用同一份计算。"""

    thresholds: np.ndarray
    tps: np.ndarray
    fps: np.ndarray
    n_pos: int
    n_neg: int


def _curve(y_arr: np.ndarray, s_arr: np.ndarray) -> _Curve:
    order = np.argsort(-s_arr, kind="mergesort")
    ys = y_arr[order]
    ss = s_arr[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1.0 - ys)
    # 并列分数必须合并到同一个阈值上：否则会凭空造出"能把并列样本分开"的
    # 操作点，报出实际达不到的敏感度。
    last = np.flatnonzero(np.append(ss[1:] != ss[:-1], True))
    return _Curve(
        thresholds=ss[last],
        tps=tp[last],
        fps=fp[last],
        n_pos=int(y_arr.sum()),
        n_neg=int(y_arr.size - y_arr.sum()),
    )


def average_precision(y, score) -> float:
    """AUC-PR（阶梯式平均精度，与 sklearn average_precision_score 同口径）。"""
    y_arr, s_arr = _check_inputs(y, score)
    if y_arr.sum() == 0:
        return float("nan")
    c = _curve(y_arr, s_arr)
    precision = c.tps / np.maximum(c.tps + c.fps, 1e-12)
    recall = c.tps / c.n_pos
    d_recall = np.diff(np.append(0.0, recall))
    return float(np.sum(d_recall * precision))


# ---------------------------------------------------------------------------
# 操作点（阈值 + 该阈值下的全部代价）
# ---------------------------------------------------------------------------
@dataclass
class OperatingPoint:
    """
    一个可上线的阈值，以及它的完整代价。

    只报敏感度不报报警率是耍流氓：敏感度 99% 但要报警 60% 的人群，
    临床上等于没有分层能力。四个数字必须捆绑出现。
    """

    label: str
    threshold: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    alert_rate: float
    n_flagged: int
    n_missed: int

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"[{self.label}] 阈值={self.threshold:.4f} "
            f"敏感度={self.sensitivity:.1%} 特异度={self.specificity:.1%} "
            f"PPV={self.ppv:.1%} 报警率={self.alert_rate:.1%} "
            f"漏诊={self.n_missed}例"
        )


def _point_at(c: _Curve, i: int, label: str) -> OperatingPoint:
    tp, fp = float(c.tps[i]), float(c.fps[i])
    n = c.n_pos + c.n_neg
    flagged = tp + fp
    fn = c.n_pos - tp
    tn = c.n_neg - fp
    return OperatingPoint(
        label=label,
        threshold=float(c.thresholds[i]),
        sensitivity=tp / c.n_pos if c.n_pos else float("nan"),
        specificity=tn / c.n_neg if c.n_neg else float("nan"),
        ppv=tp / flagged if flagged else float("nan"),
        npv=tn / (tn + fn) if (tn + fn) else float("nan"),
        alert_rate=flagged / n,
        n_flagged=int(flagged),
        n_missed=int(fn),
    )


def threshold_at_sensitivity(y, score, target: float = 0.90) -> OperatingPoint:
    """
    按目标敏感度反解阈值（规范 5"敏感度（严控漏诊）"的落地方式）。

    取【满足 sensitivity >= target 的最高阈值】—— 即在守住漏诊底线的前提下
    尽可能少报警。这是慢病早筛的标准定阈方式，也是本平台唯一允许的默认口径。
    """
    if not 0 < target <= 1:
        raise MetricsError(f"目标敏感度必须落在 (0,1]，收到 {target}")
    y_arr, s_arr = _check_inputs(y, score)
    c = _curve(y_arr, s_arr)
    if c.n_pos == 0:
        raise MetricsError("无阳性样本，无法按敏感度定阈值")
    sens = c.tps / c.n_pos
    idx = int(np.searchsorted(sens, target - 1e-12, side="left"))
    idx = min(idx, len(sens) - 1)
    op = _point_at(c, idx, label=f"敏感度≥{target:.0%}")

    # 并列概率会让阈值"跳档"：isotonic 校准的输出是阶梯函数，同一台阶上成百上千
    # 个样本概率完全相同，阈值无法落在台阶中间，只能整档跨过去。
    # 结果是敏感度远超目标（不是坏事），但特异度被连带拉低（是坏事，且很隐蔽：
    # 报告上写着"敏感度≥90%"，实际报警率却按 97% 敏感度的代价在付）。
    if op.sensitivity - target > 0.03:
        logger.warning(
            "目标敏感度 %.0f%% 的阈值实际给到 %.1f%%：预测概率存在大量并列值，"
            "阈值只能跳到下一个台阶，代价是特异度 %.1f%%、报警率 %.1f%%。"
            "若该操作点要上线，建议在模型层用未校准分数给校准输出打破并列"
            "（保持排序、消除台阶），或改用报警率口径定阈。",
            100 * target, 100 * op.sensitivity, 100 * op.specificity, 100 * op.alert_rate,
        )
    return op


def threshold_at_alert_rate(y, score, alert_rate: float = 0.10) -> OperatingPoint:
    """
    按可承受的报警率反解阈值。用于资源受限场景："我们只有能力随访 10% 的人，
    这 10% 能捞回多少阳性？"——这个问题的答案才是运营真正要的数字。
    """
    if not 0 < alert_rate <= 1:
        raise MetricsError(f"报警率必须落在 (0,1]，收到 {alert_rate}")
    y_arr, s_arr = _check_inputs(y, score)
    c = _curve(y_arr, s_arr)
    n = c.n_pos + c.n_neg
    rate = (c.tps + c.fps) / n
    idx = int(np.searchsorted(rate, alert_rate + 1e-12, side="left"))
    idx = min(idx, len(rate) - 1)
    return _point_at(c, idx, label=f"报警率≈{alert_rate:.0%}")


def youden_threshold(y, score) -> OperatingPoint:
    """Youden J（敏感度+特异度-1）最大点。只作参考：它默认漏诊与误报等价代价，
    而慢病早筛里漏诊的代价远高于误报，所以【不作为平台默认阈值】。"""
    y_arr, s_arr = _check_inputs(y, score)
    c = _curve(y_arr, s_arr)
    if c.n_pos == 0 or c.n_neg == 0:
        raise MetricsError("单一类别样本，无法计算 Youden 阈值")
    j = c.tps / c.n_pos - c.fps / c.n_neg
    return _point_at(c, int(np.argmax(j)), label="Youden-J(仅参考)")


# ---------------------------------------------------------------------------
# 校准
# ---------------------------------------------------------------------------
def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 区间。小样本分箱里 k/n±1.96*sqrt(p(1-p)/n) 会给出负数下界，不可用。"""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, center - half), min(1.0, center + half)


def calibration_table(y, prob, n_bins: int = 10, strategy: str = "quantile") -> pd.DataFrame:
    """
    校准表：每个分箱的【预测均值】vs【实际发生率】。

    strategy="quantile"（默认）按样本量等分，而不是按概率等宽 —— 低阳性率下
    概率高度集中在左端，等宽分箱会让 9 个箱空着、所有样本挤在第 1 箱，
    看不出任何东西。
    """
    y_arr, p_arr = _check_inputs(y, prob)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p_arr, np.linspace(0, 1, n_bins + 1)))
    elif strategy == "uniform":
        edges = np.linspace(p_arr.min(), p_arr.max(), n_bins + 1)
    else:
        raise MetricsError(f"未知分箱策略: {strategy}")
    if edges.size < 2:
        raise MetricsError("预测概率无变化（所有值相同），无法做校准分箱")

    idx = np.clip(np.searchsorted(edges, p_arr, side="right") - 1, 0, edges.size - 2)
    rows = []
    for b in range(edges.size - 1):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        k = int(y_arr[m].sum())
        lo, hi = _wilson_ci(k, n)
        pred = float(p_arr[m].mean())
        rows.append(
            {
                "bin": b,
                "range_lo": float(edges[b]),
                "range_hi": float(edges[b + 1]),
                "n": n,
                "n_pos": k,
                "pred_mean": pred,
                "obs_rate": k / n,
                "obs_lo": lo,
                "obs_hi": hi,
                "gap": k / n - pred,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(y, prob, n_bins: int = 10) -> float:
    """ECE：各分箱 |预测均值 - 实际发生率| 的样本量加权平均。"""
    tbl = calibration_table(y, prob, n_bins=n_bins)
    if tbl.empty:
        return float("nan")
    w = tbl["n"] / tbl["n"].sum()
    return float((w * tbl["gap"].abs()).sum())


def brier_scores(y, prob) -> tuple[float, float]:
    """
    返回 (brier, brier_skill_score)。

    BSS = 1 - brier / brier_baseline，baseline 是"所有人都预测总体阳性率"。
    BSS <= 0 意味着模型的概率数值还不如直接报人群平均值 —— 这种模型
    绝不能拿去做规范 6 的概率展示，哪怕它的 AUC 很好看（排序对≠数值对）。
    """
    y_arr, p_arr = _check_inputs(y, prob)
    brier = float(np.mean((p_arr - y_arr) ** 2))
    base = float(y_arr.mean())
    brier_base = float(np.mean((base - y_arr) ** 2))
    bss = 1.0 - brier / brier_base if brier_base > 0 else float("nan")
    return brier, bss


# ---------------------------------------------------------------------------
# 风险分层（规范 6 四级分层的验收依据）
# ---------------------------------------------------------------------------
def risk_stratification_table(
    y,
    prob,
    quantiles: tuple[float, ...] = DEFAULT_TIER_QUANTILES,
    tier_names: tuple[str, ...] = DEFAULT_TIER_NAMES,
    cutpoints: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    """
    风险分层表。每层输出：人数占比、预测均值、实际发生率（含 95%CI）、
    lift（相对总体阳性率的倍数）、cum_capture（该层及以上累计捕获的阳性占比）。

    cum_capture 是产品/运营最该看的一列：极高危 5% 的人群捕获了 40% 的阳性，
    意味着把随访资源压在这 5% 上的投入产出比是 8 倍。

    cutpoints 传入时按【绝对概率】切（上线后必须用固定阈值，不能每天按当日
    人群分位数漂移）；不传则按验证集分位数切，用于首次定阈。
    """
    y_arr, p_arr = _check_inputs(y, prob)
    if cutpoints is None:
        edges = np.quantile(p_arr, quantiles)
    else:
        edges = np.asarray(cutpoints, dtype=float)
    if len(tier_names) != len(edges) + 1:
        raise MetricsError(f"层名数量({len(tier_names)})必须比切点数({len(edges)})多 1")

    tier_idx = np.searchsorted(edges, p_arr, side="right")
    overall = float(y_arr.mean())
    n_total = y_arr.size
    n_pos_total = float(y_arr.sum())

    rows = []
    for t in range(len(tier_names)):
        m = tier_idx == t
        n = int(m.sum())
        k = int(y_arr[m].sum()) if n else 0
        lo, hi = _wilson_ci(k, n)
        # 该层及以上（更高危）累计捕获的阳性比例
        cum = float(y_arr[tier_idx >= t].sum()) / n_pos_total if n_pos_total else float("nan")
        rows.append(
            {
                "tier": tier_names[t],
                "n": n,
                "share": n / n_total,
                "pred_mean": float(p_arr[m].mean()) if n else float("nan"),
                "obs_rate": k / n if n else float("nan"),
                "obs_lo": lo,
                "obs_hi": hi,
                "lift": (k / n) / overall if n and overall > 0 else float("nan"),
                "cum_capture": cum,
            }
        )
    return pd.DataFrame(rows)


def stratification_violations(table: pd.DataFrame) -> list[str]:
    """
    检查分层单调性。返回违规描述列表（空列表 = 通过）。

    判定用【CI 是否真的反转】而不是点估计大小：小样本层的点估计本来就抖，
    对点估计做严格单调断言会让门禁频繁误伤。只有当高危层的发生率上界
    低于低危层的下界时，才算真的反了 —— 那是模型问题，不是噪声。
    """
    bad: list[str] = []
    rows = table.to_dict("records")
    for i in range(1, len(rows)):
        lo, hi = rows[i - 1], rows[i]
        if hi["n"] == 0 or lo["n"] == 0:
            bad.append(f"「{hi['tier']}」或「{lo['tier']}」层为空，分层切点无效")
            continue
        if hi["obs_hi"] < lo["obs_lo"]:
            bad.append(
                f"分层反转: 「{hi['tier']}」实际发生率 {hi['obs_rate']:.2%} "
                f"(CI上界 {hi['obs_hi']:.2%}) 显著低于「{lo['tier']}」"
                f"{lo['obs_rate']:.2%} (CI下界 {lo['obs_lo']:.2%})"
            )
    return bad


# ---------------------------------------------------------------------------
# 置信区间：患者级 bootstrap
# ---------------------------------------------------------------------------
def bootstrap_ci(
    y,
    score,
    fn,
    n_boot: int = 500,
    groups=None,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """
    百分位 bootstrap 置信区间。

    groups 传患者 ID 时做【整簇重采样】（同一患者的所有样本一起进出）。
    行级重采样会把同患者的相关性当成独立信息，CI 被系统性做窄。
    """
    y_arr, s_arr = _check_inputs(y, score)
    if n_boot <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = y_arr.size

    if groups is None:
        cluster_rows = None
        n_clusters = n
    else:
        g = pd.Series(np.asarray(groups).ravel())
        if g.size != n:
            raise MetricsError(f"groups 长度({g.size})与样本数({n})不一致")
        codes, _ = pd.factorize(g)
        order = np.argsort(codes, kind="mergesort")
        bounds = np.flatnonzero(np.append(True, np.diff(codes[order]) != 0))
        cluster_rows = np.split(order, bounds[1:])
        n_clusters = len(cluster_rows)

    vals = []
    for _ in range(n_boot):
        if cluster_rows is None:
            idx = rng.integers(0, n, size=n)
        else:
            pick = rng.integers(0, n_clusters, size=n_clusters)
            idx = np.concatenate([cluster_rows[i] for i in pick])
        yb = y_arr[idx]
        if yb.sum() == 0 or yb.sum() == yb.size:
            continue  # 单一类别的重采样轮次无法定义指标，跳过而不是记 nan
        try:
            vals.append(float(fn(yb, s_arr[idx])))
        except MetricsError:
            continue
    if not vals:
        return float("nan"), float("nan")
    arr = np.asarray(vals, dtype=float)
    return float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2))


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
@dataclass
class BinaryMetrics:
    """单个时程、单个验证层的完整评估结果。进 JSON 报告与全链路日志。"""

    label: str = ""
    n: int = 0
    n_pos: int = 0
    pos_rate: float = float("nan")

    auc_roc: float = float("nan")
    auc_roc_lo: float = float("nan")
    auc_roc_hi: float = float("nan")

    auc_pr: float = float("nan")
    auc_pr_lo: float = float("nan")
    auc_pr_hi: float = float("nan")
    auc_pr_baseline: float = float("nan")
    pr_lift: float = float("nan")

    brier: float = float("nan")
    brier_skill: float = float("nan")
    ece: float = float("nan")
    o_e_ratio: float = float("nan")
    calibrated_input: bool = True

    operating_points: dict[str, OperatingPoint] = field(default_factory=dict)
    stratification: pd.DataFrame | None = None
    _accuracy: float = float("nan")
    _accuracy_trivial: float = float("nan")

    # ------------------------------------------------------------------
    @property
    def primary(self) -> OperatingPoint | None:
        """平台默认操作点（敏感度优先）。风险分层与告警都以它为准。"""
        for k, v in self.operating_points.items():
            if k.startswith("敏感度"):
                return v
        return next(iter(self.operating_points.values()), None)

    def to_dict(self) -> dict:
        d = {
            k: v
            for k, v in asdict(self).items()
            if k not in ("operating_points", "stratification")
        }
        d["operating_points"] = {k: v.to_dict() for k, v in self.operating_points.items()}
        if self.stratification is not None:
            d["stratification"] = self.stratification.to_dict("records")
        return d

    def summary(self) -> str:
        lines = [
            f"===== 评估结果: {self.label or '(未命名)'} =====",
            f"样本 {self.n}  阳性 {self.n_pos} ({self.pos_rate:.2%})",
            f"AUC-ROC   {self.auc_roc:.4f}  95%CI [{self.auc_roc_lo:.4f}, {self.auc_roc_hi:.4f}]",
            f"AUC-PR    {self.auc_pr:.4f}  95%CI [{self.auc_pr_lo:.4f}, {self.auc_pr_hi:.4f}]"
            f"  (随机基线={self.auc_pr_baseline:.4f}, 提升 {self.pr_lift:.1f}x)",
        ]
        if self.calibrated_input:
            lines.append(
                f"校准      Brier={self.brier:.4f} (BSS={self.brier_skill:+.3f})  "
                f"ECE={self.ece:.4f}  实测/预测(O:E)={self.o_e_ratio:.3f}"
            )
        else:
            lines.append("校准      跳过（输入不是 [0,1] 概率，见 evaluate_binary 告警）")
        for op in self.operating_points.values():
            lines.append("  " + op.summary())
        # 规范"禁止只看准确率"：永远与无脑基线并排出现
        lines.append(
            f"准确率    {self._accuracy:.2%}  ← 对照：全判阴性也能拿到 "
            f"{self._accuracy_trivial:.2%}，这就是规范禁止只看准确率的原因"
        )
        if self.n_pos < MIN_TRUSTWORTHY_POSITIVES:
            lines.append(
                f"⚠ 阳性仅 {self.n_pos} 例（< {MIN_TRUSTWORTHY_POSITIVES}）："
                "置信区间宽到没有参考价值，此结果不得作为上线依据。"
            )
        if self.stratification is not None and not self.stratification.empty:
            lines.append("风险分层:")
            show = self.stratification[
                ["tier", "n", "share", "pred_mean", "obs_rate", "lift", "cum_capture"]
            ].copy()
            for c in ("share", "pred_mean", "obs_rate", "cum_capture"):
                show[c] = show[c].map(lambda v: f"{v:.2%}")
            show["lift"] = show["lift"].map(lambda v: f"{v:.2f}x")
            lines.append("  " + show.to_string(index=False).replace("\n", "\n  "))
        return "\n".join(lines)


def evaluate_binary(
    y,
    prob,
    label: str = "",
    groups=None,
    n_boot: int = 500,
    sensitivity_targets: tuple[float, ...] = (0.90, 0.80),
    alert_rates: tuple[float, ...] = (0.10,),
    n_calibration_bins: int = 10,
    tier_quantiles: tuple[float, ...] = DEFAULT_TIER_QUANTILES,
    tier_cutpoints: tuple[float, ...] | None = None,
    seed: int = 42,
) -> BinaryMetrics:
    """
    平台标准评估入口。任何对外汇报的二分类精度数字都必须出自这里。

    groups：患者 ID 数组，用于整簇 bootstrap（强烈建议传，理由见模块说明第 1 条）。
    prob 落在 [0,1] 之外时（未校准的 raw score）自动跳过全部校准类指标并告警 ——
    对 raw margin 算 Brier/ECE 得到的数字毫无意义，宁可留空也不能填个假数。
    """
    y_arr, p_arr = _check_inputs(y, prob)
    is_prob = bool((p_arr >= 0).all() and (p_arr <= 1).all())
    if not is_prob:
        logger.warning(
            "[%s] 预测分数超出 [0,1]，判定为未校准 raw score："
            "Brier/ECE/O:E/分层预测均值全部跳过。规范 6 的概率展示必须用校准后概率。",
            label or "eval",
        )

    m = BinaryMetrics(
        label=label,
        n=int(y_arr.size),
        n_pos=int(y_arr.sum()),
        pos_rate=float(y_arr.mean()),
        calibrated_input=is_prob,
    )
    if m.n_pos == 0 or m.n_pos == m.n:
        raise MetricsError(
            f"[{label}] 评估集只有单一类别（阳性 {m.n_pos}/{m.n}），所有排序指标都无定义。"
        )
    if m.n_pos < MIN_TRUSTWORTHY_POSITIVES:
        logger.warning(
            "[%s] 阳性样本仅 %d 例（< %d）：指标置信区间会宽到不可用，"
            "此结果不得作为上线依据（规范 5）。",
            label or "eval", m.n_pos, MIN_TRUSTWORTHY_POSITIVES,
        )

    m.auc_roc = roc_auc(y_arr, p_arr)
    m.auc_roc_lo, m.auc_roc_hi = bootstrap_ci(
        y_arr, p_arr, roc_auc, n_boot=n_boot, groups=groups, seed=seed
    )
    m.auc_pr = average_precision(y_arr, p_arr)
    m.auc_pr_lo, m.auc_pr_hi = bootstrap_ci(
        y_arr, p_arr, average_precision, n_boot=n_boot, groups=groups, seed=seed + 1
    )
    m.auc_pr_baseline = m.pos_rate
    m.pr_lift = m.auc_pr / m.pos_rate if m.pos_rate > 0 else float("nan")

    if is_prob:
        m.brier, m.brier_skill = brier_scores(y_arr, p_arr)
        m.ece = expected_calibration_error(y_arr, p_arr, n_bins=n_calibration_bins)
        exp = float(p_arr.sum())
        m.o_e_ratio = float(y_arr.sum()) / exp if exp > 0 else float("nan")

    for t in sensitivity_targets:
        op = threshold_at_sensitivity(y_arr, p_arr, target=t)
        m.operating_points[op.label] = op
    for a in alert_rates:
        op = threshold_at_alert_rate(y_arr, p_arr, alert_rate=a)
        m.operating_points[op.label] = op

    m.stratification = risk_stratification_table(
        y_arr, p_arr, quantiles=tier_quantiles, cutpoints=tier_cutpoints
    )

    primary = m.primary
    if primary is not None:
        pred = (p_arr >= primary.threshold).astype(float)
        m._accuracy = float((pred == y_arr).mean())
    m._accuracy_trivial = float(1.0 - m.pos_rate)
    return m


# ---------------------------------------------------------------------------
# 生存模型：C-index（规范 5）
# ---------------------------------------------------------------------------
class _Fenwick:
    """树状数组。C-index 的 O(n log n) 实现依赖它统计"风险更低的已处理样本数"。"""

    __slots__ = ("n", "tree")

    def __init__(self, n: int):
        self.n = n
        self.tree = [0] * (n + 1)

    def add(self, i: int) -> None:  # i 为 1-based
        while i <= self.n:
            self.tree[i] += 1
            i += i & -i

    def query(self, i: int) -> int:  # 前缀和 1..i
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s


def concordance_index(time, event, risk) -> float:
    """
    Harrell's C-index（规范 5 生存模型指标）。

    可比对定义（与 lifelines 同口径）：
      - T_i < T_j 且 i 发生结局  -> 可比，i 的风险应更高
      - T_i == T_j 且 i 发生结局、j 删失 -> 可比（删失者至少活到该时刻）
      - 同时刻双事件 / 双删失     -> 不可比
    风险并列记 0.5 分。

    自己实现而不是依赖 lifelines：C-index 是要写进对外报告的核心数字，
    不能因为环境缺库就静默降级成 nan；且 O(n log n) 实现让 bootstrap 置信区间
    在 3 万样本上仍然跑得动（朴素两重循环是 O(n²)，30k 样本要 9 亿次比较）。
    """
    t = np.asarray(time, dtype=float).ravel()
    e = np.asarray(event, dtype=float).ravel()
    r = np.asarray(risk, dtype=float).ravel()
    if not (t.size == e.size == r.size):
        raise MetricsError(f"time/event/risk 长度不一致: {t.size}/{e.size}/{r.size}")
    if t.size == 0:
        raise MetricsError("空样本无法计算 C-index")
    if np.isnan(t).any() or np.isnan(r).any() or np.isnan(e).any():
        raise MetricsError("time/event/risk 含 NaN")
    if not np.isin(e, (0.0, 1.0)).all():
        raise MetricsError("event 必须是 0/1")
    if (t <= 0).any():
        raise MetricsError("随访时长必须为正数")

    _, r_rank = np.unique(r, return_inverse=True)
    m = int(r_rank.max()) + 1
    order = np.argsort(-t, kind="mergesort")  # 时间降序
    ts, es, rs = t[order], e[order], r_rank[order]

    bit = _Fenwick(m)
    in_bit = 0
    concordant = 0.0
    comparable = 0.0
    n = ts.size
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ts[j + 1] == ts[i]:
            j += 1
        grp = slice(i, j + 1)
        ev_pos = np.flatnonzero(es[grp] == 1) + i
        if ev_pos.size:
            # (a) 与所有随访更久的样本比对
            for k in ev_pos:
                rk = int(rs[k]) + 1
                less = bit.query(rk - 1)
                eq = bit.query(rk) - less
                concordant += less + 0.5 * eq
                comparable += in_bit
            # (b) 与同时刻删失样本比对
            cens = np.sort(rs[grp][es[grp] == 0])
            if cens.size:
                for k in ev_pos:
                    lo = int(np.searchsorted(cens, rs[k], side="left"))
                    hi = int(np.searchsorted(cens, rs[k], side="right"))
                    concordant += lo + 0.5 * (hi - lo)
                    comparable += cens.size
        for k in range(i, j + 1):
            bit.add(int(rs[k]) + 1)
            in_bit += 1
        i = j + 1

    if comparable == 0:
        raise MetricsError("不存在可比样本对（可能全部删失），C-index 无定义")
    return float(concordant / comparable)


@dataclass
class SurvivalMetrics:
    label: str = ""
    n: int = 0
    n_events: int = 0
    event_rate: float = float("nan")
    c_index: float = float("nan")
    c_index_lo: float = float("nan")
    c_index_hi: float = float("nan")

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"===== 生存评估: {self.label or '(未命名)'} =====\n"
            f"样本 {self.n}  事件 {self.n_events} ({self.event_rate:.2%})\n"
            f"C-index   {self.c_index:.4f}  95%CI [{self.c_index_lo:.4f}, {self.c_index_hi:.4f}]"
        )


def evaluate_survival(
    time,
    event,
    risk,
    label: str = "",
    groups=None,
    n_boot: int = 200,
    seed: int = 42,
) -> SurvivalMetrics:
    """C-index + 患者级 bootstrap 置信区间。n_boot 默认比二分类低：
    C-index 单次计算成本更高，200 次足够定出区间量级。"""
    t = np.asarray(time, dtype=float).ravel()
    e = np.asarray(event, dtype=float).ravel()
    r = np.asarray(risk, dtype=float).ravel()
    sm = SurvivalMetrics(
        label=label,
        n=int(t.size),
        n_events=int(e.sum()),
        event_rate=float(e.mean()) if e.size else float("nan"),
        c_index=concordance_index(t, e, r),
    )

    if n_boot > 0:
        rng = np.random.default_rng(seed)
        if groups is None:
            clusters = [np.array([i]) for i in range(t.size)]
        else:
            g = pd.Series(np.asarray(groups).ravel())
            codes, _ = pd.factorize(g)
            order = np.argsort(codes, kind="mergesort")
            bounds = np.flatnonzero(np.append(True, np.diff(codes[order]) != 0))
            clusters = np.split(order, bounds[1:])
        vals = []
        for _ in range(n_boot):
            pick = rng.integers(0, len(clusters), size=len(clusters))
            idx = np.concatenate([clusters[i] for i in pick])
            try:
                vals.append(concordance_index(t[idx], e[idx], r[idx]))
            except MetricsError:
                continue
        if vals:
            arr = np.asarray(vals)
            sm.c_index_lo = float(np.quantile(arr, 0.025))
            sm.c_index_hi = float(np.quantile(arr, 0.975))
    return sm
