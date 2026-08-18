"""
时序特征（规范 2.3 —— 精度提升幅度最大的模块）。

规范原话："核心：单次指标不准，趋势才准，必须做时序建模。"

这句话在临床上完全正确，理由：
  单次检验值 = 真实生理水平 + 个体内生物学波动 + 分析误差 + 当日状态干扰
  其中后三项的合计变异，对某些指标（CRP CVi=42%、TG CVi=20%）能轻松盖过
  真实病理变化。所以单点值的信噪比很低。

  而趋势是对同一个体的重复测量做的差分，个体固有偏倚被消掉了，
  剩下的就是真实变化方向。同样一个 ALT=55（略高于上限），
    - 三年来一直在 50-58 之间波动 → 大概率是个体基线偏高，风险很低
    - 从 20 一路涨到 55 → 肝脏正在发生进行性损害，风险高得多
  两者的临床含义天差地别，只看单点值的模型永远学不到这个区别。

本模块产出的特征（每个指标）：
    {C}_n_obs            观测次数
    {C}_days_since_last  距最近一次检查的天数（也是随访依从性的代理变量）
    {C}_last / _prev     最近值 / 上一次值
    {C}_delta            最近两次绝对变化
    {C}_delta_rate       最近两次变化速率（每月）
    {C}_slope            全窗口 OLS 斜率（每月），log_transform 指标取对数后回归
    {C}_slope_r2         斜率拟合优度，用于区分"稳定趋势"与"剧烈震荡"
    {C}_mean/_std/_cv/_min/_max/_range
    {C}_trend            趋势标签（基于 RCV，见下）
    {C}_n_abnormal / _abnormal_ratio
    {C}_persistence      一过性/反复/持续（规范 2.4）
    {C}_max_grade        窗口内最严重的异常分级
    {C}_crossed_up       窗口内是否发生 正常→异常 的跨界事件

【趋势判定为什么必须用 RCV 而不是固定百分比阈值】
RCV（参考变化值）= 2.77 × √(CVa² + CVi²)，是检验医学的标准工具。
它回答的问题是："这两次结果的差异，有多大概率不是噪声？"
  血钠 CVi=0.7% → RCV≈3.4%，变 5% 就是真实变化（且临床上很危险）
  CRP  CVi=42% → RCV≈117%，变 50% 完全在噪声范围内
用同一个"变化超过 10% 算上升"的规则去套这两个指标，
会同时制造大量假阳性（CRP）和假阴性（血钠）。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..data.constants import (
    COL_INDEX_DATE,
    COL_INDICATOR,
    COL_MEASURED_AT,
    COL_PATIENT_ID,
    COL_SEX,
    COL_VALUE,
    FEATURE_GROUP_TEMPORAL,
    PersistencePattern,
    TrendLabel,
)
from ..data.reference import IndicatorMeta, ReferenceRegistry
from .base import BaseFeatureBuilder, FeatureSpec
from .deviation import _compute_age, _grade_of

logger = logging.getLogger(__name__)

DAYS_PER_MONTH = 30.4375


class TemporalFeatureBuilder(BaseFeatureBuilder):
    name = "temporal"

    def __init__(
        self,
        registry: ReferenceRegistry,
        indicators: list[str] | None = None,
        lookback_days: int = 1825,  # 默认回看 5 年
        max_points: int = 20,  # 每指标最多用最近 N 次，防止个别高频随访患者主导
        min_points_for_slope: int = 3,
    ):
        self.registry = registry
        self.indicators = [c.upper() for c in (indicators or registry.codes)]
        self.lookback_days = lookback_days
        self.max_points = max_points
        self.min_points_for_slope = min_points_for_slope

    # ------------------------------------------------------------------
    def build(
        self,
        cohort: pd.DataFrame,
        records: pd.DataFrame,
        **kwargs,
    ) -> tuple[pd.DataFrame, list[FeatureSpec]]:
        cohort = cohort.reset_index(drop=True)
        n = len(cohort)
        ages = _compute_age(cohort)
        sexes = (
            cohort[COL_SEX].fillna("U").astype(str).str.upper().to_numpy()
            if COL_SEX in cohort.columns
            else np.full(n, "U")
        )
        index_dates = pd.to_datetime(cohort[COL_INDEX_DATE]).to_numpy()
        patients = cohort[COL_PATIENT_ID].to_numpy()

        # 预分组：(patient_id, indicator) -> (times[], values[])，避免 O(n*m) 的重复筛选
        groups = self._index_records(records)

        feats: dict[str, np.ndarray] = {}
        specs: list[FeatureSpec] = []

        for code in self.indicators:
            meta = self.registry.get(code)
            if meta is None:
                continue
            block, block_specs = self._build_for_indicator(
                code, meta, patients, index_dates, sexes, ages, groups
            )
            if block is None:
                continue
            feats.update(block)
            specs.extend(block_specs)

        out = pd.DataFrame(feats, index=cohort.index)
        self._check_alignment(cohort, out)
        return out, specs

    # ------------------------------------------------------------------
    def _index_records(self, records: pd.DataFrame) -> dict:
        """把长表预分组成 dict，key=(pid, code)，value=(排序后的时间数组, 值数组)。"""
        if records.empty:
            return {}
        rec = records[[COL_PATIENT_ID, COL_INDICATOR, COL_VALUE, COL_MEASURED_AT]].copy()
        rec[COL_MEASURED_AT] = pd.to_datetime(rec[COL_MEASURED_AT])
        rec = rec.dropna(subset=[COL_VALUE]).sort_values(COL_MEASURED_AT)

        groups: dict = {}
        for (pid, code), grp in rec.groupby([COL_PATIENT_ID, COL_INDICATOR], observed=True):
            groups[(pid, str(code))] = (
                grp[COL_MEASURED_AT].to_numpy(),
                grp[COL_VALUE].to_numpy(dtype=float),
            )
        return groups

    # ------------------------------------------------------------------
    def _build_for_indicator(
        self,
        code: str,
        meta: IndicatorMeta,
        patients: np.ndarray,
        index_dates: np.ndarray,
        sexes: np.ndarray,
        ages: np.ndarray,
        groups: dict,
    ):
        n = len(patients)
        # 该指标是否有任何数据；没有就整块跳过，避免产出满是 NaN 的无效特征列
        if not any((p, code) in groups for p in set(patients.tolist())):
            return None, []

        F = {
            k: np.full(n, np.nan)
            for k in (
                "n_obs", "days_since_last", "last", "prev", "delta", "delta_rate",
                "slope", "slope_r2", "mean", "std", "cv", "min", "max", "range",
                "n_abnormal", "abnormal_ratio", "max_grade", "span_days",
            )
        }
        F["trend"] = np.full(n, float(TrendLabel.UNKNOWN))
        F["persistence"] = np.full(n, float(PersistencePattern.UNKNOWN))
        F["crossed_up"] = np.full(n, np.nan)

        lookback = np.timedelta64(self.lookback_days, "D")

        for i in range(n):
            key = (patients[i], code)
            g = groups.get(key)
            if g is None:
                continue
            times, values = g

            # as-of 窗口截取。这里再截一次是【纵深防御】：
            # 即使上游忘了调 as_of_filter，这里也不会让未来数据进特征。
            idx_date = index_dates[i]
            mask = (times <= idx_date) & (times >= idx_date - lookback)
            if not mask.any():
                continue
            t = times[mask]
            v = values[mask]
            if len(t) > self.max_points:
                t, v = t[-self.max_points:], v[-self.max_points:]

            self._fill_row(F, i, meta, t, v, idx_date, sexes[i], ages[i])

        prefix = code
        feats = {f"{prefix}_{k}": arr for k, arr in F.items()}
        specs = _make_specs(prefix, meta.name_cn)
        return feats, specs

    # ------------------------------------------------------------------
    def _fill_row(
        self,
        F: dict,
        i: int,
        meta: IndicatorMeta,
        t: np.ndarray,
        v: np.ndarray,
        idx_date,
        sex: str,
        age: float,
    ) -> None:
        k = len(v)
        F["n_obs"][i] = k
        F["last"][i] = v[-1]
        F["days_since_last"][i] = (idx_date - t[-1]) / np.timedelta64(1, "D")
        F["span_days"][i] = (t[-1] - t[0]) / np.timedelta64(1, "D")
        F["mean"][i] = np.mean(v)
        F["min"][i] = np.min(v)
        F["max"][i] = np.max(v)
        F["range"][i] = np.max(v) - np.min(v)

        if k >= 2:
            F["std"][i] = np.std(v, ddof=1)
            if abs(F["mean"][i]) > 1e-9:
                F["cv"][i] = F["std"][i] / abs(F["mean"][i])
            F["prev"][i] = v[-2]
            F["delta"][i] = v[-1] - v[-2]
            dt_month = (t[-1] - t[-2]) / np.timedelta64(1, "D") / DAYS_PER_MONTH
            if dt_month > 1e-6:
                F["delta_rate"][i] = F["delta"][i] / dt_month

        # ---- 异常状态序列 ----
        iv = meta.match_interval(sex, None if np.isnan(age) else float(age))
        if iv is not None:
            abn = np.array([not iv.contains(x) for x in v], dtype=bool)
            grades = np.array([int(_grade_of(meta, x, iv)) for x in v], dtype=float)
            F["n_abnormal"][i] = abn.sum()
            F["abnormal_ratio"][i] = abn.mean()
            F["max_grade"][i] = np.max(np.abs(grades)) * np.sign(
                grades[np.argmax(np.abs(grades))]
            )
            F["persistence"][i] = float(_persistence_of(abn))
            if k >= 2:
                F["crossed_up"][i] = float((not abn[0]) and abn[-1])

        # ---- 趋势与斜率 ----
        if k >= self.min_points_for_slope:
            slope, r2 = _ols_slope(t, v, log_transform=meta.log_transform)
            F["slope"][i] = slope
            F["slope_r2"][i] = r2

        if k >= 2:
            F["trend"][i] = float(_trend_of(v, t, meta))

    # ------------------------------------------------------------------


def _ols_slope(t: np.ndarray, v: np.ndarray, log_transform: bool) -> tuple[float, float]:
    """
    最小二乘斜率（单位：每月）。

    右偏指标（ALT、TG、CRP、UACR 等，log_transform=True）先取对数再回归。
    原因：这些指标的变化在生理上是【乘性】的，从 20→40 和从 200→400
    临床意义相近，但线性斜率会认为后者的变化剧烈 10 倍。
    对数化后斜率变成"每月相对增长率"，跨患者可比性大幅提升。
    """
    x = (t - t[0]) / np.timedelta64(1, "D") / DAYS_PER_MONTH
    y = v.astype(float)

    if log_transform:
        if np.any(y <= 0):
            y = np.log1p(np.clip(y, 0, None))  # 含 0 值时用 log1p
        else:
            y = np.log(y)

    if np.ptp(x) < 1e-9:
        return np.nan, np.nan

    x_mean, y_mean = x.mean(), y.mean()
    sxx = np.sum((x - x_mean) ** 2)
    if sxx < 1e-12:
        return np.nan, np.nan
    slope = np.sum((x - x_mean) * (y - y_mean)) / sxx

    y_hat = y_mean + slope * (x - x_mean)
    ss_tot = np.sum((y - y_mean) ** 2)
    r2 = 1.0 - np.sum((y - y_hat) ** 2) / ss_tot if ss_tot > 1e-12 else np.nan
    return float(slope), float(r2)


def _trend_of(v: np.ndarray, t: np.ndarray, meta: IndicatorMeta) -> TrendLabel:
    """
    基于 RCV 的趋势判定。

    比较最近值与窗口起点值的相对变化：
        rel = (v_last - v_first) / v_first
    若 |rel| <= RCV，认为变化在生物学噪声范围内 → STABLE
    否则按方向判 RISING / FALLING。

    观测点 >= 3 时额外要求斜率方向一致，避免"先大跌后大涨"被误判成上升。
    """
    if len(v) < 2:
        return TrendLabel.UNKNOWN
    base = v[0]
    if abs(base) < 1e-9:
        return TrendLabel.UNKNOWN

    rel = (v[-1] - base) / abs(base)
    rcv = meta.rcv

    if abs(rel) <= rcv:
        return TrendLabel.STABLE

    if len(v) >= 3:
        slope, _ = _ols_slope(t, v, log_transform=False)
        if np.isfinite(slope) and np.sign(slope) != np.sign(rel):
            # 端点变化与整体趋势矛盾 → 判为震荡，归入 STABLE 更保守
            return TrendLabel.STABLE

    return TrendLabel.RISING if rel > 0 else TrendLabel.FALLING


def _persistence_of(abn: np.ndarray) -> PersistencePattern:
    """
    一过性 vs 持续性判定（规范 2.4）。

    规则（按优先级）：
      从未异常                                    -> NEVER_ABNORMAL
      最近连续 >=2 次异常                          -> PERSISTENT
      异常次数 >=2 但不连续（中间回落过）           -> RECURRENT
      仅 1 次异常且最近一次已回落正常               -> TRANSIENT
      仅 1 次异常且就是最近一次（无法判断是否持续）  -> UNKNOWN
    """
    k = len(abn)
    if k == 0:
        return PersistencePattern.UNKNOWN
    if not abn.any():
        return PersistencePattern.NEVER_ABNORMAL

    if k >= 2 and abn[-1] and abn[-2]:
        return PersistencePattern.PERSISTENT

    n_abn = int(abn.sum())
    if n_abn >= 2:
        return PersistencePattern.RECURRENT
    if n_abn == 1 and not abn[-1]:
        return PersistencePattern.TRANSIENT
    return PersistencePattern.UNKNOWN


def _make_specs(prefix: str, name_cn: str) -> list[FeatureSpec]:
    g = FEATURE_GROUP_TEMPORAL
    d = [
        ("n_obs", "numeric", "窗口内观测次数", 0),
        ("days_since_last", "numeric", "距最近一次检查天数", 0),
        ("last", "numeric", "最近一次值", 0),
        ("prev", "numeric", "上一次值", 0),
        ("delta", "numeric", "最近两次绝对变化", 0),
        ("delta_rate", "numeric", "最近两次变化速率(每月)", 0),
        ("slope", "numeric", "窗口 OLS 斜率(每月)", 0),
        ("slope_r2", "numeric", "斜率拟合优度 R²(区分稳定趋势与震荡)", 0),
        ("mean", "numeric", "窗口均值", 0),
        ("std", "numeric", "窗口标准差", 0),
        ("cv", "numeric", "窗口变异系数", 0),
        ("min", "numeric", "窗口最小值", 0),
        ("max", "numeric", "窗口最大值", 0),
        ("range", "numeric", "窗口极差", 0),
        ("span_days", "numeric", "窗口首末观测间隔天数", 0),
        ("n_abnormal", "numeric", "窗口内异常次数", 1),
        ("abnormal_ratio", "numeric", "窗口内异常比例", 1),
        ("max_grade", "numeric", "窗口内最严重异常分级", 0),
        ("trend", "categorical", "趋势标签 0未知/1平稳/2上升/3下降(基于RCV)", 0),
        (
            "persistence", "categorical",
            # monotone 必须为 0：LightGBM 禁止 categorical+monotone（fatal 终止），
            # 且枚举顺序 0数据不足 < 1从未异常 并非风险序 —— 单调假设本就不成立。
            "0未知/1从未异常/2一过性/3反复/4持续", 0,
        ),
        ("crossed_up", "binary", "窗口内是否发生正常→异常跨界", 1),
    ]
    return [
        FeatureSpec(
            name=f"{prefix}_{suffix}",
            group=g,
            dtype=dt,  # type: ignore[arg-type]
            indicator=prefix,
            description=f"{name_cn} {desc}",
            monotone=mono,
        )
        for suffix, dt, desc, mono in d
    ]
