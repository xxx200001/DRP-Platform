"""
风险归因：SHAP 贡献度与 Top10 风险因子（规范 3.2 / 3.3 / 6）。

规范原话：
  - "可解释性强制开启：每一次预测输出 SHAP 特征贡献度"
  - "输出：Top10 风险贡献因子 / 风险升高、降低原因、趋势变化解读"
  - "个性化风险归因：明确告诉用户是哪几项指标导致风险升高"

模型层的 lgbm.shap_values() 只负责算出逐特征的原始贡献，本模块负责把它
变成【能给人看的东西】。这中间有四个坑，每一个踩了都会让归因变成误导：

【坑 1】SHAP 值在 raw score（对数几率）空间，展示的概率是校准后的
    两者不是同一把尺子。"ALT 的 SHAP 值是 0.35" 换算不出"ALT 让你的风险
    上升了 3.5 个百分点" —— 校准器是单调非线性的，同样的 0.35 在概率 2%
    附近和 40% 附近对应的百分点完全不同。
    所以本模块只输出【排序】与【方向】，并在 RiskAttribution 里刻意不提供
    "贡献百分点"字段。产品文案只能说"ALT 偏高是本次风险升高的主要原因"，
    不能说"ALT 贡献了 3.5%"。这条是硬约束，不是措辞偏好。

【坑 2】必须按【指标】聚合，不能逐特征展示
    规范 2 的特征工程会让一个 ALT 派生出 ALT_value / ALT_dev / ALT_slope /
    ALT_trend / ALT_persistence / ALT_status … 十几个特征。逐个列出来，
    用户看到"ALT_slope 贡献 0.12、ALT_dev 贡献 -0.03"完全无法理解。
    更危险的是：同一指标的多个特征贡献可能正负相消，只挑其中一个展示
    会得出与整体相反的结论。
    附带的好处是缓解共线性分摊：强相关特征（AST 与 AST/ALT 比值）之间
    SHAP 会把贡献切开，聚合到指标级正好把切开的部分合回去。

【坑 3】缺失必须显式说明是"没查"，不能让用户以为"正常"
    规范 1.2 的三态设计里，MISSING 本身携带信息，模型确实会给它贡献值。
    但如果展示成"血小板：降低了您的风险"，用户会理解成"我血小板没问题"，
    而真相是"你根本没查过血小板"。所以缺失指标的归因必须换一套话术，
    由 FactorContribution.is_missing 驱动。

【坑 4】Top10 覆盖率不足时，"主要因为这几项"本身就是误导
    如果 Top10 只覆盖了总绝对贡献的 30%，说明风险是弥散的 —— 几十项各出
    一点力。此时告诉用户"主要因为这 10 项"是错的。coverage 字段就是为了
    让产品层能在覆盖率低时改用"综合多项指标"的措辞。
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ..data.constants import (
    FEATURE_GROUP_CONFOUNDER,
    FEATURE_GROUP_DEMO,
    MeasureStatus,
)

logger = logging.getLogger(__name__)

#: 规范 3.3 明确要求 Top10
DEFAULT_TOP_N = 10

#: Top10 覆盖率低于此值时，"主要由这几项导致"的表述不成立
COVERAGE_WARN = 0.5


@dataclass
class FactorContribution:
    """单个风险因子（已按指标聚合）的贡献。"""

    key: str  # 指标代码或特征名，程序用
    display: str  # 展示名，给人看
    group: str  # 特征分组（临床比值 / 时序 / 人口学 …）
    shap_sum: float  # 该因子下所有特征的 SHAP 之和（raw score 空间）
    direction: int  # +1 推高风险 / -1 拉低风险 / 0 无影响
    n_features: int  # 聚合了几个特征
    value: float | None = None  # 该指标当前值（若可得）
    is_missing: bool = False  # 该指标本次未检查
    detail: dict[str, float] = field(default_factory=dict)  # 逐特征明细，供排障

    @property
    def magnitude(self) -> float:
        return abs(self.shap_sum)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["magnitude"] = self.magnitude
        return d

    def phrase(self) -> str:
        """
        单条归因的中性描述。刻意不含任何数值换算（坑 1），
        也不含诊断/治疗类措辞（规范 7）—— 输出前仍需过 compliance.assert_compliant。
        """
        if self.is_missing:
            return f"{self.display}：本次未检查，缺少该项数据会影响评估精度"
        if self.direction > 0:
            return f"{self.display}：本次评估中推高风险的主要因素之一"
        if self.direction < 0:
            return f"{self.display}：本次评估中起到降低风险作用"
        return f"{self.display}：对本次评估影响很小"


@dataclass
class RiskAttribution:
    """单次预测的完整归因结果。整条进全链路日志（规范 4.2）。"""

    base_value: float = 0.0  # 模型基准（raw score 空间的人群平均）
    total_shap: float = 0.0  # 全部特征贡献之和
    probability: float = float("nan")  # 校准后概率，仅作展示锚点
    factors: list[FactorContribution] = field(default_factory=list)
    by_group: dict[str, float] = field(default_factory=dict)
    coverage: float = float("nan")  # Top-N 占总绝对贡献的比例
    total_abs: float = 0.0

    @property
    def raising(self) -> list[FactorContribution]:
        return [f for f in self.factors if f.direction > 0]

    @property
    def lowering(self) -> list[FactorContribution]:
        return [f for f in self.factors if f.direction < 0]

    @property
    def diffuse(self) -> bool:
        """风险弥散：Top-N 解释不了大头，产品层必须改措辞（坑 4）。"""
        return not np.isnan(self.coverage) and self.coverage < COVERAGE_WARN

    def to_dict(self) -> dict:
        return {
            "base_value": self.base_value,
            "total_shap": self.total_shap,
            "probability": self.probability,
            "coverage": self.coverage,
            "total_abs": self.total_abs,
            "diffuse": self.diffuse,
            "by_group": self.by_group,
            "factors": [f.to_dict() for f in self.factors],
        }

    def summary(self) -> str:
        lines = [
            f"风险概率 {self.probability:.2%}（基准 raw={self.base_value:+.3f}，"
            f"本例合计 {self.total_shap:+.3f}）",
            f"Top{len(self.factors)} 覆盖总贡献 {self.coverage:.1%}"
            + ("  ⚠ 风险弥散，不宜表述为「主要由这几项导致」" if self.diffuse else ""),
        ]
        for i, f in enumerate(self.factors, 1):
            arrow = "↑" if f.direction > 0 else ("↓" if f.direction < 0 else "·")
            val = "未检查" if f.is_missing else (
                f"{f.value:.3g}" if f.value is not None and not np.isnan(f.value) else "—"
            )
            lines.append(
                f"  {i:2d}. {arrow} {f.display:<18s} 贡献={f.shap_sum:+.4f} "
                f"当前值={val} (聚合{f.n_features}个特征)"
            )
        return "\n".join(lines)


class AttributionEngine:
    """
    把模型的逐特征 SHAP 变成按指标聚合的 Top-N 风险因子。

    用法::

        eng = AttributionEngine(model, display_names={"ALT": "谷丙转氨酶"})
        attrs = eng.explain(X_new, probabilities=risk)
        print(attrs[0].summary())

    display_names 建议从 configs/reference_intervals.yaml 的指标中文名注入 ——
    直接把 "ALT_dev" 抛给用户是产品事故，不是技术细节。
    """

    def __init__(
        self,
        model,
        display_names: dict[str, str] | None = None,
        top_n: int = DEFAULT_TOP_N,
        status_suffix: str = "_status",
    ):
        if getattr(model, "manifest", None) is None:
            raise ValueError("模型尚未训练/加载，无法建立归因引擎")
        self.model = model
        self.manifest = model.manifest
        self.display_names = display_names or {}
        self.top_n = top_n
        self.status_suffix = status_suffix

        self._indicator_map = self.manifest.indicator_map()
        self._group_map = self.manifest.group_map()
        # 聚合键：有 indicator 的按指标聚合，没有的（人口学、干扰因子）按特征自身
        self._key_of: dict[str, str] = {}
        for name in self.manifest.names:
            ind = self._indicator_map.get(name)
            self._key_of[name] = ind if ind else name
        self._assert_shap_capable()

    # ------------------------------------------------------------------
    def _assert_shap_capable(self) -> None:
        """
        启动期校验 SHAP 能力，而不是等第一个用户请求。

        规范 3.2 把可解释性列为强制项，意味着"算不出 SHAP"等于服务不可用。
        如果只在 explain() 里报错，这个事实会在深夜第一个真实请求进来时才暴露，
        而那时链路已经上线、流量已经切过来了。构造期失败则只影响一次发布。
        """
        backend = getattr(self.model, "backend_", None)
        if backend == "lightgbm":
            return  # 内置 TreeSHAP，零额外依赖
        try:
            import shap  # noqa: F401, PLC0415
        except ImportError as e:
            raise RuntimeError(
                f"当前模型后端为 {backend!r}，计算 SHAP 需要 shap 包，但环境中没有。"
                "规范 3.2 要求每次预测强制输出贡献度，缺此能力的服务不允许上线。"
                "请安装 lightgbm（内置 TreeSHAP，推荐）或 shap；"
                "确需关闭解释的离线批量场景请设 ServiceConfig(explain=False) 并留档。"
            ) from e

    # ------------------------------------------------------------------
    def _display(self, key: str, group: str) -> str:
        if key in self.display_names:
            return self.display_names[key]
        if group in (FEATURE_GROUP_DEMO, FEATURE_GROUP_CONFOUNDER):
            return key
        return key

    def _value_and_missing(self, row: pd.Series, key: str) -> tuple[float | None, bool]:
        """
        取该指标的当前值与"是否本次未检查"。

        判定缺失优先看三态列（规范 1.2 的 _status），而不是看数值是否为 NaN ——
        数值 NaN 也可能是被单位校验拦截的 INVALID，两者对用户的话术完全不同。
        """
        status_col = f"{key}{self.status_suffix}"
        if status_col in row.index:
            try:
                st = int(row[status_col])
                if st == int(MeasureStatus.MISSING):
                    return None, True
            except (TypeError, ValueError):
                pass
        for cand in (f"{key}_value", key):
            if cand in row.index:
                v = pd.to_numeric(row[cand], errors="coerce")
                if pd.notna(v):
                    return float(v), False
                return None, status_col not in row.index
        return None, False

    # ------------------------------------------------------------------
    def explain(
        self,
        X: pd.DataFrame,
        probabilities: np.ndarray | pd.Series | None = None,
        top_n: int | None = None,
    ) -> list[RiskAttribution]:
        """
        逐样本归因。返回列表，顺序与 X 一致。

        probabilities 传校准后概率仅用于展示锚点；不传则留 NaN。
        绝不能用 SHAP 反推概率（坑 1）。
        """
        n_top = top_n or self.top_n
        contrib, base = self.model.shap_values(X)
        Xa = self.manifest.align(X.copy(), strict=False)

        keys = np.array([self._key_of[c] for c in contrib.columns])
        uniq_keys = pd.unique(keys)
        # 按聚合键求和：一次矩阵运算，避免逐样本 groupby
        agg = pd.DataFrame(
            {k: contrib.loc[:, keys == k].to_numpy().sum(axis=1) for k in uniq_keys},
            index=contrib.index,
        )

        probs = (
            np.asarray(probabilities, dtype=float).ravel()
            if probabilities is not None
            else np.full(len(contrib), np.nan)
        )
        if probs.size != len(contrib):
            raise ValueError(f"probabilities 长度({probs.size})与样本数({len(contrib)})不符")

        out: list[RiskAttribution] = []
        for i, (_idx, row) in enumerate(agg.iterrows()):
            total_abs = float(row.abs().sum())
            order = row.abs().sort_values(ascending=False).index[:n_top]

            factors: list[FactorContribution] = []
            for key in order:
                members = [c for c in contrib.columns if self._key_of[c] == key]
                group = self._group_map.get(members[0], "unknown")
                shap_sum = float(row[key])
                value, missing = self._value_and_missing(Xa.iloc[i], key)
                factors.append(
                    FactorContribution(
                        key=str(key),
                        display=self._display(str(key), group),
                        group=group,
                        shap_sum=shap_sum,
                        direction=int(np.sign(round(shap_sum, 9))),
                        n_features=len(members),
                        value=value,
                        is_missing=missing,
                        detail={m: float(contrib.iloc[i][m]) for m in members},
                    )
                )

            by_group: dict[str, float] = {}
            for c in contrib.columns:
                g = self._group_map.get(c, "unknown")
                by_group[g] = by_group.get(g, 0.0) + float(contrib.iloc[i][c])

            covered = float(sum(f.magnitude for f in factors))
            attr = RiskAttribution(
                base_value=float(base[i]),
                total_shap=float(row.sum()),
                probability=float(probs[i]),
                factors=factors,
                by_group=by_group,
                coverage=covered / total_abs if total_abs > 0 else float("nan"),
                total_abs=total_abs,
            )
            if attr.diffuse:
                logger.info(
                    "样本 %s 风险弥散：Top%d 仅覆盖 %.1f%% 的总贡献，"
                    "产品层应改用「综合多项指标」的表述。",
                    _idx, n_top, 100 * attr.coverage,
                )
            out.append(attr)
        return out

    # ------------------------------------------------------------------
    def global_importance(self, X: pd.DataFrame, top_n: int | None = None) -> pd.Series:
        """
        全局重要性 = 各指标 SHAP 绝对值的样本均值。

        比 LightGBM 的 split/gain importance 更可信：gain 会被高基数特征
        系统性抬高，而且它衡量的是"树用了多少次"，不是"对预测影响多大"。
        规范 4.3 的线上监控看的是后者。
        """
        contrib, _ = self.model.shap_values(X)
        keys = np.array([self._key_of[c] for c in contrib.columns])
        agg = pd.DataFrame(
            {k: contrib.loc[:, keys == k].to_numpy().sum(axis=1) for k in pd.unique(keys)}
        )
        imp = agg.abs().mean().sort_values(ascending=False)
        return imp.head(top_n) if top_n else imp


# ---------------------------------------------------------------------------
# 变化归因（规范 6："本次 vs 上次"对比报告 / "风险升高降低原因"）
# ---------------------------------------------------------------------------
@dataclass
class ChangeFactor:
    key: str
    display: str
    delta_shap: float
    prev_value: float | None
    curr_value: float | None

    def to_dict(self) -> dict:
        return asdict(self)

    def phrase(self) -> str:
        move = "推高" if self.delta_shap > 0 else "拉低"
        # 数值相同却贡献变了是完全可能的：模型是非线性的，同一个 ALT 值
        # 在别的指标变化后，其边际贡献会跟着变。此时绝不能编造"由 X 下降至 X"
        # 这种趋势 —— 用户一眼就能看出这句话是假的，整份报告的可信度随之崩塌。
        if (
            self.prev_value is not None
            and self.curr_value is not None
            and self.prev_value != self.curr_value
        ):
            trend = "上升" if self.curr_value > self.prev_value else "下降"
            return (
                f"{self.display}：由 {self.prev_value:.3g} {trend}至 "
                f"{self.curr_value:.3g}，本次{move}了风险"
            )
        return f"{self.display}：本次{move}了风险"


@dataclass
class ChangeAttribution:
    prev_probability: float
    curr_probability: float
    delta_probability: float
    factors: list[ChangeFactor] = field(default_factory=list)

    @property
    def rose(self) -> bool:
        return self.delta_probability > 0

    def to_dict(self) -> dict:
        return {
            "prev_probability": self.prev_probability,
            "curr_probability": self.curr_probability,
            "delta_probability": self.delta_probability,
            "rose": self.rose,
            "factors": [f.to_dict() for f in self.factors],
        }

    def summary(self) -> str:
        head = (
            f"风险由 {self.prev_probability:.2%} "
            f"{'上升' if self.rose else '下降'}至 {self.curr_probability:.2%} "
            f"({self.delta_probability:+.2%})"
        )
        return "\n".join([head] + [f"  · {f.phrase()}" for f in self.factors])


def explain_change(
    prev: RiskAttribution,
    curr: RiskAttribution,
    top_n: int = 5,
) -> ChangeAttribution:
    """
    两次预测之间的风险变化归因（规范 6 时序对比报告）。

    做差必须在【聚合后的 SHAP】上做，而不是比较两次的 Top10 名单：
    名单是排序结果，某项从第 11 名升到第 9 名会造成"新出现了一个风险因素"
    的错觉，实际它的贡献可能几乎没变。变化的大小只能由 Δ贡献 决定。

    注意 delta_probability 是两次校准概率的实际差值，与 Δ贡献 不是线性关系
    （同坑 1）；两者一起展示，但绝不互相换算。
    """
    prev_map = {f.key: f for f in prev.factors}
    curr_map = {f.key: f for f in curr.factors}
    deltas: list[ChangeFactor] = []
    for key in set(prev_map) | set(curr_map):
        p, c = prev_map.get(key), curr_map.get(key)
        d = (c.shap_sum if c else 0.0) - (p.shap_sum if p else 0.0)
        if abs(d) < 1e-12:
            continue
        deltas.append(
            ChangeFactor(
                key=key,
                display=(c or p).display,
                delta_shap=d,
                prev_value=p.value if p else None,
                curr_value=c.value if c else None,
            )
        )
    deltas.sort(key=lambda f: abs(f.delta_shap), reverse=True)
    return ChangeAttribution(
        prev_probability=prev.probability,
        curr_probability=curr.probability,
        delta_probability=curr.probability - prev.probability,
        factors=deltas[:top_n],
    )
