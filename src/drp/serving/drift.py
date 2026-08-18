"""
数据漂移监控（规范 3.2 / 4.3）。

规范原话：
  - "数据漂移监控：线上特征分布偏移自动告警"
  - "线上模型自动监控：AUC、准确率、漂移度、错误案例统计"

【为什么慢病预测平台里，漂移监控的地位比一般业务高得多】
一般业务（点击率、风控）几天内就能拿到标签，模型掉点很快会在 AUC 上暴露。
本平台预测的是 1/3/5 年结局 —— 你要等三年才能知道今天上线的模型是不是错的。
在这个长达数年的"标签真空期"里，特征漂移是【唯一】能实时拿到的信号。
它不能证明模型掉点，但它是唯一能在事故发生前发出的声音。所以本模块的
定位不是锦上添花的看板，而是规范 4.3 线上监控的主承重墙。

【四个必须做对的技术细节】

1. 分箱边界必须来自训练集，且随模型一起持久化
   用线上数据重新分箱再比，永远算出 PSI≈0（因为你是拿它自己的分位数比它自己）。
   这是 PSI 实现里最常见的 bug，而且它的症状是"监控一片绿"，没人会去查。

2. 缺失率漂移必须单列，且阈值比数值漂移更严
   数值 PSI 只在非缺失部分上计算，接口挂掉导致某项 90% 变成 NaN 时，
   剩下 10% 的分布可能完全没变 —— PSI 报 0.01，一切正常。
   而真相是上游 OCR 或对接接口坏了。规范 4.1 说"上游错一个数据，下游模型全错"，
   缺失率突变正是这句话最常见的具体形态。

3. 漂移必须按特征重要性加权
   一个模型几乎不用的特征漂到天上去也不影响预测；主力特征漂 0.15 就该报警。
   不加权的"平均 PSI"会被大量无关特征稀释，等它报警时已经出事很久了。
   重要性建议直接传 AttributionEngine.global_importance() 的结果。

4. 样本量不足时拒绝出结论
   200 条数据算出来的 PSI 抖动极大，按它告警只会训练出"监控天天报错、
   大家都不看了"的团队习惯。宁可返回 INSUFFICIENT，也不要给假信号。

【漂移了之后做什么】
本模块只负责发现和分级，不负责决策。ALERT 之后的标准动作是：
先查上游（OCR、单位、接口）—— 大部分"漂移"其实是数据管道故障；
排除故障后才考虑人群变化，走规范 6 的模型迭代流程重训。
顺序反了会导致用一批脏数据去重训模型，把故障固化进权重里。
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: PSI 行业惯例阈值：<0.1 稳定，0.1~0.25 需关注，>0.25 显著漂移
PSI_WATCH = 0.10
PSI_ALERT = 0.25

#: 缺失率漂移阈值（绝对变化）。比 PSI 更严：它通常意味着管道故障而非人群变化。
MISSING_WATCH = 0.05
MISSING_ALERT = 0.15

#: 低于此样本量拒绝出结论
MIN_SAMPLES = 500

LEVEL_OK = "OK"
LEVEL_WATCH = "WATCH"
LEVEL_ALERT = "ALERT"
LEVEL_INSUFFICIENT = "INSUFFICIENT"

_LEVEL_RANK = {LEVEL_INSUFFICIENT: -1, LEVEL_OK: 0, LEVEL_WATCH: 1, LEVEL_ALERT: 2}


class DriftError(ValueError):
    pass


# ---------------------------------------------------------------------------
# 参考分布快照
# ---------------------------------------------------------------------------
@dataclass
class FeatureProfile:
    """单个特征的训练集分布快照。"""

    name: str
    kind: str  # numeric | categorical
    missing_rate: float = 0.0
    edges: list[float] = field(default_factory=list)  # numeric: 分箱边界
    freqs: dict[str, float] = field(default_factory=dict)  # 各箱/各类占比
    n_ref: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReferenceProfile:
    """
    训练集分布快照。必须与模型一同持久化、一同上线 ——
    模型换了参考分布没换，等于拿新模型去比旧世界，监控立刻失真。
    """

    features: dict[str, FeatureProfile] = field(default_factory=dict)
    n_ref: int = 0
    created_at: str = ""
    model_version: str = ""
    n_bins: int = 10

    # ------------------------------------------------------------------
    @classmethod
    def from_training(
        cls,
        X: pd.DataFrame,
        manifest=None,
        n_bins: int = 10,
        model_version: str = "",
    ) -> "ReferenceProfile":
        """从训练特征表建立快照。categorical 由 manifest 判定，缺 manifest 时按 dtype 推断。"""
        cat_names = set(manifest.categorical_names()) if manifest is not None else set()
        prof = cls(
            n_ref=len(X),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            model_version=model_version,
            n_bins=n_bins,
        )
        for col in X.columns:
            s = X[col]
            miss = float(s.isna().mean())
            is_cat = col in cat_names or (
                not cat_names and (s.dtype == object or str(s.dtype) == "category")
            )
            if is_cat:
                vc = s.dropna().astype(str).value_counts(normalize=True)
                prof.features[col] = FeatureProfile(
                    name=col, kind="categorical", missing_rate=miss,
                    freqs={str(k): float(v) for k, v in vc.items()}, n_ref=len(X),
                )
            else:
                v = pd.to_numeric(s, errors="coerce").dropna().to_numpy()
                if v.size == 0:
                    prof.features[col] = FeatureProfile(
                        name=col, kind="numeric", missing_rate=miss, n_ref=len(X)
                    )
                    continue
                # 分位数分箱：等宽分箱在长尾指标上会把 99% 样本塞进第一箱
                edges = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)))
                edges[0], edges[-1] = -np.inf, np.inf
                counts = np.histogram(v, bins=edges)[0].astype(float)
                prof.features[col] = FeatureProfile(
                    name=col, kind="numeric", missing_rate=miss,
                    edges=[float(e) for e in edges],
                    freqs={str(i): float(c / counts.sum()) for i, c in enumerate(counts)},
                    n_ref=len(X),
                )
        logger.info("参考分布快照已建立: %d 个特征, n_ref=%d", len(prof.features), len(X))
        return prof

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_ref": self.n_ref,
            "created_at": self.created_at,
            "model_version": self.model_version,
            "n_bins": self.n_bins,
            "features": {k: v.to_dict() for k, v in self.features.items()},
        }
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "ReferenceProfile":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            features={k: FeatureProfile(**v) for k, v in d["features"].items()},
            n_ref=d["n_ref"], created_at=d.get("created_at", ""),
            model_version=d.get("model_version", ""), n_bins=d.get("n_bins", 10),
        )


# ---------------------------------------------------------------------------
# 漂移计算
# ---------------------------------------------------------------------------
def population_stability_index(
    ref_freqs: np.ndarray, cur_freqs: np.ndarray, eps: float = 1e-6
) -> float:
    """
    PSI = Σ (cur - ref) * ln(cur / ref)。

    eps 平滑必不可少：线上某个箱为空时 ln(0) 会得到 inf，
    一个空箱就能让整个特征的 PSI 变成无穷大并污染加权总分。
    """
    r = np.clip(np.asarray(ref_freqs, dtype=float), eps, None)
    c = np.clip(np.asarray(cur_freqs, dtype=float), eps, None)
    r, c = r / r.sum(), c / c.sum()
    return float(np.sum((c - r) * np.log(c / r)))


@dataclass
class FeatureDrift:
    name: str
    kind: str
    psi: float
    level: str
    missing_ref: float
    missing_cur: float
    missing_delta: float
    unseen_categories: list[str] = field(default_factory=list)
    importance: float = 0.0
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DriftReport:
    level: str = LEVEL_OK
    n_online: int = 0
    weighted_psi: float = 0.0
    max_psi: float = 0.0
    checked_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    model_version: str = ""
    features: list[FeatureDrift] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)

    @property
    def alerts(self) -> list[FeatureDrift]:
        return [f for f in self.features if f.level == LEVEL_ALERT]

    @property
    def watches(self) -> list[FeatureDrift]:
        return [f for f in self.features if f.level == LEVEL_WATCH]

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([f.to_dict() for f in self.features])

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "n_online": self.n_online,
            "weighted_psi": self.weighted_psi,
            "max_psi": self.max_psi,
            "checked_at": self.checked_at,
            "model_version": self.model_version,
            "messages": self.messages,
            "features": [f.to_dict() for f in self.features],
        }

    def summary(self, top: int = 10) -> str:
        lines = [
            f"===== 漂移监控 [{self.level}] =====",
            f"线上样本 {self.n_online}  加权PSI={self.weighted_psi:.4f}  "
            f"最大PSI={self.max_psi:.4f}  模型版本={self.model_version or '未标注'}",
        ]
        lines.extend(f"  {m}" for m in self.messages)
        if self.features:
            tbl = self.table().sort_values("psi", ascending=False).head(top)
            show = tbl[["name", "psi", "level", "missing_ref", "missing_cur", "importance"]].copy()
            for c in ("psi", "importance"):
                show[c] = show[c].map(lambda v: f"{v:.4f}")
            for c in ("missing_ref", "missing_cur"):
                show[c] = show[c].map(lambda v: f"{v:.1%}")
            lines.append(show.to_string(index=False))
        if self.level == LEVEL_ALERT:
            lines.append(
                "⚠ 处置顺序：先排查上游（OCR、单位换算、对接接口），"
                "大部分漂移其实是数据管道故障；确认无故障后再走重训流程。"
                "顺序反了会把故障固化进模型权重。"
            )
        return "\n".join(lines)


class DriftMonitor:
    """
    用法::

        profile = ReferenceProfile.from_training(X_train, manifest, model_version="v3")
        profile.save("artifacts/model_v3/reference_profile.json")

        monitor = DriftMonitor(profile, importance=engine.global_importance(X_train))
        report = monitor.check(X_online_batch)
        if report.level == "ALERT":
            alert_to_oncall(report.summary())
    """

    def __init__(
        self,
        profile: ReferenceProfile,
        importance: pd.Series | dict | None = None,
        min_samples: int = MIN_SAMPLES,
    ):
        self.profile = profile
        self.min_samples = min_samples
        if importance is None:
            self.importance = None
        else:
            imp = pd.Series(importance, dtype=float)
            total = float(imp.abs().sum())
            self.importance = (imp.abs() / total) if total > 0 else imp * 0.0

    # ------------------------------------------------------------------
    def _weight_for(self, feature_name: str) -> float:
        """
        特征权重。归因是按【指标】聚合的，而漂移是按【特征】算的，
        所以要把 ALT_dev / ALT_slope 都映射回 ALT 的重要性。
        找不到映射时给默认权重 1（宁可高估，也不要漏掉没登记的特征）。
        """
        if self.importance is None:
            return 1.0
        if feature_name in self.importance.index:
            return float(self.importance[feature_name])
        for key in self.importance.index:
            if feature_name.startswith(f"{key}_"):
                return float(self.importance[key])
        return float(self.importance.mean()) if len(self.importance) else 1.0

    # ------------------------------------------------------------------
    def check(self, X: pd.DataFrame, model_version: str = "") -> DriftReport:
        rep = DriftReport(
            n_online=len(X), model_version=model_version or self.profile.model_version
        )
        if len(X) < self.min_samples:
            rep.level = LEVEL_INSUFFICIENT
            rep.messages.append(
                f"线上样本仅 {len(X)} 条（阈值 {self.min_samples}）："
                "此规模下 PSI 抖动极大，按它告警只会制造噪声。已拒绝出结论，"
                "请积攒足够样本或拉长统计窗口。"
            )
            logger.info("漂移检查样本量不足(%d)，跳过。", len(X))
            return rep

        missing_in_online = [c for c in self.profile.features if c not in X.columns]
        if missing_in_online:
            rep.messages.append(
                f"线上特征表缺少 {len(missing_in_online)} 个参考特征: "
                f"{missing_in_online[:5]} —— 特征集不一致本身就是重大事故，"
                "请立即核对模型版本与特征清单指纹。"
            )

        weighted_num = weighted_den = 0.0
        for name, fp in self.profile.features.items():
            if name not in X.columns:
                continue
            s = X[name]
            miss_cur = float(s.isna().mean())
            miss_delta = miss_cur - fp.missing_rate
            unseen: list[str] = []
            note = ""

            if fp.kind == "numeric":
                v = pd.to_numeric(s, errors="coerce").dropna().to_numpy()
                if v.size == 0 or not fp.edges:
                    psi = 0.0
                    note = "线上该特征全部缺失，PSI 无意义，以缺失率判定为准"
                else:
                    edges = np.asarray(fp.edges, dtype=float)
                    counts = np.histogram(v, bins=edges)[0].astype(float)
                    cur = counts / counts.sum()
                    ref = np.asarray(
                        [fp.freqs.get(str(i), 0.0) for i in range(len(cur))], dtype=float
                    )
                    psi = population_stability_index(ref, cur)
            else:
                cur_vc = s.dropna().astype(str).value_counts(normalize=True)
                cats = list(dict.fromkeys(list(fp.freqs) + list(cur_vc.index)))
                ref = np.array([fp.freqs.get(c, 0.0) for c in cats], dtype=float)
                cur = np.array([float(cur_vc.get(c, 0.0)) for c in cats], dtype=float)
                psi = population_stability_index(ref, cur)
                unseen = [c for c in cur_vc.index if c not in fp.freqs]
                if unseen:
                    note = (
                        f"出现 {len(unseen)} 个训练时未见过的取值 {unseen[:3]}："
                        "通常是上游字典新增或编码变更，模型对它们没有任何学习依据"
                    )

            # 分级：缺失率与 PSI 各自判定，取更严的那个
            lvl_psi = (
                LEVEL_ALERT if psi >= PSI_ALERT
                else LEVEL_WATCH if psi >= PSI_WATCH
                else LEVEL_OK
            )
            lvl_miss = (
                LEVEL_ALERT if abs(miss_delta) >= MISSING_ALERT
                else LEVEL_WATCH if abs(miss_delta) >= MISSING_WATCH
                else LEVEL_OK
            )
            level = max((lvl_psi, lvl_miss), key=lambda x: _LEVEL_RANK[x])
            if lvl_miss == LEVEL_ALERT and lvl_psi == LEVEL_OK:
                note = (
                    f"数值分布几乎没变但缺失率从 {fp.missing_rate:.1%} 变为 {miss_cur:.1%}"
                    " —— 这是数据管道故障的典型形态，优先排查 OCR/接口，而不是重训模型"
                ) + (f"；{note}" if note else "")

            w = self._weight_for(name)
            weighted_num += w * psi
            weighted_den += w
            rep.features.append(
                FeatureDrift(
                    name=name, kind=fp.kind, psi=psi, level=level,
                    missing_ref=fp.missing_rate, missing_cur=miss_cur,
                    missing_delta=miss_delta, unseen_categories=unseen,
                    importance=w, note=note,
                )
            )

        rep.weighted_psi = weighted_num / weighted_den if weighted_den > 0 else 0.0
        rep.max_psi = max((f.psi for f in rep.features), default=0.0)

        if rep.alerts:
            rep.level = LEVEL_ALERT
        elif rep.weighted_psi >= PSI_ALERT:
            rep.level = LEVEL_ALERT
            rep.messages.append("单特征均未超阈，但加权总漂移已达告警线：整体人群发生了系统性偏移")
        elif rep.watches or rep.weighted_psi >= PSI_WATCH:
            rep.level = LEVEL_WATCH
        else:
            rep.level = LEVEL_OK

        for f in rep.alerts[:5]:
            rep.messages.append(f"[{f.name}] PSI={f.psi:.3f} 缺失率 {f.missing_ref:.1%}→{f.missing_cur:.1%}"
                                + (f"；{f.note}" if f.note else ""))
        logger.log(
            logging.WARNING if rep.level == LEVEL_ALERT else logging.INFO,
            "漂移检查完成: level=%s 加权PSI=%.4f 告警特征=%d/%d",
            rep.level, rep.weighted_psi, len(rep.alerts), len(rep.features),
        )
        return rep
