"""
三层验证协议与上线门禁（规范 5 + 规范 9）。

规范原话：
  - "必须三层验证：时间拆分验证 / K折交叉验证 / 外部独立数据集验证"
  - "慢病病情风险预测 线上真实 AUC ≥ 0.82~0.88"

本模块把这两句话变成一个【不通过就不能上线】的程序化门禁：
run_three_layer_validation() 跑完三层，ValidationGate 逐条判定，
assert_release_ready() 在任何一条 BLOCK 项失败时抛 ReleaseBlocked。
模型服务化（规范 4.3）的发布流程必须调用它，人不能绕过。

【为什么门禁要写成代码而不是写进流程文档】
"上线前请确认 AUC 达标"这种约定在赶工期时一定会被跳过，而跳过的代价
（一个 AUC 0.72 的模型给用户报 5 年风险概率）远大于晚上线一周。
把判定写死在发布路径上，是唯一有效的做法 —— 这与 leakage.py 把
"禁止随机划分"变成硬断言是同一个思路。

【三层数字的差值本身就是诊断信息】
报告会并列打印三个 AUC，并强制解读它们的差：

    时间拆分 << K折平均   → 概念漂移：数据分布随时间变化，模型学的是旧世界。
                            处方：缩短重训周期、加时间相关特征、上漂移监控。
    外部集   << 内部      → 过拟合到本院：设备型号、检测方法、人群结构被学进去了。
                            处方：检查单位归一化（规范 4.1）、剔除中心相关特征、
                            按中心做分层校准。
    三者接近但都不高      → 特征信息量不足，不是调参问题。
                            处方：回到规范 2 补时序特征与临床衍生特征。

【关于"没有外部数据集怎么办"】
规范写的是"必须"。本模块默认 require_external=True，缺外部集直接 BLOCK。
允许用 allow_missing_external=True 显式降级，但降级事实会被写进报告 JSON
且门禁结果标记为 CONDITIONAL —— 让"我们暂时没有外部数据"这件事在
上线记录里留下痕迹，而不是悄悄消失。
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.constants import COL_PATIENT_ID
from .crossval import FOLD_STD_WARN, CVReport, cross_validate, patient_stratified_kfold
from .leakage import assert_split_integrity, time_based_split
from .metrics import (
    MIN_TRUSTWORTHY_POSITIVES,
    BinaryMetrics,
    evaluate_binary,
    stratification_violations,
)

logger = logging.getLogger(__name__)

SEVERITY_BLOCK = "BLOCK"
SEVERITY_WARN = "WARN"

LAYER_TIME = "time_split"
LAYER_CV = "cross_validation"
LAYER_EXTERNAL = "external"


class ReleaseBlocked(RuntimeError):
    """上线门禁未通过。发布流程必须让这个异常传播出去，禁止 catch 后继续。"""


def _jsonable(obj):
    """numpy / pandas 标量转原生类型，保证报告能直接 json.dump。"""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, float):
        return None if np.isnan(obj) else obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


# ---------------------------------------------------------------------------
# 门禁
# ---------------------------------------------------------------------------
@dataclass
class GateResult:
    name: str
    passed: bool
    severity: str
    actual: float | None
    threshold: float | None
    message: str

    def to_dict(self) -> dict:
        return _jsonable(asdict(self))

    def line(self) -> str:
        mark = "✓" if self.passed else ("✗" if self.severity == SEVERITY_BLOCK else "!")
        return f"  {mark} [{self.severity}] {self.name}: {self.message}"


@dataclass
class ValidationGate:
    """
    上线验收阈值。默认值直接对应规范 9 的承诺下限，调低必须走评审 ——
    所以 GateResult 里会同时记录 threshold，事后能查出当时用的是什么标准。
    """

    # ---- 判别力（规范 9：线上真实 AUC ≥ 0.82） ----
    min_auc_roc: float = 0.82
    min_auc_ci_lower: float = 0.78
    min_pr_lift: float = 2.0

    # ---- 漏诊控制（规范 5："敏感度（严控漏诊）"） ----
    target_sensitivity: float = 0.90
    min_specificity_at_target: float = 0.40

    # ---- 概率可用性（规范 6 要展示概率数值） ----
    max_ece: float = 0.05
    max_oe_deviation: float = 0.20
    require_monotonic_stratification: bool = True

    # ---- 结论可信度 ----
    min_test_positives: int = MIN_TRUSTWORTHY_POSITIVES
    max_cv_auc_std: float = FOLD_STD_WARN

    # ---- 泛化 ----
    require_external: bool = True
    max_external_auc_drop: float = 0.05
    max_drift_gap: float = 0.05  # K折均值 - 时间拆分，超过即疑似概念漂移

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
@dataclass
class LayerReport:
    name: str
    kind: str
    detail: str = ""
    metrics: BinaryMetrics | None = None
    cv: CVReport | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "detail": self.detail,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "cv": self.cv.to_dict() if self.cv else None,
        }


@dataclass
class ValidationReport:
    model_id: str = ""
    horizon: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    layers: dict[str, LayerReport] = field(default_factory=dict)
    gates: list[GateResult] = field(default_factory=list)
    gate_config: dict = field(default_factory=dict)
    external_waived: bool = False

    # ------------------------------------------------------------------
    @property
    def headline(self) -> BinaryMetrics | None:
        """对外唯一口径：时间拆分层（规范 5）。"""
        lay = self.layers.get(LAYER_TIME)
        return lay.metrics if lay else None

    @property
    def headline_auc(self) -> float:
        h = self.headline
        return h.auc_roc if h else float("nan")

    @property
    def blocked(self) -> bool:
        return any((not g.passed) and g.severity == SEVERITY_BLOCK for g in self.gates)

    @property
    def status(self) -> str:
        if self.blocked:
            return "BLOCKED"
        if self.external_waived or any(not g.passed for g in self.gates):
            return "CONDITIONAL"
        return "PASS"

    def failures(self) -> list[GateResult]:
        return [g for g in self.gates if not g.passed and g.severity == SEVERITY_BLOCK]

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return _jsonable(
            {
                "model_id": self.model_id,
                "horizon": self.horizon,
                "created_at": self.created_at,
                "status": self.status,
                "headline_auc": self.headline_auc,
                "external_waived": self.external_waived,
                "gate_config": self.gate_config,
                "gates": [g.to_dict() for g in self.gates],
                "layers": {k: v.to_dict() for k, v in self.layers.items()},
            }
        )

    def save_json(self, path: str | Path) -> Path:
        """验证报告必须随模型一起归档（规范 4.2 全链路日志 / 4.3 版本管理）。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("验证报告已归档: %s (status=%s)", p, self.status)
        return p

    def summary(self) -> str:
        title = f"三层验证报告  model={self.model_id or '(未命名)'}"
        if self.horizon:
            title += f"  时程={self.horizon}"
        lines = ["=" * 78, title, f"生成时间 {self.created_at}   结论 【{self.status}】", "=" * 78]

        for key in (LAYER_TIME, LAYER_CV, LAYER_EXTERNAL):
            lay = self.layers.get(key)
            if lay is None:
                continue
            lines.append("")
            lines.append(f"--- 第 {['①','②','③'][[LAYER_TIME, LAYER_CV, LAYER_EXTERNAL].index(key)]} 层: {lay.name} ---")
            if lay.detail:
                lines.append(f"切分: {lay.detail}")
            if lay.metrics is not None:
                lines.append(lay.metrics.summary())
            if lay.cv is not None:
                lines.append(lay.cv.summary())

        lines.append("")
        lines.append("--- 三层对照（差值即诊断） ---")
        lines.append(self._diagnosis())

        lines.append("")
        lines.append(f"--- 上线门禁 ({len([g for g in self.gates if g.passed])}/{len(self.gates)} 通过) ---")
        lines.extend(g.line() for g in self.gates)
        if self.blocked:
            lines.append("")
            lines.append("⛔ 门禁未通过，禁止上线。逐条整改后重跑验证。")
        elif self.status == "CONDITIONAL":
            lines.append("")
            lines.append("⚠ 有条件通过：存在 WARN 项或外部验证被豁免，上线需负责人签字并留档。")
        return "\n".join(lines)

    def _diagnosis(self) -> str:
        h = self.headline
        cv = self.layers.get(LAYER_CV)
        ex = self.layers.get(LAYER_EXTERNAL)
        t_auc = h.auc_roc if h else float("nan")
        c_auc = cv.cv.mean_auc if (cv and cv.cv) else float("nan")
        e_auc = ex.metrics.auc_roc if (ex and ex.metrics) else float("nan")
        rows = [
            f"  时间拆分 AUC = {t_auc:.4f}   ← 对外唯一口径",
            f"  K折均值  AUC = {c_auc:.4f}   ← 仅看稳定性",
            f"  外部集   AUC = {e_auc:.4f}   ← 换院泛化能力",
        ]
        notes = []
        if not np.isnan(t_auc) and not np.isnan(c_auc) and c_auc - t_auc > 0.05:
            notes.append(
                f"  ⚠ K折比时间拆分高 {c_auc - t_auc:.3f}：疑似概念漂移。"
                "处方 —— 缩短重训周期、补时间相关特征、上线漂移监控（规范 3.2）。"
            )
        if not np.isnan(t_auc) and not np.isnan(e_auc) and t_auc - e_auc > 0.05:
            notes.append(
                f"  ⚠ 外部集比内部低 {t_auc - e_auc:.3f}：疑似过拟合到本院数据。"
                "处方 —— 复查单位归一化（规范 4.1）、剔除中心相关特征、按中心重新校准。"
            )
        if not np.isnan(t_auc) and t_auc < 0.75 and (np.isnan(c_auc) or abs(c_auc - t_auc) < 0.03):
            notes.append(
                "  ⚠ 三层接近但整体偏低：这是特征信息量不足，不是调参能救的。"
                "回到规范 2 补时序特征与临床衍生特征。"
            )
        return "\n".join(rows + notes)


# ---------------------------------------------------------------------------
# 门禁判定
# ---------------------------------------------------------------------------
def _gate(name, passed, severity, actual, threshold, message) -> GateResult:
    return GateResult(
        name=name,
        passed=bool(passed),
        severity=severity,
        actual=None if actual is None or (isinstance(actual, float) and np.isnan(actual)) else float(actual),
        threshold=None if threshold is None else float(threshold),
        message=message,
    )


def apply_gate(report: ValidationReport, gate: ValidationGate) -> list[GateResult]:
    """对已跑完的三层结果逐条判定。判定与执行分离，便于事后用不同标准复审。"""
    out: list[GateResult] = []
    h = report.headline

    if h is None:
        out.append(
            _gate("时间拆分层缺失", False, SEVERITY_BLOCK, None, None,
                  "没有时间拆分结果，对外 AUC 无口径（规范 5 第一条）")
        )
        return out

    # ---- 样本量：先判这个，样本不够时后面所有数字都不可信 ----
    out.append(
        _gate("测试集阳性样本量", h.n_pos >= gate.min_test_positives, SEVERITY_BLOCK,
              h.n_pos, gate.min_test_positives,
              f"测试集阳性 {h.n_pos} 例（要求 ≥{gate.min_test_positives}）"
              + ("" if h.n_pos >= gate.min_test_positives else "，样本不足时任何指标都无参考价值"))
    )

    # ---- 判别力 ----
    out.append(
        _gate("AUC-ROC（规范9承诺）", h.auc_roc >= gate.min_auc_roc, SEVERITY_BLOCK,
              h.auc_roc, gate.min_auc_roc,
              f"时间拆分 AUC={h.auc_roc:.4f}（承诺 ≥{gate.min_auc_roc}）")
    )
    ci_lo = h.auc_roc_lo
    out.append(
        _gate("AUC 置信区间下界", (not np.isnan(ci_lo)) and ci_lo >= gate.min_auc_ci_lower,
              SEVERITY_BLOCK, ci_lo, gate.min_auc_ci_lower,
              f"95%CI 下界={ci_lo:.4f}（要求 ≥{gate.min_auc_ci_lower}）"
              "：下界代表真实水平的悲观估计，承诺值不能建立在点估计的运气上")
    )
    out.append(
        _gate("AUC-PR 相对基线提升", h.pr_lift >= gate.min_pr_lift, SEVERITY_BLOCK,
              h.pr_lift, gate.min_pr_lift,
              f"AUC-PR={h.auc_pr:.4f}，阳性率基线={h.auc_pr_baseline:.4f}，"
              f"提升 {h.pr_lift:.2f}x（要求 ≥{gate.min_pr_lift}x）")
    )

    # ---- 漏诊控制 ----
    op = None
    for k, v in h.operating_points.items():
        if k.startswith("敏感度"):
            op = v
            break
    if op is None:
        out.append(_gate("敏感度操作点", False, SEVERITY_BLOCK, None, None, "未产出敏感度操作点"))
    else:
        out.append(
            _gate(f"特异度@敏感度{gate.target_sensitivity:.0%}",
                  op.specificity >= gate.min_specificity_at_target, SEVERITY_BLOCK,
                  op.specificity, gate.min_specificity_at_target,
                  f"守住敏感度 {op.sensitivity:.1%} 时特异度={op.specificity:.1%}、"
                  f"报警率={op.alert_rate:.1%}（特异度要求 ≥{gate.min_specificity_at_target:.0%}）"
                  "：特异度过低意味着为了不漏诊要把大半人群标成高危，临床上无法使用")
        )

    # ---- 概率可用性 ----
    if h.calibrated_input:
        out.append(
            _gate("校准误差 ECE", h.ece <= gate.max_ece, SEVERITY_BLOCK, h.ece, gate.max_ece,
                  f"ECE={h.ece:.4f}（要求 ≤{gate.max_ece}）"
                  "：规范 6 要把概率数值直接展示给用户，校准不过关就是给用户错数字")
        )
        oe_dev = abs(h.o_e_ratio - 1.0) if not np.isnan(h.o_e_ratio) else float("nan")
        out.append(
            _gate("实测/预测比 O:E", (not np.isnan(oe_dev)) and oe_dev <= gate.max_oe_deviation,
                  SEVERITY_BLOCK, h.o_e_ratio, gate.max_oe_deviation,
                  f"O:E={h.o_e_ratio:.3f}（允许偏离 ±{gate.max_oe_deviation:.0%}）"
                  + ("" if np.isnan(oe_dev) or oe_dev <= gate.max_oe_deviation
                     else ("，整体" + ("低估" if h.o_e_ratio > 1 else "高估") + "风险"))),
        )
        out.append(
            _gate("Brier 技能分 BSS", h.brier_skill > 0, SEVERITY_WARN, h.brier_skill, 0.0,
                  f"BSS={h.brier_skill:+.3f}"
                  + ("" if h.brier_skill > 0 else "：概率数值还不如直接报人群平均值"))
        )
    else:
        out.append(
            _gate("概率校准", False, SEVERITY_BLOCK, None, None,
                  "预测值不是 [0,1] 概率，无法校验校准。"
                  "请在模型层开启 calibration（LGBMConfig.calibration）后重跑")
        )

    if gate.require_monotonic_stratification and h.stratification is not None:
        viols = stratification_violations(h.stratification)
        out.append(
            _gate("风险分层单调性", not viols, SEVERITY_BLOCK, None, None,
                  "各层实际发生率随风险等级单调上升" if not viols else "；".join(viols))
        )

    # ---- 稳定性 ----
    cv_lay = report.layers.get(LAYER_CV)
    if cv_lay and cv_lay.cv:
        std = cv_lay.cv.std_auc
        out.append(
            _gate("折间 AUC 标准差", std <= gate.max_cv_auc_std, SEVERITY_WARN,
                  std, gate.max_cv_auc_std,
                  f"折间标准差={std:.4f}（要求 ≤{gate.max_cv_auc_std}）"
                  + ("" if std <= gate.max_cv_auc_std else "：结论高度依赖切分运气"))
        )
        gap = cv_lay.cv.mean_auc - h.auc_roc
        out.append(
            _gate("概念漂移间隙", gap <= gate.max_drift_gap, SEVERITY_WARN, gap, gate.max_drift_gap,
                  f"K折均值 - 时间拆分 = {gap:+.4f}"
                  + ("" if gap <= gate.max_drift_gap else "：数据分布随时间变化，需缩短重训周期"))
        )
    else:
        out.append(
            _gate("K折交叉验证", False, SEVERITY_BLOCK, None, None,
                  "缺少 K 折结果，无法判断该 AUC 是否稳定（规范 5 第二条）")
        )

    # ---- 泛化 ----
    ex_lay = report.layers.get(LAYER_EXTERNAL)
    if ex_lay and ex_lay.metrics:
        drop = h.auc_roc - ex_lay.metrics.auc_roc
        out.append(
            _gate("外部集掉点", drop <= gate.max_external_auc_drop, SEVERITY_BLOCK,
                  drop, gate.max_external_auc_drop,
                  f"外部集 AUC={ex_lay.metrics.auc_roc:.4f}，较内部下降 {drop:+.4f}"
                  f"（允许 ≤{gate.max_external_auc_drop}）")
        )
    elif gate.require_external:
        out.append(
            _gate("外部独立数据集验证", False, SEVERITY_BLOCK, None, None,
                  "规范 5 要求外部独立数据集验证（完全未参与训练），当前缺失。"
                  "确需先行上线请显式传 allow_missing_external=True，该豁免会写进报告")
        )
    else:
        out.append(
            _gate("外部独立数据集验证", True, SEVERITY_WARN, None, None,
                  "外部验证被显式豁免：本次上线未经过换院泛化检验，"
                  "必须在报告归档中留痕并限期补做")
        )
    return out


def assert_release_ready(report: ValidationReport) -> None:
    """发布流程的最后一道闸。有任何 BLOCK 项失败就抛异常，禁止 catch。"""
    fails = report.failures()
    if fails:
        detail = "\n".join(f"  ✗ {g.name}: {g.message}" for g in fails)
        raise ReleaseBlocked(
            f"模型 {report.model_id or '(未命名)'} 未通过上线门禁，"
            f"{len(fails)} 项不达标：\n{detail}\n"
            "禁止绕过本检查发布（规范 5 / 规范 9）。"
        )
    if report.status == "CONDITIONAL":
        logger.warning(
            "模型 %s 有条件通过：存在 WARN 项或外部验证豁免，需负责人签字。", report.model_id
        )


# ---------------------------------------------------------------------------
# 编排
# ---------------------------------------------------------------------------
def run_three_layer_validation(
    cohort: pd.DataFrame,
    X: pd.DataFrame,
    y,
    fit_predict,
    external: tuple[pd.DataFrame, pd.DataFrame, object] | None = None,
    gate: ValidationGate | None = None,
    model_id: str = "",
    horizon: str = "",
    time_test_size: float = 0.2,
    gap_days: int = 30,
    n_splits: int = 5,
    n_boot: int = 500,
    allow_missing_external: bool = False,
    groups_col: str = COL_PATIENT_ID,
    seed: int = 42,
) -> ValidationReport:
    """
    跑完规范 5 要求的三层验证并出具门禁报告。

    fit_predict : ``(X_train, y_train, X_test) -> 概率数组``。
        三层共用同一个回调，保证三个数字来自同一套训练逻辑 ——
        任何一层单独换配置，三层对照的差值就失去诊断意义。
        回调内部必须自包含全部 fit 步骤（采样、校准都算），否则构成预处理泄露。

    external : (cohort_ext, X_ext, y_ext)。必须是【完全未参与训练】的独立来源，
        最好来自另一家医院/另一个采集渠道 —— 同一批数据随机切一份出来叫测试集，
        不叫外部验证，起不到检验换院泛化的作用。

    返回的报告需 save_json() 随模型归档（规范 4.2 / 4.3）。
    """
    gate = gate or ValidationGate()
    if allow_missing_external:
        gate = ValidationGate(**{**gate.to_dict(), "require_external": False})

    y_arr = np.asarray(y, dtype=float).ravel()
    if np.isnan(y_arr).any():
        raise ValueError(
            "标签含 NaN（删失样本）。请先用 models.usable_mask 逐时程剔除后再验证。"
        )
    if not (len(cohort) == len(X) == y_arr.size):
        raise ValueError(f"cohort({len(cohort)}) / X({len(X)}) / y({y_arr.size}) 行数不一致")

    rep = ValidationReport(model_id=model_id, horizon=horizon, gate_config=gate.to_dict())
    groups_all = cohort[groups_col].to_numpy() if groups_col in cohort.columns else None
    label_tag = f"{model_id}{('/' + horizon) if horizon else ''}"

    # ---------------- 第 ① 层：时间拆分（对外口径） ----------------
    logger.info("========== 第①层 时间拆分验证（对外口径） ==========")
    tmp = cohort.copy()
    tmp["_y"] = y_arr
    split = time_based_split(cohort, test_size=time_test_size, gap_days=gap_days)
    assert_split_integrity(tmp, split, label_col="_y", min_test_positives=gate.min_test_positives)

    p_test = np.asarray(
        fit_predict(X.iloc[split.train_idx], y_arr[split.train_idx], X.iloc[split.test_idx]),
        dtype=float,
    ).ravel()
    m_time = evaluate_binary(
        y_arr[split.test_idx],
        p_test,
        label=f"{label_tag} 时间拆分",
        groups=groups_all[split.test_idx] if groups_all is not None else None,
        n_boot=n_boot,
        sensitivity_targets=(gate.target_sensitivity, 0.80),
        seed=seed,
    )
    rep.layers[LAYER_TIME] = LayerReport(
        name="时间拆分验证（旧数据训练 / 新数据测试）",
        kind=LAYER_TIME,
        detail=f"{split!r} {split.detail}",
        metrics=m_time,
    )

    # ---------------- 第 ② 层：K 折（稳定性） ----------------
    logger.info("========== 第②层 患者级 K 折交叉验证（稳定性） ==========")
    folds = patient_stratified_kfold(tmp, n_splits=n_splits, stratify_col="_y", seed=seed)
    cv = cross_validate(
        cohort, X, y_arr, fit_predict, folds,
        label=f"{label_tag} K折", groups_col=groups_col, n_boot=max(100, n_boot // 2), seed=seed,
    )
    rep.layers[LAYER_CV] = LayerReport(
        name=f"患者级分层 {n_splits} 折交叉验证",
        kind=LAYER_CV,
        detail=f"{n_splits} 折，按患者级标签分层",
        cv=cv,
    )

    # ---------------- 第 ③ 层：外部独立集（泛化） ----------------
    if external is not None:
        logger.info("========== 第③层 外部独立数据集验证（泛化） ==========")
        co_ex, X_ex, y_ex = external
        y_ex = np.asarray(y_ex, dtype=float).ravel()
        if np.isnan(y_ex).any():
            raise ValueError("外部集标签含 NaN，请先剔除删失样本")
        if not (len(co_ex) == len(X_ex) == y_ex.size):
            raise ValueError("外部集 cohort / X / y 行数不一致")
        # 外部验证必须用【内部全量训练】的模型：那才是真正要发布的那一个
        p_ex = np.asarray(fit_predict(X, y_arr, X_ex), dtype=float).ravel()
        m_ex = evaluate_binary(
            y_ex,
            p_ex,
            label=f"{label_tag} 外部集",
            groups=co_ex[groups_col].to_numpy() if groups_col in co_ex.columns else None,
            n_boot=n_boot,
            sensitivity_targets=(gate.target_sensitivity, 0.80),
            seed=seed + 7,
        )
        rep.layers[LAYER_EXTERNAL] = LayerReport(
            name="外部独立数据集验证（完全未参与训练）",
            kind=LAYER_EXTERNAL,
            detail=f"外部样本 {len(co_ex)} 行，阳性 {int(y_ex.sum())} 例",
            metrics=m_ex,
        )
    else:
        rep.external_waived = bool(allow_missing_external)
        logger.warning(
            "未提供外部独立数据集：规范 5 第三条未覆盖%s",
            "（已显式豁免，将写入报告）" if allow_missing_external else "，门禁将拦截上线",
        )

    rep.gates = apply_gate(rep, gate)
    logger.info("三层验证完成，结论=%s", rep.status)
    return rep
