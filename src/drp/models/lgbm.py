"""
风险概率预测主模型：LightGBM（规范 3.1）。

规范原话：
  - "风险概率预测主模型：LightGBM / Cox-PH 生存分析模型"
  - "禁止使用大模型直接输出风险概率（不稳定、不可复现、精度低）"

本模块是平台唯一的风险概率产出通道。大模型只允许出现在报告解析与
解释文案环节（规范 3.1 第二条），任何绕过本模块直接生成概率的代码
都不允许合入。

【设计决策逐条说明】

1. 后端与"模型栈固定"的关系
   生产后端固定为 LightGBM（backend="lightgbm"，缺库直接报错拒跑）。
   同时提供 sklearn HistGradientBoosting 作为【仅限开发环境】的回退
   （backend="auto" 时缺 lightgbm 会降级并打大字告警）——
   两者同为直方图 GBDT、同样原生支持 NaN 与单调约束，能让没装
   lightgbm 的开发机跑通全流程单测；但回退后端不支持 focal loss、
   不支持外部验证集早停，绝不允许出现在生产配置里。
   这不违反"禁止乱换模型栈"：换掉的从来不是算法家族，而且降级
   必须显式、带告警、写进 meta.json 可追溯。

2. 缺失值：一个 fillna 都没有
   LightGBM/HistGB 原生学习 NaN 的分裂方向。任何填充都会抹掉
   "没查过"这个信号（constants.py 里已把原因写透）。本模块若检测到
   上游偷偷填充过（无 NaN 且 _status 列大量为 MISSING）不做拦截 ——
   那是特征层测试的职责 —— 但自己绝不引入填充。

3. 单调约束默认开启（use_monotone_constraints=True）
   FeatureSpec.monotone 里声明的临床方向（HbA1c 越高风险越高等）
   直接转成 monotone_constraints。收益在样本稀疏区域最明显：
   没有约束时，老年+极端值亚组常因样本少而学出反直觉的局部下降，
   线上被用户/医生一眼识破，信任崩塌比掉几个点 AUC 更致命。

4. 概率校准（calibration，默认 isotonic）
   scale_pos_weight / focal / 欠采样都会让输出概率系统性偏移
   （排序不变、数值失真）。而规范 6 的四级风险分层、1/3/5 年概率
   展示用的是概率【数值本身】，不校准的数字给用户看就是错的。
   校准器只在【外部传入的验证集】上拟合 —— 用训练集拟合校准器
   属于 leakage.py 的泄露 4。focal 模式下强制开启，关不掉。

5. 训练/推理一致性
   predict 一律先过 manifest.align()。列错位是线上精度事故里
   最隐蔽的一类（模型不报错、AUC 直接归零），物理上堵死。
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..features.base import FeatureManifest
from .imbalance import (
    FocalBinary,
    _check_binary,
    balanced_sample_weight,
    compute_scale_pos_weight,
)

logger = logging.getLogger(__name__)

_META_FILE = "meta.json"
_MANIFEST_FILE = "manifest.json"
_CALIBRATOR_FILE = "calibrator.joblib"
_LGBM_MODEL_FILE = "model.txt"
_SKLEARN_MODEL_FILE = "model.joblib"


def _try_import_lightgbm():
    try:
        import lightgbm as lgb  # noqa: PLC0415

        return lgb
    except ImportError:
        return None


@dataclass
class LGBMConfig:
    """
    主模型配置。随模型一起持久化进 meta.json，保证任何一次线上预测
    都能溯源到完整超参（规范 4.2 全链路日志 / 4.3 版本管理）。

    默认值按"3 万条、阳性率 3%~15%、数百特征"的规范场景调过：
    小学习率 + 大叶子最小样本数 + 行列采样，防过拟合优先于训练速度。
    """

    # ---- 树结构 ----
    n_estimators: int = 3000          # 上限，实际轮数由早停决定
    learning_rate: float = 0.03
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 60       # 医疗数据必须大：小叶子=记住个体而非规律
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 2.0

    # ---- 训练控制 ----
    early_stopping_rounds: int = 200
    backend: str = "auto"             # lightgbm | sklearn | auto（生产必须 lightgbm）
    seed: int = 42

    # ---- 不均衡（规范 3.2） ----
    imbalance: str = "scale_pos_weight"  # none | scale_pos_weight | focal
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    scale_pos_weight_cap: float = 100.0

    # ---- 概率校准 ----
    calibration: str = "isotonic"     # none | isotonic | sigmoid

    # ---- 特征语义 ----
    use_monotone_constraints: bool = True
    use_categorical: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        if self.backend not in ("lightgbm", "sklearn", "auto"):
            raise ValueError(f"未知 backend: {self.backend}")
        if self.imbalance not in ("none", "scale_pos_weight", "focal"):
            raise ValueError(f"未知 imbalance 模式: {self.imbalance}")
        if self.calibration not in ("none", "isotonic", "sigmoid"):
            raise ValueError(f"未知 calibration 模式: {self.calibration}")
        if self.imbalance == "focal" and self.calibration == "none":
            raise ValueError(
                "focal loss 模式下禁止关闭概率校准：focal 的 raw 输出没有概率含义，"
                "直接展示给用户/参与风险分层就是错误数字（见模块 docstring 第 4 条）。"
            )


class LGBMRiskModel:
    """
    单一时程的二分类风险模型。1/3/5 年三个时程各训练一个实例，
    由 bank.HorizonBank 统一编排。

    用法（切分与标签构建见 leakage.py / labels.py）::

        model = LGBMRiskModel(config)
        model.fit(X_tr, y_tr, manifest, eval_set=(X_va, y_va))
        risk = model.predict_risk(X_new)          # 已校准概率
        model.save("artifacts/model_1y")
    """

    def __init__(self, config: LGBMConfig | None = None):
        self.config = config or LGBMConfig()
        self.config.validate()
        self.manifest: FeatureManifest | None = None
        self.backend_: str | None = None          # 实际生效后端
        self.booster_ = None                      # lightgbm.Booster
        self.sk_model_ = None                     # sklearn HistGB
        self.calibrator_ = None
        self.focal_: FocalBinary | None = None
        self.best_iteration_: int | None = None
        self.meta_: dict = {}

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        manifest: FeatureManifest,
        eval_set: tuple[pd.DataFrame, pd.Series | np.ndarray] | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> "LGBMRiskModel":
        """
        eval_set：必须是【时间上晚于训练集、且患者不重叠】的验证集
        （用 leakage.time_based_split 切出来的那种）。它同时承担三件事：
        早停、校准器拟合、训练过程监控。不传则三者全部停用并告警 ——
        只应出现在快速调试里。
        """
        if manifest is None or len(manifest) == 0:
            raise ValueError(
                "fit 必须传入训练特征对应的 FeatureManifest。"
                "没有清单就没有训练/推理一致性保障（base.py 模块说明第 3 条）。"
            )
        self.manifest = manifest

        y_arr = np.asarray(y, dtype=float)
        _check_binary(y_arr)
        Xa = manifest.align(X.copy(), strict=True)

        eval_pack = None
        if eval_set is not None:
            Xv, yv = eval_set
            yv_arr = np.asarray(yv, dtype=float)
            _check_binary(yv_arr)
            eval_pack = (manifest.align(Xv.copy(), strict=True), yv_arr)
        else:
            logger.warning(
                "未提供 eval_set：早停、概率校准、过拟合监控全部停用。"
                "此配置仅限调试，禁止用于产出上线模型。"
            )
            if self.config.imbalance == "focal":
                raise ValueError("focal 模式必须提供 eval_set（校准是强制的）。")

        self.backend_ = self._resolve_backend()
        pos_rate = float(y_arr.mean())
        logger.info(
            "开始训练 [backend=%s imbalance=%s calibration=%s] "
            "n_train=%d 阳性率=%.2f%% 特征=%d",
            self.backend_, self.config.imbalance, self.config.calibration,
            len(Xa), 100 * pos_rate, len(manifest),
        )

        if self.backend_ == "lightgbm":
            self._fit_lightgbm(Xa, y_arr, eval_pack, sample_weight)
        else:
            self._fit_sklearn(Xa, y_arr, eval_pack, sample_weight)

        if eval_pack is not None and self.config.calibration != "none":
            self._fit_calibrator(*eval_pack)

        self.meta_ = {
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "backend": self.backend_,
            "n_train": int(len(Xa)),
            "pos_rate_train": pos_rate,
            "n_features": len(manifest),
            "best_iteration": self.best_iteration_,
            "calibrated": self.calibrator_ is not None,
            "feature_hash": self._feature_hash(),
            "config": self.config.to_dict(),
        }
        return self

    # ------------------------------------------------------------------
    def _resolve_backend(self) -> str:
        cfg = self.config.backend
        lgb = _try_import_lightgbm()
        if cfg == "lightgbm":
            if lgb is None:
                raise RuntimeError(
                    "backend='lightgbm' 但当前环境未安装 lightgbm。"
                    "规范 3.1 模型栈固定，生产环境必须安装：pip install lightgbm。"
                    "开发调试可临时用 backend='auto' 降级到 sklearn 回退。"
                )
            return "lightgbm"
        if cfg == "sklearn":
            logger.warning("显式使用 sklearn 回退后端 —— 仅限开发环境。")
            return "sklearn"
        # auto
        if lgb is not None:
            return "lightgbm"
        logger.warning(
            "【降级告警】未检测到 lightgbm，自动回退到 sklearn "
            "HistGradientBoosting。该后端仅用于开发自测：不支持 focal loss、"
            "不支持外部验证集早停。生产部署必须安装 lightgbm（规范 3.1）。"
        )
        return "sklearn"

    # ------------------------------------------------------------------
    def _fit_lightgbm(self, X, y, eval_pack, sample_weight) -> None:
        import lightgbm as lgb  # noqa: PLC0415  已由 _resolve_backend 确认可用

        c = self.config
        params: dict = {
            "objective": "binary",
            "metric": ["auc", "binary_logloss"],
            "learning_rate": c.learning_rate,
            "num_leaves": c.num_leaves,
            "max_depth": c.max_depth,
            "min_child_samples": c.min_child_samples,
            "bagging_fraction": c.subsample,
            "bagging_freq": c.subsample_freq,
            "feature_fraction": c.colsample_bytree,
            "lambda_l1": c.reg_alpha,
            "lambda_l2": c.reg_lambda,
            "seed": c.seed,
            "deterministic": True,          # 可复现（规范 3.1 对可复现的要求）
            "force_row_wise": True,         # deterministic 所需
            "verbose": -1,
        }
        if c.use_monotone_constraints and self.manifest is not None:
            mono = self.manifest.monotone_constraints()
            if any(mono):
                # 装配期硬校验：LightGBM 对 categorical+monotone 是 C++ 级
                # fatal（直接终止进程、无 Python 堆栈）。在这里拦下并点名
                # 违规特征，把一次"进程凭空消失"变成一条能定位的报错。
                if c.use_categorical:
                    cat_set = set(self.manifest.categorical_names())
                    offenders = [
                        s.name for s, m in zip(self.manifest.specs, mono)
                        if m and s.name in cat_set
                    ]
                    if offenders:
                        raise ValueError(
                            "以下特征同时被标记为 categorical 与 monotone，"
                            f"LightGBM 禁止该组合（会 fatal 终止进程）: {offenders[:10]}"
                            f"{'…' if len(offenders) > 10 else ''}。"
                            "请修正对应 FeatureSpec（分类特征的 monotone 必须为 0）。"
                        )
                params["monotone_constraints"] = mono
                params["monotone_constraints_method"] = "advanced"
                logger.info("单调约束启用: %d/%d 个特征带方向", sum(1 for m in mono if m), len(mono))

        if c.imbalance == "scale_pos_weight":
            params["scale_pos_weight"] = compute_scale_pos_weight(y, c.scale_pos_weight_cap)
        elif c.imbalance == "focal":
            self.focal_ = FocalBinary(alpha=c.focal_alpha, gamma=c.focal_gamma)
            params["objective"] = self.focal_
            params["metric"] = ["auc"]      # AUC 只依赖排序，对 raw score 依然有效

        cats = self.manifest.categorical_names() if c.use_categorical else []
        for col in cats:
            X[col] = X[col].astype("category")

        dtrain = lgb.Dataset(X, label=y, weight=sample_weight,
                             categorical_feature=cats or "auto")
        valid_sets, valid_names = [dtrain], ["train"]
        callbacks = [lgb.log_evaluation(period=0)]
        if eval_pack is not None:
            Xv, yv = eval_pack
            for col in cats:
                Xv[col] = Xv[col].astype("category")
            dvalid = lgb.Dataset(Xv, label=yv, reference=dtrain,
                                 categorical_feature=cats or "auto")
            valid_sets.append(dvalid)
            valid_names.append("valid")
            callbacks.append(
                lgb.early_stopping(c.early_stopping_rounds, first_metric_only=True)
            )

        self.booster_ = lgb.train(
            params,
            dtrain,
            num_boost_round=c.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            feval=self.focal_.eval_metric if self.focal_ is not None else None,
            callbacks=callbacks,
        )
        self.best_iteration_ = self.booster_.best_iteration or c.n_estimators

    # ------------------------------------------------------------------
    def _fit_sklearn(self, X, y, eval_pack, sample_weight) -> None:
        from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415

        c = self.config
        if c.imbalance == "focal":
            raise RuntimeError(
                "focal loss 需要 LightGBM 自定义目标函数，sklearn 回退后端不支持。"
                "请安装 lightgbm，或改用 imbalance='scale_pos_weight'。"
            )
        mono = None
        if c.use_monotone_constraints and self.manifest is not None:
            m = self.manifest.monotone_constraints()
            mono = m if any(m) else None

        # HistGB 不接受外部验证集，早停只能在训练集内部随机切 10%。
        # 这与"时间在后的验证集"原则相悖 —— 是回退后端仅限开发环境的原因之一。
        self.sk_model_ = HistGradientBoostingClassifier(
            max_iter=c.n_estimators,
            learning_rate=c.learning_rate,
            max_leaf_nodes=c.num_leaves,
            max_depth=None if c.max_depth == -1 else c.max_depth,
            min_samples_leaf=c.min_child_samples,
            l2_regularization=c.reg_lambda,
            monotonic_cst=mono,
            early_stopping=eval_pack is not None,
            validation_fraction=0.1,
            n_iter_no_change=max(10, c.early_stopping_rounds // 10),
            class_weight="balanced" if c.imbalance == "scale_pos_weight" else None,
            random_state=c.seed,
        )
        self.sk_model_.fit(X, y, sample_weight=sample_weight)
        self.best_iteration_ = int(self.sk_model_.n_iter_)

    # ------------------------------------------------------------------
    def _fit_calibrator(self, Xv: pd.DataFrame, yv: np.ndarray) -> None:
        """校准器只用验证集拟合（训练集拟合=泄露 4）。"""
        p_uncal = self._predict_uncalibrated(Xv, already_aligned=True)
        if len(np.unique(yv)) < 2:
            logger.warning("验证集只有单一类别，无法拟合校准器，跳过。")
            return
        if self.config.calibration == "isotonic":
            from sklearn.isotonic import IsotonicRegression  # noqa: PLC0415

            self.calibrator_ = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            ).fit(p_uncal, yv)
        else:  # sigmoid / Platt
            from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

            eps = 1e-6
            logit = np.log(np.clip(p_uncal, eps, 1 - eps) / np.clip(1 - p_uncal, eps, 1 - eps))
            self.calibrator_ = LogisticRegression(C=1e6).fit(logit.reshape(-1, 1), yv)
        logger.info("概率校准器已拟合 (%s, n_valid=%d)", self.config.calibration, len(yv))

    def _apply_calibrator(self, p: np.ndarray) -> np.ndarray:
        if self.calibrator_ is None:
            return p
        from sklearn.isotonic import IsotonicRegression  # noqa: PLC0415

        if isinstance(self.calibrator_, IsotonicRegression):
            return np.asarray(self.calibrator_.predict(p))
        eps = 1e-6
        logit = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
        return np.asarray(self.calibrator_.predict_proba(logit.reshape(-1, 1))[:, 1])

    # ------------------------------------------------------------------
    # 预测
    # ------------------------------------------------------------------
    def _predict_uncalibrated(self, X: pd.DataFrame, already_aligned: bool = False) -> np.ndarray:
        if self.manifest is None:
            raise RuntimeError("模型尚未训练/加载。")
        Xa = X if already_aligned else self.manifest.align(X.copy(), strict=True)
        if self.backend_ == "lightgbm":
            if self.config.use_categorical:
                for col in self.manifest.categorical_names():
                    Xa[col] = Xa[col].astype("category")
            raw = self.booster_.predict(Xa, num_iteration=self.best_iteration_)
            if self.focal_ is not None:
                from .imbalance import _sigmoid  # noqa: PLC0415

                return _sigmoid(np.asarray(raw))
            return np.asarray(raw)
        return np.asarray(self.sk_model_.predict_proba(Xa)[:, 1])

    def predict_risk(self, X: pd.DataFrame, calibrated: bool = True) -> np.ndarray:
        """
        输出风险概率。线上一律用默认 calibrated=True；
        calibrated=False 只用于校准前后对比分析。
        """
        p = self._predict_uncalibrated(X)
        return self._apply_calibrator(p) if calibrated else p

    # ------------------------------------------------------------------
    # 可解释性（规范 3.2：SHAP 强制开启）
    # ------------------------------------------------------------------
    def shap_values(self, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """
        逐样本 SHAP 贡献（raw score 空间）。

        LightGBM 后端走内置 TreeSHAP（pred_contrib=True），零额外依赖；
        sklearn 回退后端尝试 shap 包，没有就明确报错。
        返回 (逐特征贡献 DataFrame[n, n_features], 基准值 array[n])。
        按指标/语义组聚合与 Top10 输出在 explain 模块（批次3）做。
        """
        if self.manifest is None:
            raise RuntimeError("模型尚未训练/加载。")
        Xa = self.manifest.align(X.copy(), strict=True)
        names = self.manifest.names

        if self.backend_ == "lightgbm":
            if self.config.use_categorical:
                for col in self.manifest.categorical_names():
                    Xa[col] = Xa[col].astype("category")
            contrib = self.booster_.predict(
                Xa, num_iteration=self.best_iteration_, pred_contrib=True
            )
            contrib = np.asarray(contrib)
            return pd.DataFrame(contrib[:, :-1], columns=names, index=Xa.index), contrib[:, -1]

        try:
            import shap  # noqa: PLC0415
        except ImportError as e:
            raise RuntimeError(
                "sklearn 回退后端计算 SHAP 需要安装 shap 包（pip install shap），"
                "或安装 lightgbm 使用其内置 TreeSHAP。"
            ) from e
        explainer = shap.TreeExplainer(self.sk_model_)
        sv = explainer.shap_values(Xa)
        sv = sv[1] if isinstance(sv, list) else sv
        base = np.full(len(Xa), float(np.ravel(explainer.expected_value)[-1]))
        return pd.DataFrame(np.asarray(sv), columns=names, index=Xa.index), base

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        """全局特征重要性（仅 LightGBM 后端提供；全局归因请以 SHAP 为准）。"""
        if self.backend_ != "lightgbm":
            raise RuntimeError(
                "sklearn 回退后端不提供内置全局重要性。"
                "请用 shap_values() 的均值绝对贡献（批次3 explain 模块封装）。"
            )
        imp = self.booster_.feature_importance(importance_type=importance_type)
        return pd.Series(imp, index=self.manifest.names).sort_values(ascending=False)

    # ------------------------------------------------------------------
    # 持久化（规范 4.3：版本管理的最小单元）
    # ------------------------------------------------------------------
    def _feature_hash(self) -> str:
        """特征清单指纹。加载/服务化时比对，特征集一变哈希必变。"""
        joined = "|".join(self.manifest.names) if self.manifest else ""
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]

    def save(self, path: str | Path) -> Path:
        if self.manifest is None or (self.booster_ is None and self.sk_model_ is None):
            raise RuntimeError("模型尚未训练，无法保存。")
        d = Path(path)
        d.mkdir(parents=True, exist_ok=True)

        self.manifest.save(d / _MANIFEST_FILE)
        with (d / _META_FILE).open("w", encoding="utf-8") as f:
            json.dump(self.meta_, f, ensure_ascii=False, indent=2)
        if self.calibrator_ is not None:
            joblib.dump(self.calibrator_, d / _CALIBRATOR_FILE)
        if self.backend_ == "lightgbm":
            # 不用 save_model(str(path))——LightGBM C 库不支持非 ASCII 路径
            # （Windows 下 D:\桌面\... 会变成乱码导致 "Could not open"）。
            # 改用 model_to_string() + Python open() 绕过。
            model_str = self.booster_.model_to_string(num_iteration=self.best_iteration_)
            with (d / _LGBM_MODEL_FILE).open("w", encoding="utf-8") as mf:
                mf.write(model_str)
        else:
            joblib.dump(self.sk_model_, d / _SKLEARN_MODEL_FILE)
        logger.info("模型已保存: %s (backend=%s, hash=%s)",
                    d, self.backend_, self.meta_.get("feature_hash"))
        return d

    @classmethod
    def load(cls, path: str | Path) -> "LGBMRiskModel":
        d = Path(path)
        with (d / _META_FILE).open("r", encoding="utf-8") as f:
            meta = json.load(f)
        cfg = LGBMConfig(**meta["config"])
        model = cls(cfg)
        model.meta_ = meta
        model.backend_ = meta["backend"]
        model.best_iteration_ = meta.get("best_iteration")
        model.manifest = FeatureManifest.load(d / _MANIFEST_FILE)

        if model._feature_hash() != meta.get("feature_hash"):
            raise RuntimeError(
                "特征清单指纹与 meta.json 不一致 —— 模型目录被篡改或文件错配，"
                "拒绝加载（加载错配模型的后果是全量预测静默出错）。"
            )
        if model.backend_ == "lightgbm":
            lgb = _try_import_lightgbm()
            if lgb is None:
                raise RuntimeError("该模型以 lightgbm 后端训练，加载环境必须安装 lightgbm。")
            # 不用 Booster(model_file=str(path))——LightGBM C 库不支持非 ASCII 路径。
            # 用 Python open() 读取后通过 model_str 参数加载。
            with (d / _LGBM_MODEL_FILE).open("r", encoding="utf-8") as mf:
                model.booster_ = lgb.Booster(model_str=mf.read())
            if cfg.imbalance == "focal":
                model.focal_ = FocalBinary(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
        else:
            model.sk_model_ = joblib.load(d / _SKLEARN_MODEL_FILE)
        calib = d / _CALIBRATOR_FILE
        if calib.exists():
            model.calibrator_ = joblib.load(calib)
        logger.info("模型已加载: %s (backend=%s)", d, model.backend_)
        return model
