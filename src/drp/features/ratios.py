"""
临床衍生特征：指标比值 + 公认临床复合评分（规范 2.2）。

规范原话："指标比值特征：AST/ALT、胆红素/白蛋白、血小板/胆红素等临床标准比值"。

这一层是**性价比最高的精度来源**，原因值得每个算法工程师理解清楚：

树模型（LightGBM）擅长学习阈值切分，但**极不擅长学习比值和乘积**。
要用轴平行切分逼近 "AST/ALT > 2" 这条斜线边界，需要大量的阶梯状分裂，
既消耗树深度又极易过拟合。而这些比值背后是几十年临床研究沉淀的机制知识
（AST/ALT 比值反映肝细胞损伤模式，FIB-4 反映肝纤维化程度），
直接算好喂进去，相当于免费给模型加了几百篇文献的先验。

在 3 万条量级的训练集上，这一层通常能带来 0.02~0.05 的 AUC 提升，
是所有特征工程动作里投入产出比最高的。

所有公式均标注文献出处。**禁止自创比值** —— 没有临床依据的比值组合
本质是噪声，会显著增加过拟合风险，且在可解释性输出时无法向医生交代。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ..data.constants import FEATURE_GROUP_RATIO
from ..data.reference import ReferenceRegistry
from .base import BaseFeatureBuilder, FeatureSpec

logger = logging.getLogger(__name__)

EPS = 1e-9


@dataclass(frozen=True)
class DerivedFeature:
    """一个临床衍生特征的定义。"""

    name: str
    requires: tuple[str, ...]  # 依赖的指标码，任一缺失则结果为 NaN
    fn: Callable[..., np.ndarray]
    description: str
    reference: str  # 文献出处，必填，会写进可解释性输出
    monotone: int = 0
    needs_age: bool = False
    needs_sex: bool = False


# ---------------------------------------------------------------------------
# 简单比值
# ---------------------------------------------------------------------------
def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """安全除法。分母接近 0 时返回 NaN 而非 inf —— inf 会毁掉树模型的分裂点。"""
    out = np.full(len(a), np.nan)
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > EPS)
    out[ok] = a[ok] / b[ok]
    return out


# ---------------------------------------------------------------------------
# 临床复合评分
# ---------------------------------------------------------------------------
def _fib4(age: np.ndarray, ast: np.ndarray, alt: np.ndarray, plt: np.ndarray) -> np.ndarray:
    """
    FIB-4 肝纤维化指数 = (年龄 × AST) / (血小板 × √ALT)

    Sterling RK, et al. Hepatology 2006;43(6):1317-25.
    临床切点: <1.45 排除进展性纤维化, >3.25 提示进展性纤维化。
    对 NAFLD、慢乙肝的纤维化进展预测有很强的判别力。
    """
    out = np.full(len(age), np.nan)
    ok = (
        np.isfinite(age) & np.isfinite(ast) & np.isfinite(alt) & np.isfinite(plt)
        & (plt > EPS) & (alt > EPS)
    )
    out[ok] = (age[ok] * ast[ok]) / (plt[ok] * np.sqrt(alt[ok]))
    return out


def _apri(ast: np.ndarray, plt: np.ndarray, uln_ast: float) -> np.ndarray:
    """
    APRI = (AST / AST正常值上限 × 100) / 血小板计数

    Wai CT, et al. Hepatology 2003;38(2):518-26.
    注意 uln_ast 必须取【本实验室】的 AST 上限，不是固定 40 ——
    这正是为什么本函数要从 registry 拿参考区间而不是写死常数。
    """
    out = np.full(len(ast), np.nan)
    ok = np.isfinite(ast) & np.isfinite(plt) & (plt > EPS) & (uln_ast > EPS)
    out[ok] = (ast[ok] / uln_ast * 100.0) / plt[ok]
    return out


def _egfr_ckd_epi_2021(crea_umol: np.ndarray, age: np.ndarray, sex: np.ndarray) -> np.ndarray:
    """
    eGFR，CKD-EPI 2021 去种族版公式。

    Inker LA, et al. N Engl J Med 2021;385:1737-1749.

        eGFR = 142 × min(Scr/κ,1)^α × max(Scr/κ,1)^(-1.200) × 0.9938^Age × 1.012(女性)
        κ = 0.7(女) / 0.9(男)，α = -0.241(女) / -0.302(男)，Scr 单位 mg/dL

    为什么必须算 eGFR 而不是直接用肌酐：
    肌酐受肌肉量影响极大，同样是 90 μmol/L，在肌肉发达的年轻男性是正常，
    在消瘦老年女性可能已经是 CKD 3 期。eGFR 把年龄性别的影响用生理模型
    显式扣除了，是 CKD 分期的国际标准，判别力远高于原始肌酐。
    """
    n = len(crea_umol)
    out = np.full(n, np.nan)
    scr = crea_umol / 88.4  # μmol/L -> mg/dL

    is_f = np.asarray(sex) == "F"
    is_m = np.asarray(sex) == "M"
    known = is_f | is_m
    ok = np.isfinite(scr) & np.isfinite(age) & (scr > EPS) & known
    if not ok.any():
        return out

    kappa = np.where(is_f, 0.7, 0.9)
    alpha = np.where(is_f, -0.241, -0.302)
    ratio = np.divide(scr, kappa, out=np.full(n, np.nan), where=ok)

    lo = np.minimum(ratio, 1.0)
    hi = np.maximum(ratio, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        val = 142.0 * (lo**alpha) * (hi**-1.200) * (0.9938**age)
    val = np.where(is_f, val * 1.012, val)
    out[ok] = val[ok]
    return out


def _tyg(tg_mmol: np.ndarray, glu_mmol: np.ndarray) -> np.ndarray:
    """
    TyG 甘油三酯-葡萄糖指数 = ln[TG(mg/dL) × 空腹血糖(mg/dL) / 2]

    Simental-Mendía LE, et al. Metab Syndr Relat Disord 2008;6(4):299-304.
    胰岛素抵抗的简易替代指标，与 HOMA-IR 高度相关但只需常规生化，
    是 2 型糖尿病发病预测的强预测因子。
    """
    tg_mg = tg_mmol / 0.01129
    glu_mg = glu_mmol / 0.05551
    out = np.full(len(tg_mmol), np.nan)
    ok = np.isfinite(tg_mg) & np.isfinite(glu_mg) & (tg_mg > EPS) & (glu_mg > EPS)
    out[ok] = np.log(tg_mg[ok] * glu_mg[ok] / 2.0)
    return out


def _homa_ir(glu_mmol: np.ndarray, ins: np.ndarray) -> np.ndarray:
    """HOMA-IR = 空腹血糖(mmol/L) × 空腹胰岛素(μIU/mL) / 22.5
    Matthews DR, et al. Diabetologia 1985;28(7):412-9."""
    out = np.full(len(glu_mmol), np.nan)
    ok = np.isfinite(glu_mmol) & np.isfinite(ins) & (glu_mmol > EPS) & (ins > EPS)
    out[ok] = glu_mmol[ok] * ins[ok] / 22.5
    return out


def _map_bp(sbp: np.ndarray, dbp: np.ndarray) -> np.ndarray:
    """平均动脉压 MAP = DBP + (SBP - DBP)/3。比单看收缩压更能反映器官灌注压。"""
    out = np.full(len(sbp), np.nan)
    ok = np.isfinite(sbp) & np.isfinite(dbp)
    out[ok] = dbp[ok] + (sbp[ok] - dbp[ok]) / 3.0
    return out


# ---------------------------------------------------------------------------
# 特征注册表
# ---------------------------------------------------------------------------
def build_derived_registry(registry: ReferenceRegistry) -> list[DerivedFeature]:
    """
    构造衍生特征定义表。需要参考区间的评分（如 APRI）在这里绑定实验室特定上限。
    """
    ast_meta = registry.get("AST")
    uln_ast = 40.0
    if ast_meta is not None:
        iv = ast_meta.match_interval("M", 40)
        if iv is not None and iv.upper is not None:
            uln_ast = float(iv.upper)

    defs: list[DerivedFeature] = [
        # ---------------- 肝脏 ----------------
        DerivedFeature(
            "ratio_AST_ALT",
            ("AST", "ALT"),
            lambda AST, ALT: _safe_div(AST, ALT),
            "De Ritis 比值(AST/ALT)。>2 提示酒精性肝病/肝硬化，<1 多见于病毒性肝炎",
            "De Ritis F. Clin Chim Acta 1957;2:70-4",
        ),
        DerivedFeature(
            "score_FIB4",
            ("AST", "ALT", "PLT"),
            lambda AST, ALT, PLT, age: _fib4(age, AST, ALT, PLT),
            "FIB-4 肝纤维化指数。<1.45 排除进展性纤维化，>3.25 高度提示",
            "Sterling RK, Hepatology 2006;43:1317-25",
            monotone=1,
            needs_age=True,
        ),
        DerivedFeature(
            "score_APRI",
            ("AST", "PLT"),
            lambda AST, PLT: _apri(AST, PLT, uln_ast),
            f"APRI 天冬氨酸转氨酶血小板比值指数(ULN_AST={uln_ast})",
            "Wai CT, Hepatology 2003;38:518-26",
            monotone=1,
        ),
        DerivedFeature(
            "ratio_TBIL_ALB",
            ("TBIL", "ALB"),
            lambda TBIL, ALB: _safe_div(TBIL, ALB),
            "胆红素/白蛋白比值。反映肝脏合成与排泄功能的综合失代偿程度",
            "Wu SJ, et al. Medicine 2019;98:e14834",
            monotone=1,
        ),
        DerivedFeature(
            "ratio_PLT_TBIL",
            ("PLT", "TBIL"),
            lambda PLT, TBIL: _safe_div(PLT, TBIL),
            "血小板/胆红素比值。门脉高压与肝功能储备的联合指标",
            "临床常用肝硬化评估比值",
            monotone=-1,
        ),
        DerivedFeature(
            "ratio_ALB_GLB",
            ("ALB", "TP"),
            lambda ALB, TP: _safe_div(ALB, np.where(np.isfinite(TP), TP - ALB, np.nan)),
            "白球比 A/G。降低提示慢性肝病或免疫球蛋白增多",
            "全国临床检验操作规程",
            monotone=-1,
        ),
        # ---------------- 肾脏 ----------------
        DerivedFeature(
            "score_eGFR",
            ("CREA",),
            lambda CREA, age, sex: _egfr_ckd_epi_2021(CREA, age, sex),
            "eGFR 估算肾小球滤过率(CKD-EPI 2021 去种族版)",
            "Inker LA, NEJM 2021;385:1737-49",
            monotone=-1,
            needs_age=True,
            needs_sex=True,
        ),
        DerivedFeature(
            "ratio_UREA_CREA",
            ("UREA", "CREA"),
            lambda UREA, CREA: _safe_div(UREA * 1000.0, CREA),
            "尿素/肌酐比值。升高提示肾前性因素(脱水/消化道出血)而非肾实质损害",
            "临床常用鉴别指标",
        ),
        DerivedFeature(
            "ratio_UA_CREA",
            ("UA", "CREA"),
            lambda UA, CREA: _safe_div(UA, CREA),
            "尿酸/肌酐比值。用于区分尿酸生成过多与排泄减少",
            "临床常用比值",
        ),
        # ---------------- 血脂 / 代谢 ----------------
        DerivedFeature(
            "calc_nonHDLC",
            ("TC", "HDLC"),
            lambda TC, HDLC: np.where(np.isfinite(TC) & np.isfinite(HDLC), TC - HDLC, np.nan),
            "非高密度脂蛋白胆固醇 = TC - HDL-C。ASCVD 风险预测优于 LDL-C 单项",
            "中国成人血脂异常防治指南(2023年修订版)",
            monotone=1,
        ),
        DerivedFeature(
            "ratio_TC_HDLC",
            ("TC", "HDLC"),
            lambda TC, HDLC: _safe_div(TC, HDLC),
            "Castelli 指数 I (TC/HDL-C)。心血管风险综合指标",
            "Castelli WP, Am J Med 1977;62:707-14",
            monotone=1,
        ),
        DerivedFeature(
            "ratio_LDLC_HDLC",
            ("LDLC", "HDLC"),
            lambda LDLC, HDLC: _safe_div(LDLC, HDLC),
            "Castelli 指数 II (LDL-C/HDL-C)",
            "Castelli WP, Am J Med 1977;62:707-14",
            monotone=1,
        ),
        DerivedFeature(
            "ratio_TG_HDLC",
            ("TG", "HDLC"),
            lambda TG, HDLC: _safe_div(TG, HDLC),
            "TG/HDL-C 比值。胰岛素抵抗与小而密 LDL 的替代标志",
            "Gaziano JM, Circulation 1997;96:2520-5",
            monotone=1,
        ),
        DerivedFeature(
            "score_TyG",
            ("TG", "GLU"),
            lambda TG, GLU: _tyg(TG, GLU),
            "TyG 甘油三酯-葡萄糖指数。胰岛素抵抗简易替代指标",
            "Simental-Mendía LE, Metab Syndr Relat Disord 2008;6:299-304",
            monotone=1,
        ),
        DerivedFeature(
            "score_HOMA_IR",
            ("GLU", "INS"),
            lambda GLU, INS: _homa_ir(GLU, INS),
            "HOMA-IR 胰岛素抵抗指数",
            "Matthews DR, Diabetologia 1985;28:412-9",
            monotone=1,
        ),
        # ---------------- 炎症 / 血液 ----------------
        DerivedFeature(
            "ratio_NLR",
            ("NEUT", "LYMPH"),
            lambda NEUT, LYMPH: _safe_div(NEUT, LYMPH),
            "中性粒细胞/淋巴细胞比值 NLR。全身炎症反应与不良预后的强预测因子",
            "Zahorec R. Bratisl Lek Listy 2001;102:5-14",
            monotone=1,
        ),
        DerivedFeature(
            "ratio_PLR",
            ("PLT", "LYMPH"),
            lambda PLT, LYMPH: _safe_div(PLT, LYMPH),
            "血小板/淋巴细胞比值 PLR。炎症与血栓风险联合指标",
            "Templeton AJ, Cancer Epidemiol Biomarkers Prev 2014;23:1204-12",
            monotone=1,
        ),
        # ---------------- 血压 ----------------
        DerivedFeature(
            "calc_MAP",
            ("SBP", "DBP"),
            lambda SBP, DBP: _map_bp(SBP, DBP),
            "平均动脉压 MAP。器官灌注压的核心指标",
            "生理学标准公式",
            monotone=1,
        ),
        DerivedFeature(
            "calc_PP",
            ("SBP", "DBP"),
            lambda SBP, DBP: np.where(
                np.isfinite(SBP) & np.isfinite(DBP), SBP - DBP, np.nan
            ),
            "脉压差 = SBP - DBP。增大提示动脉硬化程度加重",
            "Franklin SS, Circulation 1999;100:354-60",
            monotone=1,
        ),
    ]
    return defs


class ClinicalRatioBuilder(BaseFeatureBuilder):
    """
    临床衍生特征构造器。

    输入依赖 DeviationFeatureBuilder 产出的 {CODE}_value 列 ——
    即已经过单位换算和参考区间对齐的最近一次值。
    这样保证比值计算用的单位一定是 canonical 单位（否则 AST/ALT 还好，
    但 TyG 这种要求 mg/dL 的公式会算出完全错误的数）。
    """

    name = "clinical_ratio"

    def __init__(self, registry: ReferenceRegistry, enabled: list[str] | None = None):
        self.registry = registry
        self.definitions = build_derived_registry(registry)
        if enabled is not None:
            keep = set(enabled)
            self.definitions = [d for d in self.definitions if d.name in keep]

    def build(
        self,
        cohort: pd.DataFrame,
        records: pd.DataFrame,
        value_frame: pd.DataFrame | None = None,
        **kwargs,
    ) -> tuple[pd.DataFrame, list[FeatureSpec]]:
        if value_frame is None:
            raise ValueError(
                "ClinicalRatioBuilder 需要 value_frame（DeviationFeatureBuilder 的输出）。"
                "请通过 FeaturePipeline 调用，不要单独使用。"
            )

        cohort = cohort.reset_index(drop=True)
        n = len(cohort)
        from .deviation import _compute_age

        age = _compute_age(cohort)
        sex = (
            cohort["sex"].fillna("U").astype(str).str.upper().to_numpy()
            if "sex" in cohort.columns
            else np.full(n, "U")
        )

        feats: dict[str, np.ndarray] = {}
        specs: list[FeatureSpec] = []
        skipped: list[str] = []

        for d in self.definitions:
            cols = [f"{code}_value" for code in d.requires]
            if any(c not in value_frame.columns for c in cols):
                skipped.append(d.name)
                continue

            args = [value_frame[c].to_numpy(dtype=float) for c in cols]
            kw = {}
            if d.needs_age:
                kw["age"] = age
            if d.needs_sex:
                kw["sex"] = sex

            try:
                arr = np.asarray(d.fn(*args, **kw), dtype=float)
            except Exception as e:  # noqa: BLE001
                logger.error("衍生特征 %s 计算失败: %s", d.name, e)
                skipped.append(d.name)
                continue

            arr[~np.isfinite(arr)] = np.nan  # inf 一律转 NaN
            feats[d.name] = arr
            specs.append(
                FeatureSpec(
                    name=d.name,
                    group=FEATURE_GROUP_RATIO,
                    dtype="numeric",
                    indicator="+".join(d.requires),
                    description=f"{d.description} [依据: {d.reference}]",
                    monotone=d.monotone,
                )
            )

        if skipped:
            logger.info("跳过 %d 个衍生特征（依赖指标缺失）: %s", len(skipped), skipped)

        out = pd.DataFrame(feats, index=cohort.index)
        self._check_alignment(cohort, out)
        return out, specs
