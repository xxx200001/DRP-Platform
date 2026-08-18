"""
单位换算 + 量级校验 + 生理极限拦截（规范 4.1）。

规范原话："上游错一个数据，下游模型全错，必须强校验。"

这一层是整个平台的第一道关口，也是线上事故最高发的地方。真实化验单里出现过的坑：
  - 肌酐写 1.2，单位 mg/dL —— 不换算直接入模，模型以为这人肌酐 1.2 μmol/L（不可能）
  - 血糖写 108，单位漏填 —— 实际是 mg/dL，按 mmol/L 读就是致命高血糖
  - 血小板写 250000，单位 /uL —— 按 10^9/L 读就是 250000，超生理极限
  - OCR 把 "4.5" 认成 "45"，小数点丢了
  - 单位写成 "umol/l" / "μmol/L" / "uMol/L" 三种大小写

处理策略分三档，严格按顺序：
  1. 单位已知 -> 换算到 canonical_unit
  2. 单位缺失/无法识别 -> 尝试量级推断（只在 magnitude_fix=True 且结果唯一时生效）
  3. 换算后仍超生理极限 -> 判 INVALID，拒绝入模，写告警日志

第 3 档是硬拦截：宁可丢一条数据（走 MISSING 三态，模型能处理），
也绝不能让一个错误数值进特征层。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum

from .reference import IndicatorMeta, ReferenceRegistry

logger = logging.getLogger(__name__)


class ValidationCode(str, Enum):
    """校验结果码。全部写入全链路日志，用于错误样本复盘（规范 4.2）。"""

    OK = "OK"
    UNIT_CONVERTED = "UNIT_CONVERTED"  # 单位识别并换算成功
    UNIT_ASSUMED_CANONICAL = "UNIT_ASSUMED_CANONICAL"  # 单位缺失，按标准单位处理且数值合理
    MAGNITUDE_FIXED = "MAGNITUDE_FIXED"  # 量级自动纠错（x1000 / /1000 等）
    UNIT_UNKNOWN = "UNIT_UNKNOWN"  # 单位无法识别且数值不合理
    OUT_OF_PLAUSIBLE = "OUT_OF_PLAUSIBLE"  # 超生理极限
    NOT_A_NUMBER = "NOT_A_NUMBER"
    UNKNOWN_INDICATOR = "UNKNOWN_INDICATOR"
    CRITICAL_VALUE = "CRITICAL_VALUE"  # 危急值：数据有效，但要走告警通道


@dataclass
class ValidationResult:
    indicator_code: str | None
    value: float | None  # 已换算到 canonical_unit；None 表示拒绝入模
    unit: str | None
    code: ValidationCode
    is_valid: bool
    is_critical: bool = False
    original_value: float | None = None
    original_unit: str | None = None
    detail: str = ""

    def to_log_dict(self) -> dict:
        """全链路日志用。每一条数据的处理过程都必须可回溯。"""
        return {
            "indicator_code": self.indicator_code,
            "value": self.value,
            "unit": self.unit,
            "code": self.code.value,
            "is_valid": self.is_valid,
            "is_critical": self.is_critical,
            "original_value": self.original_value,
            "original_unit": self.original_unit,
            "detail": self.detail,
        }


# 常见量级纠错候选因子。只用 10 的整数次幂，不做任意缩放 —— 任意缩放会把
# 真实的异常值"修"成正常值，属于灾难性错误。
_MAGNITUDE_FACTORS = (0.000001, 0.001, 0.01, 0.1, 10.0, 100.0, 1000.0, 1000000.0)


def normalize_unit(unit: str | None) -> str:
    """
    单位字符串归一化。吃掉大小写、希腊字母 μ、全角、空格差异。
    注意不能吃掉 mg/L 和 mg/dL 的区别 —— 那是 10 倍差。
    """
    if unit is None:
        return ""
    s = str(unit).strip()
    s = s.replace("μ", "u").replace("µ", "u").replace("Μ", "u")
    s = s.replace("１０", "10").replace("＾", "^").replace("／", "/")
    s = s.replace(" ", "").replace("\u3000", "")
    # 统一 dL/DL/dl -> dL, L/l -> L，但保留 d 前缀
    s = s.replace("DL", "dL").replace("dl", "dL")
    s = s.replace("MG", "mg").replace("Mg", "mg")
    s = s.replace("MMOL", "mmol").replace("MMol", "mmol").replace("Mmol", "mmol")
    s = s.replace("UMOL", "umol").replace("UMol", "umol").replace("Umol", "umol")
    s = s.replace("MMHG", "mmHg").replace("mmhg", "mmHg")
    if s.endswith("/l"):
        s = s[:-2] + "/L"
    return s


class UnitValidator:
    """单位与数值校验器。无状态，可跨线程复用。"""

    def __init__(self, registry: ReferenceRegistry):
        self.registry = registry
        # 预建归一化后的换算表，避免每条数据都做字符串处理
        self._conv: dict[str, dict[str, float]] = {}
        for code in registry.codes:
            meta = registry.require(code)
            table = {normalize_unit(meta.canonical_unit): 1.0}
            for u, f in meta.unit_conversions.items():
                table[normalize_unit(u)] = f
            self._conv[code] = table

    # ------------------------------------------------------------------
    def validate(
        self,
        raw_name: str,
        raw_value: object,
        raw_unit: str | None = None,
    ) -> ValidationResult:
        """
        单条检验结果的完整校验。raw_name 可以是标准码，也可以是任意别名。
        """
        code = self.registry.resolve_alias(raw_name) or (
            raw_name.upper() if raw_name and raw_name.upper() in self.registry else None
        )
        if code is None:
            return ValidationResult(
                indicator_code=None,
                value=None,
                unit=None,
                code=ValidationCode.UNKNOWN_INDICATOR,
                is_valid=False,
                original_unit=raw_unit,
                detail=f"未登记的指标名: {raw_name!r}。请补充到 reference_intervals.yaml 的 aliases。",
            )

        meta = self.registry.require(code)
        value = _to_float(raw_value)
        if value is None:
            return ValidationResult(
                indicator_code=code,
                value=None,
                unit=None,
                code=ValidationCode.NOT_A_NUMBER,
                is_valid=False,
                original_unit=raw_unit,
                detail=f"数值无法解析: {raw_value!r}",
            )

        unit_key = normalize_unit(raw_unit)
        conv_table = self._conv[code]

        # ---- 第 1 档：单位已知，直接换算 ----
        if unit_key and unit_key in conv_table:
            factor = conv_table[unit_key]
            converted = value * factor
            return self._finalize(
                meta,
                converted,
                original_value=value,
                original_unit=raw_unit,
                code=(
                    ValidationCode.OK
                    if factor == 1.0
                    else ValidationCode.UNIT_CONVERTED
                ),
                detail=("" if factor == 1.0 else f"{raw_unit} -> {meta.canonical_unit} (x{factor})"),
            )

        # ---- 第 2 档：单位未知或缺失 ----
        if meta.is_plausible(value):
            # 按标准单位读取时数值合理，接受但标记（便于后续统计单位缺失率）
            return self._finalize(
                meta,
                value,
                original_value=value,
                original_unit=raw_unit,
                code=(
                    ValidationCode.OK
                    if not unit_key
                    and False  # 单位缺失一律标记，不能静默当作 OK
                    else ValidationCode.UNIT_ASSUMED_CANONICAL
                ),
                detail=f"单位缺失或未识别({raw_unit!r})，按 {meta.canonical_unit} 处理，数值在生理范围内",
            )

        # ---- 量级自动纠错 ----
        if meta.magnitude_fix:
            fixed = self._try_magnitude_fix(meta, value)
            if fixed is not None:
                return self._finalize(
                    meta,
                    fixed,
                    original_value=value,
                    original_unit=raw_unit,
                    code=ValidationCode.MAGNITUDE_FIXED,
                    detail=f"量级纠错 {value} -> {fixed} ({meta.canonical_unit})",
                )

        return ValidationResult(
            indicator_code=code,
            value=None,
            unit=meta.canonical_unit,
            code=(ValidationCode.UNIT_UNKNOWN if unit_key else ValidationCode.OUT_OF_PLAUSIBLE),
            is_valid=False,
            original_value=value,
            original_unit=raw_unit,
            detail=(
                f"{meta.name_cn}={value} {raw_unit or '(无单位)'} 超出生理极限 "
                f"{meta.plausible_range}，拒绝入模"
            ),
        )

    # ------------------------------------------------------------------
    def _try_magnitude_fix(self, meta: IndicatorMeta, value: float) -> float | None:
        """
        量级纠错。只在【唯一一个因子】能把数值拉回生理范围时才生效。
        有多个候选就放弃 —— 猜错比丢弃危险得多。
        """
        hits = [value * f for f in _MAGNITUDE_FACTORS if meta.is_plausible(value * f)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            logger.warning(
                "指标 %s 值 %s 存在多个量级纠错候选 %s，放弃自动纠错并拒绝入模",
                meta.code,
                value,
                hits,
            )
        return None

    def _finalize(
        self,
        meta: IndicatorMeta,
        value: float,
        *,
        original_value: float,
        original_unit: str | None,
        code: ValidationCode,
        detail: str,
    ) -> ValidationResult:
        if not meta.is_plausible(value):
            return ValidationResult(
                indicator_code=meta.code,
                value=None,
                unit=meta.canonical_unit,
                code=ValidationCode.OUT_OF_PLAUSIBLE,
                is_valid=False,
                original_value=original_value,
                original_unit=original_unit,
                detail=f"换算后 {value} 仍超出生理极限 {meta.plausible_range}",
            )
        critical = meta.is_critical(value)
        if critical:
            logger.warning(
                "危急值告警: %s = %s %s (危急阈值 low=%s high=%s)",
                meta.name_cn,
                value,
                meta.canonical_unit,
                meta.critical_low,
                meta.critical_high,
            )
        return ValidationResult(
            indicator_code=meta.code,
            value=value,
            unit=meta.canonical_unit,
            code=code,
            is_valid=True,
            is_critical=critical,
            original_value=original_value,
            original_unit=original_unit,
            detail=detail,
        )


def _to_float(v: object) -> float | None:
    """
    宽松数值解析。处理化验单常见写法：
      "<0.01" -> 0.01   ">1000" -> 1000   "3.5↑" -> 3.5   "12,345" -> 12345
    注意 "<" ">" 的处理是保守的：取边界值本身，不做半值替换。
    """
    if v is None:
        return None
    if isinstance(v, (int, float)):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    s = str(v).strip()
    if not s:
        return None
    for ch in "<>≤≥=约±":
        s = s.replace(ch, "")
    s = s.replace(",", "").replace("↑", "").replace("↓", "").replace("H", "").replace("L", "")
    s = s.strip()
    try:
        f = float(s)
    except ValueError:
        return None
    return None if (math.isnan(f) or math.isinf(f)) else f
