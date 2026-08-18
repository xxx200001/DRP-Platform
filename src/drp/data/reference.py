"""
参考区间引擎（规范 2.1 / 2.2 的地基）。

职责：
  1. 加载 configs/reference_intervals.yaml，构建指标元数据注册表
  2. 按 (指标, 性别, 年龄) 匹配最具体的参考区间
  3. 提供 RCV（参考变化值）计算，供时序模块判定"变化是否真实"
  4. 支持多实验室版本（lab_id），换医院不用改代码

设计要点 —— 为什么 lab_id 要做成一等公民：
不同医院检验科的参考区间不同。如果模型在 A 院数据上训练，偏离度特征是按
A 院区间算的；上线到 B 院却继续用 A 院区间，特征分布会整体平移，AUC 直接
掉一大截，而且监控只能看到"漂移告警"却找不到根因。所以参考区间必须跟着
数据源走，并且在全链路日志里记录用了哪个版本（规范 4.2）。
"""

from __future__ import annotations

import functools
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .constants import SEX_FEMALE, SEX_MALE, SEX_UNKNOWN

logger = logging.getLogger(__name__)

_SEX_ANY = "ANY"


@dataclass(frozen=True)
class RefInterval:
    """一条具体的参考区间。lower/upper 允许为 None 表示单侧区间。"""

    lower: float | None
    upper: float | None
    sex: str
    age_low: float
    age_high: float

    @property
    def has_both_bounds(self) -> bool:
        return self.lower is not None and self.upper is not None

    @property
    def center(self) -> float | None:
        if not self.has_both_bounds:
            return None
        return (self.lower + self.upper) / 2.0  # type: ignore[operator]

    @property
    def half_width(self) -> float | None:
        if not self.has_both_bounds:
            return None
        hw = (self.upper - self.lower) / 2.0  # type: ignore[operator]
        return hw if hw > 0 else None

    def contains(self, value: float) -> bool:
        if self.lower is not None and value < self.lower:
            return False
        if self.upper is not None and value > self.upper:
            return False
        return True


@dataclass(frozen=True)
class GradeMultiplier:
    mild: float = 1.0
    moderate: float = 2.0
    severe: float = 3.0


@dataclass
class IndicatorMeta:
    """单个指标的全部元数据。"""

    code: str
    name_cn: str
    aliases: tuple[str, ...]
    canonical_unit: str
    unit_conversions: dict[str, float]
    plausible_range: tuple[float, float]
    magnitude_fix: bool
    critical_low: float | None
    critical_high: float | None
    grade_multiplier: GradeMultiplier
    cv_intra: float
    cv_analytical: float
    log_transform: bool
    intervals: tuple[RefInterval, ...] = field(default=())

    # ---------------- 参考区间匹配 ----------------
    def match_interval(self, sex: str, age: float | None) -> RefInterval | None:
        """
        匹配规则：先按性别精确匹配（M/F），再按年龄区间落点。
        同时命中多条时取【年龄跨度最窄】的一条 —— 越窄越具体。
        性别未知时只用 ANY 区间；若没有 ANY 区间则退化取男女区间的并集。
        若年龄超出所有分段（如 0 岁 / 未设生日 / 极端年龄），稳健降级匹配最适区间，保证参考区间永不落空。
        """
        if not self.intervals:
            return None

        age_val = age if age is not None else -1.0

        def age_ok(iv: RefInterval) -> bool:
            if age is None or age <= 0:
                return True
            return iv.age_low <= age_val < iv.age_high

        sex_norm = (sex or SEX_UNKNOWN).upper()

        if sex_norm in (SEX_MALE, SEX_FEMALE):
            candidates = [
                iv for iv in self.intervals if iv.sex in (sex_norm, _SEX_ANY) and age_ok(iv)
            ]
            # 性别专属区间优先于 ANY
            specific = [iv for iv in candidates if iv.sex == sex_norm]
            pool = specific or candidates
        else:
            pool = [iv for iv in self.intervals if iv.sex == _SEX_ANY and age_ok(iv)]
            if not pool:
                merged = self._merge_sex_intervals(age_ok)
                if merged is not None:
                    return merged

        if not pool:
            # 稳健兜底：若精准分段落空（如 age 极值），取最相近的性别/通用区间
            if sex_norm in (SEX_MALE, SEX_FEMALE):
                fallback = [iv for iv in self.intervals if iv.sex in (sex_norm, _SEX_ANY)]
                specific = [iv for iv in fallback if iv.sex == sex_norm]
                pool = specific or fallback
            else:
                pool = list(self.intervals)

        if not pool:
            return None
        return min(pool, key=lambda iv: iv.age_high - iv.age_low)

    def _merge_sex_intervals(self, age_ok) -> RefInterval | None:
        """性别未知的兜底：取男女区间并集，避免把正常值误判成异常。"""
        pool = [iv for iv in self.intervals if age_ok(iv)]
        if not pool:
            return None
        lowers = [iv.lower for iv in pool if iv.lower is not None]
        uppers = [iv.upper for iv in pool if iv.upper is not None]
        return RefInterval(
            lower=min(lowers) if lowers else None,
            upper=max(uppers) if uppers else None,
            sex=_SEX_ANY,
            age_low=min(iv.age_low for iv in pool),
            age_high=max(iv.age_high for iv in pool),
        )

    # ---------------- 校验 ----------------
    def is_plausible(self, value: float) -> bool:
        lo, hi = self.plausible_range
        return lo <= value <= hi

    def is_critical(self, value: float) -> bool:
        if self.critical_low is not None and value <= self.critical_low:
            return True
        if self.critical_high is not None and value >= self.critical_high:
            return True
        return False

    # ---------------- RCV ----------------
    @functools.cached_property
    def rcv(self) -> float:
        """
        参考变化值 Reference Change Value（双侧 95%）：
            RCV = 1.96 * sqrt(2) * sqrt(CVa^2 + CVi^2) ≈ 2.77 * sqrt(CVa^2 + CVi^2)

        含义：两次检测之间的相对变化超过 RCV，才能认为是"真实的生理变化"，
        否则只是分析误差 + 个体内生物学波动。

        这是本平台判定趋势（上升/下降/平稳）的唯一依据。
        绝对不要用"变化 > 5%"这种拍脑袋阈值 —— 血钠 CVi 只有 0.7%，变 5%
        已经是危急情况；而 CRP 的 CVi 高达 42%，变 5% 纯属噪声。
        用同一个阈值会同时制造大量假阳性和假阴性。
        """
        return 2.77 * math.sqrt(self.cv_analytical**2 + self.cv_intra**2)


class ReferenceRegistry:
    """指标注册表。全平台唯一入口，禁止在业务代码里硬编码参考区间。"""

    def __init__(self, config: dict[str, Any]):
        self.version: str = config.get("version", "unknown")
        self.source: str = config.get("source", "")
        self.lab_id: str = config.get("lab_id", "__DEFAULT__")
        self._defaults: dict[str, Any] = config.get("defaults", {}) or {}
        self._indicators: dict[str, IndicatorMeta] = {}
        self._alias_index: dict[str, str] = {}

        for code, raw in (config.get("indicators") or {}).items():
            meta = self._build_meta(code, raw)
            self._indicators[meta.code] = meta
            for alias in meta.aliases:
                key = normalize_alias(alias)
                if key in self._alias_index and self._alias_index[key] != meta.code:
                    raise ValueError(
                        f"别名冲突: '{alias}' 同时指向 {self._alias_index[key]} 和 {meta.code}。"
                        "归一化词典必须一对一，否则 OCR 结构化会把两个指标混淆。"
                    )
                self._alias_index[key] = meta.code

        logger.info(
            "参考区间注册表加载完成: version=%s lab_id=%s 指标数=%d 别名数=%d",
            self.version,
            self.lab_id,
            len(self._indicators),
            len(self._alias_index),
        )

    # ---------------- 构造 ----------------
    def _build_meta(self, code: str, raw: dict[str, Any]) -> IndicatorMeta:
        d = self._defaults
        gm_raw = raw.get("grade_multiplier") or d.get("grade_multiplier") or {}
        crit = raw.get("critical") or {}
        plausible = raw.get("plausible_range")
        if not plausible or len(plausible) != 2:
            raise ValueError(f"指标 {code} 缺少 plausible_range，生理极限拦截无法工作")

        intervals: list[RefInterval] = []
        for iv in raw.get("reference") or []:
            age = iv.get("age", [0, 120])
            intervals.append(
                RefInterval(
                    lower=_as_float(iv.get("lower")),
                    upper=_as_float(iv.get("upper")),
                    sex=str(iv.get("sex", _SEX_ANY)).upper(),
                    age_low=float(age[0]),
                    age_high=float(age[1]),
                )
            )

        return IndicatorMeta(
            code=code.upper(),
            name_cn=raw.get("name_cn", code),
            aliases=tuple(raw.get("aliases") or [code]),
            canonical_unit=raw.get("canonical_unit", ""),
            unit_conversions={str(k): float(v) for k, v in (raw.get("unit_conversions") or {}).items()},
            plausible_range=(float(plausible[0]), float(plausible[1])),
            magnitude_fix=bool(raw.get("magnitude_fix", d.get("magnitude_fix", True))),
            critical_low=_as_float(crit.get("low")),
            critical_high=_as_float(crit.get("high")),
            grade_multiplier=GradeMultiplier(
                mild=float(gm_raw.get("mild", 1.0)),
                moderate=float(gm_raw.get("moderate", 2.0)),
                severe=float(gm_raw.get("severe", 3.0)),
            ),
            cv_intra=float(raw.get("cv_intra", d.get("cv_intra", 0.10))),
            cv_analytical=float(raw.get("cv_analytical", d.get("cv_analytical", 0.05))),
            log_transform=bool(raw.get("log_transform", d.get("log_transform", False))),
            intervals=tuple(intervals),
        )

    # ---------------- 查询 ----------------
    def get(self, code: str) -> IndicatorMeta | None:
        return self._indicators.get(code.upper())

    def require(self, code: str) -> IndicatorMeta:
        meta = self.get(code)
        if meta is None:
            raise KeyError(f"未注册指标: {code}。请先在 reference_intervals.yaml 中登记。")
        return meta

    def resolve_alias(self, raw_name: str) -> str | None:
        """OCR 结构化归一化入口（规范 4.1）：任意别名 -> 标准指标码。"""
        return self._alias_index.get(normalize_alias(raw_name))

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(self._indicators.keys())

    def __contains__(self, code: str) -> bool:
        return code.upper() in self._indicators

    def __len__(self) -> int:
        return len(self._indicators)

    # ---------------- 加载 ----------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "ReferenceRegistry":
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            return cls(yaml.safe_load(f))

    @classmethod
    def from_lab(cls, base_dir: str | Path, lab_id: str) -> "ReferenceRegistry":
        """
        按实验室加载。约定：
            configs/reference_intervals.yaml            默认版本
            configs/labs/<lab_id>/reference_intervals.yaml   本院覆盖版本
        """
        base = Path(base_dir)
        lab_path = base / "labs" / lab_id / "reference_intervals.yaml"
        if lab_path.exists():
            return cls.from_yaml(lab_path)
        logger.warning(
            "未找到实验室 %s 的专属参考区间，回退到默认版本。"
            "上线前必须补齐，否则偏离度特征会带系统性偏差。",
            lab_id,
        )
        return cls.from_yaml(base / "reference_intervals.yaml")


def to_halfwidth(s: str) -> str:
    """
    全角转半角。中文化验单 OCR 输出里全角字母数字极其常见
    （"ＡＬＴ" "ＨｂＡ１ｃ"），不处理会导致大量指标识别失败，
    进而被当成"未知指标"丢弃 —— 表现为数据拒绝率莫名其妙地高。

    Unicode 全角 ASCII 区: U+FF01~U+FF5E 对应半角 U+0021~U+007E，偏移 0xFEE0。
    表意空格 U+3000 单独处理。
    """
    out = []
    for ch in s:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:
            out.append(chr(code - 0xFEE0))
        elif code == 0x3000:
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def normalize_alias(name: str) -> str:
    """
    别名归一化：全角转半角、去空格、去分隔符、统一大写。
    化验单 OCR 出来的名字千奇百怪，这一层要尽可能吃掉格式噪声。

    注意不要过度归一化：不能去掉数字（HbA1c 的 1），
    也不能去掉连字符以外的有意义符号（HDL-C 去掉 - 后仍可区分，但 β 不能丢）。
    """
    if name is None:
        return ""
    s = to_halfwidth(str(name)).strip().upper()
    for ch in " \t_·．.-()（）:：,，/":
        s = s.replace(ch, "")
    return s


def _as_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    return float(v)
