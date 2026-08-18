"""
干扰因子剔除（规范 2.4）。

规范原话：
  - 用药标记、近期感染标记、熬夜/临时异常干扰权重降级
  - 模型特征层区分：一过性异常 vs 持续性异常

【核心设计决策：标记而非剔除】

很多人第一反应是把受干扰的样本删掉。这是错的，原因有两条：

  1. 选择偏倚。服药人群、感染人群不是随机缺失的，删掉他们等于让训练分布
     偏离真实人群分布。而且服药人群往往就是高危人群——把他们删了，
     模型在最需要它的人身上反而没见过数据。

  2. 线上无法执行。线上来一个正在吃他汀的用户，你不能拒绝服务。

正确做法是三件事一起做：
  a) 把干扰状态本身变成特征（模型可以学"在服他汀且 LDL 仍高 = 更危险"）
  b) 给受影响指标打 confounded 标记
  c) 输出 reliability 权重，供下游可解释性模块在归因时降级展示

产出特征：
  med_{class}                药物类别标记 0/1
  med_n_classes              用药类别总数（多重用药本身是共病负担的代理变量）
  acute_{state}              急性状态标记 0/1
  phys_{state}               生理状态标记 0/1
  {CODE}_confounded          该指标是否受任一干扰影响 0/1
  {CODE}_reliability         该指标可靠度 0~1，1 表示无干扰
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..data.constants import (
    COL_INDEX_DATE,
    COL_INDICATOR,
    COL_MEASURED_AT,
    COL_PATIENT_ID,
    COL_VALUE,
    FEATURE_GROUP_CONFOUNDER,
)
from .base import BaseFeatureBuilder, FeatureSpec

logger = logging.getLogger(__name__)

# 用药记录表的标准列名
COL_MED_NAME = "medication_name"
COL_MED_START = "start_date"
COL_MED_END = "end_date"


@dataclass(frozen=True)
class Affect:
    indicator: str
    direction: str  # up / down
    strength: float


@dataclass
class MedicationClass:
    key: str
    name_cn: str
    keywords: tuple[str, ...]
    affects: tuple[Affect, ...]


@dataclass
class StateRule:
    key: str
    name_cn: str
    window_days: int
    triggers: tuple[tuple[str, str, float], ...]  # (indicator, op, value)
    unreliable: tuple[tuple[str, float], ...]  # (indicator, strength)


class ConfounderConfig:
    def __init__(self, cfg: dict[str, Any]):
        self.version = cfg.get("version", "unknown")
        self.med_classes: list[MedicationClass] = [
            MedicationClass(
                key=k,
                name_cn=v.get("name_cn", k),
                keywords=tuple(str(x).lower() for x in (v.get("keywords") or [])),
                affects=tuple(
                    Affect(a["indicator"].upper(), a.get("direction", "down"), float(a.get("strength", 0.5)))
                    for a in (v.get("affects") or [])
                ),
            )
            for k, v in (cfg.get("medication_classes") or {}).items()
        ]
        self.acute_states = self._parse_states(cfg.get("acute_states") or {})
        self.phys_states = self._parse_states(cfg.get("physiological_states") or {})

    @staticmethod
    def _parse_states(raw: dict[str, Any]) -> list[StateRule]:
        out = []
        for k, v in raw.items():
            out.append(
                StateRule(
                    key=k,
                    name_cn=v.get("name_cn", k),
                    window_days=int(v.get("window_days", 30)),
                    triggers=tuple(
                        (t["indicator"].upper(), t.get("op", ">"), float(t["value"]))
                        for t in (v.get("triggers") or [])
                    ),
                    unreliable=tuple(
                        (u["indicator"].upper(), float(u.get("strength", 0.5)))
                        for u in (v.get("unreliable") or [])
                    ),
                )
            )
        return out

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ConfounderConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls(yaml.safe_load(f))


class ConfounderFeatureBuilder(BaseFeatureBuilder):
    name = "confounder"

    def __init__(self, config: ConfounderConfig, indicators: list[str] | None = None):
        self.config = config
        self.indicators = [c.upper() for c in indicators] if indicators else None

    # ------------------------------------------------------------------
    def build(
        self,
        cohort: pd.DataFrame,
        records: pd.DataFrame,
        medications: pd.DataFrame | None = None,
        state_flags: pd.DataFrame | None = None,
        **kwargs,
    ) -> tuple[pd.DataFrame, list[FeatureSpec]]:
        """
        medications : 可选，列 patient_id / medication_name / start_date / end_date
                      end_date 为空表示仍在服用
        state_flags : 可选，列 patient_id / index_date / pregnancy / non_fasting /
                      strenuous_exercise 等 0/1 列，来自问卷或采血登记
        """
        cohort = cohort.reset_index(drop=True)
        n = len(cohort)
        index_dates = pd.to_datetime(cohort[COL_INDEX_DATE]).to_numpy()
        patients = cohort[COL_PATIENT_ID].to_numpy()

        feats: dict[str, np.ndarray] = {}
        specs: list[FeatureSpec] = []

        # 每个指标累计的"不可靠强度"，最后转成 reliability
        indicator_codes = set(self.indicators or [])
        if not indicator_codes and not records.empty:
            indicator_codes = set(records[COL_INDICATOR].astype(str).unique())
        penalty: dict[str, np.ndarray] = {c: np.zeros(n) for c in indicator_codes}

        # ---------------- 用药标记 ----------------
        med_active = self._resolve_medications(patients, index_dates, medications)
        n_classes = np.zeros(n)
        for mc in self.config.med_classes:
            flag = med_active.get(mc.key)
            if flag is None:
                continue
            feats[f"med_{mc.key}"] = flag.astype(float)
            specs.append(
                FeatureSpec(
                    name=f"med_{mc.key}",
                    group=FEATURE_GROUP_CONFOUNDER,
                    dtype="binary",
                    description=f"索引日期时正在使用{mc.name_cn}",
                )
            )
            n_classes += flag
            for a in mc.affects:
                if a.indicator in penalty:
                    penalty[a.indicator] = np.maximum(penalty[a.indicator], flag * a.strength)

        if med_active:
            feats["med_n_classes"] = n_classes
            specs.append(
                FeatureSpec(
                    name="med_n_classes",
                    group=FEATURE_GROUP_CONFOUNDER,
                    dtype="numeric",
                    description="索引日期时使用的药物类别数(多重用药=共病负担代理变量)",
                    monotone=1,
                )
            )

        # ---------------- 急性状态 ----------------
        for rule in self.config.acute_states:
            flag = self._detect_state(rule, patients, index_dates, records)
            feats[f"acute_{rule.key}"] = flag
            specs.append(
                FeatureSpec(
                    name=f"acute_{rule.key}",
                    group=FEATURE_GROUP_CONFOUNDER,
                    dtype="binary",
                    description=f"索引日期前{rule.window_days}天内存在{rule.name_cn}",
                )
            )
            for ind, strength in rule.unreliable:
                if ind in penalty:
                    penalty[ind] = np.maximum(penalty[ind], flag * strength)

        # ---------------- 生理状态 ----------------
        for rule in self.config.phys_states:
            flag = self._read_state_flag(rule.key, cohort, state_flags)
            if flag is None:
                continue
            feats[f"phys_{rule.key}"] = flag
            specs.append(
                FeatureSpec(
                    name=f"phys_{rule.key}",
                    group=FEATURE_GROUP_CONFOUNDER,
                    dtype="binary",
                    description=f"索引日期时处于{rule.name_cn}状态",
                )
            )
            for ind, strength in rule.unreliable:
                if ind in penalty:
                    penalty[ind] = np.maximum(penalty[ind], flag * strength)

        # ---------------- 可靠度输出 ----------------
        for code, pen in penalty.items():
            if pen.max() == 0:
                continue  # 该指标从未被任何干扰影响，不产出常量列
            feats[f"{code}_confounded"] = (pen > 0).astype(float)
            feats[f"{code}_reliability"] = 1.0 - pen
            specs.extend(
                [
                    FeatureSpec(
                        name=f"{code}_confounded",
                        group=FEATURE_GROUP_CONFOUNDER,
                        dtype="binary",
                        indicator=code,
                        description=f"{code} 受用药/急性状态干扰",
                    ),
                    FeatureSpec(
                        name=f"{code}_reliability",
                        group=FEATURE_GROUP_CONFOUNDER,
                        dtype="numeric",
                        indicator=code,
                        description=f"{code} 可靠度 0~1(1=无干扰)。供归因输出时降级展示",
                    ),
                ]
            )

        out = pd.DataFrame(feats, index=cohort.index)
        self._check_alignment(cohort, out)
        return out, specs

    # ------------------------------------------------------------------
    def _resolve_medications(
        self,
        patients: np.ndarray,
        index_dates: np.ndarray,
        medications: pd.DataFrame | None,
    ) -> dict[str, np.ndarray]:
        """
        判定索引日期时每个患者在用哪些药物类别。

        时间窗判定必须严格：只有 start_date <= index_date 且
        (end_date 为空 或 end_date >= index_date) 才算在用。
        用 start_date > index_date 的记录会构成未来数据泄露。
        """
        if medications is None or medications.empty:
            return {}

        required = {COL_PATIENT_ID, COL_MED_NAME}
        if not required.issubset(medications.columns):
            logger.warning("用药表缺少必需列 %s，跳过用药特征", required - set(medications.columns))
            return {}

        med = medications.copy()
        med[COL_MED_START] = (
            pd.to_datetime(med[COL_MED_START], errors="coerce")
            if COL_MED_START in med.columns
            else pd.NaT
        )
        med[COL_MED_END] = (
            pd.to_datetime(med[COL_MED_END], errors="coerce")
            if COL_MED_END in med.columns
            else pd.NaT
        )
        med["_name_lower"] = med[COL_MED_NAME].astype(str).str.lower()

        # 药名 -> 类别（一个药名可能命中多个类别，全部保留）
        med["_classes"] = med["_name_lower"].map(self._classify_med)

        by_patient: dict = {}
        for pid, grp in med.groupby(COL_PATIENT_ID, observed=True):
            by_patient[pid] = grp

        out = {mc.key: np.zeros(len(patients), dtype=bool) for mc in self.config.med_classes}
        for i, (pid, idx_date) in enumerate(zip(patients, index_dates)):
            grp = by_patient.get(pid)
            if grp is None:
                continue
            idx_ts = pd.Timestamp(idx_date)
            started = grp[COL_MED_START].isna() | (grp[COL_MED_START] <= idx_ts)
            not_ended = grp[COL_MED_END].isna() | (grp[COL_MED_END] >= idx_ts)
            active = grp[started & not_ended]
            for cls_list in active["_classes"]:
                for k in cls_list:
                    out[k][i] = True

        # 丢掉全 0 的类别，避免产出常量列（常量列对树模型无用且污染 SHAP）
        return {k: v for k, v in out.items() if v.any()}

    def _classify_med(self, name_lower: str) -> tuple[str, ...]:
        hits = []
        for mc in self.config.med_classes:
            if any(kw in name_lower for kw in mc.keywords):
                hits.append(mc.key)
        return tuple(hits)

    # ------------------------------------------------------------------
    def _detect_state(
        self,
        rule: StateRule,
        patients: np.ndarray,
        index_dates: np.ndarray,
        records: pd.DataFrame,
    ) -> np.ndarray:
        """基于检验值触发规则判定急性状态。窗口内任一触发条件命中即为 1。"""
        n = len(patients)
        flag = np.zeros(n)
        if records.empty or not rule.triggers:
            return flag

        trigger_inds = {t[0] for t in rule.triggers}
        rec = records[records[COL_INDICATOR].astype(str).isin(trigger_inds)]
        if rec.empty:
            return flag

        rec = rec.copy()
        rec[COL_MEASURED_AT] = pd.to_datetime(rec[COL_MEASURED_AT])
        by_patient = {pid: g for pid, g in rec.groupby(COL_PATIENT_ID, observed=True)}
        window = pd.Timedelta(days=rule.window_days)

        for i, (pid, idx_date) in enumerate(zip(patients, index_dates)):
            g = by_patient.get(pid)
            if g is None:
                continue
            idx_ts = pd.Timestamp(idx_date)
            w = g[(g[COL_MEASURED_AT] <= idx_ts) & (g[COL_MEASURED_AT] >= idx_ts - window)]
            if w.empty:
                continue
            for ind, op, val in rule.triggers:
                sub = w[w[COL_INDICATOR].astype(str) == ind][COL_VALUE]
                if sub.empty:
                    continue
                hit = (sub > val).any() if op == ">" else (sub < val).any()
                if hit:
                    flag[i] = 1.0
                    break
        return flag

    @staticmethod
    def _read_state_flag(
        key: str, cohort: pd.DataFrame, state_flags: pd.DataFrame | None
    ) -> np.ndarray | None:
        """生理状态来自问卷/采血登记，无法从检验值推断，缺失就不产出该特征。"""
        if key in cohort.columns:
            return pd.to_numeric(cohort[key], errors="coerce").fillna(0).to_numpy(dtype=float)
        if state_flags is not None and key in state_flags.columns:
            merged = cohort[[COL_PATIENT_ID, COL_INDEX_DATE]].merge(
                state_flags, on=[COL_PATIENT_ID, COL_INDEX_DATE], how="left"
            )
            return pd.to_numeric(merged[key], errors="coerce").fillna(0).to_numpy(dtype=float)
        return None
