"""
数据清洗管线（规范 1.1 / 1.2 / 4.1）。

输入：长表 DataFrame，一行一条检验结果
    patient_id | indicator_code(或别名) | value | unit | measured_at
输出：清洗后的长表 + 逐条校验报告 + 清洗统计

清洗顺序不可调换，每一步都依赖上一步的结果：
    1. 结构校验（必需列、类型、时间可解析）
    2. 指标名归一化 + 单位换算 + 生理极限拦截
    3. 去重（同患者同指标同时刻多条）
    4. 时间轴异常拦截（未来时间、早于出生日期）
    5. 三态状态标注（MISSING / NORMAL / ABNORMAL / INVALID）

关于"清洗误诊噪声样本"（规范 1.1 最后一条）：
标签层面的噪声清洗放在 cohort.py 里做，不在这里。原因是误诊判定需要结局
标签和随访信息，属于队列构建阶段的事，混进数值清洗会造成职责纠缠，
而且极易引入标签泄露（用未来信息清洗当前特征）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .constants import (
    COL_INDICATOR,
    COL_MEASURED_AT,
    COL_PATIENT_ID,
    COL_UNIT,
    COL_VALUE,
    MeasureStatus,
)
from .reference import ReferenceRegistry
from .units import UnitValidator, ValidationCode

logger = logging.getLogger(__name__)

REQUIRED_COLS = (COL_PATIENT_ID, COL_INDICATOR, COL_VALUE, COL_MEASURED_AT)

# 清洗后长表新增列
COL_STATUS = "status"
COL_VALID = "is_valid"
COL_CRITICAL = "is_critical"
COL_VCODE = "validation_code"
COL_ORIG_VALUE = "original_value"
COL_ORIG_UNIT = "original_unit"


class DuplicatePolicy:
    """同患者 + 同指标 + 同时刻 出现多条记录时的处理策略。"""

    MEDIAN = "median"  # 取中位数，抗单点 OCR 错误，默认
    LAST = "last"  # 取最后一条（按录入顺序）
    MAX_ABS_DEVIATION = "max_dev"  # 取偏离参考区间最远的（保守，宁可高估风险）


@dataclass
class CleaningReport:
    """清洗统计。必须打到监控里 —— 这些比率的突变是最早的数据质量告警信号。"""

    n_input: int = 0
    n_output: int = 0
    n_unknown_indicator: int = 0
    n_not_a_number: int = 0
    n_out_of_plausible: int = 0
    n_unit_converted: int = 0
    n_unit_assumed: int = 0
    n_magnitude_fixed: int = 0
    n_critical: int = 0
    n_duplicates_merged: int = 0
    n_future_timestamp: int = 0
    unknown_names: dict[str, int] = field(default_factory=dict)
    rejected_samples: list[dict] = field(default_factory=list)

    @property
    def reject_rate(self) -> float:
        return 0.0 if self.n_input == 0 else 1.0 - self.n_output / self.n_input

    def summary(self) -> str:
        return (
            f"清洗完成: 输入 {self.n_input} 条 -> 输出 {self.n_output} 条 "
            f"(拒绝率 {self.reject_rate:.2%})\n"
            f"  未知指标 {self.n_unknown_indicator} | 非数值 {self.n_not_a_number} | "
            f"超生理极限 {self.n_out_of_plausible}\n"
            f"  单位换算 {self.n_unit_converted} | 单位缺失按标准处理 {self.n_unit_assumed} | "
            f"量级纠错 {self.n_magnitude_fixed}\n"
            f"  危急值 {self.n_critical} | 重复合并 {self.n_duplicates_merged} | "
            f"未来时间戳 {self.n_future_timestamp}"
        )

    def assert_quality(self, max_reject_rate: float = 0.05) -> None:
        """
        训练前的质量闸门。拒绝率过高说明上游 OCR 或词典有系统性问题，
        这时候硬训出来的模型线上必然拉胯，直接抛异常挡住。
        """
        if self.reject_rate > max_reject_rate:
            raise ValueError(
                f"数据拒绝率 {self.reject_rate:.2%} 超过阈值 {max_reject_rate:.2%}。"
                f"最常见未知指标: {sorted(self.unknown_names.items(), key=lambda x: -x[1])[:10]}。"
                "请先修复归一化词典/单位映射，不要带病训练。"
            )


class LabDataCleaner:
    def __init__(
        self,
        registry: ReferenceRegistry,
        duplicate_policy: str = DuplicatePolicy.MEDIAN,
        max_rejected_samples: int = 500,
    ):
        self.registry = registry
        self.validator = UnitValidator(registry)
        self.duplicate_policy = duplicate_policy
        self.max_rejected_samples = max_rejected_samples

    # ------------------------------------------------------------------
    def clean(
        self,
        df: pd.DataFrame,
        demographics: pd.DataFrame | None = None,
        now: pd.Timestamp | None = None,
    ) -> tuple[pd.DataFrame, CleaningReport]:
        """
        demographics: 可选，含 patient_id / sex / birth_date，
                      用于状态标注时匹配年龄性别分层参考区间。
                      缺失时状态只能标 NORMAL/ABNORMAL 的兜底并集判断。
        """
        report = CleaningReport(n_input=len(df))
        self._check_schema(df)

        work = df.copy()
        work[COL_MEASURED_AT] = pd.to_datetime(work[COL_MEASURED_AT], errors="coerce")
        n_bad_time = int(work[COL_MEASURED_AT].isna().sum())
        if n_bad_time:
            logger.warning("丢弃 %d 条时间戳无法解析的记录", n_bad_time)
            work = work[work[COL_MEASURED_AT].notna()]

        # 未来时间戳拦截：这是数据回流环节最常见的脏数据来源之一
        now = now or pd.Timestamp.now()
        future_mask = work[COL_MEASURED_AT] > now
        report.n_future_timestamp = int(future_mask.sum())
        if report.n_future_timestamp:
            logger.warning("丢弃 %d 条时间戳在未来的记录", report.n_future_timestamp)
            work = work[~future_mask]

        # ---- 逐条校验 ----
        rows = self._validate_rows(work, report)
        if not rows:
            return _empty_clean_frame(), report

        out = pd.DataFrame(rows)

        # ---- 去重 ----
        out, n_merged = self._deduplicate(out)
        report.n_duplicates_merged = n_merged

        # ---- 三态标注 ----
        out = self._annotate_status(out, demographics)

        report.n_output = len(out)
        logger.info(report.summary())
        return out.reset_index(drop=True), report

    # ------------------------------------------------------------------
    def _check_schema(self, df: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"输入长表缺少必需列: {missing}。标准列名见 data/constants.py，"
                "全平台统一，禁止各模块自定义。"
            )

    def _validate_rows(self, work: pd.DataFrame, report: CleaningReport) -> list[dict]:
        has_unit = COL_UNIT in work.columns
        rows: list[dict] = []

        pids = work[COL_PATIENT_ID].to_numpy()
        names = work[COL_INDICATOR].to_numpy()
        values = work[COL_VALUE].to_numpy()
        units = work[COL_UNIT].to_numpy() if has_unit else np.full(len(work), None)
        times = work[COL_MEASURED_AT].to_numpy()

        for pid, name, val, unit, ts in zip(pids, names, values, units, times):
            res = self.validator.validate(name, val, unit)

            if res.code is ValidationCode.UNKNOWN_INDICATOR:
                report.n_unknown_indicator += 1
                key = str(name)
                report.unknown_names[key] = report.unknown_names.get(key, 0) + 1
            elif res.code is ValidationCode.NOT_A_NUMBER:
                report.n_not_a_number += 1
            elif res.code in (ValidationCode.OUT_OF_PLAUSIBLE, ValidationCode.UNIT_UNKNOWN):
                report.n_out_of_plausible += 1
            elif res.code is ValidationCode.UNIT_CONVERTED:
                report.n_unit_converted += 1
            elif res.code is ValidationCode.UNIT_ASSUMED_CANONICAL:
                report.n_unit_assumed += 1
            elif res.code is ValidationCode.MAGNITUDE_FIXED:
                report.n_magnitude_fixed += 1

            if res.is_critical:
                report.n_critical += 1

            if not res.is_valid:
                if len(report.rejected_samples) < self.max_rejected_samples:
                    d = res.to_log_dict()
                    d[COL_PATIENT_ID] = pid
                    d["raw_name"] = name
                    d[COL_MEASURED_AT] = ts
                    report.rejected_samples.append(d)
                continue

            rows.append(
                {
                    COL_PATIENT_ID: pid,
                    COL_INDICATOR: res.indicator_code,
                    COL_VALUE: res.value,
                    COL_UNIT: res.unit,
                    COL_MEASURED_AT: ts,
                    COL_VALID: True,
                    COL_CRITICAL: res.is_critical,
                    COL_VCODE: res.code.value,
                    COL_ORIG_VALUE: res.original_value,
                    COL_ORIG_UNIT: res.original_unit,
                }
            )
        return rows

    def _deduplicate(self, out: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        keys = [COL_PATIENT_ID, COL_INDICATOR, COL_MEASURED_AT]
        dup_mask = out.duplicated(subset=keys, keep=False)
        n_dup_rows = int(dup_mask.sum())
        if n_dup_rows == 0:
            return out, 0

        n_groups = int(out[dup_mask].groupby(keys, observed=True).ngroups)
        logger.info("发现 %d 条重复记录（%d 组），按策略 %s 合并", n_dup_rows, n_groups, self.duplicate_policy)

        if self.duplicate_policy == DuplicatePolicy.LAST:
            out = out.drop_duplicates(subset=keys, keep="last")
        elif self.duplicate_policy == DuplicatePolicy.MEDIAN:
            agg = {
                COL_VALUE: "median",
                COL_UNIT: "first",
                COL_VALID: "first",
                COL_CRITICAL: "max",
                COL_VCODE: "first",
                COL_ORIG_VALUE: "median",
                COL_ORIG_UNIT: "first",
            }
            out = out.groupby(keys, as_index=False, observed=True).agg(agg)
        elif self.duplicate_policy == DuplicatePolicy.MAX_ABS_DEVIATION:
            out = out.sort_values(COL_VALUE).drop_duplicates(subset=keys, keep="last")
        else:
            raise ValueError(f"未知去重策略: {self.duplicate_policy}")

        return out, n_dup_rows - len(out[out.duplicated(subset=keys, keep=False)])

    def _annotate_status(
        self, out: pd.DataFrame, demographics: pd.DataFrame | None
    ) -> pd.DataFrame:
        """
        标注三态。这里只区分 NORMAL / ABNORMAL —— MISSING 是在特征宽表
        拼接阶段才产生的（没有记录 = 缺失），INVALID 已在上一步被过滤掉。
        """
        sex_map: dict = {}
        birth_map: dict = {}
        if demographics is not None and len(demographics):
            demo = demographics.set_index(COL_PATIENT_ID)
            if "sex" in demo.columns:
                sex_map = demo["sex"].to_dict()
            if "birth_date" in demo.columns:
                birth_map = pd.to_datetime(demo["birth_date"], errors="coerce").to_dict()

        statuses = np.empty(len(out), dtype=np.int8)
        for i, (pid, code, val, ts) in enumerate(
            zip(
                out[COL_PATIENT_ID].to_numpy(),
                out[COL_INDICATOR].to_numpy(),
                out[COL_VALUE].to_numpy(),
                out[COL_MEASURED_AT].to_numpy(),
            )
        ):
            meta = self.registry.get(code)
            if meta is None:
                statuses[i] = MeasureStatus.INVALID
                continue
            sex = sex_map.get(pid)
            birth = birth_map.get(pid)
            age = _age_at(birth, ts)
            iv = meta.match_interval(sex or "U", age)
            if iv is None:
                statuses[i] = MeasureStatus.NORMAL  # 无区间可判，保守按正常，但值仍保留
            else:
                statuses[i] = MeasureStatus.NORMAL if iv.contains(val) else MeasureStatus.ABNORMAL

        out = out.copy()
        out[COL_STATUS] = statuses
        return out


def _age_at(birth, ts) -> float | None:
    if birth is None or pd.isna(birth):
        return None
    delta = pd.Timestamp(ts) - pd.Timestamp(birth)
    return delta.days / 365.25


def _empty_clean_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            COL_PATIENT_ID,
            COL_INDICATOR,
            COL_VALUE,
            COL_UNIT,
            COL_MEASURED_AT,
            COL_VALID,
            COL_CRITICAL,
            COL_VCODE,
            COL_ORIG_VALUE,
            COL_ORIG_UNIT,
            COL_STATUS,
        ]
    )
