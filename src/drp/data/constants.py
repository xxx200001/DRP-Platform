"""
核心常量定义。

对应规范 1.2：数据库必须区分【缺失 / 检查过正常 / 检查过异常】三种状态，
禁止统一填均值、填 0。

为什么这条是死命令（工程师必须理解，否则会自作主张去 fillna）：
医疗数据的缺失是 MNAR（非随机缺失）。医生给你开某项检查，这个"开了检查"
的动作本身就是强信号——说明医生临床上怀疑了什么。把缺失填成均值，等于
人为抹掉了这个信号，同时给模型灌进一个从未发生过的假观测值。
线下 AUC 会因为分布变平滑而虚高，线上遇到真实缺失模式立刻崩。

正确做法：
  1. 数值列保留 NaN —— LightGBM / XGBoost 原生支持 NaN 分裂方向学习
  2. 额外产出 `_status` 三态列，把"有没有测"这件事显式喂给模型
  3. 严禁 SimpleImputer(strategy='mean') 之类的全局填充
"""

from __future__ import annotations

from enum import IntEnum


class MeasureStatus(IntEnum):
    """指标三态。数值刻意用 IntEnum，方便直接进模型做 categorical 特征。"""

    MISSING = 0  # 未检查 —— 没有这条记录
    NORMAL = 1  # 检查过，落在该年龄/性别参考区间内
    ABNORMAL = 2  # 检查过，超出参考区间
    INVALID = 3  # 检查过，但数值被校验拦截（超生理极限 / 单位错误且无法纠正）

    @classmethod
    def label_cn(cls, v: "MeasureStatus | int") -> str:
        return {0: "未检查", 1: "检查正常", 2: "检查异常", 3: "数据无效"}[int(v)]


class AbnormalGrade(IntEnum):
    """
    异常分级（规范 2.2）。带符号：负数=偏低，正数=偏高。
    分级边界由 configs/reference_intervals.yaml 里每个指标的 grade_multiplier 决定，
    不写死在代码里——因为不同指标的临床危险梯度完全不同。
    例如 ALT 超上限 3 倍才算中度，而血钾超上限 1.2 倍就已经是危急值。
    """

    SEVERE_LOW = -3
    MODERATE_LOW = -2
    MILD_LOW = -1
    NORMAL = 0
    MILD_HIGH = 1
    MODERATE_HIGH = 2
    SEVERE_HIGH = 3

    @classmethod
    def label_cn(cls, v: "AbnormalGrade | int") -> str:
        return {
            -3: "重度偏低",
            -2: "中度偏低",
            -1: "轻度偏低",
            0: "正常",
            1: "轻度偏高",
            2: "中度偏高",
            3: "重度偏高",
        }[int(v)]


class TrendLabel(IntEnum):
    """
    时序趋势标签（规范 2.3）。
    注意 UNKNOWN 和 STABLE 必须分开：只测过 1 次 ≠ 测过 5 次都很平稳，
    后者的临床信息量大得多，合并会丢信息。
    """

    UNKNOWN = 0  # 观测点不足，无法判断
    STABLE = 1  # 变化幅度未超过 RCV（生物学变异阈值）
    RISING = 2
    FALLING = 3

    @classmethod
    def label_cn(cls, v: "TrendLabel | int") -> str:
        return {0: "数据不足", 1: "平稳", 2: "上升", 3: "下降"}[int(v)]


class PersistencePattern(IntEnum):
    """
    一过性异常 vs 持续性异常（规范 2.4）。

    临床意义天差地别：一次感冒导致的 CRP 升高，和连续三次随访都升高的 CRP，
    对慢病进展的预测价值完全不同。模型特征层必须能区分，否则会把大量
    急性一过性波动学成慢病风险信号，线上假阳性暴涨。
    """

    UNKNOWN = 0
    NEVER_ABNORMAL = 1  # 从未异常
    TRANSIENT = 2  # 一过性：异常出现过但未连续，且已回落
    RECURRENT = 3  # 反复性：多次异常但中间有回落
    PERSISTENT = 4  # 持续性：最近连续 >=2 次异常

    @classmethod
    def label_cn(cls, v: "PersistencePattern | int") -> str:
        return {0: "数据不足", 1: "从未异常", 2: "一过性异常", 3: "反复异常", 4: "持续异常"}[int(v)]


# ---------------------------------------------------------------------------
# 长表标准列名。全平台统一，禁止各模块自定义别名。
# ---------------------------------------------------------------------------
COL_PATIENT_ID = "patient_id"
COL_INDICATOR = "indicator_code"
COL_VALUE = "value"
COL_UNIT = "unit"
COL_MEASURED_AT = "measured_at"
COL_INDEX_DATE = "index_date"
COL_SEX = "sex"
COL_BIRTH_DATE = "birth_date"
COL_AGE = "age"

SEX_MALE = "M"
SEX_FEMALE = "F"
SEX_UNKNOWN = "U"

# 特征分组标签，用于 SHAP 归因聚合、漂移监控分组、以及前端"风险归因"展示
FEATURE_GROUP_DEMO = "demographic"
FEATURE_GROUP_RAW = "raw_value"
FEATURE_GROUP_STATUS = "status"
FEATURE_GROUP_DEVIATION = "deviation"
FEATURE_GROUP_RATIO = "clinical_ratio"
FEATURE_GROUP_TEMPORAL = "temporal"
FEATURE_GROUP_CONFOUNDER = "confounder"

ALL_FEATURE_GROUPS = (
    FEATURE_GROUP_DEMO,
    FEATURE_GROUP_RAW,
    FEATURE_GROUP_STATUS,
    FEATURE_GROUP_DEVIATION,
    FEATURE_GROUP_RATIO,
    FEATURE_GROUP_TEMPORAL,
    FEATURE_GROUP_CONFOUNDER,
)
