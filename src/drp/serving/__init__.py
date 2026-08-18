from .attribution import (
    COVERAGE_WARN,
    DEFAULT_TOP_N,
    AttributionEngine,
    ChangeAttribution,
    ChangeFactor,
    FactorContribution,
    RiskAttribution,
    explain_change,
)
from .audit import (
    PII_FIELD_NAMES,
    PII_PATTERNS,
    AuditLogger,
    PIIError,
    PredictionRecord,
    assert_no_pii,
    pseudonymize,
    scan_pii,
)
from .compliance import (
    DISCLAIMER,
    FORBIDDEN_TERMS,
    ComplianceError,
    Violation,
    assert_compliant,
    attach_disclaimer,
    is_compliant,
    safe_fallback,
    scan,
)
from .drift import (
    LEVEL_ALERT,
    LEVEL_INSUFFICIENT,
    LEVEL_OK,
    LEVEL_WATCH,
    MISSING_ALERT,
    MISSING_WATCH,
    PSI_ALERT,
    PSI_WATCH,
    DriftMonitor,
    DriftReport,
    FeatureDrift,
    FeatureProfile,
    ReferenceProfile,
    population_stability_index,
)
from .referral import (
    DEPARTMENT_RULES,
    IndicatorFinding,
    Priority,
    ReferralAdvice,
    ReferralEngine,
    ReferralItem,
    extract_demographics,
    grade_value,
)
from .trend import (
    IndicatorComparison,
    IndicatorSeries,
    RiskTrajectory,
    RiskTrajectoryPoint,
    TrendEngine,
    TrendReport,
    build_trend_report,
    render_trend_text,
    risk_trajectory_from_audit,
)
from .feedback import (
    FeedbackOrchestrator,
    FollowUpFeedback,
    RetrainJob,
    ReviewCase,
    ReviewQueue,
)
from .rollout import (
    ABComparison,
    RoutingDecision,
    TrafficRouter,
    build_ab_comparison,
)
from .service import (
    PredictionResult,
    RiskPredictionService,
    RiskTierScheme,
    ServiceConfig,
)

__all__ = [
    # 归因（规范 3.2 / 3.3 / 6）
    "AttributionEngine", "RiskAttribution", "FactorContribution",
    "explain_change", "ChangeAttribution", "ChangeFactor",
    "DEFAULT_TOP_N", "COVERAGE_WARN",
    # 漂移监控（规范 3.2 / 4.3）
    "ReferenceProfile", "FeatureProfile", "DriftMonitor", "DriftReport",
    "FeatureDrift", "population_stability_index",
    "PSI_WATCH", "PSI_ALERT", "MISSING_WATCH", "MISSING_ALERT",
    "LEVEL_OK", "LEVEL_WATCH", "LEVEL_ALERT", "LEVEL_INSUFFICIENT",
    # 全链路日志与脱敏（规范 4.2 / 1.2 / 1.3）
    "AuditLogger", "PredictionRecord", "PIIError",
    "scan_pii", "assert_no_pii", "pseudonymize",
    "PII_PATTERNS", "PII_FIELD_NAMES",
    # 合规（规范 7）
    "ComplianceError", "Violation", "scan", "is_compliant", "assert_compliant",
    "attach_disclaimer", "safe_fallback", "DISCLAIMER", "FORBIDDEN_TERMS",
    # 服务入口（规范 4.2 / 4.3 / 6 / 7）
    "RiskPredictionService", "ServiceConfig", "RiskTierScheme", "PredictionResult",
    # 智能就医建议（规范 6）
    "ReferralEngine", "ReferralAdvice", "ReferralItem", "IndicatorFinding",
    "Priority", "DEPARTMENT_RULES", "grade_value", "extract_demographics",
    # 趋势追踪与时序对比报告（规范 6）
    "TrendEngine", "TrendReport", "build_trend_report", "render_trend_text",
    "IndicatorComparison", "IndicatorSeries",
    "RiskTrajectory", "RiskTrajectoryPoint", "risk_trajectory_from_audit",
    # 随访回流与复盘（规范 1.3 / 4.2）
    "FollowUpFeedback", "FeedbackOrchestrator", "ReviewCase", "ReviewQueue", "RetrainJob",
    # 模型灰度与 A/B（规范 4.3）
    "TrafficRouter", "RoutingDecision", "ABComparison", "build_ab_comparison",
]
