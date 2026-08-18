"""
服务层单元测试（批次4）。

这一层的 bug 有个共同特征：**不会让任何东西报错**。
归因聚合错了，用户看到的是一份措辞流畅但归因错误的报告；
PSI 分箱用错了，监控永远一片绿；日志少存一个字段，要等到出事故那天
才发现查不下去。所以本文件的重点是精确断言，而不是"跑通就行"：

    - 归因：逐特征贡献之和必须等于聚合值（浮点级精确）
    - 漂移：缺失率突变必须在 PSI 为 0 时依然告警
    - 日志：不可重建的记录必须写不进去；明文 PII 必须写不进去
    - 服务：日志写失败必须让预测失败，不能"结果照发"

运行方式：
    pytest tests/test_serving.py -q
    python tests/test_serving.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import pytest

    _HAVE_PYTEST = True
except ImportError:  # pragma: no cover
    _HAVE_PYTEST = False

    class _Skip(Exception):
        pass

    class pytest:  # type: ignore[no-redef]
        @staticmethod
        @contextmanager
        def raises(exc):
            try:
                yield
            except exc:
                return
            raise AssertionError(f"预期抛出 {exc.__name__} 但没有抛出")

        @staticmethod
        def importorskip(name):
            try:
                return __import__(name)
            except ImportError:
                raise _Skip(name)


from drp.data.constants import (  # noqa: E402
    FEATURE_GROUP_DEMO,
    FEATURE_GROUP_DEVIATION,
    MeasureStatus,
)
from drp.features.base import FeatureManifest, FeatureSpec  # noqa: E402
from drp.serving import (  # noqa: E402
    LEVEL_ALERT,
    LEVEL_INSUFFICIENT,
    LEVEL_OK,
    AttributionEngine,
    AuditLogger,
    ChangeFactor,
    ComplianceError,
    DriftMonitor,
    PIIError,
    PredictionRecord,
    ReferenceProfile,
    RiskPredictionService,
    RiskTierScheme,
    ServiceConfig,
    assert_compliant,
    attach_disclaimer,
    explain_change,
    is_compliant,
    population_stability_index,
    pseudonymize,
    safe_fallback,
    scan,
    scan_pii,
)

SALT = "unit-test-salt"
FEATS = {
    "ALT_value": 0.5, "ALT_dev": 0.9, "ALT_slope": 0.2,
    "AST_value": -0.4, "AST_dev": 0.6,
    "PLT_value": -0.7, "PLT_status": 0.3,
    "age": 0.15,
}


# ===========================================================================
# 桩模型：SHAP 由已知线性规则生成，聚合结果可被精确断言
# ===========================================================================
def _manifest() -> FeatureManifest:
    mf = FeatureManifest()
    for c in FEATS:
        ind = c.split("_")[0] if "_" in c else None
        mf.add(
            FeatureSpec(
                name=c,
                group=FEATURE_GROUP_DEVIATION if ind else FEATURE_GROUP_DEMO,
                dtype="numeric",
                indicator=ind,
                description=c,
            )
        )
    return mf


class StubModel:
    """线性打分 + 线性 SHAP。真实 TreeSHAP 的正确性由模型层保证，这里只测归因编排。"""

    def __init__(self, manifest=None, weights=None, backend: str = "lightgbm"):
        self.manifest = manifest or _manifest()
        self.w = weights or FEATS
        self.backend_ = backend
        self.meta_ = {
            "feature_hash": "abc123", "backend": backend,
            "calibrated": True, "trained_at": "2026-08-15T00:00:00Z",
        }

    def _raw(self, X):
        Xa = self.manifest.align(X.copy(), strict=False)
        return sum(self.w.get(c, 0.0) * Xa[c].fillna(0.0) for c in Xa.columns) - 2.0

    def predict_risk(self, X, calibrated: bool = True):
        return (1 / (1 + np.exp(-self._raw(X)))).to_numpy()

    def shap_values(self, X):
        Xa = self.manifest.align(X.copy(), strict=False)
        contrib = pd.DataFrame(
            {c: self.w.get(c, 0.0) * Xa[c].fillna(0.0) for c in Xa.columns}, index=Xa.index
        )
        return contrib, np.full(len(Xa), -2.0)


def _X(n=5, seed=0):
    r = np.random.default_rng(seed)
    return pd.DataFrame({c: r.normal(0, 1, n) for c in FEATS})


def _tmp() -> Path:
    return Path(tempfile.mkdtemp())


# ===========================================================================
# 1. 合规拦截（规范 7）
# ===========================================================================
def test_forbidden_terms_are_caught():
    bad = [
        "您已确诊为脂肪肝。",
        "建议服用水飞蓟宾，每日 3 次。",
        "该方案可根治本病，保证不会复发。",
        "请按处方用药，剂量遵医嘱。",
    ]
    for text in bad:
        assert not is_compliant(text), f"未拦住: {text}"
        with pytest.raises(ComplianceError):
            assert_compliant(text, source="unit")


def test_allowed_phrases_are_not_false_positives():
    """规范 6 要求输出就医建议与推荐检查，误伤它们会让产品变哑巴。"""
    good = [
        "就医建议：建议就诊消化内科，推荐检查项目为肝功能、腹部超声。",
        "建议复查该项指标，并建议随访观察变化趋势。",
        "如有不适请前往正规医疗机构，由执业医师作出判断。",
        "本结果不构成任何医疗意见。",
    ]
    for text in good:
        assert is_compliant(text), f"误伤: {text} -> {[str(v) for v in scan(text)]}"


def test_violation_reports_category_and_context():
    v = scan("您已确诊为慢性肝病，建议服用保肝药。")
    cats = {x.category for x in v}
    assert "下诊断" in cats and "开药与剂量" in cats
    assert all(x.context for x in v)


def test_disclaimer_attached_once_and_fallback_is_safe():
    t = attach_disclaimer("风险分层为中危。")
    assert attach_disclaimer(t) == t  # 幂等，不叠加
    fb = safe_fallback("高危")
    assert is_compliant(fb) and "高危" in fb


# ===========================================================================
# 2. 脱敏与全链路日志（规范 1.2 / 1.3 / 4.2）
# ===========================================================================
def test_scan_pii_detects_values_and_field_names():
    assert scan_pii({"note": "电话13800138000"})
    assert scan_pii({"ocr": {"id": "110101199001011234"}})
    assert scan_pii({"contact": "a.b@example.com"})
    assert scan_pii({"姓名": "张三"})  # 字段名黑名单，内容无所谓
    assert scan_pii({"phone": "无"})
    assert scan_pii([{"deep": {"x": "13900139000"}}])
    assert not scan_pii({"ALT_value": 42.0, "trend": "上升", "code": "ALT"})


def test_pseudonymize_is_stable_salt_scoped_and_irreversible():
    a, b = pseudonymize("P001", SALT), pseudonymize("P001", SALT)
    c = pseudonymize("P001", "other-salt")
    d = pseudonymize("P002", SALT)
    assert a == b, "同盐同 ID 必须稳定，否则无法做趋势追踪"
    assert a != c, "换盐必须切断关联"
    assert a != d and "P001" not in a
    with pytest.raises(ValueError):
        pseudonymize("P001", "")  # 空盐等于没盐


def test_audit_rejects_plaintext_pii():
    audit = AuditLogger(_tmp(), salt=SALT)
    rec = PredictionRecord(
        model_version="v1", feature_hash="h", features={"a": 1.0},
        ocr_result={"报告": "联系电话 13800138000"},
    )
    with pytest.raises(PIIError):
        audit.log(rec)


def test_audit_rejects_unreplayable_record():
    """只记概率的日志在事故复盘时等于没有日志。"""
    audit = AuditLogger(_tmp(), salt=SALT)
    with pytest.raises(ValueError):
        audit.log(PredictionRecord(probability=0.3, risk_tier="中危"))


def test_audit_roundtrip_and_find():
    audit = AuditLogger(_tmp(), salt=SALT)
    ids = []
    for i in range(3):
        rec = PredictionRecord(
            pseudo_id=audit.pseudonymize(f"P{i}"), model_version="v1",
            feature_hash="h", features={"ALT_dev": float(i)},
            probability=0.1 * i, risk_tier="中危",
        )
        ids.append(audit.log(rec))
    df = audit.load_day()
    assert len(df) == 3 and set(df["trace_id"]) == set(ids)
    got = audit.find(ids[1])
    assert got is not None and got.features["ALT_dev"] == 1.0 and got.is_replayable()


def test_attach_outcome_appends_instead_of_overwriting():
    """append-only 是审计日志不可篡改性的全部来源，回填不能原地改写历史行。"""
    root = _tmp()
    audit = AuditLogger(root, salt=SALT)
    tid = audit.log(
        PredictionRecord(model_version="v1", feature_hash="h", features={"a": 1.0},
                         probability=0.8, risk_tier="极高危")
    )
    n_before = len(list(root.glob("*.jsonl"))[0].read_text(encoding="utf-8").strip().split("\n"))
    assert audit.attach_outcome(tid, event=1, days=400)
    n_after = len(list(root.glob("*.jsonl"))[0].read_text(encoding="utf-8").strip().split("\n"))
    assert n_after == n_before + 1, "回填必须追加新行，不能改写原行"

    latest = AuditLogger.dedup_latest(audit.load_day())
    assert len(latest) == 1 and latest.iloc[0]["outcome_event"] == 1


def test_to_training_frame_keeps_only_labeled_records():
    audit = AuditLogger(_tmp(), salt=SALT)
    t1 = audit.log(PredictionRecord(model_version="v1", feature_hash="h",
                                    features={"ALT_dev": 1.0, "age": 50.0},
                                    probability=0.7, risk_tier="高危"))
    audit.log(PredictionRecord(model_version="v1", feature_hash="h",
                               features={"ALT_dev": 0.2, "age": 30.0},
                               probability=0.1, risk_tier="低危"))
    audit.attach_outcome(t1, event=1, days=300)
    tf = audit.to_training_frame()
    assert len(tf) == 1  # 未回流的那条不进训练表
    assert tf.iloc[0]["outcome_event"] == 1
    assert "ALT_dev" in tf.columns and "age" in tf.columns


def test_corrupt_line_does_not_break_the_whole_day():
    root = _tmp()
    audit = AuditLogger(root, salt=SALT)
    audit.log(PredictionRecord(model_version="v1", feature_hash="h", features={"a": 1.0}))
    path = list(root.glob("*.jsonl"))[0]
    with path.open("a", encoding="utf-8") as f:
        f.write("{坏行不是合法 JSON\n")
    audit.log(PredictionRecord(model_version="v1", feature_hash="h", features={"a": 2.0}))
    assert len(audit.load_day()) == 2  # 坏行跳过，其余可读


def test_replay_candidates_selects_error_samples():
    audit = AuditLogger(_tmp(), salt=SALT)
    specs = [("极高危", 0.9, 0), ("极高危", 0.85, 1), ("低危", 0.02, 1)]
    for tier, p, ev in specs:
        tid = audit.log(PredictionRecord(model_version="v1", feature_hash="h",
                                         features={"a": p}, probability=p, risk_tier=tier))
        audit.attach_outcome(tid, event=ev, days=500)
    df = audit.load_day()
    fp = AuditLogger.replay_candidates(df, tier="极高危", outcome_event=0)
    fn = AuditLogger.replay_candidates(df, tier="低危", outcome_event=1)
    assert len(fp) == 1 and len(fn) == 1


# ===========================================================================
# 3. 风险归因（规范 3.2 / 3.3 / 6）
# ===========================================================================
def test_aggregation_by_indicator_is_exact():
    """一个指标的贡献必须精确等于它派生的所有特征之和 —— 差一个特征就是错的归因。"""
    model = StubModel()
    X = _X(4, seed=1)
    eng = AttributionEngine(model, top_n=8)
    attrs = eng.explain(X, probabilities=model.predict_risk(X))
    contrib, _ = model.shap_values(X)

    for i, attr in enumerate(attrs):
        for f in attr.factors:
            members = [c for c in contrib.columns if c == f.key or c.startswith(f"{f.key}_")]
            assert abs(f.shap_sum - contrib.iloc[i][members].sum()) < 1e-12
            assert f.n_features == len(members)


def test_top_n_and_coverage_are_consistent():
    model = StubModel()
    X = _X(3, seed=2)
    eng = AttributionEngine(model, top_n=2)
    attr = eng.explain(X, probabilities=model.predict_risk(X))[0]
    assert len(attr.factors) == 2
    covered = sum(f.magnitude for f in attr.factors)
    assert abs(attr.coverage - covered / attr.total_abs) < 1e-12
    # 只取 2 个因子时覆盖率必然不高 -> 应判定为风险弥散
    assert attr.diffuse == (attr.coverage < 0.5)


def test_missing_indicator_is_phrased_as_not_checked():
    """展示成"降低了风险"会让用户以为该项正常，而真相是他根本没查（规范 1.2 三态）。"""
    model = StubModel()
    X = _X(2, seed=3)
    X.loc[0, "PLT_status"] = int(MeasureStatus.MISSING)
    X.loc[0, "PLT_value"] = np.nan
    eng = AttributionEngine(model, top_n=8, display_names={"PLT": "血小板"})
    attr = eng.explain(X, probabilities=model.predict_risk(X))[0]
    plt = next(f for f in attr.factors if f.key == "PLT")
    assert plt.is_missing and "未检查" in plt.phrase()
    assert plt.display == "血小板"


def test_direction_signs_match_contribution():
    model = StubModel()
    X = _X(6, seed=4)
    eng = AttributionEngine(model, top_n=8)
    for attr in eng.explain(X, probabilities=model.predict_risk(X)):
        for f in attr.factors:
            assert f.direction == int(np.sign(round(f.shap_sum, 9)))
        assert all(f.shap_sum > 0 for f in attr.raising)
        assert all(f.shap_sum < 0 for f in attr.lowering)


def test_explain_change_ranks_by_delta_not_by_rank_movement():
    """名次变化不等于贡献变化：只有 Δ贡献 才能决定"这次为什么涨了"。"""
    model = StubModel()
    X = _X(2, seed=5)
    X.loc[1] = X.loc[0]
    X.loc[1, "ALT_value"] = X.loc[0, "ALT_value"] + 4.0  # 只动 ALT
    eng = AttributionEngine(model, top_n=8)
    a0, a1 = eng.explain(X, probabilities=model.predict_risk(X))
    ch = explain_change(a0, a1, top_n=3)
    assert ch.rose and ch.factors[0].key == "ALT"
    assert ch.factors[0].delta_shap > 0
    assert abs(ch.delta_probability - (a1.probability - a0.probability)) < 1e-12
    assert "上升" in ch.factors[0].phrase()

    # 数值没变而贡献变了时，绝不能编造趋势（模型非线性，这种情况真实存在）
    same = ChangeFactor(key="AST", display="谷草转氨酶", delta_shap=0.2,
                        prev_value=30.0, curr_value=30.0)
    assert "上升" not in same.phrase() and "下降" not in same.phrase()


def test_engine_fails_fast_when_shap_unavailable():
    """能力缺失必须在启动时暴露，而不是在第一个用户请求时（规范 3.2）。"""
    try:
        import shap  # noqa: F401
        return  # 装了 shap 就没有这个失败路径
    except ImportError:
        pass
    with pytest.raises(RuntimeError):
        AttributionEngine(StubModel(backend="sklearn"))


def test_global_importance_is_aggregated_and_sorted():
    model = StubModel()
    imp = AttributionEngine(model, top_n=5).global_importance(_X(200, seed=6))
    assert set(imp.index) == {"ALT", "AST", "PLT", "age"}
    assert list(imp) == sorted(imp, reverse=True)


# ===========================================================================
# 4. 漂移监控（规范 3.2 / 4.3）
# ===========================================================================
def test_psi_is_zero_for_identical_distribution_and_positive_for_shift():
    ref = np.array([0.2, 0.3, 0.3, 0.2])
    assert population_stability_index(ref, ref) < 1e-9
    assert population_stability_index(ref, np.array([0.5, 0.3, 0.15, 0.05])) > 0.2


def test_psi_handles_empty_online_bin_without_inf():
    """线上某箱为空时 ln(0)=-inf 会污染整个加权总分，必须被平滑掉。"""
    v = population_stability_index(np.array([0.25] * 4), np.array([0.5, 0.5, 0.0, 0.0]))
    assert np.isfinite(v) and v > 0


def test_bins_come_from_training_so_shift_is_detectable():
    """
    最常见的 PSI 实现 bug 是用线上数据重新分箱 —— 那等于拿它自己比它自己，
    永远算出 0，监控一片绿。这里断言明显偏移必须被检出。
    """
    r = np.random.default_rng(7)
    ref = pd.DataFrame({c: r.normal(0, 1, 4000) for c in FEATS})
    prof = ReferenceProfile.from_training(ref, _manifest(), model_version="v1")
    online = pd.DataFrame({c: r.normal(0, 1, 2000) for c in FEATS})
    online["ALT_dev"] = r.normal(3.0, 1, 2000)  # 明显偏移
    rep = DriftMonitor(prof).check(online)
    row = rep.table().set_index("name").loc["ALT_dev"]
    assert row["psi"] > 1.0 and row["level"] == LEVEL_ALERT
    assert rep.level == LEVEL_ALERT


def test_missing_rate_spike_alerts_even_when_psi_is_zero():
    """管道故障的典型形态：剩余数据分布没变，但大半变成了 NaN。"""
    r = np.random.default_rng(8)
    ref = pd.DataFrame({c: r.normal(0, 1, 4000) for c in FEATS})
    prof = ReferenceProfile.from_training(ref, _manifest())
    online = pd.DataFrame({c: r.normal(0, 1, 2000) for c in FEATS})
    online.loc[:1400, "AST_dev"] = np.nan  # 70% 缺失，非缺失部分分布不变
    rep = DriftMonitor(prof).check(online)
    row = rep.table().set_index("name").loc["AST_dev"]
    assert row["psi"] < 0.1, "非缺失部分分布没变，PSI 本就应该接近 0"
    assert row["level"] == LEVEL_ALERT, "但缺失率突变必须独立告警"
    assert "管道故障" in row["note"] or "OCR" in row["note"]


def test_insufficient_samples_refuses_to_conclude():
    r = np.random.default_rng(9)
    prof = ReferenceProfile.from_training(pd.DataFrame({c: r.normal(0, 1, 2000) for c in FEATS}))
    rep = DriftMonitor(prof).check(pd.DataFrame({c: r.normal(5, 1, 50) for c in FEATS}))
    assert rep.level == LEVEL_INSUFFICIENT
    assert not rep.features, "拒绝出结论时不应给出任何逐特征数字"


def test_unseen_category_is_reported():
    ref = pd.DataFrame({"sex": ["M"] * 3000 + ["F"] * 3000})
    mf = FeatureManifest()
    mf.add(FeatureSpec(name="sex", group=FEATURE_GROUP_DEMO, dtype="categorical"))
    prof = ReferenceProfile.from_training(ref, mf)
    online = pd.DataFrame({"sex": ["M"] * 900 + ["F"] * 900 + ["U"] * 200})
    row = DriftMonitor(prof).check(online).table().set_index("name").loc["sex"]
    assert "U" in row["unseen_categories"]


def test_importance_weighting_downweights_useless_features():
    """无关特征漂到天上也不该拉响整体警报；主力特征漂一点就要。"""
    r = np.random.default_rng(10)
    ref = pd.DataFrame({c: r.normal(0, 1, 4000) for c in FEATS})
    prof = ReferenceProfile.from_training(ref, _manifest())
    online = pd.DataFrame({c: r.normal(0, 1, 2000) for c in FEATS})
    online["age"] = r.normal(4.0, 1, 2000)  # 只有 age 漂

    heavy_age = DriftMonitor(prof, importance={"ALT": 0.01, "AST": 0.01, "PLT": 0.01, "age": 1.0})
    light_age = DriftMonitor(prof, importance={"ALT": 1.0, "AST": 1.0, "PLT": 1.0, "age": 0.001})
    assert heavy_age.check(online).weighted_psi > 10 * light_age.check(online).weighted_psi


def test_reference_profile_roundtrip():
    r = np.random.default_rng(11)
    prof = ReferenceProfile.from_training(
        pd.DataFrame({c: r.normal(0, 1, 3000) for c in FEATS}), _manifest(), model_version="v9"
    )
    p = prof.save(_tmp() / "profile.json")
    back = ReferenceProfile.load(p)
    assert back.model_version == "v9" and back.n_ref == 3000
    assert set(back.features) == set(prof.features)
    assert back.features["ALT_dev"].edges == prof.features["ALT_dev"].edges


# ===========================================================================
# 5. 服务入口（规范 4.2 / 4.3 / 6 / 7）
# ===========================================================================
def _service(root=None, **cfg):
    model = StubModel()
    r = np.random.default_rng(12)
    scheme = RiskTierScheme.from_probabilities(
        model.predict_risk(pd.DataFrame({c: r.normal(0, 1, 5000) for c in FEATS})),
        source="oof_3y",
    )
    audit = AuditLogger(root or _tmp(), salt=SALT)
    svc = RiskPredictionService(
        model, scheme, audit=audit,
        display_names={"ALT": "谷丙转氨酶", "AST": "谷草转氨酶", "PLT": "血小板"},
        config=ServiceConfig(model_version="liver_v3", horizon="3y", **cfg),
    )
    return svc, audit, model, scheme


def test_tier_scheme_validates_and_assigns():
    s = RiskTierScheme(cutpoints=(0.05, 0.15, 0.40))
    # 区间左闭右开，切点值归入更高一层：边缘个体宁可多提示一级
    assert s.assign(0.01) == "低危" and s.assign(0.049) == "低危"
    assert s.assign(0.05) == "中危" and s.assign(0.99) == "极高危"
    assert s.assign_many([0.01, 0.2, 0.9]) == ["低危", "高危", "极高危"]
    with pytest.raises(ValueError):
        RiskTierScheme(cutpoints=(0.4, 0.1))  # 非单调
    with pytest.raises(ValueError):
        RiskTierScheme(cutpoints=(0.1, 0.2), names=("A", "B"))  # 层名数不匹配


def test_tier_scheme_is_fixed_not_recomputed_online():
    """
    切点必须固化：同一个人指标没变，风险等级不能因为"今天来的人更健康"而变。
    """
    _svc, _audit, model, scheme = _service()
    X = _X(1, seed=13)
    tier_alone = scheme.assign(float(model.predict_risk(X)[0]))
    # 把这条样本混进一批更健康的人里，等级必须不变
    r = np.random.default_rng(14)
    healthy = pd.DataFrame({c: r.normal(-2, 0.5, 500) for c in FEATS})
    mixed = pd.concat([X, healthy], ignore_index=True)
    assert scheme.assign_many(model.predict_risk(mixed))[0] == tier_alone


def test_tier_scheme_roundtrip():
    s = RiskTierScheme(cutpoints=(0.03, 0.12, 0.35), source="oof_3y")
    back = RiskTierScheme.load(s.save(_tmp() / "tiers.json"))
    assert back.cutpoints == s.cutpoints and back.source == "oof_3y"


def test_predict_returns_attribution_and_writes_full_chain_log():
    root = _tmp()
    svc, audit, _model, _s = _service(root)
    X = _X(3, seed=15)
    res = svc.predict(X, patient_ids=["P1", "P2", "P3"], raw_refs=["oss://1", "oss://2", "oss://3"])

    assert len(res) == 3
    for r_ in res:
        assert 0 <= r_.probability <= 1 and r_.risk_tier
        assert r_.attribution is not None and r_.attribution.factors
        assert is_compliant(r_.narrative)

    df = audit.load_day()
    assert len(df) == 3
    rec = audit.find(res[0].trace_id)
    # 规范 4.2 要求的因果链要素齐全
    assert rec.is_replayable()
    assert rec.model_version == "liver_v3" and rec.feature_hash == "abc123"
    assert set(rec.features) == set(X.columns)
    assert rec.attribution["factors"] and rec.risk_tier == res[0].risk_tier
    assert rec.raw_ref == "oss://1" and rec.pseudo_id and "P1" not in rec.pseudo_id


def test_strict_audit_requires_a_logger():
    model = StubModel()
    with pytest.raises(ValueError):
        RiskPredictionService(model, RiskTierScheme(), audit=None)


def test_audit_failure_aborts_the_prediction():
    """结果照发、日志掉了 = 制造一次永远查不清的预测。"""
    svc, audit, _m, _s = _service()

    def boom(_rec):
        raise OSError("磁盘满")

    audit.log = boom
    with pytest.raises(OSError):
        svc.predict(_X(1, seed=16), patient_ids=["P1"])


def test_non_strict_audit_allows_offline_batch():
    svc, audit, _m, _s = _service()
    svc.config.strict_audit = False
    audit.log = lambda _rec: (_ for _ in ()).throw(OSError("磁盘满"))
    res = svc.predict(_X(2, seed=17), patient_ids=["P1", "P2"])
    assert len(res) == 2  # 显式降级后允许继续


def test_nan_probability_aborts_instead_of_being_treated_as_low_risk():
    svc, _a, model, _s = _service()
    model.predict_risk = lambda X, calibrated=True: np.full(len(X), np.nan)
    with pytest.raises(RuntimeError):
        svc.predict(_X(2, seed=18), patient_ids=["P1", "P2"])


def test_narrative_carries_disclaimer_and_mentions_missing_items():
    svc, _a, _m, _s = _service()
    X = _X(1, seed=19)
    X.loc[0, "PLT_status"] = int(MeasureStatus.MISSING)
    X.loc[0, "PLT_value"] = np.nan
    r_ = svc.predict(X, patient_ids=["P1"])[0]
    assert "不构成任何医疗意见" in r_.narrative
    assert "未检查" in r_.narrative
    assert is_compliant(r_.narrative)


def test_drift_level_is_carried_into_prediction_records():
    """事故复盘时要能看到"这次预测发生在漂移告警期间"。"""
    root = _tmp()
    svc, audit, _m, _s = _service(root)
    r = np.random.default_rng(20)
    prof = ReferenceProfile.from_training(
        pd.DataFrame({c: r.normal(0, 1, 4000) for c in FEATS}), _manifest()
    )
    svc.drift_monitor = DriftMonitor(prof)
    online = pd.DataFrame({c: r.normal(0, 1, 2000) for c in FEATS})
    online["ALT_dev"] = r.normal(3.0, 1, 2000)
    assert svc.check_drift(online).level == LEVEL_ALERT

    res = svc.predict(_X(1, seed=21), patient_ids=["P1"])
    assert res[0].drift_level == LEVEL_ALERT
    assert audit.find(res[0].trace_id).drift_level == LEVEL_ALERT


def test_empty_input_returns_empty_without_side_effects():
    root = _tmp()
    svc, audit, _m, _s = _service(root)
    assert svc.predict(pd.DataFrame(columns=list(FEATS))) == []
    assert audit.load_day().empty


def test_explain_disabled_is_allowed_but_loud():
    svc, _a, _m, _s = _service(explain=False)
    res = svc.predict(_X(1, seed=22), patient_ids=["P1"])
    assert res[0].attribution is None  # 显式关闭后不产出归因
    assert res[0].risk_tier


# ===========================================================================
# 无 pytest 环境的直跑入口
# ===========================================================================
if __name__ == "__main__":
    import logging
    import traceback

    logging.basicConfig(level=logging.CRITICAL)
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    n_ok = n_fail = n_skip = 0
    for name, fn in fns:
        try:
            kwargs = {}
            if "tmp_path" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
                kwargs["tmp_path"] = tempfile.mkdtemp()
            fn(**kwargs)
            print(f"PASS  {name}")
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            if not _HAVE_PYTEST and type(e).__name__ == "_Skip":
                print(f"SKIP  {name} (缺 {e})")
                n_skip += 1
            else:
                print(f"FAIL  {name}: {e}")
                traceback.print_exc()
                n_fail += 1
    print(f"\n{n_ok} passed, {n_fail} failed, {n_skip} skipped")
    sys.exit(1 if n_fail else 0)
