"""
应用层集成测试：小规模自举（单时程 + 开发闸）→ 全链路 HTTP 走查。

这不是单元测试，是"整机点火"：患者建档 → OCR 报告解析入库 → 真实特征
管线 → 预测（灰度路由+审计留痕）→ 趋势 → 随访回流 → 管理台。
自举用缩小参数（n=1200、仅 3y、开发闸）控制耗时；生产默认参数的门禁
是规范 9 全量标准，由 run_app.py 首启走。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from drp.validation import ValidationGate

from app.bootstrap import BOOTSTRAP_VERSION, run_bootstrap
from app.server import SAMPLE_REPORT, build_server

DEV_GATE = ValidationGate(
    min_auc_roc=0.75, min_auc_ci_lower=0.70, min_pr_lift=1.3,
    min_specificity_at_target=0.20, max_ece=0.12,
    max_oe_deviation=0.35, min_test_positives=20,
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    data = Path(tempfile.mkdtemp()) / "app_data"
    run_bootstrap(data, n_patients=1200, seed=7,
                  horizons=(("3y", 1095),), gate=DEV_GATE)
    app = build_server(data)
    return TestClient(app)


def _mk_patient(client, pid="MRN-001", sex="M", birth="1968-05-20"):
    return client.post("/api/patients",
                       json={"patient_id": pid, "sex": sex, "birth_date": birth})


# ===========================================================================
class TestMetaAndPatients:
    def test_meta(self, client):
        m = client.get("/api/meta").json()
        assert m["active_version"] == BOOTSTRAP_VERSION
        assert m["horizons"] == ["3y"]
        assert any(i["code"] == "ALT" for i in m["indicators"])
        assert "不构成任何医疗意见" in m["disclaimer"]
        assert "ALT" in m["sample_report"]

    def test_create_and_list(self, client):
        r = _mk_patient(client)
        assert r.status_code == 200, r.text
        assert _mk_patient(client).status_code == 409  # 重复
        lst = client.get("/api/patients").json()
        assert any(p["patient_id"] == "MRN-001" for p in lst)

    def test_patient_id_with_pii_rejected(self, client):
        r = client.post("/api/patients", json={
            "patient_id": "13800138000", "sex": "F", "birth_date": "1990-01-01"})
        assert r.status_code == 422
        assert "个人信息" in r.json()["detail"]

    def test_unknown_patient_404(self, client):
        assert client.get("/api/patients/nobody/records").status_code == 404


# ===========================================================================
class TestProfileAndMedications:
    """规范 2.1（档案）与 2.4（用药/采血登记）的应用层通路。"""

    def test_create_with_profile_and_meta_catalog(self, client):
        m = client.get("/api/meta").json()
        assert "hx_diabetes" in m["profile_fields"]["history"]
        assert "fh_cad" in m["profile_fields"]["family_history"]

        r = client.post("/api/patients", json={
            "patient_id": "MRN-PROF", "sex": "M", "birth_date": "1966-06-06",
            "height_cm": 172, "weight_kg": 81, "smoking_status": 2,
            "smoking_pack_years": 20, "hx_diabetes": 1, "hx_gout": 0,
            "fh_cad": 1})
        assert r.status_code == 200, r.text
        p = r.json()
        assert p["height_cm"] == 172 and p["hx_diabetes"] == 1
        assert p["hx_gout"] == 0 and p["hx_cad"] is None  # 三态：0 与 NULL 不同

    def test_profile_update_overwrites_and_withdraws(self, client):
        r = client.put("/api/patients/MRN-PROF/profile",
                       json={"height_cm": 172, "weight_kg": 81})
        assert r.status_code == 200
        p = r.json()
        assert p["hx_diabetes"] is None, "整档覆盖：未带字段应回到未采集"
        assert p["weight_kg"] == 81
        assert client.put("/api/patients/nobody/profile", json={}).status_code == 404

    def test_profile_rejects_implausible_values(self, client):
        r = client.put("/api/patients/MRN-PROF/profile", json={"height_cm": 20})
        assert r.status_code == 422

    def test_medication_crud_and_pii_gate(self, client):
        pii = client.post("/api/patients/MRN-PROF/medications",
                          json={"medication_name": "他汀 13800138000"})
        assert pii.status_code == 422 and "个人信息" in pii.json()["detail"]

        bad = client.post("/api/patients/MRN-PROF/medications", json={
            "medication_name": "二甲双胍", "start_date": "2026-02-01",
            "end_date": "2026-01-01"})
        assert bad.status_code == 422

        ok = client.post("/api/patients/MRN-PROF/medications", json={
            "medication_name": "阿托伐他汀钙片", "start_date": "2026-01-01"})
        assert ok.status_code == 200
        med = ok.json()
        assert med["end_date"] is None

        rows = client.get("/api/patients/MRN-PROF/medications").json()
        assert any(x["id"] == med["id"] for x in rows)

        # 归属校验：不能通过别的患者路径删除
        _mk_patient(client, pid="MRN-OTHER", sex="F", birth="1990-01-01")
        wrong = client.delete(f"/api/patients/MRN-OTHER/medications/{med['id']}")
        assert wrong.status_code == 404
        gone = client.delete(f"/api/patients/MRN-PROF/medications/{med['id']}")
        assert gone.status_code == 200
        assert client.get("/api/patients/MRN-PROF/medications").json() == []


# ===========================================================================
class TestReportParse:
    def test_pii_text_rejected(self, client):
        r = client.post("/api/reports/parse", json={
            "patient_id": "MRN-001",
            "text": "患者手机 13800138000\nALT 62 U/L 9-50 ↑",
            "measured_at": "2026-08-10"})
        assert r.status_code == 422
        assert "个人信息" in r.json()["detail"]

    def test_sample_report_full_chain(self, client):
        r = client.post("/api/reports/parse", json={
            "patient_id": "MRN-001", "text": SAMPLE_REPORT,
            "measured_at": "2026-06-01"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["stored"] >= 15                # 大部分行成功入库
        assert body["parse"]["n_unmatched"] >= 1   # "神秘未知指标"
        # 6。8 中文句号被修复入库：GLU 存在且值为 7.2
        recs = client.get("/api/patients/MRN-001/records").json()
        glu = [x for x in recs if x["indicator_code"] == "GLU"]
        assert glu and abs(glu[0]["value"] - 7.2) < 1e-9

    def test_second_visit_for_trend(self, client):
        text = ("丙氨酸氨基转移酶 ALT 88 U/L 9-50 ↑\n"
                "葡萄糖 GLU 8.4 mmol/L 3.9-6.1 ↑\n"
                "甘油三酯 TG 3.4 mmol/L 0.4-1.7 ↑\n"
                "血小板计数 PLT 160 10^9/L 125-350\n")
        r = client.post("/api/reports/parse", json={
            "patient_id": "MRN-001", "text": text, "measured_at": "2026-08-10"})
        assert r.status_code == 200 and r.json()["stored"] >= 4


# ===========================================================================
class TestPredictAndTrend:
    def test_predict_full_chain(self, client):
        r = client.post("/api/predict", json={"patient_id": "MRN-001"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model_version"] == BOOTSTRAP_VERSION
        assert body["arm"] in ("active", "canary")
        assert len(body["results"]) == 1
        res = body["results"][0]
        assert res["horizon"] == "3y"
        assert 0.0 <= res["probability"] <= 1.0
        assert res["risk_tier"] in ("低危", "中危", "高危", "极高危")
        assert "不构成任何医疗意见" in res["narrative"]
        assert len(res["top_factors"]) > 0
        # 就医建议：示例报告一堆异常，必然有条目
        assert len(body["referral"]["items"]) >= 1

    def test_predict_without_records_422(self, client):
        _mk_patient(client, pid="MRN-EMPTY", sex="F", birth="1985-03-03")
        r = client.post("/api/predict", json={"patient_id": "MRN-EMPTY"})
        assert r.status_code == 422
        assert "暂无化验记录" in r.json()["detail"]

    def test_predict_with_profile_meds_and_flags(self, client):
        """
        规范 2.1 + 2.4 的整机通路：档案 + 在用药 + 采血登记随预测请求进入
        真实特征管线。自举模型的 manifest 未必包含这些列（align 会按清单
        对齐），本测试守的是【通路不炸、结果合法】——特征是否被模型消费
        由训练清单决定，那是生产训练的职责。
        """
        client.put("/api/patients/MRN-001/profile", json={
            "height_cm": 171, "weight_kg": 85, "smoking_status": 2,
            "hx_diabetes": 1, "fh_cad": 1})
        client.post("/api/patients/MRN-001/medications", json={
            "medication_name": "阿托伐他汀", "start_date": "2025-01-01"})
        r = client.post("/api/predict", json={
            "patient_id": "MRN-001", "non_fasting": True,
            "strenuous_exercise": False})   # pregnancy 不登记 → 特征层不产出
        assert r.status_code == 200, r.text
        res = r.json()["results"][0]
        assert 0.0 <= res["probability"] <= 1.0
        assert res["risk_tier"] in ("低危", "中危", "高危", "极高危")

    def test_second_predict_then_trend(self, client):
        client.post("/api/predict", json={"patient_id": "MRN-001"})
        t = client.get("/api/patients/MRN-001/trend").json()
        assert "3y" in t["risk_trajectories"]
        assert len(t["risk_trajectories"]["3y"]["points"]) >= 2
        assert len(t["comparisons"]) >= 3          # ALT/GLU/TG/PLT 两次
        alt = next(c for c in t["comparisons"] if c["code"] == "ALT")
        assert alt["is_real_change"] is True and alt["direction"] == "上升"
        assert "风险走势" in t["rendered_text"]
        assert "不构成任何医疗意见" in t["rendered_text"]


# ===========================================================================
class TestFeedbackAndAdmin:
    def test_feedback_roundtrip(self, client):
        t = client.get("/api/patients/MRN-001/trend").json()
        trace = t["risk_trajectories"]["3y"]["points"][0]["trace_id"]
        bad = client.post("/api/feedback", json={
            "trace_id": trace, "event_occurred": True,
            "days_since_prediction": 200, "consented": False})
        assert bad.status_code == 422 and "授权" in bad.json()["detail"]
        ok = client.post("/api/feedback", json={
            "trace_id": trace, "event_occurred": True,
            "days_since_prediction": 200, "consented": True})
        assert ok.status_code == 200 and ok.json()["accepted"] is True
        missing = client.post("/api/feedback", json={
            "trace_id": "no-such", "event_occurred": False,
            "days_since_prediction": 5, "consented": True})
        assert missing.status_code == 404

    def test_admin_versions_and_review(self, client):
        v = client.get("/api/admin/versions").json()
        assert BOOTSTRAP_VERSION in v["versions"]
        assert v["versions"][BOOTSTRAP_VERSION]["status"] == "ACTIVE"
        q = client.get("/api/admin/review-queue").json()
        assert q["n_total_labeled"] >= 1           # 上面刚回流了一条
        assert "summary" in q

    def test_admin_drift(self, client):
        r = client.get("/api/admin/drift", params={"horizon": "3y"}).json()
        assert "level" in r and r["n_online"] >= 1
        bad = client.get("/api/admin/drift", params={"horizon": "9y"})
        assert bad.status_code == 422

    def test_admin_ab_insufficient(self, client):
        r = client.get("/api/admin/ab", params={
            "champion": BOOTSTRAP_VERSION, "challenger": "ghost"}).json()
        assert r["verdict"] == "insufficient_data"
        assert "三层验证门禁" in r["summary"]

    def test_static_index_served(self, client):
        r = client.get("/")
        assert r.status_code == 200 and "病情预测平台" in r.text
        assert client.get("/vendor/echarts.min.js").status_code == 200
