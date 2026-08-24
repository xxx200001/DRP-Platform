# -*- coding: utf-8 -*-
"""
V3.3 回归：干预建议与 AI 兜底分析必须是【数据驱动】的 ——
两份不同的化验单要得到不同的输出；真实数值要写进句子里；
仅有一次观测的异常（如首份报告的 ALP↑）不能被吞掉；
所有生成文案不得命中合规禁用词。

只依赖 pandas + pyyaml + 标准库（不需要 lightgbm / fastapi），
可与 tests/test_reports_db.py 一样在最小环境直接运行：
    PYTHONPATH=src python -m pytest tests/test_dynamic_advice.py -q
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drp.data.constants import COL_INDICATOR, COL_MEASURED_AT, COL_SEX, COL_VALUE  # noqa: E402
from drp.data.reference import ReferenceRegistry  # noqa: E402
from drp.serving.compliance import scan  # noqa: E402
from drp.serving.llm_advisor import (  # noqa: E402
    _build_expert_knowledge_analysis,
    generate_llm_trend_analysis,
    resolve_llm_env,
)
from drp.serving.trend import TrendEngine, build_trend_report  # noqa: E402


@pytest.fixture(scope="module")
def registry() -> ReferenceRegistry:
    return ReferenceRegistry.from_yaml(ROOT / "configs" / "reference_intervals.yaml")


def _frame(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{COL_INDICATOR: c, COL_MEASURED_AT: pd.Timestamp(d), COL_VALUE: v,
          "patient_id": "P", "unit": "", "status": 0}
         for c, d, v in rows]
    )


#: 与真实样张同构的"肝功能型"患者（何鑫式：转氨酶/GGT/ALP 高 + 血脂异常）
LIVER_ROWS = [
    ("ALT", "2025-10-06", 94), ("ALT", "2025-10-27", 112),
    ("AST", "2025-10-06", 59), ("AST", "2025-10-27", 68),
    ("GGT", "2025-10-06", 36), ("GGT", "2025-10-27", 63),
    ("ALP", "2025-10-27", 103),                 # 只有一次观测：不许被吞掉
    ("TG", "2025-10-06", 2.13), ("TG", "2025-10-27", 1.88),
    ("HDLC", "2025-10-06", 0.98), ("HDLC", "2025-10-27", 0.75),
    ("TC", "2025-10-06", 4.79), ("TC", "2025-10-27", 5.45),
]

#: "肾/尿酸 + 血糖型"患者，用来证明两份数据 → 两套不同建议
RENAL_ROWS = [
    ("UA", "2025-10-06", 480), ("UA", "2025-10-27", 512),
    ("CREA", "2025-10-06", 105), ("CREA", "2025-10-27", 118),
    ("GLU", "2025-10-06", 6.9), ("GLU", "2025-10-27", 7.4),
]

DEMO_F25 = pd.DataFrame([{COL_SEX: "F", "age": 25.0, "patient_id": "P"}])
DEMO_M55 = pd.DataFrame([{COL_SEX: "M", "age": 55.0, "patient_id": "P"}])


def _report(registry, rows, demo):
    return build_trend_report(TrendEngine(registry), _frame(rows), demographics=demo)


def _all_text(obj) -> str:
    """把结构化结果里的全部字符串拼在一起，供合规扫描。"""
    return json.dumps(obj, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# 干预建议（trend.build_interventions via build_trend_report）
# ---------------------------------------------------------------------------
class TestInterventionsAreDataDriven:
    def test_two_patients_get_different_systems(self, registry):
        liver = _report(registry, LIVER_ROWS, DEMO_F25)
        renal = _report(registry, RENAL_ROWS, DEMO_M55)
        sys_l = {iv.system for iv in liver.interventions}
        sys_r = {iv.system for iv in renal.interventions}
        assert "肝胆代谢防护" in sys_l and "心血管与脂代谢管理" in sys_l
        assert "肝胆代谢防护" not in sys_r
        assert "肾脏机能与尿酸排泄" in sys_r and "糖代谢与血糖调控" in sys_r
        assert sys_l != sys_r

    def test_real_values_embedded_in_sentences(self, registry):
        liver = _report(registry, LIVER_ROWS, DEMO_F25)
        card = next(iv for iv in liver.interventions if iv.system == "肝胆代谢防护")
        blob = "；".join(card.details) + "；".join(card.diet_advice) + card.followup_cycle
        assert "112" in blob, "ALT 真实值必须写进句子"
        assert "戒酒" in "".join(card.diet_advice)
        # GGT 63 vs 上限 → 数值出现在饮食建议里
        assert "63" in "".join(card.diet_advice)
        # 随访周期点名最严重指标
        assert "丙氨酸氨基转移酶" in card.followup_cycle

    def test_single_observation_abnormal_not_swallowed(self, registry):
        """ALP 只有一次观测（comparisons 里不存在），仍必须进入干预卡。"""
        liver = _report(registry, LIVER_ROWS, DEMO_F25)
        card = next(iv for iv in liver.interventions if iv.system == "肝胆代谢防护")
        assert any("碱性磷酸酶" in t for t in card.target_indicators)
        assert any("首次记录" in d for d in card.details)

    def test_stats_worst_and_over_ratio(self, registry):
        liver = _report(registry, LIVER_ROWS, DEMO_F25)
        card = next(iv for iv in liver.interventions if iv.system == "肝胆代谢防护")
        st = card.stats
        assert st["n_abnormal"] >= 4
        assert st["worst"]["name_cn"] == "丙氨酸氨基转移酶"
        assert "倍" in st["worst"]["over"]

    def test_renal_card_has_purine_and_water_but_no_alcohol_liver_line(self, registry):
        renal = _report(registry, RENAL_ROWS, DEMO_M55)
        card = next(iv for iv in renal.interventions if iv.system == "肾脏机能与尿酸排泄")
        diet = "".join(card.diet_advice)
        assert "嘌呤" in diet and "512" in diet
        assert "戒酒" not in diet  # 肝胆卡的专属句不得串到肾脏卡

    def test_all_generated_text_is_compliant(self, registry):
        for rows, demo in ((LIVER_ROWS, DEMO_F25), (RENAL_ROWS, DEMO_M55)):
            rep = _report(registry, rows, demo)
            blob = _all_text([iv.to_dict() for iv in rep.interventions])
            hits = scan(blob)
            assert not hits, f"干预文案命中禁用词: {[str(h) for h in hits]}"

    def test_all_normal_gets_stable_card_with_count(self, registry):
        rows = [("ALT", "2025-10-06", 20), ("ALT", "2025-10-27", 22),
                ("GLU", "2025-10-27", 5.0)]
        rep = _report(registry, rows, DEMO_F25)
        assert len(rep.interventions) == 1
        card = rep.interventions[0]
        assert card.level == "平稳维持"
        assert "2 项" in card.target_indicators[0]  # ALT + GLU 两项全正常

    def test_snapshot_exposed_with_ref_bounds(self, registry):
        rep = _report(registry, LIVER_ROWS, DEMO_F25)
        snap = {e["code"]: e for e in rep.latest_snapshot}
        assert snap["ALT"]["value"] == 112 and snap["ALT"]["grade"] > 0
        assert snap["ALT"]["ref_high"] is not None
        assert snap["ALP"]["n_points"] == 1


# ---------------------------------------------------------------------------
# AI 兜底引擎（llm_advisor，无密钥环境）
# ---------------------------------------------------------------------------
def _ai_for(registry, rows, demo, age, sex_cn, traj=None, tier=None):
    rep = _report(registry, rows, demo)
    return generate_llm_trend_analysis(
        {"name": "P", "sex": sex_cn, "age": age, "n_records": len(rows)},
        [c.to_dict() for c in rep.comparisons],
        traj or {},
        snapshot=rep.latest_snapshot,
        risk_tier=tier,
        span_label="21天",
    )


class TestAdvisorFallbackIsDataDriven:
    @pytest.fixture(autouse=True)
    def _no_keys(self, monkeypatch):
        for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
            monkeypatch.delenv(k, raising=False)

    def test_no_key_means_offline_engine(self, registry):
        assert resolve_llm_env()["provider"] is None
        ai = _ai_for(registry, LIVER_ROWS, DEMO_F25, 25, "女")
        assert ai["source"] == "AI_CLINICAL_KNOWLEDGE_ENGINE"
        assert "llm_narrative_text" not in ai

    def test_target_heart_rate_uses_real_age(self, registry):
        ai = _ai_for(registry, LIVER_ROWS, DEMO_F25, 25, "女")
        ex = "".join(it for s in ai["lifestyle_interventions"] for it in s["items"])
        assert "117~136" in ex          # (220-25)*0.6 ~ *0.7
        ai55 = _ai_for(registry, RENAL_ROWS, DEMO_M55, 55, "男")
        ex55 = "".join(it for s in ai55["lifestyle_interventions"] for it in s["items"])
        assert "99~115" in ex55         # (220-55)*0.6 ~ *0.7

    def test_diet_sections_follow_the_report(self, registry):
        ai_l = _ai_for(registry, LIVER_ROWS, DEMO_F25, 25, "女")
        ai_r = _ai_for(registry, RENAL_ROWS, DEMO_M55, 55, "男")
        titles_l = [s["title"] for s in ai_l["diet_interventions"]]
        titles_r = [s["title"] for s in ai_r["diet_interventions"]]
        assert any("肝胆" in t for t in titles_l)
        assert not any("肝胆" in t for t in titles_r)
        assert any("尿酸" in t or "肾" in t for t in titles_r)
        blob_l = _all_text(ai_l["diet_interventions"])
        assert "112" in blob_l and "GGT 63" in blob_l

    def test_followup_dept_and_cycle_follow_data(self, registry):
        ai_l = _ai_for(registry, LIVER_ROWS, DEMO_F25, 25, "女", tier="高危")
        assert "消化内科" in ai_l["followup_plan"]["recommend_dept"]
        assert ai_l["followup_plan"]["cycle_short"] == "2~4 周后"
        assert any("丙氨酸氨基转移酶" in x
                   for x in ai_l["followup_plan"]["cycle_short_items"])
        ai_r = _ai_for(registry, RENAL_ROWS, DEMO_M55, 55, "男", tier="中危")
        assert "肾内科" in ai_r["followup_plan"]["recommend_dept"]
        assert "消化内科" not in ai_r["followup_plan"]["recommend_dept"]
        assert ai_r["followup_plan"]["cycle_short"] == "1~2 个月后"

    def test_red_flags_only_for_hit_systems(self, registry):
        ai_l = _ai_for(registry, LIVER_ROWS, DEMO_F25, 25, "女")
        blob = "".join(ai_l["red_flags"])
        assert "巩膜发黄" in blob
        assert "内分泌科" not in blob        # 无血糖异常 → 不给糖尿病红旗
        ai_r = _ai_for(registry, RENAL_ROWS, DEMO_M55, 55, "男")
        blob_r = "".join(ai_r["red_flags"])
        assert "肾内科" in blob_r and "内分泌科" in blob_r
        assert "巩膜发黄" not in blob_r

    def test_risk_trajectory_narrated_first_to_last(self, registry):
        traj = {"3y": {"points": [
            {"at": "2025-10-06", "probability": 0.41, "risk_tier": "中危"},
            {"at": "2025-10-27", "probability": 0.63, "risk_tier": "高危"},
        ]}}
        ai = _ai_for(registry, LIVER_ROWS, DEMO_F25, 25, "女", traj=traj)
        blob = "".join(ai["risk_trajectory_summary"])
        assert "41.0%" in blob and "63.0%" in blob
        assert "2025-10-06" in blob and "2025-10-27" in blob
        mech = "".join(ai["pathology_mechanism"])
        assert "AI 仅作解释" in mech

    def test_mechanism_mentions_values_and_span(self, registry):
        ai = _ai_for(registry, LIVER_ROWS, DEMO_F25, 25, "女")
        mech = "".join(ai["pathology_mechanism"])
        assert "21天" in mech and "112" in mech
        assert "肝细胞" in mech
        assert "胰岛" not in mech            # 无血糖命中 → 不谈糖代谢机制

    def test_patient_summary_counts_abnormal(self, registry):
        ai = _ai_for(registry, LIVER_ROWS, DEMO_F25, 25, "女")
        assert "项指标异常" in ai["patient_summary"]
        assert "跨度 21天" in ai["patient_summary"]

    def test_all_generated_text_is_compliant(self, registry):
        for rows, demo, age, sx in ((LIVER_ROWS, DEMO_F25, 25, "女"),
                                    (RENAL_ROWS, DEMO_M55, 55, "男")):
            ai = _ai_for(registry, rows, demo, age, sx)
            hits = scan(_all_text(ai))
            assert not hits, f"AI 兜底文案命中禁用词: {[str(h) for h in hits]}"

    def test_all_normal_says_stable(self, registry):
        rows = [("ALT", "2025-10-06", 20), ("ALT", "2025-10-27", 22)]
        ai = _ai_for(registry, rows, DEMO_F25, 25, "女")
        assert ai["n_abnormal"] == 0
        assert "均在参考区间内" in ai["pathology_mechanism"][0]
        assert any("通用原则" in s["title"] for s in ai["diet_interventions"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
