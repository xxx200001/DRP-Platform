"""
批次5测试：OCR结构化引擎（lexicon/parser）+ 智能就医建议（referral）。

测试哲学延续前四批：每个测试钉死一条【会在真实线上出事故】的行为，
不测实现细节。歧义拒绝、短键防护、量级疑点降置信、合规断言 ——
这些是本批次的安全承重墙，谁改坏了 CI 必须立刻红。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from drp.data.cleaning import COL_STATUS, LabDataCleaner
from drp.data.constants import (
    COL_INDICATOR,
    COL_MEASURED_AT,
    COL_PATIENT_ID,
    COL_UNIT,
    COL_VALUE,
    AbnormalGrade,
    MeasureStatus,
)
from drp.data.reference import ReferenceRegistry
from drp.features.deviation import _grade_of
from drp.ingest import (
    IndicatorLexicon,
    LabReportParser,
    fold_confusions,
    parse_lab_text,
    rows_to_frame,
)
from drp.serving.compliance import DISCLAIMER, is_compliant
from drp.serving.referral import (
    DEPARTMENT_RULES,
    Priority,
    ReferralEngine,
    grade_value,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "reference_intervals.yaml"


@pytest.fixture(scope="module")
def registry() -> ReferenceRegistry:
    return ReferenceRegistry.from_yaml(CONFIG)


@pytest.fixture(scope="module")
def lexicon(registry) -> IndicatorLexicon:
    return IndicatorLexicon(registry)


# ===========================================================================
# 词典引擎
# ===========================================================================
class TestLexicon:
    def test_exact_alias_chinese(self, lexicon):
        m = lexicon.lookup("谷丙转氨酶")
        assert m.code == "ALT" and m.confidence == 1.0 and m.method == "exact"

    def test_exact_fullwidth_case_space(self, lexicon):
        assert lexicon.lookup("ａｌｔ").code == "ALT"
        assert lexicon.lookup("  HbA1c ").code == "HBA1C"
        assert lexicon.lookup("γ-GT").code == "GGT"

    def test_confusion_fold_digit_letter(self, lexicon):
        """OCR 把 1 认成 l/I、0 认成 O 的确定性折叠。"""
        m = lexicon.lookup("HBAlC")   # 1 -> l
        assert m.code == "HBA1C" and m.method in ("exact", "fold")
        m2 = lexicon.lookup("A1T")    # L -> 1
        assert m2.code == "ALT" and m2.confidence >= 0.95

    def test_fuzzy_single_edit_chinese(self, lexicon):
        """谷丙转氨"梅"：一个错别字，长度>=4，候选唯一 -> 命中。"""
        m = lexicon.lookup("谷丙转氨梅")
        assert m.code == "ALT" and m.method == "fuzzy" and m.confidence == pytest.approx(0.85)

    def test_fuzzy_ambiguity_rejected_hdlc_ldlc(self, lexicon):
        """HDLC/LDLC 编辑距离为1：查询 XDLC 同时命中两者 -> 必须拒绝。"""
        m = lexicon.lookup("XDLC")
        assert m.code is None
        assert m.method == "ambiguous"
        assert set(m.candidates) >= {"HDLC", "LDLC"}

    def test_short_key_never_fuzzy(self, lexicon):
        """红线1：K/NA/CA 这类短键，错一个字符就必须拒绝，绝不猜。"""
        assert lexicon.lookup("MA").code is None      # NA 的1-edit
        assert lexicon.lookup("X").code is None
        assert lexicon.lookup("K").code == "K"        # 精确仍然通

    def test_digit_edit_rejected(self, lexicon):
        """红线2：数字位差异不允许模糊匹配。HBA2C 不能配到 HBA1C。"""
        m = lexicon.lookup("HBA2C")
        # 2->Z 折叠后 HBAZC 与 HBA1C 折叠键 HBAIC 不同；模糊层数字位又被禁
        assert m.code is None

    def test_fold_is_consistent(self):
        assert fold_confusions("HBA1C") == fold_confusions("HBAIC") == fold_confusions("HBALC")

    def test_unknown_name(self, lexicon):
        assert lexicon.lookup("量子波动速读酶").code is None


# ===========================================================================
# 行解析器
# ===========================================================================
OCR_SAMPLE = """\
XX市第一人民医院 检验报告单
姓名:张三  性别:男  年龄:45  样本号:20260815001
丙氨酸氨基转移酶 ALT      45    U/L     0-40      ↑
葡萄糖(GLU)  6。8 mmol/L  参考值:3.9-6.1  H
血小板计数 PLT 250 10^9/L 125-350
白细胞计数6.5x10^9/L
肌酐 CREA 98 μmol/L 57-97
血钾 K 3.8 mmol/L 3.5-5.3
乙肝表面抗原  阴性
神秘未知指标 12.3 U/L
审核者:李四  报告时间:2026-08-15
"""


@pytest.fixture(scope="module")
def parsed(registry):
    parser = LabReportParser(registry)
    return parser.parse(OCR_SAMPLE)


class TestParser:

    def test_meta_lines_skipped(self, parsed):
        _, report = parsed
        assert report.n_meta_skipped >= 3  # 医院头、姓名行、审核行

    def test_basic_line_with_flag_and_ref(self, parsed):
        rows, _ = parsed
        alt = next(r for r in rows if r.indicator_code == "ALT")
        assert alt.value == 45.0
        assert alt.unit == "U/L"
        assert alt.flag == "↑"
        assert alt.printed_ref == (0.0, 40.0)
        assert alt.confidence >= 0.95
        # 参考区间的 0 和 40 绝不能被误认为结果值
        assert alt.value not in (0.0, 40.0)

    def test_ocr_decimal_dot_fixed(self, parsed):
        """6。8 -> 6.8（OCR 句号小数点）。"""
        rows, _ = parsed
        glu = next(r for r in rows if r.indicator_code == "GLU")
        assert glu.value == pytest.approx(6.8)
        assert glu.flag == "H"

    def test_glued_name_value(self, parsed):
        rows, _ = parsed
        wbc = next(r for r in rows if r.indicator_code == "WBC")
        assert wbc.value == pytest.approx(6.5)
        assert "10^9/L" in (wbc.unit or "")

    def test_qualitative_row_not_ingestible(self, parsed):
        rows, report = parsed
        qual = next(r for r in rows if r.is_qualitative)
        assert "乙肝表面抗原" in (qual.matched_name or "")
        assert not qual.ingestible
        assert report.n_qualitative == 1

    def test_unknown_indicator_reported(self, parsed):
        rows, report = parsed
        unk = next(r for r in rows if "神秘" in r.raw_line)
        assert unk.indicator_code is None
        assert "indicator_unmatched" in unk.issues
        assert report.n_unmatched >= 1

    def test_scale_suspect_lowers_confidence_and_quarantines(self, registry):
        """
        肌酐打印区间 0.5-1.2（mg/dL 量级）vs 注册表 57-97（μmol/L）：
        量级差 ~70 倍 -> 标记疑点、压置信、进复核队列、不入帧。
        这是"OCR丢小数点/单位错"的交叉证据防线。
        """
        text = "肌酐 CREA 1.1 mg/dL 0.5-1.2"
        parser = LabReportParser(registry)
        rows, report = parser.parse(text)
        row = rows[0]
        assert row.indicator_code == "CREA"
        assert "unit_or_scale_suspect" in row.issues
        assert row.confidence < 0.75
        assert report.n_review == 1 and report.n_ingested == 0
        frame = rows_to_frame(rows, "p1", "2026-08-01")
        assert frame.empty

    def test_unit_missing_lowers_confidence(self, registry):
        parser = LabReportParser(registry)
        rows, _ = parser.parse("血红蛋白 HGB 145")
        row = rows[0]
        assert row.indicator_code == "HGB"
        assert "unit_missing" in row.issues
        assert row.confidence == pytest.approx(0.9)

    def test_end_to_end_into_cleaning_pipeline(self, registry):
        """
        全链路：OCR文本 -> 解析 -> 标准长表 -> LabDataCleaner
        单位换算(μmol/L原样、10^9/L原样)与三态标注由既有管线完成。
        """
        frame, report, _rows = parse_lab_text(
            OCR_SAMPLE, registry, patient_id="p001", measured_at="2026-08-10"
        )
        assert report.n_ingested == len(frame) >= 5
        demo = pd.DataFrame(
            [{COL_PATIENT_ID: "p001", "sex": "M", "birth_date": pd.Timestamp("1981-03-01")}]
        )
        pipeline = LabDataCleaner(registry)
        cleaned, creport = pipeline.clean(frame, demographics=demo)
        assert creport.n_output == len(frame)
        alt = cleaned[cleaned[COL_INDICATOR] == "ALT"].iloc[0]
        assert alt[COL_VALUE] == 45.0
        # 45 岁男性 ALT 参考上限 50 —— 分性别区间生效，45 判 NORMAL
        assert alt[COL_STATUS] == MeasureStatus.NORMAL
        glu = cleaned[cleaned[COL_INDICATOR] == "GLU"].iloc[0]
        assert glu[COL_VALUE] == pytest.approx(6.8)
        assert glu[COL_STATUS] == MeasureStatus.ABNORMAL  # 上限 6.1
        k = cleaned[cleaned[COL_INDICATOR] == "K"].iloc[0]
        assert k[COL_STATUS] == MeasureStatus.NORMAL


# ===========================================================================
# 智能就医建议
# ===========================================================================
def _cleaned_frame(rows: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                COL_PATIENT_ID: "p1",
                COL_INDICATOR: code,
                COL_VALUE: val,
                COL_UNIT: None,
                COL_MEASURED_AT: pd.Timestamp("2026-08-01"),
            }
            for code, val in rows
        ]
    )


class TestReferral:
    def test_grade_parity_with_deviation_module(self, registry):
        """
        单一真源守护：referral.grade_value 与 features.deviation._grade_of
        在全指标 × 多倍率网格上必须逐点一致。
        """
        for code in registry.codes:
            meta = registry.require(code)
            iv = meta.match_interval(sex="M", age=45.0)
            if iv is None:
                continue
            probes = []
            if iv.upper is not None:
                probes += [iv.upper * f for f in (0.9, 1.05, 1.3, 2.0, 3.5, 6.0)]
            if iv.lower is not None and iv.lower > 0:
                probes += [iv.lower / f for f in (1.05, 1.4, 2.2, 4.0)]
            for v in probes:
                assert grade_value(meta, v, "M", 45.0) == _grade_of(meta, v, iv), (
                    f"{code} 在 {v} 处分级不一致"
                )

    def test_every_indicator_has_department_rule(self, registry):
        """新指标入 yaml 必须同步归组，防止异常了却给不出建议。"""
        covered = {c for rule in DEPARTMENT_RULES for c in rule.codes}
        missing = [c for c in registry.codes if c not in covered]
        assert not missing, f"以下指标未归属科室规则: {missing}"

    def test_critical_potassium_is_urgent(self, registry):
        """血钾 6.8（危急值 high=6.5）-> 急迫级 + 尽快就诊措辞。"""
        eng = ReferralEngine(registry)
        advice = eng.advise(_cleaned_frame([("K", 6.8)]))
        assert len(advice.items) == 1
        item = advice.items[0]
        assert item.priority == Priority.URGENT
        assert "肾内科" in item.department
        assert "建议尽快" in advice.rendered_text

    def test_mild_liver_is_routine_with_checkups(self, registry):
        eng = ReferralEngine(registry)
        advice = eng.advise(_cleaned_frame([("ALT", 48.0), ("GGT", 70.0)]))
        item = next(it for it in advice.items if it.group == "肝功能")
        assert item.priority == Priority.ROUTINE
        assert "消化内科" in item.department
        assert any("肝功能" in c for c in item.checkups)
        assert len(item.reasons) == 2

    def test_tier_escalates_priority(self, registry):
        """同样的轻度异常，整体分层"极高危"时提级为 SOON，并附整体建议。"""
        eng = ReferralEngine(registry)
        base = eng.advise(_cleaned_frame([("ALT", 48.0)]))
        assert base.items[0].priority == Priority.ROUTINE
        hot = eng.advise(_cleaned_frame([("ALT", 48.0)]), risk_tier="极高危")
        assert hot.items[0].priority == Priority.SOON
        assert hot.general_note and "极高危" in hot.general_note

    def test_high_tier_no_findings_still_notes(self, registry):
        eng = ReferralEngine(registry)
        advice = eng.advise(_cleaned_frame([("ALT", 20.0)]), risk_tier="高危")
        assert advice.items == []
        assert advice.general_note and "高危" in advice.general_note
        assert DISCLAIMER in advice.rendered_text

    def test_rendered_text_is_compliant(self, registry):
        """合规硬闸：富场景渲染文本必须无禁用词、带免责声明。"""
        eng = ReferralEngine(registry)
        advice = eng.advise(
            _cleaned_frame(
                [("K", 6.8), ("GLU", 15.0), ("ALT", 300.0), ("CREA", 200.0), ("TG", 6.0)]
            ),
            risk_tier="极高危",
        )
        assert is_compliant(advice.rendered_text)
        assert DISCLAIMER in advice.rendered_text
        for banned in ("确诊", "治疗", "用药", "服用"):
            assert banned not in advice.rendered_text

    def test_max_items_cap_and_priority_order(self, registry):
        eng = ReferralEngine(registry, max_items=3)
        advice = eng.advise(
            _cleaned_frame(
                [("K", 6.8), ("ALT", 48.0), ("GLU", 7.0), ("PLT", 80.0), ("TG", 3.0), ("CRP", 20.0)]
            )
        )
        assert len(advice.items) == 3
        priorities = [int(it.priority) for it in advice.items]
        assert priorities == sorted(priorities, reverse=True)
        assert advice.items[0].group == "电解质"  # 危急值组必须排最前

    def test_latest_record_wins(self, registry):
        """同指标多次记录，只按最近一次评估。"""
        df = pd.concat(
            [
                _cleaned_frame([("ALT", 300.0)]).assign(measured_at=pd.Timestamp("2026-01-01")),
                _cleaned_frame([("ALT", 25.0)]).assign(measured_at=pd.Timestamp("2026-08-01")),
            ]
        )
        eng = ReferralEngine(registry)
        advice = eng.advise(df)
        assert advice.items == []  # 最近一次已正常
