# -*- coding: utf-8 -*-
"""
V3.3 验收测试 · 真实样张的指标必须【全部】识别入库。

用户要求原文："你至少保证我给你的这些图片的指标全部可以正常识别"。
本测试把这条要求钉死成 CI 断言：8 份真实化验单（德清县武康健康保健集团
临检中心）逐份解析，期望入库的指标码一个都不能少。

fixture 是人工核对过的识别文本（见 tests/fixtures_real_reports.py），
刻意保留真实畸变：裸序号列、↑↓ 箭头、"结果 参考区间 单位"列序、
HR 互认前缀、比值行、定性行、影像描述。

分工声明（出问题时据此定位）：
  · 本测试红 → 词典/参考区间/解析器有缺口，与拍照质量无关；
  · 本测试绿但实拍失败 → 问题在 OCR 引擎侧（方向/清晰度/双栏/模型）。

运行：python3 tests/test_real_reports.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from drp.data.reference import ReferenceRegistry  # noqa: E402
from drp.ingest import parse_lab_text  # noqa: E402
from tests.fixtures_real_reports import REPORTS  # noqa: E402

CONFIG = ROOT / "configs" / "reference_intervals.yaml"


class TestRealReportsCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg = ReferenceRegistry.from_yaml(CONFIG)

    def _parse(self, rep):
        frame, report, rows = parse_lab_text(
            rep["text"], self.reg, patient_id="P_REAL",
            measured_at=pd.Timestamp(rep["measured_at"]),
        )
        got = set(frame["indicator_code"]) if len(frame) else set()
        return frame, report, got

    def test_every_expected_indicator_ingested(self):
        """逐份：期望的指标必须全部入库，一个都不能少。"""
        missing_all = {}
        for rep in REPORTS:
            _, _, got = self._parse(rep)
            miss = rep["expect"] - got
            if miss:
                missing_all[rep["name"]] = sorted(miss)
        self.assertEqual(missing_all, {}, f"以下报告有指标未能识别入库：{missing_all}")

    def test_totals(self):
        """总量断言：8 份合计 61 项定量指标全部入库。"""
        total_expect = sum(len(r["expect"]) for r in REPORTS)
        total_got = 0
        for rep in REPORTS:
            _, _, got = self._parse(rep)
            total_got += len(got & rep["expect"])
        self.assertEqual(total_got, total_expect)
        self.assertGreaterEqual(total_expect, 61)

    def test_no_phantom_indicators(self):
        """不得凭空多出指标（定性行/影像描述/比值行不许乱入量化帧）。"""
        for rep in REPORTS:
            _, _, got = self._parse(rep)
            extra = got - rep["expect"]
            self.assertEqual(extra, set(), f"{rep['name']} 多出了不该入库的指标：{extra}")

    def test_qualitative_and_imaging_do_not_crash(self):
        """抗核抗体（全定性+滴度 1:160）与腹部超声：不报错、不入量化帧。"""
        for name in ("抗核抗体全套(定性为主)", "腹部超声(影像描述)"):
            rep = next(r for r in REPORTS if r["name"] == name)
            frame, report, got = self._parse(rep)
            self.assertEqual(got, set())
            self.assertGreater(report.n_lines, 0)   # 确实读了行，不是空转

    def test_key_values_are_correct(self):
        """抽查关键值：确保取到的是结果值，不是序号/参考区间/名称里的数字。"""
        checks = [
            ("生化32项", "ALT", 94.0), ("生化32项", "K", 4.07),
            ("生化32项", "CKMB", 12.0), ("生化32项", "RBP", 27.8),
            ("心肌酶谱", "AST", 58.0), ("心肌酶谱", "CK", 75.0),
            ("肝功能15项", "ALT", 112.0), ("肝功能15项", "LAP", 37.0),
            ("血脂四项", "HDLC", 0.75), ("血脂四项", "TG", 1.88),
            # C3/C4 名字自带数字，旧逻辑会把 "3" 当值 —— 回归守卫
            ("免疫全套", "C3", 152.9), ("免疫全套", "C4", 37.0),
            ("免疫全套", "IGA", 3.59),
            # HR 互认前缀行
            ("肝功能(HR互认·旋转拍摄)", "ALT", 97.0),
            ("肝功能(HR互认·旋转拍摄)", "GGT", 64.0),
        ]
        for rep_name, code, expected in checks:
            rep = next(r for r in REPORTS if r["name"] == rep_name)
            frame, _, got = self._parse(rep)
            self.assertIn(code, got, f"{rep_name} 未识别 {code}")
            val = float(frame[frame["indicator_code"] == code]["value"].iloc[0])
            self.assertAlmostEqual(
                val, expected, places=2,
                msg=f"{rep_name}·{code} 取值错误：得到 {val}，应为 {expected}",
            )


class TestParserFormatRegression(unittest.TestCase):
    """既有支持格式不得回退（_find_value / HR 剥离的副作用守卫）。"""

    @classmethod
    def setUpClass(cls):
        cls.reg = ReferenceRegistry.from_yaml(CONFIG)

    def _one(self, text):
        frame, report, rows = parse_lab_text(
            text, self.reg, patient_id="P", measured_at=pd.Timestamp("2025-10-06")
        )
        return frame, report, rows

    def test_name_value_glued_still_parses(self):
        """名值粘连 "白细胞计数6.5x10^9/L"：无独立数值 token 时退回旧行为。"""
        frame, _, _ = self._one("白细胞计数6.5x10^9/L")
        self.assertIn("WBC", set(frame["indicator_code"]))
        self.assertAlmostEqual(float(frame["value"].iloc[0]), 6.5, places=2)

    def test_classic_formats(self):
        for text, code, val in [
            ("丙氨酸氨基转移酶 ALT  45  U/L  0-40  ↑", "ALT", 45.0),
            ("葡萄糖(GLU)  6.8 mmol/L 参考值:3.9-6.1 H", "GLU", 6.8),
            ("血小板计数 250 10^9/L 125-350", "PLT", 250.0),
        ]:
            frame, _, _ = self._one(text)
            got = set(frame["indicator_code"])
            self.assertIn(code, got, text)
            v = float(frame[frame["indicator_code"] == code]["value"].iloc[0])
            self.assertAlmostEqual(v, val, places=2)

    def test_magnitude_suspect_still_goes_to_review(self):
        """mg/dL 肌酐的量级疑点必须仍然拦进人工复核（不是回退，是既有设计）。"""
        frame, report, rows = self._one("肌酐 CREA 1.2 mg/dL 0.5-1.2")
        self.assertEqual(len(frame), 0)
        self.assertEqual(report.n_review, 1)
        self.assertIn("unit_or_scale_suspect", rows[0].issues)


if __name__ == "__main__":
    unittest.main(verbosity=2)
