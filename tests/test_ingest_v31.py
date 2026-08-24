# -*- coding: utf-8 -*-
"""
V3.1 · 真实化验单行格式的端到端解析回归（需完整依赖环境，pytest 运行）。

背景：接入 8 份真实化验单照片后暴露三类行级畸变，全部来自「OCR 文字框
重排」后的产物形态，本文件把它们钉死：

  1. 裸序号行           "1 总胆汁酸 4.0 <10.0 μmol/L"
       序号列是独立 OCR 框，若不剥离，取值逻辑抓到的第一个数字是序号本身
       （值=1），整行数据报废 —— 这是"识别不完整"的直接来源之一。
  2. 值后紧跟裸单侧区间  真实列序常为「结果 参考区间 单位」，区间无"参考"
       前缀不会被摘走，单位在更后面，需要向后搜索兜底。
  3. 序号点号 vs 小数点  "4.07 mmol/L" 的 "4." 绝不能被当成序号剥掉。

离线可跑的镜像（ast 提取 _preprocess）见 tests/test_ocr_layout.py::TestPreprocess；
本文件走完整 parse_lab_text 链路，锁的是端到端行为。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from drp.data.reference import ReferenceRegistry
from drp.ingest import parse_lab_text

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "reference_intervals.yaml"


@pytest.fixture(scope="module")
def registry() -> ReferenceRegistry:
    return ReferenceRegistry.from_yaml(CONFIG)


def _parse(registry, text: str):
    frame, report, rows = parse_lab_text(
        text, registry, patient_id="P_TEST", measured_at=pd.Timestamp("2025-10-06")
    )
    return frame, report, rows


def _row_of(frame, code: str):
    hit = frame[frame["indicator_code"] == code]
    assert len(hit) == 1, f"{code} 应恰好入库一行，实得 {len(hit)}"
    return hit.iloc[0]


class TestBareSeqLines:
    """裸序号行：值必须是真实结果，不是序号。"""

    def test_seq_not_taken_as_value(self, registry):
        frame, report, _ = _parse(registry, "9 丙氨酸氨基转移酶 112 7-40 U/L")
        assert report.n_ingested == 1
        r = _row_of(frame, "ALT")
        assert r["value"] == pytest.approx(112.0)
        assert r["unit"] == "U/L"

    def test_two_panel_reconstructed_block(self, registry):
        # ocr_layout 双栏拆分后的典型产物：逐行独立、含裸序号与箭头
        text = "\n".join([
            "2 总胆红素 9.8 <26.0 μmol/L",
            "9 丙氨酸氨基转移酶 112 ↑ 7-40 U/L",
            "10 天门冬氨酸氨基转移酶 68 ↑ 13-40 U/L",
            "13 γ-谷氨酰转移酶 63 ↑ 7-45 U/L",
        ])
        frame, report, _ = _parse(registry, text)
        assert report.n_ingested >= 3          # TBIL/ALT/AST/GGT 主体必须在列
        assert _row_of(frame, "ALT")["value"] == pytest.approx(112.0)
        assert _row_of(frame, "AST")["value"] == pytest.approx(68.0)


class TestUnitFallbackSearch:
    """值后紧跟裸单侧区间时，单位向后搜索兜底。"""

    def test_bare_oneside_ref_then_unit(self, registry):
        # 按值定位行，不依赖词典命中 —— 锁的是词法层行为
        _, _, rows = _parse(registry, "总胆红素 9.8 <26.0 μmol/L")
        row = next(r for r in rows if r.value == pytest.approx(9.8))
        assert row.unit == "μmol/L"            # 不是 None，也没把 <26.0 吃进值

    def test_lonely_HL_not_taken_as_unit(self, registry):
        _, _, rows = _parse(registry, "血红蛋白 98 H 115-150")
        row = next(r for r in rows if r.value == pytest.approx(98.0))
        assert row.unit != "H"                 # 孤立 H 是高低标记，不是单位


class TestDecimalNotEaten:
    """序号点号规则不得吞掉小数：4.07 的 '4.' 曾被剥离致值变 07。"""

    def test_value_leading_line(self, registry):
        _, _, rows = _parse(registry, "4.07 mmol/L 3.50-5.30")
        vals = [r.value for r in rows if r.value is not None]
        assert pytest.approx(4.07) in vals     # 值完整保留（名缺失可不入库）

    def test_punct_seq_still_stripped(self, registry):
        frame, report, _ = _parse(registry, "2、葡萄糖 5.14 mmol/L 3.90-6.10")
        assert _row_of(frame, "GLU")["value"] == pytest.approx(5.14)
