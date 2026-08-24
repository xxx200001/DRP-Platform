# -*- coding: utf-8 -*-
"""
V3.1 回归测试 · OCR 版面重排 + 解析器预处理。

对应用户实测反馈：
  「识别不准确不完整」（横版照片乱序 / 双栏右列丢失 / 裸序号吃掉数值）
  「上传两张不知道是否都识别到」（双栏丢列是漏识别来源之一）

  方向评分            → TestTextWeight        （转错方向中文全灭，得分应数量级下降）
  双栏逐行拆分        → TestPanelReconstruct  （模拟肝功能 15 项左右分栏真实版面）
  单栏不误拆          → TestSingleColumn
  PII 脱敏            → TestRedact
  裸序号预处理        → TestPreprocess        （"1 总胆汁酸 4.0" 不能把序号当数值）

依赖说明：ocr_layout 为纯标准库模块直接 import；parser._preprocess 所在模块
顶层 import pandas（本环境不可用），故用 ast 从源码提取【生产代码本体】
（_preprocess + 其依赖 to_halfwidth）执行后测试 —— 测的就是线上那份实现。
"""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ocr_layout import (  # noqa: E402
    cluster_rows,
    find_panel_split,
    reconstruct_lines,
    redact_pii_text,
    text_weight,
)


# ---------------------------------------------------------------- 工具
def _box(text, x0, y0, w=60, h=22):
    return {"text": text, "x0": x0, "y0": y0, "x1": x0 + w, "y1": y0 + h}


def _load_preprocess():
    """ast 提取 parser._preprocess 与 reference.to_halfwidth（均为纯函数）。"""
    import re as _re

    def _pull(path: Path, names: set[str]) -> list[ast.stmt]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        out = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                out.append(node)
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in names for t in node.targets
            ):
                out.append(node)
        return out

    body = _pull(ROOT / "src/drp/data/reference.py", {"to_halfwidth", "_HALFWIDTH"})
    body += _pull(ROOT / "src/drp/ingest/parser.py", {"_preprocess"})
    assert any(isinstance(n, ast.FunctionDef) and n.name == "_preprocess" for n in body)
    ns = {"re": _re}
    exec(compile(ast.Module(body=body, type_ignores=[]), "<extract>", "exec"), ns)
    return ns["_preprocess"]


# ---------------------------------------------------------------- 方向评分
class TestTextWeight(unittest.TestCase):
    def test_cjk_dominates(self):
        good = text_weight("丙氨酸氨基转移酶") * 0.9      # 正确方向：中文 × 高置信
        bad = text_weight("|||") * 0.3                     # 转错方向：符号渣 × 低置信
        self.assertGreater(good, bad * 10)

    def test_empty(self):
        self.assertEqual(text_weight(""), 0.0)


# ---------------------------------------------------------------- 双栏重排
def _two_panel_items():
    """模拟真实肝功能单：页宽 1700，左栏 x∈[60,700]，右栏 x∈[900,1600]。
    每个视觉行左右各一条记录，各记录由 序号/名称/值/区间/单位 多个框组成。"""
    rows = [
        # (y, 左栏cells, 右栏cells)
        (100, [("1", 60), ("总胆汁酸", 110), ("4.0", 380), ("<10.0", 470), ("μmol/L", 590)],
              [("9", 900), ("丙氨酸氨基转移酶", 950), ("112", 1280), ("7-40", 1380), ("U/L", 1500)]),
        (140, [("2", 60), ("总胆红素", 110), ("9.8", 380), ("<26.0", 470), ("μmol/L", 590)],
              [("10", 900), ("天门冬氨酸氨基转移", 950), ("68", 1280), ("13-40", 1380), ("U/L", 1500)]),
        (180, [("3", 60), ("直接胆红素", 110), ("5.2", 380), ("<8.0", 470), ("μmol/L", 590)],
              [("12", 900), ("碱性磷酸酶", 950), ("103", 1280), ("35-100", 1380), ("U/L", 1500)]),
        (220, [("4", 60), ("间接胆红素", 110), ("4.6", 380), ("3.0-14.0", 470), ("μmol/L", 590)],
              [("13", 900), ("γ-谷氨酰基转移酶", 950), ("63", 1280), ("7-45", 1380), ("U/L", 1500)]),
    ]
    items = []
    for y, left, right in rows:
        for txt, x in left + right:
            items.append(_box(txt, x, y))
    return items, 1700.0


class TestPanelReconstruct(unittest.TestCase):
    def test_split_detected_and_rows_paired(self):
        items, w = _two_panel_items()
        self.assertIsNotNone(find_panel_split(items, w))
        lines, layout = reconstruct_lines(items, w)
        self.assertEqual(layout, "two_panel")
        # 4 个视觉行 × 左右两栏 = 8 条独立记录行，右栏一条不丢
        self.assertEqual(len(lines), 8)
        # 逐行配对正确：值跟着自己栏的名称走
        self.assertIn("总胆汁酸", lines[0]);  self.assertIn("4.0", lines[0])
        self.assertNotIn("112", lines[0])     # 右栏的值绝不能混进左栏行
        self.assertIn("丙氨酸氨基转移酶", lines[1]); self.assertIn("112", lines[1])
        self.assertIn("碱性磷酸酶", lines[5]); self.assertIn("103", lines[5])

    def test_scrambled_input_order_irrelevant(self):
        items, w = _two_panel_items()
        items = list(reversed(items))          # 打乱输入顺序（OCR 返回序不可依赖）
        lines, _ = reconstruct_lines(items, w)
        self.assertIn("总胆汁酸", lines[0])
        self.assertIn("γ-谷氨酰基转移酶", lines[7])

    def test_wide_title_does_not_block_gap(self):
        items, w = _two_panel_items()
        # 横贯整页的医院抬头（宽 > 0.55 页宽）不应破坏分栏探测
        items.append(_box("德清县某健康保健集团临检中心报告单", 100, 30, w=1500, h=30))
        self.assertIsNotNone(find_panel_split(items, w))


class TestSingleColumn(unittest.TestCase):
    def test_no_false_split(self):
        items = []
        for i, y in enumerate(range(100, 460, 40)):
            items.extend([
                _box(f"指标{i}", 60, y), _box("5.4", 380, y),
                _box("3.0-14.0", 470, y), _box("mmol/L", 620, y),
            ])
        lines, layout = reconstruct_lines(items, 800.0)
        self.assertEqual(layout, "single")
        self.assertEqual(len(lines), 9)
        self.assertTrue(lines[0].startswith("指标0"))

    def test_cluster_rows_tolerance(self):
        # 同一行内 ±4px 抖动应聚在一起
        items = [_box("甘油三脂", 60, 100), _box("1.88", 300, 104), _box("mmol/L", 420, 97)]
        rows = cluster_rows(items)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]), 3)


# ---------------------------------------------------------------- 脱敏
class TestRedact(unittest.TestCase):
    def test_name_and_patterns(self):
        text = "姓 名：何鑫  性别：女\n联系电话 13812345678\n采集时间：2025/10/06 09:39"
        out, n = redact_pii_text(text)
        self.assertNotIn("何鑫", out)
        self.assertNotIn("13812345678", out)
        self.assertIn("[已脱敏]", out)
        self.assertIn("采集时间：2025/10/06", out)   # 日期行必须原样保留
        self.assertEqual(n, 2)

    def test_clean_text_untouched(self):
        text = "丙氨酸氨基转移酶 112 U/L 7-40"
        out, n = redact_pii_text(text)
        self.assertEqual(out, text)
        self.assertEqual(n, 0)


# ---------------------------------------------------------------- 解析器预处理
class TestPreprocess(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pre = staticmethod(_load_preprocess())

    def test_bare_seq_stripped(self):
        # OCR 独立序号框重排后的典型行：序号 + 空格 + 指标名
        self.assertEqual(self.pre("1 总胆汁酸 4.0 <10.0 μmol/L"),
                         "总胆汁酸 4.0 <10.0 μmol/L")
        self.assertEqual(self.pre("16 AST/ALT 0.63"), "AST/ALT 0.63")

    def test_value_leading_line_untouched(self):
        # 以真实数值开头的行（小数点在 1-3 位内出现）绝不能被当序号剥掉
        self.assertTrue(self.pre("4.07 mmol/L 3.50-5.30").startswith("4.07"))

    def test_old_punct_seq_still_works(self):
        self.assertEqual(self.pre("2、总胆红素 9.8"), "总胆红素 9.8")


if __name__ == "__main__":
    unittest.main(verbosity=2)
