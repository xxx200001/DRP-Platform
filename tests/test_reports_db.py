# -*- coding: utf-8 -*-
"""
V3 改版回归测试 · 报告管理 DB 层 + OCR 报告日期识别。

对应技术核查：
  核查项 1（多份报告独立保存、无覆盖）      → TestReportsDB.test_eight_reports_independent
  核查项 3（时间字段一致性：改日期必须联动）→ TestReportsDB.test_update_date_syncs_lab_records
  改动 1（逐份 删除 / 重新识别）            → test_delete_cascades / test_reparse_clear_and_counts
  改动 2（OCR 自动识别检查日期）            → TestDetectReportDate.*

依赖说明：仅用标准库（sqlite3 / unittest / ast），不需要 fastapi/pytest —— 
_detect_report_date 定义在 app/server.py（顶部 import fastapi），无法在无依赖
环境整体导入，故用 ast 从源码中【提取生产代码本体】执行后测试，保证测的
就是线上那份实现而不是复制品。
"""
from __future__ import annotations

import ast
import sys
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.db import AppDB  # noqa: E402


# ---------------------------------------------------------------- 工具

def _load_detect_fn():
    """从 app/server.py 源码中提取 _DATE_LINE_KEYWORDS 与 _detect_report_date。"""
    src = (ROOT / "app" / "server.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    wanted: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_DATE_LINE_KEYWORDS"
            for t in node.targets
        ):
            wanted.append(node)
        if isinstance(node, ast.FunctionDef) and node.name == "_detect_report_date":
            wanted.append(node)
    assert len(wanted) == 2, "server.py 中未找到日期识别实现"
    mod = ast.Module(body=wanted, type_ignores=[])
    ns = {"datetime": datetime, "timezone": timezone, "date": date}
    exec(compile(mod, filename="<server.py:extract>", mode="exec"), ns)
    return ns["_detect_report_date"]


def _mk_db() -> AppDB:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db = AppDB(tmp.name)
    db.create_patient("P001", sex="F", birth_date="1962-05-01")
    return db


def _ingest(db: AppDB, measured_at: str, codes: list[str]) -> int:
    rid = db.insert_report(
        "P001", raw_text=f"报告 {measured_at}", measured_at=measured_at,
        n_ingested=len(codes), n_review=0, n_unmatched=0,
    )
    db.insert_lab_records([
        {"patient_id": "P001", "indicator_code": c, "value": 1.0 + i,
         "unit": "U/L", "measured_at": measured_at, "status": "ok",
         "report_id": rid}
        for i, c in enumerate(codes)
    ])
    return rid


# ---------------------------------------------------------------- DB 层

class TestReportsDB(unittest.TestCase):
    def setUp(self):
        self.db = _mk_db()

    def tearDown(self):
        self.db.close()

    def test_eight_reports_independent(self):
        """核查项 1：同一天连传 8 份 → 8 行 reports、记录互不覆盖。"""
        rids = [_ingest(self.db, f"2025-0{m}-10", ["ALT", "AST"]) for m in range(1, 9)]
        self.assertEqual(len(set(rids)), 8)
        reps = self.db.list_reports("P001")
        self.assertEqual(len(reps), 8)
        self.assertTrue(all(r["n_stored"] == 2 for r in reps))
        # 升序按真实检查日期
        self.assertEqual([r["measured_at"] for r in reps],
                         sorted(r["measured_at"] for r in reps))
        self.assertEqual(len(self.db.records_for_patient("P001")), 16)

    def test_update_date_syncs_lab_records(self):
        """核查项 3：改报告日期必须联动其名下全部 lab_records.measured_at。"""
        rid = _ingest(self.db, "2025-03-10", ["ALT", "AST", "GGT"])
        _ingest(self.db, "2025-06-10", ["ALT"])  # 别的报告不受影响
        self.db.update_report_date(rid, "2025-04-01")
        self.assertEqual(self.db.get_report(rid)["measured_at"], "2025-04-01")
        dates = {r["measured_at"] for r in self.db.records_for_patient("P001")}
        self.assertEqual(dates, {"2025-04-01", "2025-06-10"})

    def test_delete_cascades(self):
        rid1 = _ingest(self.db, "2025-03-10", ["ALT", "AST"])
        rid2 = _ingest(self.db, "2025-06-10", ["ALT"])
        n = self.db.delete_report(rid1)
        self.assertEqual(n, 2)
        self.assertIsNone(self.db.get_report(rid1))
        self.assertEqual([r["id"] for r in self.db.list_reports("P001")], [rid2])
        self.assertEqual(len(self.db.records_for_patient("P001")), 1)

    def test_reparse_clear_and_counts(self):
        """重新识别 = 清空旧记录（报告行保留）→ 重灌 → 更新计数。"""
        rid = _ingest(self.db, "2025-03-10", ["ALT", "AST"])
        self.assertEqual(self.db.clear_report_records(rid), 2)
        self.assertIsNotNone(self.db.get_report(rid))          # 报告行还在
        self.assertEqual(self.db.list_reports("P001")[0]["n_stored"], 0)
        self.db.insert_lab_records([{
            "patient_id": "P001", "indicator_code": "ALT", "value": 88.0,
            "unit": "U/L", "measured_at": "2025-03-10", "status": "ok",
            "report_id": rid}])
        self.db.update_report_counts(rid, n_ingested=1, n_review=0, n_unmatched=1)
        rep = self.db.list_reports("P001")[0]
        self.assertEqual((rep["n_ingested"], rep["n_unmatched"], rep["n_stored"]),
                         (1, 1, 1))


# ---------------------------------------------------------------- OCR 日期识别

class TestDetectReportDate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detect = staticmethod(_load_detect_fn())

    def test_keyword_line_wins(self):
        d, src = self.detect("姓名:张三 出生日期:1962-05-01\n采样时间：2025年10月07日 08:31")
        self.assertEqual(d, "2025-10-07")
        self.assertIn("采样", src)

    def test_keyword_priority_over_plain_hits(self):
        # 报告行日期更晚，但"采样"优先级高于"报告"
        d, _ = self.detect("采样日期 2025/10/07\n报告日期 2025/10/09")
        self.assertEqual(d, "2025-10-07")

    def test_dotted_and_cn_formats(self):
        self.assertEqual(self.detect("检验日期 2025.1.7")[0], "2025-01-07")
        self.assertEqual(self.detect("送检：2025-10-07")[0], "2025-10-07")

    def test_fallback_max_any_date(self):
        d, src = self.detect("2024-12-01 参考\n体检 2025-02-03")
        self.assertEqual(d, "2025-02-03")
        self.assertIn("文本", src)

    def test_future_and_invalid_rejected(self):
        self.assertEqual(self.detect("采样时间 2031-01-01")[0], None)   # 未来
        self.assertEqual(self.detect("采样时间 2025-02-30")[0], None)   # 非法
        self.assertEqual(self.detect("")[0], None)

    def test_birthdate_not_picked(self):
        # 只有出生日期（<2000）→ 识别不出，交给用户填
        self.assertEqual(self.detect("出生日期 1962-05-01")[0], None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
