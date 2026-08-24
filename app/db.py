"""
应用数据库层（SQLite，标准库 sqlite3，零新增依赖）。

【与 drp 核心库存储的分工 —— 谁是事实来源】
  audit/ JSONL   : 预测因果链的【不可篡改事实来源】（规范 4.2 append-only）。
  registry/      : 模型版本状态的【唯一账本】（models/registry.py）。
  app.db (本层)  : 应用可查询状态 —— 患者档案、化验记录、报告、预测索引。
                   predictions 表只是审计日志的**查询索引**（按患者/时间检索
                   用），任何数字与 JSONL 冲突时以 JSONL 为准；本表可随时
                   从审计日志重建（rebuild 语义），因此丢失不构成事故。

【PII 纪律（规范 1.2）】
  本库【结构上没有】姓名/手机号/身份证字段 —— 不是"约定不填"，是没有列。
  patient_id 为业务侧标识（工号/卡号等非明文 PII 编号）；原始报告文本入库
  前必须过 serving.audit.scan_pii（server 层强制），含 PII 直接 422 拒收。

【并发】
  WAL 模式 + 每请求独立连接（sqlite3 连接不跨线程复用），check_same_thread
  =False 仅用于 FastAPI 线程池场景下的只读/短写；所有写操作单语句自提交，
  避免长事务。单机部署（本应用的定位）下这套足够；多实例部署请换 PostgreSQL
  并保持本层接口不变。
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id  TEXT PRIMARY KEY,
    sex         TEXT NOT NULL CHECK (sex IN ('M','F')),
    birth_date  TEXT NOT NULL,          -- ISO 日期
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reports (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id  TEXT NOT NULL REFERENCES patients(patient_id),
    raw_text    TEXT NOT NULL,          -- 入库前已过 scan_pii（server 层强制）
    measured_at TEXT NOT NULL,
    n_ingested  INTEGER NOT NULL,
    n_review    INTEGER NOT NULL,
    n_unmatched INTEGER NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS lab_records (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id     TEXT NOT NULL REFERENCES patients(patient_id),
    indicator_code TEXT NOT NULL,
    value          REAL NOT NULL,       -- canonical 单位下的数值（清洗后）
    unit           TEXT NOT NULL,
    measured_at    TEXT NOT NULL,
    status         INTEGER NOT NULL,    -- data.constants.MeasureStatus
    report_id      INTEGER REFERENCES reports(id)
);
CREATE INDEX IF NOT EXISTS idx_records_patient ON lab_records(patient_id, measured_at);

CREATE TABLE IF NOT EXISTS predictions (
    trace_id      TEXT PRIMARY KEY,     -- 与审计 JSONL 同 trace_id，可互查
    patient_id    TEXT NOT NULL REFERENCES patients(patient_id),
    horizon       TEXT NOT NULL,
    probability   REAL NOT NULL,
    risk_tier     TEXT NOT NULL,
    model_version TEXT NOT NULL,
    arm           TEXT,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pred_patient ON predictions(patient_id, created_at);

CREATE TABLE IF NOT EXISTS feedback (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id       TEXT NOT NULL,
    event_occurred INTEGER NOT NULL,
    days           REAL NOT NULL,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS medications (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      TEXT NOT NULL REFERENCES patients(patient_id),
    medication_name TEXT NOT NULL,   -- 入库前必须过 scan_pii（server 层强制）
    start_date      TEXT,            -- ISO 日期；NULL=起始不详（判定时视为已开始）
    end_date        TEXT,            -- NULL=仍在服用（confounders 的时间窗语义）
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_med_patient ON medications(patient_id);
"""

# ---------------------------------------------------------------------------
# 患者档案列（规范 2.1）。列名与 drp.features.demographics 的输入约定一一对应，
# 改任何一边都必须同步另一边 —— cohort 列名对不上，特征就静默变成"未采集"。
#
# 三态纪律：既往史/家族史存 INTEGER 且【允许 NULL】。
#   1 = 有   0 = 无   NULL = 未采集
# 绝不允许用 0 兼表"无"和"没问"—— 那会把三态压成二态，正是规范 1.2 明令禁止的。
# ---------------------------------------------------------------------------
PROFILE_COLUMNS: dict[str, str] = {
    # 体格
    "height_cm": "REAL",
    "weight_kg": "REAL",
    "waist_cm": "REAL",
    # 生活方式
    "smoking_status": "INTEGER",        # 0从不/1已戒/2现吸
    "smoking_pack_years": "REAL",
    "drinking_status": "INTEGER",       # 0从不/1偶尔/2经常/3每日
    "drinking_g_per_week": "REAL",
    "exercise_freq_per_week": "REAL",
    "sleep_hours": "REAL",
    # 既往史（drp.features.demographics.HISTORY_FIELDS）
    "hx_hypertension": "INTEGER",
    "hx_diabetes": "INTEGER",
    "hx_hyperlipidemia": "INTEGER",
    "hx_cad": "INTEGER",
    "hx_stroke": "INTEGER",
    "hx_ckd": "INTEGER",
    "hx_hbv": "INTEGER",
    "hx_fatty_liver": "INTEGER",
    "hx_cancer": "INTEGER",
    "hx_gout": "INTEGER",
    # 家族史（FAMILY_HISTORY_FIELDS）
    "fh_diabetes": "INTEGER",
    "fh_hypertension": "INTEGER",
    "fh_cad": "INTEGER",
    "fh_stroke": "INTEGER",
    "fh_cancer": "INTEGER",
    "fh_ckd": "INTEGER",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AppDB:
    """每个实例持有单一连接；FastAPI 层通过 per-request 依赖创建。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(_SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """
        幂等迁移：给老库的 patients 表补齐档案列（规范 2.1）。

        executescript 里的 CREATE TABLE IF NOT EXISTS 对【已存在】的表不生效，
        所以历史部署的 patients 表没有档案列 —— 这里按 PRAGMA 实际列 diff 补。
        新列全部允许 NULL：老患者的档案字段自然落在"未采集"态，语义正确，
        不需要（也不允许）任何回填。
        """
        have = {r["name"] for r in self.conn.execute("PRAGMA table_info(patients)")}
        for col, sqltype in PROFILE_COLUMNS.items():
            if col not in have:
                self.conn.execute(f"ALTER TABLE patients ADD COLUMN {col} {sqltype}")
                logger.info("迁移: patients 表新增档案列 %s %s", col, sqltype)

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------ 患者
    def create_patient(
        self, patient_id: str, sex: str, birth_date: str, profile: dict | None = None
    ) -> dict:
        cols = ["patient_id", "sex", "birth_date", "created_at"]
        vals = [patient_id, sex.upper(), birth_date, _now()]
        for col in PROFILE_COLUMNS:
            cols.append(col)
            vals.append((profile or {}).get(col))
        marks = ",".join("?" * len(cols))
        self.conn.execute(
            f"INSERT INTO patients({','.join(cols)}) VALUES ({marks})", vals
        )
        self.conn.commit()
        return self.get_patient(patient_id)

    def update_profile(self, patient_id: str, profile: dict) -> dict:
        """
        整档覆盖写（不是 patch）：表单每次提交全部档案字段，
        未填 = None = 未采集。这让"把某项从『有』改回『未采集』"成为可能 ——
        patch 语义做不到这一点，而三态里"撤回一个回答"是真实需求。
        """
        sets = ", ".join(f"{c}=?" for c in PROFILE_COLUMNS)
        vals = [profile.get(c) for c in PROFILE_COLUMNS] + [patient_id]
        cur = self.conn.execute(
            f"UPDATE patients SET {sets} WHERE patient_id=?", vals
        )
        self.conn.commit()
        if cur.rowcount == 0:
            raise KeyError(patient_id)
        return self.get_patient(patient_id)

    def get_patient(self, patient_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM patients WHERE patient_id=?", (patient_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_patients(self) -> list[dict]:
        rows = self.conn.execute(
            """SELECT p.*,
                      (SELECT COUNT(*) FROM lab_records r WHERE r.patient_id=p.patient_id) AS n_records,
                      (SELECT MAX(created_at) FROM predictions q WHERE q.patient_id=p.patient_id) AS last_predicted_at
               FROM patients p ORDER BY p.created_at DESC"""
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ 报告与记录
    def insert_report(
        self, patient_id: str, raw_text: str, measured_at: str,
        n_ingested: int, n_review: int, n_unmatched: int,
    ) -> int:
        cur = self.conn.execute(
            """INSERT INTO reports(patient_id, raw_text, measured_at,
                                   n_ingested, n_review, n_unmatched, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (patient_id, raw_text, measured_at, n_ingested, n_review, n_unmatched, _now()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def insert_lab_records(self, rows: list[dict]) -> int:
        """rows: [{patient_id, indicator_code, value, unit, measured_at, status, report_id}]"""
        if not rows:
            return 0
        self.conn.executemany(
            """INSERT INTO lab_records(patient_id, indicator_code, value, unit,
                                       measured_at, status, report_id)
               VALUES (:patient_id, :indicator_code, :value, :unit,
                       :measured_at, :status, :report_id)""",
            rows,
        )
        self.conn.commit()
        return len(rows)

    def records_for_patient(self, patient_id: str) -> list[dict]:
        rows = self.conn.execute(
            """SELECT indicator_code, value, unit, measured_at, status
               FROM lab_records WHERE patient_id=? ORDER BY measured_at""",
            (patient_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ 报告管理（改版·改动 1）
    # 需求：用户连续上传 7-8 份报告后，必须能看到"每一份都独立入库、没有覆盖"，
    # 并且可以逐份 查看原文 / 修改检查日期 / 删除 / 重新识别。
    # 注意：本表【有意】不存原始图片 —— 化验单图片上通常印有姓名等明文 PII，
    # 与规范 1.2 冲突；可回看的是已过 scan_pii 的 OCR/粘贴文本（raw_text）。
    def list_reports(self, patient_id: str) -> list[dict]:
        """该患者全部报告（按真实检查日期 measured_at 升序），附每份实际入库指标数。"""
        rows = self.conn.execute(
            """SELECT r.id, r.measured_at, r.created_at,
                      r.n_ingested, r.n_review, r.n_unmatched,
                      (SELECT COUNT(*) FROM lab_records l WHERE l.report_id = r.id) AS n_stored
               FROM reports r WHERE r.patient_id=?
               ORDER BY r.measured_at, r.id""",
            (patient_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_report(self, report_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM reports WHERE id=?", (report_id,)
        ).fetchone()
        return dict(row) if row else None

    def update_report_date(self, report_id: int, measured_at: str) -> None:
        """
        修改一份报告的真实检查日期。必须【同事务】联动它名下所有 lab_records 的
        measured_at —— 趋势/时间轴/纵向特征全部按 lab_records.measured_at 计算，
        只改 reports 表会造成两处时间对不上（正是本次核查项 3 要杜绝的问题）。
        """
        self.conn.execute(
            "UPDATE reports SET measured_at=? WHERE id=?", (measured_at, report_id)
        )
        self.conn.execute(
            "UPDATE lab_records SET measured_at=? WHERE report_id=?",
            (measured_at, report_id),
        )
        self.conn.commit()

    def delete_report(self, report_id: int) -> int:
        """删除报告及其名下全部指标记录。返回删除的指标记录数。"""
        cur = self.conn.execute(
            "DELETE FROM lab_records WHERE report_id=?", (report_id,)
        )
        n = cur.rowcount
        self.conn.execute("DELETE FROM reports WHERE id=?", (report_id,))
        self.conn.commit()
        return int(n)

    def clear_report_records(self, report_id: int) -> int:
        """清空一份报告名下的指标记录（重新识别前调用），报告行保留。"""
        cur = self.conn.execute(
            "DELETE FROM lab_records WHERE report_id=?", (report_id,)
        )
        self.conn.commit()
        return int(cur.rowcount)

    def update_report_counts(
        self, report_id: int, n_ingested: int, n_review: int, n_unmatched: int
    ) -> None:
        self.conn.execute(
            "UPDATE reports SET n_ingested=?, n_review=?, n_unmatched=? WHERE id=?",
            (n_ingested, n_review, n_unmatched, report_id),
        )
        self.conn.commit()

    # ------------------------------------------------------------------ 用药（规范 2.4）
    def add_medication(
        self, patient_id: str, medication_name: str,
        start_date: str | None, end_date: str | None,
    ) -> dict:
        cur = self.conn.execute(
            """INSERT INTO medications(patient_id, medication_name, start_date, end_date, created_at)
               VALUES (?,?,?,?,?)""",
            (patient_id, medication_name, start_date, end_date, _now()),
        )
        self.conn.commit()
        return self.get_medication(int(cur.lastrowid))

    def get_medication(self, med_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM medications WHERE id=?", (med_id,)
        ).fetchone()
        return dict(row) if row else None

    def medications_for_patient(self, patient_id: str) -> list[dict]:
        rows = self.conn.execute(
            """SELECT * FROM medications WHERE patient_id=?
               ORDER BY COALESCE(end_date, '9999-12-31') DESC, start_date DESC, id DESC""",
            (patient_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_medication(self, med_id: int) -> bool:
        cur = self.conn.execute("DELETE FROM medications WHERE id=?", (med_id,))
        self.conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------ 预测索引
    def index_prediction(
        self, trace_id: str, patient_id: str, horizon: str,
        probability: float, risk_tier: str, model_version: str, arm: str | None,
    ) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO predictions
               (trace_id, patient_id, horizon, probability, risk_tier,
                model_version, arm, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (trace_id, patient_id, horizon, probability, risk_tier,
             model_version, arm, _now()),
        )
        self.conn.commit()

    def predictions_for_patient(self, patient_id: str) -> list[dict]:
        rows = self.conn.execute(
            """SELECT * FROM predictions WHERE patient_id=?
               ORDER BY created_at""",
            (patient_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def latest_prediction_group(self, patient_id: str) -> list[dict]:
        """最近一次预测（同一时刻的多时程为一组，按 created_at 最大值取组）。"""
        rows = self.conn.execute(
            """SELECT * FROM predictions
               WHERE patient_id=? AND created_at =
                     (SELECT MAX(created_at) FROM predictions WHERE patient_id=?)
               ORDER BY horizon""",
            (patient_id, patient_id),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ 回流
    def insert_feedback(self, trace_id: str, event_occurred: bool, days: float) -> None:
        self.conn.execute(
            "INSERT INTO feedback(trace_id, event_occurred, days, created_at) VALUES (?,?,?,?)",
            (trace_id, int(event_occurred), float(days), _now()),
        )
        self.conn.commit()

    # ------------------------------------------------------------------ 统计
    def stats(self) -> dict:
        q = lambda sql: self.conn.execute(sql).fetchone()[0]  # noqa: E731
        return {
            "patients": q("SELECT COUNT(*) FROM patients"),
            "lab_records": q("SELECT COUNT(*) FROM lab_records"),
            "reports": q("SELECT COUNT(*) FROM reports"),
            "predictions": q("SELECT COUNT(*) FROM predictions"),
            "feedback": q("SELECT COUNT(*) FROM feedback"),
        }
