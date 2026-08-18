"""
病情预测平台 · 应用后端（FastAPI）。

【本层与 serving/api.py 的关系】
serving/api.py 是"纯预测微服务"形态（特征由调用方给）。本层是完整应用：
患者档案、OCR 报告结构化入库、按患者实时走真实特征管线、多时程预测、
趋势报告、随访回流、管理台（版本/灰度/复盘/漂移/AB）。两层都只做编排，
概率/归因/合规/日志全部发生在 RiskPredictionService 那条不可分割路径里。

【启动前置】必须先完成自举（run_app.py 会自动处理）：
    python run_app.py            # 首次自动 bootstrap（数分钟）后启动
本模块的 build_server() 在未自举时直接报错并给出指令——服务器不做
"现场悄悄训练"这种慢启动魔法。

【规范 7 部署清单（勿删）】
  1) 仅监听内网地址；2) 公网入口由网关强制 HTTPS 并禁用明文 HTTP；
  3) DRP_PII_SALT 必须以环境变量注入（默认 dev 盐只许开发用，启动会告警）。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from drp.data.cleaning import COL_STATUS, LabDataCleaner
from drp.data.constants import (
    COL_BIRTH_DATE,
    COL_INDEX_DATE,
    COL_INDICATOR,
    COL_MEASURED_AT,
    COL_PATIENT_ID,
    COL_SEX,
    COL_UNIT,
    COL_VALUE,
)
from drp.data.reference import ReferenceRegistry
from drp.features import ConfounderConfig, FeaturePipeline, grade_of
from drp.features.demographics import FAMILY_HISTORY_FIELDS, HISTORY_FIELDS
from drp.ingest import parse_lab_text
from drp.models import HorizonBank, ModelRegistry, RegistryError
from drp.serving.audit import AuditLogger, scan_pii
from drp.serving.compliance import DISCLAIMER
from drp.serving.drift import DriftMonitor, ReferenceProfile
from drp.serving.feedback import FeedbackOrchestrator, FollowUpFeedback
from drp.serving.referral import ReferralEngine
from drp.serving.rollout import TrafficRouter, build_ab_comparison
from drp.serving.service import RiskPredictionService, RiskTierScheme, ServiceConfig
from drp.serving.trend import TrendEngine, build_trend_report

from .bootstrap import _PIPELINE_CONFIG, is_bootstrapped
from .db import PROFILE_COLUMNS, AppDB

logger = logging.getLogger(__name__)

_DEFAULT_SALT = "dev-salt-change-me"

#: 界面"一键填充"的示例化验单（合成内容，覆盖多种版式与一处未知指标）
SAMPLE_REPORT = """XX市第一人民医院 检验报告单
样本号:20260815001  科室:体检中心
丙氨酸氨基转移酶 ALT      48    U/L     9-50
天门冬氨酸氨基转移酶 AST  48 U/L 15-40 ↑
γ-谷氨酰转移酶 GGT 88 U/L 10-60 ↑
葡萄糖(GLU)  7。2 mmol/L  参考值:3.9-6.1  H
糖化血红蛋白 HbA1c 6.8 % 4.0-6.0 ↑
甘油三酯 TG 2.9 mmol/L 0.4-1.7 ↑
低密度脂蛋白胆固醇 LDL-C 3.9 mmol/L 1.3-3.4 ↑
高密度脂蛋白胆固醇 HDL-C 0.9 mmol/L 1.0-1.6 ↓
血小板计数 PLT 168 10^9/L 125-350
血红蛋白 HGB 142 g/L 130-175
肌酐 CREA 88 μmol/L 57-97
尿酸 UA 452 μmol/L 208-428 ↑
C反应蛋白 CRP 6.2 mg/L 0-5 ↑
白蛋白 ALB 43 g/L 40-55
收缩压 SBP 146 mmHg 90-140 ↑
舒张压 DBP 92 mmHg 60-90 ↑
体质指数 BMI 27.8 kg/m2 18.5-24
神秘未知指标 12.3 U/L
审核者:系统  报告时间:2026-08-15
"""


# 请求模型必须位于模块作用域。开启 postponed annotations 时，FastAPI 无法
# 可靠解析应用工厂内部的局部类，会把 JSON body 错判为名为 body 的查询参数。
class ProfileIn(BaseModel):
    """
    患者基础档案（规范 2.1）。字段名与 drp.features.demographics 的 cohort
    列名一一对应（也是 app.db.PROFILE_COLUMNS 的键）—— 三处任何一处改名，
    另两处必须同步，否则该字段线上静默退化为"未采集"。

    三态纪律：既往史/家族史 None=未采集、0=无、1=有。表单每次整档提交，
    留空即 None —— 所以"把某项从『有』撤回到『未采集』"是被支持的操作。
    """
    # 体格
    height_cm: float | None = Field(default=None, gt=50, lt=250)
    weight_kg: float | None = Field(default=None, gt=10, lt=400)
    waist_cm: float | None = Field(default=None, gt=30, lt=250)
    # 生活方式
    smoking_status: int | None = Field(default=None, ge=0, le=2)   # 0从不/1已戒/2现吸
    smoking_pack_years: float | None = Field(default=None, ge=0, le=200)
    drinking_status: int | None = Field(default=None, ge=0, le=3)  # 0从不/1偶尔/2经常/3每日
    drinking_g_per_week: float | None = Field(default=None, ge=0, le=3000)
    exercise_freq_per_week: float | None = Field(default=None, ge=0, le=30)
    sleep_hours: float | None = Field(default=None, ge=0, le=24)
    # 既往史
    hx_hypertension: int | None = Field(default=None, ge=0, le=1)
    hx_diabetes: int | None = Field(default=None, ge=0, le=1)
    hx_hyperlipidemia: int | None = Field(default=None, ge=0, le=1)
    hx_cad: int | None = Field(default=None, ge=0, le=1)
    hx_stroke: int | None = Field(default=None, ge=0, le=1)
    hx_ckd: int | None = Field(default=None, ge=0, le=1)
    hx_hbv: int | None = Field(default=None, ge=0, le=1)
    hx_fatty_liver: int | None = Field(default=None, ge=0, le=1)
    hx_cancer: int | None = Field(default=None, ge=0, le=1)
    hx_gout: int | None = Field(default=None, ge=0, le=1)
    # 家族史
    fh_diabetes: int | None = Field(default=None, ge=0, le=1)
    fh_hypertension: int | None = Field(default=None, ge=0, le=1)
    fh_cad: int | None = Field(default=None, ge=0, le=1)
    fh_stroke: int | None = Field(default=None, ge=0, le=1)
    fh_cancer: int | None = Field(default=None, ge=0, le=1)
    fh_ckd: int | None = Field(default=None, ge=0, le=1)


class PatientIn(ProfileIn):
    patient_id: str = Field(
        min_length=1, max_length=64,
        description="业务侧编号（工号/卡号等），禁止填姓名/证件/手机号",
    )
    sex: str = Field(pattern="^[MFmf]$")
    birth_date: str


class MedicationIn(BaseModel):
    """
    用药记录（规范 2.4）。药名自由文本，特征层按 configs/confounders.yaml
    的关键词词典归入类别 —— 应用层只负责如实记录，不做分类。
    end_date 留空 = 仍在服用（confounders 时间窗判定的约定）。
    """
    medication_name: str = Field(min_length=1, max_length=120)
    start_date: str | None = None
    end_date: str | None = None


class ReportIn(BaseModel):
    patient_id: str
    text: str = Field(min_length=5)
    measured_at: str


class PredictIn(BaseModel):
    patient_id: str
    # 采血登记（规范 2.4 生理状态）。描述的是【本次预测时点】的状态，
    # 属于每次就诊的登记项而非患者档案 —— 妊娠/空腹会变，所以随请求传，
    # 不落 patients 表。None = 未登记（特征层不产出该项，与"否"严格区分）。
    non_fasting: bool | None = None
    strenuous_exercise: bool | None = None
    pregnancy: bool | None = None


class FeedbackIn(BaseModel):
    trace_id: str
    event_occurred: bool
    days_since_prediction: float
    consented: bool


class CanaryIn(BaseModel):
    version: str
    traffic_pct: float


class VersionIn(BaseModel):
    version: str


# ---------------------------------------------------------------------------
# 服务器状态：一次加载，随请求复用
# ---------------------------------------------------------------------------
@dataclass
class ServerState:
    app_data: Path
    db: AppDB
    ref: ReferenceRegistry
    cleaner: LabDataCleaner
    pipeline: FeaturePipeline
    audit: AuditLogger
    registry: ModelRegistry
    router: TrafficRouter
    referral: ReferralEngine
    trend: TrendEngine
    feedback: FeedbackOrchestrator
    display_names: dict[str, str]
    horizons: list[str] = field(default_factory=list)
    banks: dict[str, HorizonBank] = field(default_factory=dict)          # version -> bank
    services: dict[tuple[str, str], RiskPredictionService] = field(default_factory=dict)
    tier_schemes: dict[str, RiskTierScheme] = field(default_factory=dict)
    monitors: dict[str, DriftMonitor] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def ensure_version_loaded(self, version: str) -> None:
        """按需装载某版本的 HorizonBank 与逐时程服务（灰度/回滚后调用）。"""
        bank = self.banks.get(version)
        if bank is None:
            info = self.registry.get(version)
            bank = HorizonBank.load(info.bank_dir)
            self.banks[version] = bank
        for h, _days in bank.horizons:
            key = (version, h)
            if key not in self.services:
                if h not in self.tier_schemes:
                    raise RuntimeError(f"版本 {version} 包含未知时程 {h}，缺少风险分层切点")
                self.services[key] = RiskPredictionService(
                    model=bank.models[h],
                    tier_scheme=self.tier_schemes[h],
                    audit=self.audit,
                    drift_monitor=self.monitors.get(h),
                    display_names=self.display_names,
                    config=ServiceConfig(model_version=version, horizon=h),
                )
        logger.info("版本 %s 已装载（%d 个时程服务）", version, len(bank.horizons))

    def recent_audit_days(self, n: int = 14) -> list[str]:
        days = []
        for p in sorted(self.audit.root.glob("*.jsonl"), reverse=True):
            try:
                date.fromisoformat(p.stem)
                days.append(p.stem)
            except ValueError:
                continue
            if len(days) >= n:
                break
        return days

    def find_trace_day(self, trace_id: str) -> str | None:
        for d in self.recent_audit_days(30):
            if self.audit.find(trace_id, day=d) is not None:
                return d
        return None


def build_state(app_data: str | Path) -> ServerState:
    app_data = Path(app_data)
    if not is_bootstrapped(app_data):
        raise RuntimeError(
            f"应用数据目录 {app_data} 尚未完成模型自举。"
            "请先运行: python run_app.py --bootstrap （或直接 python run_app.py 自动处理）"
        )
    salt = os.environ.get("DRP_PII_SALT", _DEFAULT_SALT)
    if salt == _DEFAULT_SALT:
        logger.warning("正在使用开发默认 PII 盐！生产环境必须注入环境变量 DRP_PII_SALT。")

    root = Path(__file__).resolve().parents[1]
    ref = ReferenceRegistry.from_yaml(root / "configs" / "reference_intervals.yaml")
    conf_cfg = ConfounderConfig.from_yaml(root / "configs" / "confounders.yaml")
    registry = ModelRegistry(app_data / "registry")
    active = registry.get_active()
    if active is None:
        raise RuntimeError("注册表中没有 ACTIVE 版本——自举产物不完整，请删除数据目录后重新自举")

    audit = AuditLogger(app_data / "audit", salt=salt)
    serving_dir = app_data / "serving"

    # 先加载逐时程的切点与漂移基线（跨版本共享：同一特征空间）
    probe_bank = HorizonBank.load(active.bank_dir)
    horizons = [h for h, _ in probe_bank.horizons]
    tier_schemes = {h: RiskTierScheme.load(serving_dir / f"{h}.tier.json") for h in horizons}
    monitors = {
        h: DriftMonitor(ReferenceProfile.load(serving_dir / f"{h}.drift.json"))
        for h in horizons
    }

    display_names = {code: ref.require(code).name_cn for code in ref.codes}
    state = ServerState(
        app_data=app_data,
        db=AppDB(app_data / "app.db"),
        ref=ref,
        cleaner=LabDataCleaner(ref),
        pipeline=FeaturePipeline(ref, confounder_config=conf_cfg, config=_PIPELINE_CONFIG),
        audit=audit,
        registry=registry,
        router=TrafficRouter(registry, routing_salt=os.environ.get("DRP_ROUTING_SALT", "drp-app-routing-v1")),
        referral=ReferralEngine(ref),
        trend=TrendEngine(ref),
        feedback=FeedbackOrchestrator(audit),
        display_names=display_names,
        horizons=horizons,
        tier_schemes=tier_schemes,
        monitors=monitors,
    )
    state.banks[active.version] = probe_bank
    state.ensure_version_loaded(active.version)  # 幂等：补建 services
    canary = registry.get_canary()
    if canary is not None:
        state.ensure_version_loaded(canary.version)
    return state


# ---------------------------------------------------------------------------
# 应用工厂
# ---------------------------------------------------------------------------
def build_server(app_data: str | Path):
    from fastapi import FastAPI, HTTPException
    from fastapi.staticfiles import StaticFiles

    st = build_state(app_data)
    app = FastAPI(title="病情预测平台", version="1.0",
                  description="内网应用；公网入口必须由网关强制 HTTPS（规范 7）。")
    app.state.drp = st

    # ---------------- 小工具 ----------------
    def _patient_or_404(pid: str) -> dict:
        p = st.db.get_patient(pid)
        if p is None:
            raise HTTPException(404, f"患者 {pid} 不存在")
        return p

    def _demo_frame(p: dict) -> pd.DataFrame:
        """
        患者 → cohort 行。档案列（规范 2.1）必须整套带上：这是特征层拿到
        吸烟史/BMI/家族史的唯一通道，漏一列该特征就静默退化为"未采集"。
        NULL 原样传（pandas 变 NaN），demographics 构造器把 NaN 当第三态。
        """
        row = {
            COL_PATIENT_ID: p["patient_id"], COL_SEX: p["sex"],
            COL_BIRTH_DATE: pd.Timestamp(p["birth_date"]),
        }
        for col in PROFILE_COLUMNS:
            row[col] = p.get(col)
        return pd.DataFrame([row])

    def _age_of(p: dict) -> float:
        """按当前日期计算年龄；参考区间是年龄分层的，必须逐次现算。"""
        born = pd.Timestamp(p["birth_date"])
        today = pd.Timestamp(datetime.now(timezone.utc).date())
        return float((today - born).days) / 365.25

    def _records_frame(pid: str) -> pd.DataFrame:
        rows = st.db.records_for_patient(pid)
        if not rows:
            return pd.DataFrame(
                columns=[COL_PATIENT_ID, COL_INDICATOR, COL_VALUE, COL_UNIT, COL_MEASURED_AT, COL_STATUS]
            )
        df = pd.DataFrame(rows)
        df[COL_PATIENT_ID] = pid
        df[COL_MEASURED_AT] = pd.to_datetime(df[COL_MEASURED_AT])
        return df

    # ---------------- 元信息 ----------------
    @app.get("/api/meta")
    def meta():
        active = st.registry.get_active()
        canary = st.registry.get_canary()
        return {
            "horizons": st.horizons,
            "indicators": [
                {"code": c, "name_cn": st.ref.require(c).name_cn,
                 "unit": st.ref.require(c).canonical_unit}
                for c in st.ref.codes
            ],
            "disclaimer": DISCLAIMER,
            "sample_report": SAMPLE_REPORT,
            "active_version": active.version if active else None,
            "canary": ({"version": canary.version, "traffic_pct": canary.traffic_pct}
                       if canary else None),
            "stats": st.db.stats(),
            # 分层切点随模型固化（service.RiskTierScheme），展示层必须用真实切点画
            # 分层刻度，不允许前端自己编一套 0/25/50/75（规范 6 风险分级分层）。
            "tiers": {
                h: {"cutpoints": list(st.tier_schemes[h].cutpoints),
                    "names": list(st.tier_schemes[h].names),
                    "source": st.tier_schemes[h].source}
                for h in st.horizons if h in st.tier_schemes
            },
            # 档案字段目录（规范 2.1）。直接从特征层的字段表生成，
            # 保证表单字段与模型入参永远同源 —— 前端不允许自己维护一份。
            "profile_fields": {
                "history": HISTORY_FIELDS,
                "family_history": FAMILY_HISTORY_FIELDS,
            },
        }

    # ---------------- 患者 ----------------
    @app.post("/api/patients")
    def create_patient(body: PatientIn):
        hits = scan_pii({"patient_id": body.patient_id})
        if hits:
            raise HTTPException(422, "patient_id 疑似包含明文个人信息（证件/手机号等），"
                                     "请改用非 PII 业务编号（规范 1.2）")
        if st.db.get_patient(body.patient_id):
            raise HTTPException(409, f"患者 {body.patient_id} 已存在")
        try:
            pd.Timestamp(body.birth_date)
        except Exception as e:
            raise HTTPException(422, f"birth_date 无法解析: {e}") from e
        profile = {c: getattr(body, c) for c in PROFILE_COLUMNS}
        return st.db.create_patient(body.patient_id, body.sex.upper(),
                                    body.birth_date, profile=profile)

    @app.put("/api/patients/{pid}/profile")
    def update_profile(pid: str, body: ProfileIn):
        """
        整档覆盖写（规范 2.1）。留空的字段回到"未采集"——三态里
        "撤回一个回答"是真实需求，patch 语义做不到。
        """
        _patient_or_404(pid)
        return st.db.update_profile(pid, body.model_dump())

    # ---------------- 用药（规范 2.4 干扰因子的数据入口） ----------------
    @app.get("/api/patients/{pid}/medications")
    def list_medications(pid: str):
        _patient_or_404(pid)
        return st.db.medications_for_patient(pid)

    @app.post("/api/patients/{pid}/medications")
    def add_medication(pid: str, body: MedicationIn):
        _patient_or_404(pid)
        # 药名是自由文本，是本表唯一可能夹带 PII 的地方 —— 与报告文本同规拒收
        if scan_pii({"medication_name": body.medication_name}):
            raise HTTPException(422, "药名字段疑似包含明文个人信息，已拒收（规范 1.2）")
        for label, v in (("start_date", body.start_date), ("end_date", body.end_date)):
            if v is not None:
                try:
                    pd.Timestamp(v)
                except Exception as e:
                    raise HTTPException(422, f"{label} 无法解析: {e}") from e
        if body.start_date and body.end_date and body.end_date < body.start_date:
            raise HTTPException(422, "end_date 早于 start_date")
        return st.db.add_medication(pid, body.medication_name.strip(),
                                    body.start_date, body.end_date)

    @app.delete("/api/patients/{pid}/medications/{med_id}")
    def delete_medication(pid: str, med_id: int):
        _patient_or_404(pid)
        med = st.db.get_medication(med_id)
        if med is None or med["patient_id"] != pid:
            raise HTTPException(404, f"用药记录 {med_id} 不存在")
        st.db.delete_medication(med_id)
        return {"deleted": med_id}

    @app.get("/api/patients")
    def list_patients():
        return st.db.list_patients()

    @app.get("/api/patients/{pid}/records")
    def patient_records(pid: str):
        """
        历史化验记录。附带【该患者年龄性别下】的参考区间与分级，
        因为同一个 ALT=62 在 30 岁男性和 70 岁女性身上不是同一件事，
        前端没有年龄性别分层的参考表，也不该自己算（规范 2.1 分层归一化）。
        分级一律走 features.grade_of，与模型特征同口径。
        """
        patient = _patient_or_404(pid)
        rows = st.db.records_for_patient(pid)
        if not rows:
            return rows
        age = _age_of(patient)
        sex = str(patient["sex"]).upper()
        for r in rows:
            meta = st.ref.get(r["indicator_code"])
            r["name_cn"] = meta.name_cn if meta else r["indicator_code"]
            r["ref_low"] = r["ref_high"] = r["grade"] = None
            if meta is None:
                continue
            iv = meta.match_interval(sex, age)
            if iv is None:
                continue
            r["ref_low"] = iv.lower
            r["ref_high"] = iv.upper
            r["grade"] = int(grade_of(meta, float(r["value"]), iv))
        return rows

    # ---------------- 报告解析入库（规范 4.1 全链路真实执行） ----------------
    @app.post("/api/reports/parse")
    def parse_report(body: ReportIn):
        patient = _patient_or_404(body.patient_id)
        pii = scan_pii({"raw_text": body.text})
        if pii:
            raise HTTPException(
                422, "报告文本包含明文个人信息，已拒收（规范 1.2 全程脱敏）。"
                     "请先在上游去标识化：删除姓名/证件/手机号后重试。"
            )
        try:
            measured_at = pd.Timestamp(body.measured_at)
        except Exception as e:
            raise HTTPException(422, f"measured_at 无法解析: {e}") from e

        frame, preport, rows = parse_lab_text(
            body.text, st.ref, patient_id=body.patient_id, measured_at=measured_at
        )
        stored = 0
        cleaning_summary = None
        if not frame.empty:
            cleaned, creport = st.cleaner.clean(frame, demographics=_demo_frame(patient))
            cleaning_summary = creport.summary()
            report_id = st.db.insert_report(
                body.patient_id, body.text, str(measured_at),
                preport.n_ingested, preport.n_review, preport.n_unmatched,
            )
            stored = st.db.insert_lab_records([
                {
                    "patient_id": body.patient_id,
                    "indicator_code": r[COL_INDICATOR],
                    "value": float(r[COL_VALUE]),
                    "unit": str(r[COL_UNIT]),
                    "measured_at": str(r[COL_MEASURED_AT]),
                    "status": int(r[COL_STATUS]),
                    "report_id": report_id,
                }
                for _, r in cleaned.iterrows()
            ])
        else:
            report_id = st.db.insert_report(
                body.patient_id, body.text, str(measured_at),
                0, preport.n_review, preport.n_unmatched,
            )

        return {
            "report_id": report_id,
            "stored": stored,
            "parse": preport.to_log_dict(),
            "cleaning_summary": cleaning_summary,
            "rows": [r.to_log_dict() for r in rows],
        }

    # ---------------- 预测（真实特征管线 -> 灰度路由 -> 多时程服务） ----------------
    @app.post("/api/predict")
    def predict(body: PredictIn):
        patient = _patient_or_404(body.patient_id)
        recs = _records_frame(body.patient_id)
        if recs.empty:
            raise HTTPException(422, "该患者暂无化验记录，请先在工作台录入报告")

        try:
            decision = st.router.decide(body.patient_id)
        except RegistryError as e:
            raise HTTPException(503, str(e)) from e
        st.ensure_version_loaded(decision.version)
        bank = st.banks[decision.version]

        cohort_row = {
            COL_PATIENT_ID: patient["patient_id"],
            COL_INDEX_DATE: pd.Timestamp(datetime.now(timezone.utc).date()),
            COL_SEX: patient["sex"],
            COL_BIRTH_DATE: pd.Timestamp(patient["birth_date"]),
        }
        # 档案列（规范 2.1）：与 _demo_frame 同源，漏一列该特征静默变"未采集"
        for col in PROFILE_COLUMNS:
            cohort_row[col] = patient.get(col)
        # 采血登记（规范 2.4 生理状态）：只在【登记过】时写入列。
        # None 不写列 → 特征层不产出该项；False 写 0.0 → 明确的"否"。
        # 两者必须区分，这正是三态纪律在请求级数据上的延伸。
        for key, val in (("non_fasting", body.non_fasting),
                         ("strenuous_exercise", body.strenuous_exercise),
                         ("pregnancy", body.pregnancy)):
            if val is not None:
                cohort_row[key] = 1.0 if val else 0.0
        cohort = pd.DataFrame([cohort_row])

        # 用药表（规范 2.4）：confounders 按 start/end 与索引日期做严格时间窗
        # 判定，end_date 为空 = 仍在服用。空表传 None，等价且省一次拷贝。
        meds = st.db.medications_for_patient(body.patient_id)
        medications = (
            pd.DataFrame([{
                COL_PATIENT_ID: body.patient_id,
                "medication_name": m["medication_name"],
                "start_date": m["start_date"],
                "end_date": m["end_date"],
            } for m in meds])
            if meds else None
        )

        first_h = st.horizons[0]
        manifest = bank.models[first_h].manifest
        X = st.pipeline.transform(cohort, recs, medications=medications, manifest=manifest)

        structured_note = {"n_records": int(len(recs)),
                           "n_indicators": int(recs[COL_INDICATOR].nunique()),
                           "n_active_medications": int(len(meds))}
        results = []
        for h in st.horizons:
            svc = st.services[(decision.version, h)]
            r = svc.predict(X, patient_ids=[body.patient_id],
                            structured=[structured_note])[0]
            st.db.index_prediction(
                r.trace_id, body.patient_id, h, r.probability, r.risk_tier,
                decision.version, decision.arm,
            )
            top = []
            if r.attribution is not None:
                for f in r.attribution.factors[:8]:
                    top.append({"display": f.display, "direction": f.direction,
                                "magnitude": f.magnitude, "is_missing": f.is_missing})
            results.append({
                "horizon": h, "trace_id": r.trace_id,
                "probability": r.probability, "risk_tier": r.risk_tier,
                "narrative": r.narrative, "degraded": r.degraded,
                "top_factors": top,
            })

        probs = [x["probability"] for x in results]
        monotonic_note = None
        if any(probs[i] > probs[i + 1] + 1e-9 for i in range(len(probs) - 1)):
            monotonic_note = ("提示：各时程概率出现轻微倒挂，属独立模型的统计波动；"
                              "展示层已按时程排序原样呈现，未做修饰。")

        tier_for_referral = next(
            (x["risk_tier"] for x in results if x["horizon"] == "3y"),
            results[-1]["risk_tier"],
        )
        advice = st.referral.advise(
            recs, demographics=_demo_frame(patient), risk_tier=tier_for_referral
        )

        return {
            "patient_id": body.patient_id,
            "model_version": decision.version,
            "arm": decision.arm,
            "results": results,
            "monotonic_note": monotonic_note,
            "referral": advice.to_dict(),
        }

    # ---------------- 趋势报告（规范 6） ----------------
    @app.get("/api/patients/{pid}/trend")
    def patient_trend(pid: str):
        patient = _patient_or_404(pid)
        recs = _records_frame(pid)
        pseudo = st.audit.pseudonymize(pid)
        report = build_trend_report(
            st.trend, recs, demographics=_demo_frame(patient),
            audit=st.audit, pseudo_id=pseudo, horizons=tuple(st.horizons),
        )
        return report.to_dict()

    # ---------------- 随访回流（规范 1.3） ----------------
    @app.post("/api/feedback")
    def submit_feedback(body: FeedbackIn):
        try:
            fb = FollowUpFeedback(
                trace_id=body.trace_id, event_occurred=body.event_occurred,
                days_since_prediction=body.days_since_prediction,
                consented=body.consented,
            )
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        day = st.find_trace_day(body.trace_id)
        if day is None:
            raise HTTPException(404, f"trace_id={body.trace_id} 未找到对应预测记录")
        ok = st.feedback.ingest_followup(fb, day=day)
        if not ok:
            raise HTTPException(404, f"trace_id={body.trace_id} 回填失败")
        st.db.insert_feedback(body.trace_id, body.event_occurred, body.days_since_prediction)
        return {"accepted": True, "trace_id": body.trace_id}

    # ---------------- 管理台 ----------------
    @app.get("/api/admin/versions")
    def admin_versions():
        return {"versions": st.registry.to_dict(), "loaded": sorted(st.banks)}

    @app.post("/api/admin/canary")
    def admin_canary(body: CanaryIn):
        try:
            info = st.registry.set_canary(body.version, traffic_pct=body.traffic_pct)
            st.ensure_version_loaded(body.version)
        except (RegistryError, ValueError, KeyError) as e:
            raise HTTPException(422, str(e)) from e
        return info.to_dict()

    @app.post("/api/admin/promote")
    def admin_promote(body: VersionIn):
        try:
            info = st.registry.promote(body.version)
            st.ensure_version_loaded(body.version)
        except (RegistryError, KeyError) as e:
            raise HTTPException(422, str(e)) from e
        return info.to_dict()

    @app.post("/api/admin/rollback")
    def admin_rollback():
        try:
            info = st.registry.rollback()
            st.ensure_version_loaded(info.version)
        except RegistryError as e:
            raise HTTPException(422, str(e)) from e
        return info.to_dict()

    @app.get("/api/admin/review-queue")
    def admin_review_queue():
        q = st.feedback.build_review_queue(days=st.recent_audit_days())
        return q.to_dict() | {"summary": q.summary()}

    @app.get("/api/admin/drift")
    def admin_drift(horizon: str):
        if horizon not in st.monitors:
            raise HTTPException(422, f"未知时程 {horizon}，可选: {st.horizons}")
        active = st.registry.get_active()
        feats: list[dict] = []
        for d in st.recent_audit_days():
            for rec in st.audit.iter_records(d):
                if rec.horizon == horizon and rec.features:
                    feats.append(rec.features)
                if len(feats) >= 500:
                    break
            if len(feats) >= 500:
                break
        if not feats:
            return {"level": "INSUFFICIENT", "n_online": 0,
                    "messages": ["暂无该时程的线上预测记录，无法评估漂移"]}
        Xo = pd.DataFrame(feats)
        rep = st.monitors[horizon].check(Xo, model_version=active.version if active else "")
        return rep.to_dict()

    @app.get("/api/admin/ab")
    def admin_ab(champion: str, challenger: str, horizon: str | None = None):
        comp = build_ab_comparison(
            st.audit, champion, challenger,
            days=st.recent_audit_days(), horizon=horizon,
        )
        return comp.to_dict() | {"summary": comp.summary()}

    # ---------------- OCR 图片识别化验单（本地 RapidOCR + Vision 兜底） ----------------
    @app.post("/api/ocr")
    async def ocr_lab_image(body: dict):
        """
        接收 base64 编码的化验单图片，使用本地 RapidOCR / AI 视觉模型识别文字，
        返回格式化文本（直接填入报告解析框）。

        请求体: {"image": "<base64 string>"}
        响应:   {"text": "ALT 62 U/L 9-50\n...", "count": 5}
        """
        import asyncio
        image_b64 = body.get("image", "")
        if not image_b64:
            raise HTTPException(400, "缺少 image 字段（base64 编码图片）")

        loop = asyncio.get_running_loop()
        extracted_text = await loop.run_in_executor(
            None, _ocr_extract_image, image_b64
        )

        lines = [l.strip() for l in extracted_text.splitlines() if l.strip()]
        return {
            "text": extracted_text,
            "count": len(lines),
        }

    # ---------------- 前端静态托管（必须最后挂载，避免吞掉 /api） ----------------
    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return app


_rapid_ocr_engine = None


def _get_rapid_ocr():
    global _rapid_ocr_engine
    if _rapid_ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _rapid_ocr_engine = RapidOCR()
        except Exception as e:
            logger.warning("[OCR] 无法初始化 RapidOCR: %s", e)
    return _rapid_ocr_engine


def _ocr_extract_image(image_b64: str) -> str:
    """
    通用图片 OCR 识别：
    1. 优先使用本地 RapidOCR（极速、离线、零网络依赖、高精度中文/英文识别）
    2. 若本地引擎未返回结果，尝试 Claude Vision API 兜底
    """
    import base64
    import cv2

    clean_b64 = image_b64
    if "," in image_b64:
        clean_b64 = image_b64.split(",", 1)[1]

    # --- 1. 本地 RapidOCR 识别 ---
    try:
        engine = _get_rapid_ocr()
        if engine is not None:
            raw = base64.b64decode(clean_b64)
            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                results, _ = engine(img)
                if results:
                    items = []
                    for box, text, score in results:
                        txt = text.strip()
                        if not txt:
                            continue
                        pts = np.array(box)
                        x_min = float(np.min(pts[:, 0]))
                        y_min = float(np.min(pts[:, 1]))
                        x_max = float(np.max(pts[:, 0]))
                        y_max = float(np.max(pts[:, 1]))
                        y_center = (y_min + y_max) / 2.0
                        height = max(y_max - y_min, 1.0)
                        items.append({
                            "text": txt,
                            "x_min": x_min,
                            "y_min": y_min,
                            "x_max": x_max,
                            "y_max": y_max,
                            "y_center": y_center,
                            "h": height,
                        })

                    if items:
                        # 按照 y_center 排序，基于行中心点锚定聚类，彻底杜绝行雪球粘连
                        items.sort(key=lambda it: it["y_center"])
                        lines = []
                        for it in items:
                            matched = None
                            for l in lines:
                                if abs(it["y_center"] - l["anchor_y"]) <= max(min(it["h"], l["h"]) * 0.45, 5.0):
                                    matched = l
                                    break
                            if matched:
                                matched["items"].append(it)
                                matched["anchor_y"] = sum(x["y_center"] for x in matched["items"]) / len(matched["items"])
                                matched["h"] = sum(x["h"] for x in matched["items"]) / len(matched["items"])
                            else:
                                lines.append({"anchor_y": it["y_center"], "h": it["h"], "items": [it]})

                        # 按行纵向从上到下排序，行内横向从左到右排序
                        lines.sort(key=lambda l: l["anchor_y"])
                        text_lines = []
                        for l in lines:
                            l["items"].sort(key=lambda x: x["x_min"])
                            line_text = "  ".join(x["text"] for x in l["items"]).strip()
                            if line_text:
                                text_lines.append(line_text)

                        final_text = "\n".join(text_lines)
                        logger.info("[OCR] RapidOCR 成功精确聚类识别 %d 行文本", len(text_lines))
                        return final_text
    except Exception as e:
        logger.warning("[OCR] RapidOCR 执行异常 (%s)，尝试 Vision 兜底...", e)

    # --- 2. 尝试 Vision API 兜底 ---
    vision_items = _ai_vision_ocr_extract(image_b64)
    if vision_items:
        lines = []
        for ind in vision_items:
            name = ind.get("name_raw", "")
            val = ind.get("value", "")
            unit = ind.get("unit", "")
            ref = ind.get("reference", "")
            line = f"{name} {val} {unit}"
            if ref:
                line += f" {ref}"
            lines.append(line)
        return "\n".join(lines)

    return ""


def _ai_vision_ocr_extract(image_b64: str) -> list:
    """
    调用 AI 视觉模型识别化验单图片中的检验指标。
    """
    import urllib.request
    import json
    import re

    api_key = os.environ.get(
        "ANTHROPIC_API_KEY",
        "sk-HcQuMphdXJMXangi05KHQ6cZLERVPzTLWAOTPYzMYshjisZu",
    )
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://daodun.cc")

    media_type = "image/jpeg"
    clean_b64 = image_b64
    if "," in image_b64:
        header, clean_b64 = image_b64.split(",", 1)
        if "png" in header:
            media_type = "image/png"
        elif "webp" in header:
            media_type = "image/webp"

    prompt_text = (
        "你是一位专业医学检验单 OCR 识别员。识别图片中的所有临床检验指标名称、数值和单位。\n"
        "严格只返回 JSON 格式在 Markdown codeblock 中：\n"
        '[{"name_raw": "丙氨酸氨基转移酶(ALT)", "value": 68, "unit": "U/L", "reference": "9-50"}]'
    )

    try:
        url = f"{base_url.rstrip('/')}/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        req_body = {
            "model": os.environ.get("VISION_MODEL", "claude-sonnet-4-20250514"),
            "max_tokens": 2048,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": clean_b64,
                            },
                        },
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ],
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(req_body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            txt = res_data["content"][0]["text"].strip()
            txt = re.sub(r"^```[a-z]*\s*", "", txt, flags=re.MULTILINE)
            txt = re.sub(r"\s*```$", "", txt, flags=re.MULTILINE)
            parsed = json.loads(txt)
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed
    except Exception as err:
        logger.warning("[OCR] Vision API 异常 (%s)", err)

    return []
