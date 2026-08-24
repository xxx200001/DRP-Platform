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


class ReportDateIn(BaseModel):
    measured_at: str


class PredictIn(BaseModel):
    patient_id: str
    # 本次最关注的问题（改版·改动 2）。只影响结果页的【回答顺序】——
    # 后台永远全身扫描，concern 不裁剪任何分析。
    # 可选值: liver / cardio / glucose / renal / other / all（None 视同 all）
    concern: str | None = None
    # 采血登记（规范 2.4 生理状态）。描述的是【本次预测时点】的状态，
    # 属于每次就诊的登记项而非患者档案 —— 妊娠/空腹会变，所以随请求传，
    # 不落 patients 表。None = 未登记（特征层不产出该项，与"否"严格区分）。
    non_fasting: bool | None = None
    strenuous_exercise: bool | None = None
    pregnancy: bool | None = None


# ---------------------------------------------------------------------------
# 展示层分组与关注点映射（改版·改动 2/3）
# 仅用于时间轴统计与"风险总览"的口语化归类；临床科室推荐仍由
# ReferralEngine 的规则表负责，两者读者不同、允许粒度不同。
# ---------------------------------------------------------------------------
_TIMELINE_GROUPS: dict[str, tuple[str, ...]] = {
    "血脂": ("TG", "TC", "LDLC", "HDLC"),
    "肝功能": ("ALT", "AST", "GGT", "TBIL", "ALB"),
    "肾功能": ("CREA", "UREA", "UA", "UACR"),
    "血糖": ("GLU", "HBA1C", "INS"),
    "血压": ("SBP", "DBP"),
    "血常规": ("PLT", "HGB", "WBC"),
    "炎症指标": ("CRP",),
    "体格": ("BMI",),
}

#: concern 值 -> ReferralEngine 的指标组名（items[].group）
_CONCERN_TO_GROUPS: dict[str, tuple[str, ...]] = {
    "liver": ("肝功能",),
    "cardio": ("血脂", "血压"),
    "glucose": ("血糖代谢",),
    "renal": ("肾功能", "电解质"),
}
_CONCERN_LABEL: dict[str, str] = {
    "liver": "肝脏", "cardio": "心血管", "glucose": "血糖代谢",
    "renal": "肾脏", "other": "其他", "all": "全面分析",
}

#: V3.1 改动：用户反馈"没有说以后可能会成为什么病"。
#: 按 ReferralEngine 的组名给出【长期未干预时的典型发展方向】。措辞纪律：
#:   · 这是流行病学意义上的风险方向提示，不是对个体的诊断/断言 ——
#:     前端固定追加"（风险提示，非诊断）"，后端文案里不写"会得/将患"。
#:   · 由服务端统一下发，前端不许自己编病名（与 model_card 同一条纪律）。
_GROUP_DIRECTION: dict[str, str] = {
    "肝功能": "脂肪肝加重、肝纤维化，长期可进展为慢性肝病",
    "血脂": "动脉粥样硬化，升高冠心病、脑卒中等心脑血管疾病风险",
    "血糖代谢": "胰岛素抵抗加重，可能发展为 2 型糖尿病及其并发症",
    "肾功能": "慢性肾脏病（CKD）进展，肾功能逐步下降",
    "血压": "高血压病，累及心、脑、肾等靶器官",
    "电解质": "水电解质紊乱相关的心律失常与肾脏问题",
    "血常规": "贫血或血液系统异常持续（需结合临床进一步明确）",
    "炎症指标": "慢性炎症负担，与心血管代谢疾病风险相关",
    "体重管理": "肥胖相关代谢综合征（血糖、血脂、血压联动恶化）",
}


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
    # 模型卡（改版·核查项 4）：预测终点与"是否演示模型"必须能被展示层读到。
    model_card: dict = field(default_factory=dict)
    # 回溯风险时间轴缓存：key=(pid, n_records, last_measured_at)。
    # 数据一变 key 就变，天然失效；单机 demo 规模无需 LRU。
    risk_timeline_cache: dict = field(default_factory=dict)

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

    # ------------------------------------------------------------------
    # 模型卡（核查项 4：1Y/3Y/5Y 到底预测什么，必须能说清楚）。
    # 概率本身是真实 LightGBM+等渗校准输出（非硬编码），但当前 ACTIVE 若是
    # 开发自举版本，训练数据为【合成纵向病历】、结局为合成综合慢病事件 ——
    # 界面必须明确标注"演示模型/非临床验证"，不允许包装成疾病发生概率。
    development_only = False
    note = active.notes or ""
    meta_path = app_data / "bootstrap_meta.json"
    if meta_path.exists():
        try:
            import json
            bmeta = json.loads(meta_path.read_text(encoding="utf-8"))
            if bmeta.get("version") == active.version:
                development_only = bool(bmeta.get("development_only", False))
                note = bmeta.get("note", note)
        except Exception:  # 模型卡是展示信息，读取失败不阻断服务
            logger.warning("bootstrap_meta.json 读取失败，模型卡按注册表 notes 兜底")
    if not development_only and "合成" in note:
        development_only = True  # 注册表 notes 里钉着的那句话就是判据
    state.model_card = {
        "endpoint_label": "综合心血管代谢及慢病相关风险进展",
        "endpoint_detail": (
            "各时程概率 = 模型估计的「未来 N 年内发生综合慢病事件」的校准概率。"
            "由该患者全部历史检查数据经真实特征管线（含时序趋势特征）计算得出。"
        ),
        "development_only": development_only,
        "development_note": (
            "当前为演示模型：训练数据为合成纵向病历，概率为统计演示值，"
            "未经过真实临床队列验证，不代表真实疾病发生概率。"
            if development_only else ""
        ),
        "model_version": active.version,
        "attribution_method": "TreeSHAP（按指标聚合，仅输出排序与方向）",
    }
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
            # 模型卡（核查项 4）：预测终点 + 演示模型标识，展示层必须原样呈现
            "model_card": st.model_card,
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

        st.risk_timeline_cache.clear()
        return {
            "report_id": report_id,
            "stored": stored,
            "parse": preport.to_log_dict(),
            "cleaning_summary": cleaning_summary,
            "rows": [r.to_log_dict() for r in rows],
        }

    # ---------------- 报告管理（改版·改动 1：8 份报告逐份可见、可管理） ----------------
    def _report_or_404(report_id: int) -> dict:
        rep = st.db.get_report(report_id)
        if rep is None:
            raise HTTPException(404, f"报告 {report_id} 不存在")
        return rep

    @app.get("/api/patients/{pid}/reports")
    def list_reports(pid: str):
        """
        该患者的历史检查资料清单 + 累计状态。
        summary 用于渲染「已上传 N 份报告｜YYYY.MM—YYYY.MM｜识别 X 条指标」。
        列表按【真实检查日期】排序（核查项 3），不按上传时间。
        """
        _patient_or_404(pid)
        reports = st.db.list_reports(pid)
        dates = [str(r["measured_at"])[:10] for r in reports if r.get("measured_at")]
        return {
            "reports": reports,
            "summary": {
                "n_reports": len(reports),
                "n_stored_total": int(sum(r["n_stored"] for r in reports)),
                "first_date": min(dates) if dates else None,
                "last_date": max(dates) if dates else None,
            },
        }

    @app.get("/api/reports/{report_id}")
    def get_report(report_id: int):
        """查看报告原文（OCR/粘贴文本）。原始图片有意不落库：图片含姓名等明文
        PII，与规范 1.2 冲突；可回看的是已过 scan_pii 的识别文本。"""
        rep = _report_or_404(report_id)
        return {
            "id": rep["id"], "patient_id": rep["patient_id"],
            "measured_at": rep["measured_at"], "created_at": rep["created_at"],
            "raw_text": rep["raw_text"],
            "n_ingested": rep["n_ingested"], "n_review": rep["n_review"],
            "n_unmatched": rep["n_unmatched"],
        }

    @app.patch("/api/reports/{report_id}")
    def update_report_date(report_id: int, body: ReportDateIn):
        """修改检查日期。DB 层保证报告与其名下全部指标记录的 measured_at 联动。"""
        _report_or_404(report_id)
        try:
            ts = pd.Timestamp(body.measured_at)
        except Exception as e:
            raise HTTPException(422, f"measured_at 无法解析: {e}") from e
        if ts > pd.Timestamp(datetime.now(timezone.utc).date()) + pd.Timedelta(days=1):
            raise HTTPException(422, "检查日期不能晚于今天")
        st.db.update_report_date(report_id, str(ts))
        st.risk_timeline_cache.clear()
        return {"id": report_id, "measured_at": str(ts)}

    @app.delete("/api/reports/{report_id}")
    def delete_report(report_id: int):
        _report_or_404(report_id)
        n = st.db.delete_report(report_id)
        st.risk_timeline_cache.clear()
        return {"deleted": report_id, "records_removed": n}

    @app.post("/api/reports/{report_id}/reparse")
    def reparse_report(report_id: int):
        """重新识别：对已入库的原文重新走解析+清洗管线（词典/解析器升级后使用），
        替换该报告名下的指标记录；检查日期保持不变。"""
        rep = _report_or_404(report_id)
        patient = _patient_or_404(rep["patient_id"])
        measured_at = pd.Timestamp(rep["measured_at"])
        frame, preport, rows = parse_lab_text(
            rep["raw_text"], st.ref, patient_id=rep["patient_id"], measured_at=measured_at
        )
        st.db.clear_report_records(report_id)
        stored = 0
        if not frame.empty:
            cleaned, _ = st.cleaner.clean(frame, demographics=_demo_frame(patient))
            stored = st.db.insert_lab_records([
                {
                    "patient_id": rep["patient_id"],
                    "indicator_code": r[COL_INDICATOR],
                    "value": float(r[COL_VALUE]),
                    "unit": str(r[COL_UNIT]),
                    "measured_at": str(r[COL_MEASURED_AT]),
                    "status": int(r[COL_STATUS]),
                    "report_id": report_id,
                }
                for _, r in cleaned.iterrows()
            ])
        st.db.update_report_counts(
            report_id, preport.n_ingested, preport.n_review, preport.n_unmatched
        )
        st.risk_timeline_cache.clear()
        return {
            "report_id": report_id, "stored": stored,
            "parse": preport.to_log_dict(),
            "rows": [r.to_log_dict() for r in rows],
        }

    # ---------------- 健康时间轴（改版·改动 1：上传完先"数据确认"再预测） ----------------
    @app.get("/api/patients/{pid}/timeline")
    def patient_timeline(pid: str):
        """
        「已建立个人健康时间轴」的数据确认视图：
        数据跨度 / 报告数 / 有效记录数 / 可连续比较指标数 / 各系统时间点覆盖。
        全部按【真实检查日期】统计（核查项 3）。
        """
        _patient_or_404(pid)
        reports = st.db.list_reports(pid)
        rows = st.db.records_for_patient(pid)
        dates = sorted({str(r["measured_at"])[:10] for r in rows})
        by_code_dates: dict[str, set] = {}
        for r in rows:
            by_code_dates.setdefault(r["indicator_code"], set()).add(
                str(r["measured_at"])[:10]
            )
        comparable = sorted(
            c for c, ds in by_code_dates.items() if len(ds) >= 2
        )
        groups = []
        for gname, codes in _TIMELINE_GROUPS.items():
            gdates: set = set()
            for c in codes:
                gdates |= by_code_dates.get(c, set())
            if gdates:
                groups.append({
                    "group": gname,
                    "n_timepoints": len(gdates),
                    "codes_present": [c for c in codes if c in by_code_dates],
                })
        groups.sort(key=lambda g: -g["n_timepoints"])

        span_days = 0
        span_label = "—"
        if len(dates) >= 2:
            span_days = int(
                (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days
            )
            yy, rem = divmod(span_days, 365)
            mm = round(rem / 30.44)
            if mm == 12:
                yy, mm = yy + 1, 0
            span_label = (f"{yy}年" if yy else "") + (f"{mm}个月" if mm else "")
            span_label = span_label or "不足1个月"

        return {
            "n_reports": len(reports),
            "n_records": len(rows),
            "n_dates": len(dates),
            "first_date": dates[0] if dates else None,
            "last_date": dates[-1] if dates else None,
            "span_days": span_days,
            "span_label": span_label,
            "n_comparable_indicators": len(comparable),
            "comparable_indicators": comparable,
            "groups": groups,
            # 纵向趋势分析的门槛：至少两个真实检查时间点
            "longitudinal_ready": len(dates) >= 2,
        }

    # ---------------- 回溯风险时间轴（改版·改动 5/6：X 轴 = 真实检查日期） ----------------
    @app.get("/api/patients/{pid}/risk-timeline")
    def patient_risk_timeline(pid: str):
        """
        按【真实检查日期】回溯计算的历史风险轨迹：对每个检查时间点 t，仅用
        measured_at <= t 的记录、以 t 为索引日重跑真实特征管线并推理。

        与审计走势（trend.risk_trajectory_from_audit，按预测发生时间）的分工：
          · 审计走势回答"模型当时怎么判断"，用于复盘，保留在管理/随访链路；
          · 本视图回答"随着一次次体检，风险如何演变"，X 轴是检查日期 ——
            正是核查项 3 与改动 5 要求的用户侧曲线。
        本视图是派生分析，不写审计、不产生 trace，因此直接走 bank.predict_risk
        而不是 RiskPredictionService（后者的审计落盘是不可分割路径）。
        """
        patient = _patient_or_404(pid)
        recs = _records_frame(pid)
        if recs.empty:
            return {"points": [], "basis": "retrospective_current_model",
                    "model_version": None, "note": "暂无化验记录"}

        cache_key = (pid, int(len(recs)), str(recs[COL_MEASURED_AT].max()))
        cached = st.risk_timeline_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            decision = st.router.decide(pid)
            version = decision.version
        except RegistryError:
            active = st.registry.get_active()
            if active is None:
                raise HTTPException(503, "注册表中没有 ACTIVE 版本")
            version = active.version
        st.ensure_version_loaded(version)
        bank = st.banks[version]
        manifest = bank.models[st.horizons[0]].manifest

        meds = st.db.medications_for_patient(pid)
        medications = (
            pd.DataFrame([{
                COL_PATIENT_ID: pid,
                "medication_name": m["medication_name"],
                "start_date": m["start_date"],
                "end_date": m["end_date"],
            } for m in meds])
            if meds else None
        )

        exam_dates = sorted(recs[COL_MEASURED_AT].dt.normalize().unique())
        # 单机 demo 保护：时间点极多时等距抽稀，但首末两点必须保留
        max_points = 16
        if len(exam_dates) > max_points:
            idx = np.linspace(0, len(exam_dates) - 1, max_points).round().astype(int)
            exam_dates = [exam_dates[i] for i in sorted(set(idx.tolist()))]

        points = []
        for d in exam_dates:
            d = pd.Timestamp(d)
            sub = recs[recs[COL_MEASURED_AT] <= d + pd.Timedelta(hours=23, minutes=59)]
            if sub.empty:
                continue
            cohort_row = {
                COL_PATIENT_ID: patient["patient_id"],
                COL_INDEX_DATE: d,
                COL_SEX: patient["sex"],
                COL_BIRTH_DATE: pd.Timestamp(patient["birth_date"]),
            }
            for col in PROFILE_COLUMNS:
                cohort_row[col] = patient.get(col)
            cohort = pd.DataFrame([cohort_row])
            try:
                X = st.pipeline.transform(
                    cohort, sub, medications=medications, manifest=manifest
                )
            except Exception as e:  # 单点失败不拖垮整条时间轴
                logger.warning("[risk-timeline] %s@%s 特征失败: %s", pid, d.date(), e)
                continue
            horizons_out = {}
            for h in st.horizons:
                p = float(bank.models[h].predict_risk(X)[0])
                horizons_out[h] = {
                    "probability": p,
                    "risk_tier": st.tier_schemes[h].assign(p),
                }
            points.append({
                "at": str(d.date()),
                "n_records_used": int(len(sub)),
                "horizons": horizons_out,
            })

        payload = {
            "basis": "retrospective_current_model",
            "model_version": version,
            "points": points,
            "note": (
                "历史轨迹为按各检查日期、仅用该日期之前数据、以当前模型回溯计算；"
                "属派生分析视图，不写入审计与随访链路。"
                "「模型当时的判断」请看管理台的审计走势。"
            ),
        }
        st.risk_timeline_cache[cache_key] = payload
        return payload

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

        # -------- 风险总览（改动 3）：先给优先级排序，再平铺科室 --------
        # 排序沿用 ReferralEngine 的临床优先级（priority + 最大分级），
        # 每一项都补上"为什么排这里"：涉及指标、当前分级、较上次的真实变化方向。
        comparisons = {
            c.code: c for c in st.trend.compare_latest(
                recs, demographics=_demo_frame(patient)
            )
        }
        rank_labels = ["首要关注", "第二关注", "第三关注"]
        overview_items = []
        for i, it in enumerate(advice.items):
            inds = []
            for f in it.findings:
                comp = comparisons.get(f.code)
                trend_txt = None
                if comp is not None:
                    if comp.is_real_change:
                        trend_txt = comp.direction + ("·较上次加重" if comp.worsened else "")
                    else:
                        trend_txt = "较上次平稳"
                inds.append({
                    "code": f.code, "name_cn": f.name_cn,
                    "value": f.value, "unit": f.unit, "grade": f.grade,
                    "trend": trend_txt,
                    "worsened": bool(comp.worsened) if comp else False,
                })
            why_bits = [f"{len(inds)} 项相关指标异常"]
            n_worse = sum(1 for x in inds if x["worsened"])
            if n_worse:
                why_bits.append(f"其中 {n_worse} 项较上次真实恶化")
            overview_items.append({
                "rank": i + 1,
                "rank_label": rank_labels[i] if i < 3 else "需要留意",
                "group": it.group,
                "department": it.department,
                "priority": int(it.priority),
                "priority_label": it.priority.label_cn(),
                "indicators": inds,
                "why": "、".join(why_bits),
                "checkups": list(it.checkups),
                # V3.1：长期未干预时的可能发展方向（风险提示，非诊断）
                "direction": _GROUP_DIRECTION.get(
                    it.group, "相关系统慢性疾病风险随时间累积"
                ),
            })
        # "目前相对稳定"：有数据、且未触发任何异常建议的系统。
        # 组名口径对齐：Referral 的"血糖代谢"/"体重管理"映射到展示组名。
        abnormal_groups = {it.group for it in advice.items}
        present_codes = set(recs[COL_INDICATOR].unique())
        _alias = {"血糖代谢": "血糖", "体重管理": "体格"}
        abnormal_display = {_alias.get(g, g) for g in abnormal_groups}
        stable_groups = [
            g for g, codes in _TIMELINE_GROUPS.items()
            if (set(codes) & present_codes) and g not in abnormal_display
        ]

        # -------- 本次最关注的问题（改动 2）：先回答用户关注，再报额外发现 --------
        concern = (body.concern or "all").lower()
        concern_groups = _CONCERN_TO_GROUPS.get(concern, ())
        concern_items = [x for x in overview_items if x["group"] in concern_groups]
        if concern in ("all", "other") or not concern_groups:
            concern_answer = None
        elif concern_items:
            concern_answer = {
                "status": "abnormal",
                "items": [x["rank"] for x in concern_items],
                "text": (
                    f"你关注的「{_CONCERN_LABEL.get(concern, concern)}」方向"
                    f"存在需要关注的异常，详见下方第 "
                    f"{'、'.join(str(x['rank']) for x in concern_items)} 项。"
                ),
            }
        else:
            concern_answer = {
                "status": "normal",
                "items": [],
                "text": (
                    f"你关注的「{_CONCERN_LABEL.get(concern, concern)}」方向：本次已入库的"
                    f"相关指标未见明显异常；系统同时完成了全身各系统扫描，"
                    f"额外发现见下方总览。"
                ),
            }

        # -------- 预测依据（改动 4）：数据范围 + 预测终点 + 演示标识 --------
        reports = st.db.list_reports(body.patient_id)
        exam_dates = sorted({str(x)[:10] for x in recs[COL_MEASURED_AT]})
        by_code_dates: dict[str, set] = {}
        for _, rr in recs.iterrows():
            by_code_dates.setdefault(rr[COL_INDICATOR], set()).add(str(rr[COL_MEASURED_AT])[:10])
        n_comparable = sum(1 for ds in by_code_dates.values() if len(ds) >= 2)
        prediction_context = {
            "n_reports": len(reports),
            "n_records": int(len(recs)),
            "n_dates": len(exam_dates),
            "first_date": exam_dates[0] if exam_dates else None,
            "last_date": exam_dates[-1] if exam_dates else None,
            "n_comparable_indicators": int(n_comparable),
            **st.model_card,
        }

        return {
            "patient_id": body.patient_id,
            "model_version": decision.version,
            "arm": decision.arm,
            "concern": concern,
            "concern_label": _CONCERN_LABEL.get(concern, concern),
            "concern_answer": concern_answer,
            "results": results,
            "monotonic_note": monotonic_note,
            "risk_overview": {
                "items": overview_items,
                "stable_groups": stable_groups,
            },
            "prediction_context": prediction_context,
            "referral": advice.to_dict(),
        }

    # ---------------- 趋势报告与 AI 大模型深度分析 ----------------
    @app.get("/api/patients/{pid}/trend")
    def patient_trend(pid: str):
        patient = _patient_or_404(pid)
        recs = _records_frame(pid)
        pseudo = st.audit.pseudonymize(pid)
        # recent_n=24：指标曲线要能画出用户上传的【全部】真实检查时间点
        # （改动 5"近3次曲线 → 指标历史趋势"），时间范围筛选交给前端。
        report = build_trend_report(
            st.trend, recs, demographics=_demo_frame(patient),
            recent_n=24,
            audit=st.audit, pseudo_id=pseudo, horizons=tuple(st.horizons),
        )
        data = report.to_dict()

        # 接入 AI 大模型临床深度解读与干预方案
        p_info = {
            "name": patient.get("name", patient.get("patient_id", pid)),
            "sex": patient.get("sex", "未知"),
            "age": patient.get("age", "—"),
            "n_records": patient.get("n_records", 0),
        }
        factors = []
        if report.change_attribution:
            factors = [f.phrase() for f in report.change_attribution.factors]

        try:
            from drp.serving.llm_advisor import generate_llm_trend_analysis
            ai_data = generate_llm_trend_analysis(
                p_info,
                data.get("comparisons", []),
                data.get("risk_trajectories", {}),
                factors=factors,
            )
            data["ai_analysis"] = ai_data
        except Exception as e:
            logger.warning("[Trend] AI 大模型分析生成异常: %s", e)
            data["ai_analysis"] = None

        return data

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
        ocr = await loop.run_in_executor(None, _ocr_extract_image, image_b64)

        extracted_text = ocr["text"]
        lines = [l.strip() for l in extracted_text.splitlines() if l.strip()]
        detected = _detect_report_date(extracted_text)
        return {
            "text": extracted_text,
            "count": len(lines),
            # 改版需求：上传报告 → OCR 识别 → 「识别到检查日期：xxxx」确认/修改；
            # 识别不出（None）时前端再让用户填写。
            "detected_date": detected[0],
            "detected_date_source": detected[1],
            # V3.1：方向/版式/脱敏透明化，前端在待确认卡片上如实展示。
            "engine": ocr["engine"],
            "rotation": ocr["rotation"],          # 已自动旋转的角度（0=原图方向）
            "layout": ocr["layout"],              # two_panel = 左右双栏已逐行拆开
            "n_redacted": ocr["n_redacted"],
        }

    # ---------------- 前端静态托管（必须最后挂载，避免吞掉 /api） ----------------
    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return app


#: 报告日期识别的关键词行（优先级从高到低）。化验单上通常同时印有
#: 采集时间/接收时间/报告时间；对"这份数据属于哪一天"而言，采样/采集
#: 时间最接近真实检验时点，报告时间次之。
_DATE_LINE_KEYWORDS = ("采样", "采集", "抽血", "送检", "检验日期", "报告", "审核", "日期", "时间")


def _detect_report_date(text: str) -> tuple[str | None, str | None]:
    """
    从 OCR 文本里识别真实检查日期。返回 (ISO 日期 | None, 命中说明 | None)。

    识别不到就老老实实返回 None 让用户填 —— 宁可让用户确认一次，
    也不能把错误日期悄悄写进时间轴（核查项 3 的反面教材就是时间字段错）。
    """
    import re as _re

    if not text:
        return None, None
    today = datetime.now(timezone.utc).date()
    pat = _re.compile(
        r"(20\d{2})\s*[年./\-]\s*(\d{1,2})\s*[月./\-]\s*(\d{1,2})\s*日?"
    )

    def _valid(y: int, m: int, d: int):
        try:
            dt = date(y, m, d)
        except ValueError:
            return None
        if dt.year < 2000 or dt > today:
            return None
        return dt

    keyword_hits: list[tuple[date, str]] = []
    any_hits: list[date] = []
    for line in text.splitlines():
        for m in pat.finditer(line):
            dt = _valid(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if dt is None:
                continue
            any_hits.append(dt)
            for kw in _DATE_LINE_KEYWORDS:
                if kw in line:
                    keyword_hits.append((dt, kw))
                    break
    if keyword_hits:
        # 关键词行内取最早出现的高优先关键词；同优先级取最大日期（末次时间戳）
        best_kw_rank = min(_DATE_LINE_KEYWORDS.index(k) for _, k in keyword_hits)
        cands = [d for d, k in keyword_hits
                 if _DATE_LINE_KEYWORDS.index(k) == best_kw_rank]
        pick = max(cands)
        return pick.isoformat(), f"识别自「{_DATE_LINE_KEYWORDS[best_kw_rank]}」行"
    if any_hits:
        return max(any_hits).isoformat(), "识别自文本中的日期"
    return None, None


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


def _rapid_items(engine, img) -> list[dict]:
    """跑一次 RapidOCR，把结果转成 ocr_layout 的 item 字典（含置信度）。"""
    results, _ = engine(img)
    items: list[dict] = []
    for box, text, score in results or []:
        txt = (text or "").strip()
        if not txt:
            continue
        pts = np.array(box)
        items.append({
            "text": txt,
            "x0": float(np.min(pts[:, 0])), "y0": float(np.min(pts[:, 1])),
            "x1": float(np.max(pts[:, 0])), "y1": float(np.max(pts[:, 1])),
            "score": float(score) if score is not None else 0.5,
        })
    return items


def _ocr_extract_image(image_b64: str) -> dict:
    """
    通用图片 OCR 识别，返回结构化结果：
      {"text", "engine", "rotation"(顺时针角度), "layout", "n_redacted"}

    V3.1 三处升级（对应用户实测反馈"识别不准确不完整"）：
      · 方向自适应：横版报告竖着拍是常态。对 0/90/180/270 四个方向各跑一遍，
        按 Σ(置信度×文本权重) 取最优 —— 转错方向时中文识别近乎全灭，
        评分差距是数量级的，误选概率极低。四跑的代价（秒级）换准确率，
        对单机 Demo 是正确取舍；生产可先跑 0°、低分再补跑其余方向。
      · 双栏重排：左右分栏逐行拆开输出，一行一指标，右栏不再丢失。
      · 入库前脱敏：姓名/手机号/证件号等替换为 [已脱敏]（原图本就不落库）。
    """
    import base64
    import cv2

    from .ocr_layout import (
        redact_pii_text, reconstruct_lines, text_weight,
    )

    clean_b64 = image_b64.split(",", 1)[1] if "," in image_b64 else image_b64

    # --- 1. 本地 RapidOCR：四方向试跑取最优 ---
    try:
        engine = _get_rapid_ocr()
        if engine is not None:
            raw = base64.b64decode(clean_b64)
            arr = np.frombuffer(raw, dtype=np.uint8)
            img0 = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img0 is not None:
                best = None  # (score, rotation_deg, items, page_w)
                for k, deg in ((0, 0), (1, 270), (2, 180), (3, 90)):
                    img = img0 if k == 0 else np.ascontiguousarray(np.rot90(img0, k))
                    items = _rapid_items(engine, img)
                    sc = sum(it["score"] * text_weight(it["text"]) for it in items)
                    logger.info("[OCR] 方向 %d°: %d 框, 得分 %.0f", deg, len(items), sc)
                    if best is None or sc > best[0]:
                        best = (sc, deg, items, float(img.shape[1]))
                if best and best[2]:
                    _, deg, items, page_w = best
                    lines, layout = reconstruct_lines(items, page_w)
                    text, n_red = redact_pii_text("\n".join(lines))
                    logger.info(
                        "[OCR] RapidOCR 取方向 %d°，重排 %d 行（%s），脱敏 %d 处",
                        deg, len(lines), layout, n_red,
                    )
                    return {
                        "text": text, "engine": "rapidocr",
                        "rotation": deg, "layout": layout, "n_redacted": n_red,
                    }
    except Exception as e:
        logger.warning("[OCR] RapidOCR 执行异常 (%s)，尝试 Vision 兜底...", e)

    # --- 2. Vision API 兜底 ---
    vision = _ai_vision_ocr_extract(image_b64)
    if vision:
        lines = []
        collected_at = ""
        if isinstance(vision, dict):
            collected_at = str(vision.get("collected_at") or "").strip()
            vision_items = vision.get("indicators") or []
        else:  # 兼容旧格式：模型直接返回指标数组
            vision_items = vision
        for ind in vision_items:
            name = ind.get("name_raw", "")
            val = ind.get("value", "")
            unit = ind.get("unit", "")
            ref = ind.get("reference", "")
            line = f"{name} {val} {unit}"
            if ref:
                line += f" {ref}"
            lines.append(line)
        if collected_at:
            # 让 _detect_report_date 的「采集」关键词能命中 —— 视觉兜底
            # 此前只回传指标行，日期识别在这条路径上是断的。
            lines.append(f"采集时间：{collected_at}")
        from .ocr_layout import redact_pii_text as _red
        text, n_red = _red("\n".join(lines))
        return {
            "text": text, "engine": "vision",
            "rotation": 0, "layout": "single", "n_redacted": n_red,
        }

    return {"text": "", "engine": "none", "rotation": 0, "layout": "single", "n_redacted": 0}


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
        "你是一位专业医学检验单 OCR 识别员。任务：\n"
        "1. 图片可能整体旋转了 90°/180°/270°，请按正确阅读方向理解后再抄录；\n"
        "2. 化验单可能是左右双栏排版，务必按表格行把【项目-结果-参考区间-单位】\n"
        "   正确配对，绝不能把左栏的项目配右栏的数值；\n"
        "3. 忽略化验单纸张之外的一切背景文字（报纸、宣传页、手指等）；\n"
        "4. 抄录『采集时间/采样时间』的日期（没有则留空字符串）；\n"
        "5. 不要输出姓名等个人信息。\n"
        "严格只返回如下 JSON（Markdown codeblock 中）：\n"
        '{"collected_at": "2025/10/06", "indicators": '
        '[{"name_raw": "丙氨酸氨基转移酶(ALT)", "value": 68, "unit": "U/L", "reference": "9-50"}]}'
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
            if isinstance(parsed, dict) and parsed.get("indicators"):
                return parsed
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed
    except Exception as err:
        logger.warning("[OCR] Vision API 异常 (%s)", err)

    return []
