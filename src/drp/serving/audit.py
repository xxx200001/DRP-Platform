"""
全链路日志与数据回流（规范 4.2 / 1.2 / 1.3）。

规范原话：
  - "每一次预测永久留存：原始上传数据、OCR结果、结构化结果、所有特征值、
     模型输出概率、风险等级、SHAP权重"
  - "用于错误样本复盘、模型迭代、问题定位"
  - "所有用户医疗数据全程脱敏存储，不可存储明文身份证、手机号"
  - "用户授权后，匿名化保存每次预测输入、模型结果、用户后续随访反馈"

【判断日志够不够的唯一标准：能不能把那次预测原地重建出来】
线上出一起"用户说结果不对"的事故，你需要回答的问题是：当时用的哪个模型、
哪些特征值、OCR 认成了什么、为什么给出这个概率。只存概率和时间戳的日志
在这种时刻等于没有日志。所以 PredictionRecord 存的是一整条因果链：
    原始件引用 -> OCR 结果 -> 结构化结果 -> 全部特征值
    -> model_version + feature_hash -> 概率 -> 风险等级 -> SHAP 权重
其中 feature_hash 是关键：它一变就说明特征集变了，此时拿新模型解释旧记录
必然对不上，日志系统必须能立刻指出这一点，而不是让人查三天。

【脱敏：检测到明文 PII 是拒写并报错，不是静默清洗】
静默清洗看起来更"稳"，实则是把上游的漏洞盖住了 —— 这次身份证被悄悄擦掉，
下次它会以"备注字段"的形式再漏一遍，而且没人知道。拒写会让问题在开发阶段
就暴露在写日志的那一行，这正是规范 1.2 要的效果。
唯一的例外是 patient_id：它必须保留关联能力（同一人的多次预测要能串起来做
趋势追踪，规范 6），所以用带盐 HMAC 做假名化 —— 可关联、不可逆。
盐必须来自环境/配置，硬编码进代码等于没盐。

【为什么是 append-only JSONL 按天分片】
医疗日志的核心诉求是不可篡改与可追溯，不是查询性能。
JSONL 追加写不需要锁、天然抗并发；按天分片让归档、抽样复盘、按保留期
删除都变成文件操作。真正的分析走离线导入数仓，而不是在这个文件上查。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PIIError(ValueError):
    """检测到明文个人身份信息。必须在上游修掉，不允许在日志层静默清洗。"""


# ---------------------------------------------------------------------------
# PII 检测
# ---------------------------------------------------------------------------
#: 明文 PII 正则。宁可误报也不能漏报 —— 误报会被开发者当场发现并修，
#: 漏报会安静地躺在磁盘上直到出事。
PII_PATTERNS: dict[str, re.Pattern] = {
    "身份证": re.compile(r"(?<!\d)(\d{17}[\dXx]|\d{15})(?!\d)"),
    "手机号": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    "邮箱": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "银行卡": re.compile(r"(?<!\d)\d{16,19}(?!\d)"),
}

#: 字段名黑名单：这些字段无论内容是什么都不允许进日志。
PII_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "name", "real_name", "patient_name", "姓名", "真实姓名",
        "id_card", "idcard", "identity", "身份证", "身份证号",
        "phone", "mobile", "tel", "手机号", "电话", "联系方式",
        "address", "住址", "家庭住址", "email", "邮箱",
        "emergency_contact", "紧急联系人", "bank_card", "银行卡",
    }
)


#: 系统生成的不透明标识符字段名（uuid hex / hash 摘要），不是自由文本。
#: 它们不可能承载真实 PII，但十六进制字符序列有不可忽略的概率
#: （trace_id 是 32 位 uuid4 hex，任意连续 16 位落入纯数字子集的概率约
#: 0.5%，规模化后每天都会真实发生）随机产生连续 16~19 位纯数字子串，
#: 被"银行卡"正则误伤，进而在 strict_audit=True 下把一次完全正常的预测
#: 误判成"检测到PII"而中断（service.py"日志写失败=预测失败"的设计
#: 意图是拦截真实 PII，不是拦截系统自己生成的哈希碰撞噪声）。
#: 字段名黑名单（PII_FIELD_NAMES）对它们依然生效：如果哪天有人把这几个
#: 名字错用来存真实姓名，仍会被拦——这里只跳过【内容正则】扫描。
_OPAQUE_ID_FIELDS: frozenset[str] = frozenset({"trace_id", "feature_hash", "pseudo_id"})


def scan_pii(obj, path: str = "") -> list[str]:
    """递归扫描任意可 JSON 化结构，返回违规描述列表（空 = 干净）。"""
    hits: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k)
            if key.strip().lower() in PII_FIELD_NAMES:
                hits.append(f"{path}.{key}: 字段名属于 PII 黑名单，禁止入库")
                continue
            if key.strip().lower() in _OPAQUE_ID_FIELDS:
                continue  # 系统生成的不透明标识符，跳过内容正则扫描（见上方常量注释）
            hits.extend(scan_pii(v, f"{path}.{key}"))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            hits.extend(scan_pii(v, f"{path}[{i}]"))
    elif isinstance(obj, str):
        for label, pat in PII_PATTERNS.items():
            m = pat.search(obj)
            if m:
                hits.append(f"{path}: 疑似明文{label}（{m.group()[:4]}***）")
    return hits


def assert_no_pii(obj, source: str = "") -> None:
    hits = scan_pii(obj)
    if hits:
        raise PIIError(
            f"检测到 {len(hits)} 处明文个人信息（来源={source or '未标注'}）：\n"
            + "\n".join(f"  - {h}" for h in hits[:5])
            + "\n规范 1.2 要求全程脱敏存储。请在上游去标识化后再写日志，"
            "不要在此处静默清洗 —— 那只会把漏洞盖住。"
        )


def pseudonymize(patient_id: str, salt: str) -> str:
    """
    带盐 HMAC 假名化。同一个盐下同一个人始终得到同一个假名（可做趋势追踪），
    但无法从假名反推真实 ID。换盐即切断关联，可用于按项目隔离。
    """
    if not salt:
        raise ValueError(
            "假名化必须提供盐。空盐等于直接暴露 ID 的哈希，"
            "在患者 ID 空间不大时可被彩虹表还原。请从环境变量注入。"
        )
    return hmac.new(salt.encode("utf-8"), str(patient_id).encode("utf-8"), hashlib.sha256).hexdigest()[:24]


# ---------------------------------------------------------------------------
# 记录
# ---------------------------------------------------------------------------
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if np.isnan(v) else v
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime, date)):
        return o.isoformat()
    if o is None or isinstance(o, (str, int, bool)):
        return o
    return str(o)


@dataclass
class PredictionRecord:
    """
    一次预测的完整因果链（规范 4.2）。字段顺序即数据流向。

    raw_ref 存的是原始件的【引用】（对象存储 key），不是内容本身：
    原始报告图片可能有几 MB，塞进 JSONL 会让日志文件迅速不可用，
    而且图片里往往带着姓名等 PII，属于必须隔离存储的资产。
    """

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    )
    pseudo_id: str = ""  # 假名化后的患者标识
    horizon: str = ""

    # ---- 上游链路 ----
    raw_ref: str = ""  # 原始上传件在对象存储中的 key
    ocr_result: dict = field(default_factory=dict)  # OCR 识别产物（已脱敏）
    structured: dict = field(default_factory=dict)  # 结构化 + 单位归一化产物
    features: dict = field(default_factory=dict)  # 全部特征值（含 NaN）

    # ---- 模型 ----
    model_version: str = ""
    feature_hash: str = ""
    backend: str = ""
    calibrated: bool = True

    # ---- 输出 ----
    probability: float = float("nan")
    risk_tier: str = ""
    attribution: dict = field(default_factory=dict)  # SHAP 权重与 Top 因子

    # ---- 监控与回流 ----
    drift_level: str = ""
    degraded: bool = False  # 是否走了兜底/降级路径
    notes: dict = field(default_factory=dict)

    # ---- 回流标签（规范 1.3，随访反馈到达后回填） ----
    outcome_event: int | None = None
    outcome_days: float | None = None
    outcome_updated_at: str = ""

    def to_dict(self) -> dict:
        return _jsonable(asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "PredictionRecord":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def is_replayable(self) -> bool:
        """能否原地重建这次预测。缺任何一项，事故复盘时就断链了。"""
        return bool(self.model_version and self.feature_hash and self.features)


# ---------------------------------------------------------------------------
# 日志器
# ---------------------------------------------------------------------------
class AuditLogger:
    """
    append-only JSONL 全链路日志，按天分片。

    用法::

        audit = AuditLogger("/data/drp/audit", salt=os.environ["DRP_PII_SALT"])
        rec = PredictionRecord(pseudo_id=audit.pseudonymize("P12345"), ...)
        audit.log(rec)

        # 事故复盘
        df = audit.load_day("2026-08-15")
        bad = audit.replay_candidates(df, tier="极高危", outcome_event=0)
    """

    def __init__(
        self,
        root: str | Path,
        salt: str,
        strict_pii: bool = True,
        require_replayable: bool = True,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.salt = salt
        self.strict_pii = strict_pii
        self.require_replayable = require_replayable
        if not salt:
            raise ValueError("AuditLogger 必须提供 salt（规范 1.2 假名化要求），建议从环境变量注入")

    # ------------------------------------------------------------------
    def pseudonymize(self, patient_id: str) -> str:
        return pseudonymize(patient_id, self.salt)

    def _path_for(self, day: str | date | None = None) -> Path:
        d = day or datetime.now(timezone.utc).date()
        if isinstance(d, date):
            d = d.isoformat()
        return self.root / f"{d}.jsonl"

    # ------------------------------------------------------------------
    def log(self, record: PredictionRecord) -> str:
        """
        写入一条记录，返回 trace_id。

        写失败必须让调用方感知（规范 4.2 "永久留存"是硬要求）：
        service 层默认把它当作预测失败处理，而不是"日志掉了但结果照发"——
        一次没有日志的预测，等于一次永远查不清的潜在事故。
        """
        payload = record.to_dict()
        if self.strict_pii:
            assert_no_pii(payload, source=f"PredictionRecord[{record.trace_id}]")
        if self.require_replayable and not record.is_replayable():
            raise ValueError(
                f"记录 {record.trace_id} 不可重建（缺 model_version / feature_hash / features）。"
                "只记概率的日志在事故复盘时等于没有日志（规范 4.2）。"
            )
        path = self._path_for()
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return record.trace_id

    # ------------------------------------------------------------------
    def iter_records(self, day: str | date | None = None) -> Iterator[PredictionRecord]:
        path = self._path_for(day)
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield PredictionRecord.from_dict(json.loads(line))
                except json.JSONDecodeError:
                    # 单行损坏不能让整天的日志不可读 —— 记录位置后跳过
                    logger.error("日志解析失败: %s 第 %d 行已跳过", path, i)

    def load_day(self, day: str | date | None = None) -> pd.DataFrame:
        rows = [r.to_dict() for r in self.iter_records(day)]
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def find(self, trace_id: str, day: str | date | None = None) -> PredictionRecord | None:
        found = None
        for r in self.iter_records(day):
            if r.trace_id == trace_id:
                found = r
        return found

    # ------------------------------------------------------------------
    def history_for_patient(
        self,
        pseudo_id: str,
        since: str | date | None = None,
        until: str | date | None = None,
        horizon: str | None = None,
    ) -> pd.DataFrame:
        """
        单患者跨天历史记录（规范 6：趋势追踪 / 时序对比报告的数据来源）。

        按天分片是写入侧的最优设计（append-only、故障域小），但趋势追踪要看
        "同一个人这半年的所有记录"，天然是跨天查询。这里先用文件名
        （YYYY-MM-DD.jsonl）做日期范围过滤，再精确匹配 pseudo_id —— 对天数
        很多的部署，调用方应传 since/until 收窄范围，避免全量扫描全部日志。

        返回已按 dedup_latest 折叠（结局回填会为同一 trace_id 追加新行，
        只有最后一行数字是准的）、按 created_at 升序排列的记录。
        """
        since_d = pd.Timestamp(since).date() if since else None
        until_d = pd.Timestamp(until).date() if until else None

        frames: list[pd.DataFrame] = []
        for path in sorted(self.root.glob("*.jsonl")):
            try:
                day = date.fromisoformat(path.stem)
            except ValueError:
                continue  # 非日期命名的文件（未来可能的索引/元文件），跳过
            if since_d and day < since_d:
                continue
            if until_d and day > until_d:
                continue
            df = self.load_day(day)
            if not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        out = out[out["pseudo_id"] == pseudo_id]
        if horizon is not None and "horizon" in out.columns:
            out = out[out["horizon"] == horizon]
        if out.empty:
            return out
        out = self.dedup_latest(out)
        if "created_at" in out.columns:
            out = out.sort_values("created_at").reset_index(drop=True)
        return out

    # ------------------------------------------------------------------
    def attach_outcome(
        self,
        trace_id: str,
        event: int,
        days: float,
        day: str | date | None = None,
    ) -> bool:
        """
        回填随访结局（规范 1.3 数据回流）。

        append-only 语义下不原地改写，而是追加一条带同 trace_id 的结局记录；
        读取时以最后一条为准。改写历史行会破坏审计日志的不可篡改性，
        而这正是医疗场景选择 append-only 的全部理由。
        """
        rec = self.find(trace_id, day)
        if rec is None:
            logger.warning("回填结局失败：未找到 trace_id=%s", trace_id)
            return False
        rec.outcome_event = int(event)
        rec.outcome_days = float(days)
        rec.outcome_updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        # 结局更新必须留在原预测分片。若写入“今天”的分片，调用方按原日期
        # 查询时会看不到更新；跨天拼接顺序不同时还可能让旧记录覆盖新记录。
        with self._path_for(day).open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
        return True

    # ------------------------------------------------------------------
    @staticmethod
    def dedup_latest(df: pd.DataFrame) -> pd.DataFrame:
        """同 trace_id 取最后一条（结局回填会产生多条）。"""
        if df.empty or "trace_id" not in df.columns:
            return df
        return df.drop_duplicates(subset="trace_id", keep="last").reset_index(drop=True)

    @staticmethod
    def replay_candidates(
        df: pd.DataFrame,
        tier: str | None = None,
        outcome_event: int | None = None,
        prob_range: tuple[float, float] | None = None,
        limit: int = 200,
    ) -> pd.DataFrame:
        """
        错误样本复盘筛选（规范 4.2 明确用途）。

        两类最值钱的样本：
          - 判了极高危但随访无事件（假阳性）→ 查是不是一过性异常没被剔除（规范 2.4）
          - 判了低危却发生了事件（假阴性/漏诊）→ 查缺失项、查特征覆盖不足
        漏诊样本的优先级永远高于假阳性：规范 5 把敏感度放在特异度前面。
        """
        out = AuditLogger.dedup_latest(df)
        if out.empty:
            return out
        if tier is not None and "risk_tier" in out.columns:
            out = out[out["risk_tier"] == tier]
        if outcome_event is not None and "outcome_event" in out.columns:
            out = out[out["outcome_event"] == outcome_event]
        if prob_range is not None and "probability" in out.columns:
            lo, hi = prob_range
            out = out[(out["probability"] >= lo) & (out["probability"] <= hi)]
        return out.head(limit).reset_index(drop=True)

    # ------------------------------------------------------------------
    def to_training_frame(self, days: list[str] | None = None) -> pd.DataFrame:
        """
        把带回流标签的日志导出成可直接训练的样本表（规范 1.3 线上真实样本库）。

        只取有结局标签的记录。注意：线上回流样本的分布与训练集不同
        （只有用过产品的人才在里面），直接混进训练集会引入选择偏倚 ——
        必须作为独立数据源单独评估，或做加权，这一步由训练脚本负责，
        本函数只保证"导出的是干净、可用、带标签的样本"。
        """
        frames = [self.load_day(d) for d in (days or [datetime.now(timezone.utc).date().isoformat()])]
        df = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
        if df.empty:
            return df
        df = self.dedup_latest(df)
        if "outcome_event" not in df.columns:
            return pd.DataFrame()
        labeled = df[df["outcome_event"].notna()].reset_index(drop=True)
        if labeled.empty:
            return labeled
        feats = pd.json_normalize(labeled["features"]).reset_index(drop=True)
        meta = labeled[["pseudo_id", "created_at", "model_version", "probability",
                        "risk_tier", "outcome_event", "outcome_days"]].reset_index(drop=True)
        logger.info(
            "导出回流样本 %d 条（含标签），特征 %d 列。"
            "提醒：线上回流人群存在选择偏倚，需单独评估或加权，勿直接并入训练集。",
            len(labeled), feats.shape[1],
        )
        return pd.concat([meta, feats], axis=1)
