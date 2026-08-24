"""持久化模型注册表：验证门禁、版本状态、灰度与可回滚发布。"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from ..validation import ValidationReport, assert_release_ready

_STATE_FILE = "registry.json"
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class RegistryError(RuntimeError):
    """注册、发布或回滚操作不满足安全约束。"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ModelVersionInfo:
    version: str
    bank_dir: Path
    created_at: str = field(default_factory=_now)
    status: str = "STAGING"
    validation_status: str = "PASS"
    headline_auc: dict[str, float] = field(default_factory=dict)
    validation_reports: dict[str, str] = field(default_factory=dict)
    notes: str = ""
    traffic_pct: float = 0.0

    def to_dict(self) -> dict:
        out = asdict(self)
        out["bank_dir"] = str(self.bank_dir)
        return out

    @classmethod
    def from_dict(cls, data: Mapping, base_dir: Path) -> "ModelVersionInfo":
        payload = dict(data)
        bank = Path(payload["bank_dir"])
        payload["bank_dir"] = bank if bank.is_absolute() else (base_dir / bank).resolve()
        payload["headline_auc"] = {
            str(k): float(v) for k, v in payload.get("headline_auc", {}).items()
        }
        payload["traffic_pct"] = float(payload.get("traffic_pct", 0.0))
        return cls(**payload)


class ModelRegistry:
    """以一个原子 JSON 账本维护模型版本；同一时刻最多一个 ACTIVE/CANARY。"""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._path = self.root / _STATE_FILE
        self._lock = threading.RLock()
        if not self._path.exists():
            self._write({"schema_version": 1, "versions": {}, "active_history": []})
        self._validate_state(self._read())

    def _read(self) -> dict:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RegistryError(f"模型注册表损坏或不可读: {self._path}: {exc}") from exc
        data.setdefault("versions", {})
        data.setdefault("active_history", [])
        return data

    def _write(self, state: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self._path)

    @staticmethod
    def _validate_state(state: Mapping) -> None:
        versions = state.get("versions", {})
        active = [v for v in versions.values() if v.get("status") == "ACTIVE"]
        canary = [v for v in versions.values() if v.get("status") == "CANARY"]
        if len(active) > 1 or len(canary) > 1:
            raise RegistryError("注册表违反唯一性约束：ACTIVE/CANARY 各最多一个")

    def _serial_bank_dir(self, bank_dir: Path) -> str:
        resolved = bank_dir.resolve()
        try:
            return str(resolved.relative_to(self.root.parent.resolve()))
        except ValueError:
            return str(resolved)

    def _info(self, raw: Mapping) -> ModelVersionInfo:
        return ModelVersionInfo.from_dict(raw, self.root.parent)

    def register(
        self,
        version: str,
        bank_dir: str | Path,
        reports: Mapping[str, ValidationReport],
        notes: str = "",
    ) -> ModelVersionInfo:
        """登记已通过门禁的 HorizonBank；任何 BLOCK 报告都会中断注册。"""
        if not _VERSION_RE.fullmatch(version):
            raise RegistryError("版本号仅允许字母、数字、点、下划线和连字符，长度 1-128")
        bank = Path(bank_dir)
        if not bank.is_dir():
            raise RegistryError(f"模型目录不存在: {bank}")
        if not reports:
            raise RegistryError("注册模型必须附带至少一份三层验证报告")
        for report in reports.values():
            assert_release_ready(report)

        with self._lock:
            state = self._read()
            if version in state["versions"]:
                raise RegistryError(f"模型版本已存在，禁止覆盖: {version}")
            report_dir = self.root / "reports" / version
            saved: dict[str, str] = {}
            for horizon, report in reports.items():
                path = report.save_json(report_dir / f"{horizon}.json")
                saved[str(horizon)] = str(path.relative_to(self.root))
            statuses = {r.status for r in reports.values()}
            info = ModelVersionInfo(
                version=version,
                bank_dir=bank.resolve(),
                validation_status="CONDITIONAL" if "CONDITIONAL" in statuses else "PASS",
                headline_auc={str(h): float(r.headline_auc) for h, r in reports.items()},
                validation_reports=saved,
                notes=str(notes),
            )
            payload = info.to_dict()
            payload["bank_dir"] = self._serial_bank_dir(bank)
            state["versions"][version] = payload
            self._validate_state(state)
            self._write(state)
            return self.get(version)

    def get(self, version: str) -> ModelVersionInfo:
        raw = self._read()["versions"].get(version)
        if raw is None:
            raise KeyError(f"未知模型版本: {version}")
        return self._info(raw)

    def _find_status(self, status: str) -> ModelVersionInfo | None:
        for raw in self._read()["versions"].values():
            if raw.get("status") == status:
                return self._info(raw)
        return None

    def get_active(self) -> ModelVersionInfo | None:
        return self._find_status("ACTIVE")

    def get_canary(self) -> ModelVersionInfo | None:
        return self._find_status("CANARY")

    def set_canary(self, version: str, traffic_pct: float) -> ModelVersionInfo:
        pct = float(traffic_pct)
        if not (0.0 < pct <= 100.0):
            raise ValueError("灰度流量必须在 (0, 100] 之间")
        with self._lock:
            state = self._read()
            versions = state["versions"]
            if version not in versions:
                raise RegistryError(f"未知模型版本: {version}")
            if versions[version]["status"] == "ACTIVE":
                raise RegistryError("ACTIVE 版本无需再设为灰度")
            for name, raw in versions.items():
                if raw.get("status") == "CANARY" and name != version:
                    raw["status"] = "STAGING"
                    raw["traffic_pct"] = 0.0
            versions[version]["status"] = "CANARY"
            versions[version]["traffic_pct"] = pct
            self._validate_state(state)
            self._write(state)
        return self.get(version)

    def promote(self, version: str) -> ModelVersionInfo:
        with self._lock:
            state = self._read()
            versions = state["versions"]
            if version not in versions:
                raise RegistryError(f"未知模型版本: {version}")
            previous = next(
                (name for name, raw in versions.items() if raw.get("status") == "ACTIVE"), None
            )
            if previous == version:
                return self._info(versions[version])
            if previous is not None:
                versions[previous]["status"] = "RETIRED"
                versions[previous]["traffic_pct"] = 0.0
                state["active_history"].append(previous)
            for name, raw in versions.items():
                if raw.get("status") == "CANARY" and name != version:
                    raw["status"] = "STAGING"
                    raw["traffic_pct"] = 0.0
            versions[version]["status"] = "ACTIVE"
            versions[version]["traffic_pct"] = 100.0
            self._validate_state(state)
            self._write(state)
        return self.get(version)

    def rollback(self) -> ModelVersionInfo:
        with self._lock:
            state = self._read()
            versions = state["versions"]
            current = next(
                (name for name, raw in versions.items() if raw.get("status") == "ACTIVE"), None
            )
            history = state.get("active_history", [])
            target = None
            while history:
                candidate = history.pop()
                if candidate in versions and candidate != current:
                    target = candidate
                    break
            if target is None:
                retired = sorted(
                    ((raw.get("created_at", ""), name) for name, raw in versions.items()
                     if raw.get("status") == "RETIRED" and name != current),
                    reverse=True,
                )
                target = retired[0][1] if retired else None
            if target is None:
                raise RegistryError("没有可回滚的历史 ACTIVE 版本")
            if current is not None:
                versions[current]["status"] = "RETIRED"
                versions[current]["traffic_pct"] = 0.0
            for raw in versions.values():
                if raw.get("status") == "CANARY":
                    raw["status"] = "STAGING"
                    raw["traffic_pct"] = 0.0
            versions[target]["status"] = "ACTIVE"
            versions[target]["traffic_pct"] = 100.0
            state["active_history"] = history
            self._validate_state(state)
            self._write(state)
        return self.get(target)

    def to_dict(self) -> dict[str, dict]:
        return {name: self._info(raw).to_dict() for name, raw in self._read()["versions"].items()}
