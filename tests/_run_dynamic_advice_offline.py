# -*- coding: utf-8 -*-
"""无 pytest 环境下的最小驱动：逐个执行 test_dynamic_advice 的用例。"""
import os
import sys
import types
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

# pytest 桩：只需 fixture 装饰器与 main 占位
stub = types.ModuleType("pytest")
stub.fixture = lambda *a, **k: (lambda f: f)
stub.main = lambda *a, **k: 0
sys.modules.setdefault("pytest", stub)

import test_dynamic_advice as T  # noqa: E402

for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
    os.environ.pop(k, None)

registry = T.ReferenceRegistry.from_yaml(ROOT / "configs" / "reference_intervals.yaml")

passed, failed = 0, []
for cls_name in ("TestInterventionsAreDataDriven", "TestAdvisorFallbackIsDataDriven"):
    cls = getattr(T, cls_name)
    inst = cls()
    for name in sorted(n for n in dir(inst) if n.startswith("test_")):
        try:
            getattr(inst, name)(registry)
            passed += 1
            print(f"  PASS {cls_name}.{name}")
        except Exception:
            failed.append(f"{cls_name}.{name}")
            print(f"  FAIL {cls_name}.{name}")
            traceback.print_exc()

print(f"\n{passed} passed, {len(failed)} failed")
sys.exit(1 if failed else 0)
