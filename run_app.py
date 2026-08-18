"""
病情预测平台 · 一键启动。

    python run_app.py                 # 首次自动完成模型自举（数分钟），随后启动服务
    python run_app.py --bootstrap     # 只做自举，不启动
    python run_app.py --port 8899     # 指定端口

数据目录默认 ./app_data（可用 --data 覆盖）。删除该目录 = 全新重来。
生产部署：注入 DRP_PII_SALT，网关强制 HTTPS（规范 7，勿省略）。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("run_app")


def main() -> None:
    ap = argparse.ArgumentParser(description="病情预测平台应用")
    ap.add_argument("--data", default="app_data", help="应用数据目录（默认 ./app_data）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--bootstrap", action="store_true", help="只执行自举后退出")
    ap.add_argument("--n-patients", type=int, default=2600, help="自举合成患者数")
    args = ap.parse_args()

    from app.bootstrap import is_bootstrapped, run_bootstrap

    data = Path(args.data)
    if not is_bootstrapped(data):
        log.info("检测到首次启动，开始模型自举（真实清洗/特征/三层验证，约 2~5 分钟）…")
        meta = run_bootstrap(data, n_patients=args.n_patients)
        log.info("自举完成: version=%s auc=%s status=%s",
                 meta["version"], meta["headline_auc"], meta["validation_status"])
    else:
        log.info("检测到已完成自举的数据目录: %s", data)

    if args.bootstrap:
        return

    import uvicorn

    from app.server import build_server

    app = build_server(data)
    log.info("前端地址: http://%s:%d/  （API 文档: /docs）", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
