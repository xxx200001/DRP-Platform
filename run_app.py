"""
病情预测平台 · 一键启动。

    python run_app.py                 # 首次自动完成模型自举（数分钟），随后启动服务
    python run_app.py --bootstrap     # 只做自举，不启动
    python run_app.py --port 8899     # 指定端口

数据目录默认 ./app_data（可用 --data 覆盖）。删除该目录 = 全新重来。
生产部署：注入 DRP_PII_SALT，网关强制 HTTPS（规范 7，勿省略）。
"""

from __future__ import annotations

import os
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

def _load_dotenv(path: Path) -> int:
    """
    零依赖 .env 加载（V3.5）。

    根因回填：此前应用【从未】读取 .env——用户把 ANTHROPIC_API_KEY 写进
    .env 后，os.environ 里依然是空，Vision OCR 与在线 LLM 全部静默跳过、
    一路走离线兜底，且没有任何提示。"配了密钥却没效果"即源于此。

    规则：KEY=VALUE 逐行读取；# 开头为注释；已存在的环境变量【不覆盖】
    （显式 export 的优先级最高）；值两侧的引号剥掉。返回载入条数。
    """
    import os
    if not path.exists():
        return 0
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip("'\"")
        if k and k not in os.environ:
            os.environ[k] = v
            n += 1
    return n


_N_ENV = _load_dotenv(Path(__file__).resolve().parent / ".env")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("run_app")

# 启动即声明 AI 能力状态——"密钥到底有没有生效"不许再是谜
import os as _os  # noqa: E402
log.info(".env 已载入 %d 项环境变量", _N_ENV)
if _os.environ.get("ANTHROPIC_API_KEY"):
    log.info("AI 视觉识别: ✅ 已启用 (model=%s, endpoint=%s)",
             _os.environ.get("VISION_MODEL", "claude-sonnet-4-6"),
             _os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com"))
else:
    log.warning("AI 视觉识别: ❌ 未配置 ANTHROPIC_API_KEY —— "
                "OCR 将只用本地引擎（对横拍/模糊照片识别质量明显更差）")


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
