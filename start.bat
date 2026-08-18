@echo off
chcp 65001 >nul
title DRP 病情预测平台 - 启动中...
cd /d "%~dp0"

echo ========================================================
echo        DRP 病情预测平台 · 一键启动服务
echo ========================================================
echo.

if not exist ".venv" (
    echo [1/3] 首次运行，正在创建虚拟环境...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    echo [2/3] 正在安装依赖包（首次需要约 1~2 分钟）...
    python -m pip install -U pip
    pip install -e .
    pip install rapidocr-onnxruntime opencv-python httpx
) else (
    call .venv\Scripts\activate.bat
)

echo [3/3] 正在启动预测平台服务...
start "" "http://127.0.0.1:8000"

python run_app.py --host 0.0.0.0 --port 8000

pause
