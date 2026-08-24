@echo off
chcp 65001 >nul
title DRP 病情预测平台 - 正在开启穿透...
cd /d "%~dp0"

echo ========================================================
echo        DRP 病情预测平台 · 一键开启手机公网穿透
echo ========================================================
echo.

if not exist ".venv" (
    echo 正在创建虚拟环境并安装依赖...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    python -m pip install -U pip
    pip install -e .
    pip install rapidocr-onnxruntime opencv-python httpx qrcode
) else (
    call .venv\Scripts\activate.bat
)

python start_tunnel.py
pause
