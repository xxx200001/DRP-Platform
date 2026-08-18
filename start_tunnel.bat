@echo off
chcp 65001 >nul
title DRP Disease Risk Prediction Platform - Tunnel
cd /d "%~dp0"

if not exist ".venv" (
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
