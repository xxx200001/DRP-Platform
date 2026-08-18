#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DRP 病情预测平台 · 一键穿透启动脚本
自动启动本地服务并拉起 Cloudflare Tunnel 公网穿透，输出手机可直接访问的 HTTPS 链接与二维码。
"""

import os
import re
import sys
import time
import shutil
import signal
import socket
import urllib.request
import webbrowser
import subprocess
from pathlib import Path

PORT = 8000
LOCAL_URL = f"http://127.0.0.1:{PORT}"
ROOT_DIR = Path(__file__).resolve().parent


def is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.6)
        return s.connect_ex((host, port)) == 0


def find_cloudflared() -> str | None:
    path = shutil.which("cloudflared")
    if path:
        return path
    # 常用 Windows 路径兜底
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/WinGet/Links/cloudflared.exe",
        Path("C:/Program Files/cloudflared/cloudflared.exe"),
        Path("C:/Program Files (x86)/cloudflared/cloudflared.exe"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def main():
    print("=" * 64)
    print("      DRP 病情预测平台 · 内网穿透启动程序")
    print("=" * 64)

    procs = []

    def cleanup(sig=None, frame=None):
        print("\n正在停止所有后台服务...")
        for p in procs:
            try:
                p.terminate()
                p.wait(timeout=2)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        print("所有服务已安全退出。")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # 1. 检查或启动本地服务
    if is_port_open(PORT):
        print(f"[1/2] 检测到本地服务已在运行: {LOCAL_URL}")
    else:
        print(f"[1/2] 正在启动本地 DRP 预测平台服务 (端口 {PORT})...")
        py_exe = sys.executable
        srv_cmd = [py_exe, str(ROOT_DIR / "run_app.py"), "--host", "0.0.0.0", "--port", str(PORT)]
        srv_proc = subprocess.Popen(
            srv_cmd,
            cwd=str(ROOT_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        procs.append(srv_proc)

        # 等待服务端口就绪
        for _ in range(60):
            if is_port_open(PORT):
                break
            time.sleep(0.5)
        else:
            print("⚠️ 本地服务启动超时，请检查 run_app.py 是否有错误。")

    # 2. 检查并启动 Cloudflare Tunnel
    cf_bin = find_cloudflared()
    if not cf_bin:
        print("\n❌ 未找到 cloudflared 可执行文件。")
        print("请在终端运行以下命令安装：")
        print("  Windows: winget install --id Cloudflare.cloudflared")
        print("  或下载: https://github.com/cloudflare/cloudflared/releases/latest")
        if not is_port_open(PORT):
            print(f"\n你仍可使用本地地址访问: {LOCAL_URL}")
        input("\n按回车键退出...")
        return

    print(f"[2/2] 正在创建公网安全隧道 (Cloudflare Tunnel)...")
    tunnel_cmd = [cf_bin, "tunnel", "--url", LOCAL_URL]
    tunnel_proc = subprocess.Popen(
        tunnel_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    procs.append(tunnel_proc)

    tunnel_url = None
    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

    # 读取输出匹配穿透公网地址
    start_time = time.time()
    while time.time() - start_time < 30:
        line = tunnel_proc.stderr.readline()
        if not line and tunnel_proc.poll() is not None:
            break
        match = url_pattern.search(line)
        if match:
            tunnel_url = match.group(0)
            break

    if tunnel_url:
        print("\n" + "=" * 64)
        print(" 🎉 穿透成功！手机与外网均可直接访问：")
        print("=" * 64)
        print(f"  📱 手机/公网访问地址:  {tunnel_url}")
        print(f"  💻 电脑本地访问地址:    {LOCAL_URL}")
        print(f"  📖 API 文档 (Swagger):  {tunnel_url}/docs")
        print("=" * 64)
        print("提示：手机连接 4G/5G 或任意 Wi-Fi，用手机浏览器打开上方链接")
        print("     即可使用【手机相册选图】或【拍照】进行化验单智能识别与风险预测。")
        print("=" * 64 + "\n")

        # 尝试打印终端二维码
        try:
            import qrcode
            qr = qrcode.QRCode(border=1)
            qr.add_data(tunnel_url)
            qr.print_ascii(invert=True)
            print()
        except ImportError:
            pass

        # 自动在浏览器打开
        try:
            webbrowser.open(tunnel_url)
        except Exception:
            pass

        print("服务正在持续运行中 (按 Ctrl + C 可停止服务)...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            cleanup()
    else:
        print("❌ 获取穿透公网地址失败，请检查网络连接。")
        print(f"本地服务仍可正常使用: {LOCAL_URL}")
        input("\n按回车键退出...")


if __name__ == "__main__":
    main()
