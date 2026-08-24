#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DRP 病情预测平台 · 一键穿透启动程序
自动启动本地后端服务并拉起 Cloudflare Tunnel 公网穿透，自动输出手机可直接访问的 HTTPS 链接与二维码。
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

# 确保在 Windows 控制台或管道输出时 UTF-8 正常显示
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

PORT = 8000
LOCAL_URL = f"http://127.0.0.1:{PORT}"
ROOT_DIR = Path(__file__).resolve().parent


def is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.6)
        return s.connect_ex((host, port)) == 0


def find_cloudflared() -> str | None:
    # 1. 优先使用本地仓库捆绑的 bin/cloudflared.exe
    local_bin = ROOT_DIR / "bin" / ("cloudflared.exe" if sys.platform == "win32" else "cloudflared")
    if local_bin.exists():
        return str(local_bin)

    # 2. 检查系统 PATH
    path = shutil.which("cloudflared")
    if path:
        return path

    # 3. 常用 Windows 默认路径
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/WinGet/Links/cloudflared.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/cloudflared/cloudflared.exe",
        Path("C:/Program Files/cloudflared/cloudflared.exe"),
        Path("C:/Program Files (x86)/cloudflared/cloudflared.exe"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def ensure_cloudflared() -> str | None:
    cf = find_cloudflared()
    if cf:
        return cf

    print("[提示] 未检测到穿透组件，正在自动下载适配二进制...")
    local_bin = ROOT_DIR / "bin" / ("cloudflared.exe" if sys.platform == "win32" else "cloudflared")
    local_bin.parent.mkdir(parents=True, exist_ok=True)

    urls = [
        "https://ghproxy.net/https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe",
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe",
    ]
    for url in urls:
        try:
            print(f"正在从 {url} 下载...")
            urllib.request.urlretrieve(url, str(local_bin))
            if sys.platform != "win32":
                os.chmod(str(local_bin), 0o755)
            print("✅ 穿透组件下载完成！")
            return str(local_bin)
        except Exception as e:
            print(f"下载失败 ({e})，尝试备用地址...")
            continue
    return None


def main():
    print("=" * 64)
    print("      DRP 病情预测平台 · 一键穿透与服务启动")
    print("=" * 64)

    procs = []

    def cleanup(sig=None, frame=None):
        print("\n正在安全退出所有后台进程...")
        for p in procs:
            try:
                p.terminate()
                p.wait(timeout=2)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        print("所有服务已安全关闭。")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # 1. 检查或自动启动本地后端
    if is_port_open(PORT):
        print(f"[1/2] 检测到本地预测服务已在运行: {LOCAL_URL}")
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

        # 等待服务端口响应
        for _ in range(60):
            if is_port_open(PORT):
                break
            time.sleep(0.5)
        else:
            print("⚠️ 本地服务启动超时，请检查 run_app.py 是否有错误。")

    # 2. 检查或下载 Cloudflare Tunnel
    cf_bin = ensure_cloudflared()
    if not cf_bin:
        print("\n❌ 无法获取 cloudflared 穿透组件。")
        print("请在终端运行：winget install --id Cloudflare.cloudflared")
        print(f"本地服务仍可正常使用: {LOCAL_URL}")
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

    # 捕获输出中的穿透域名
    start_time = time.time()
    while time.time() - start_time < 35:
        line = tunnel_proc.stderr.readline()
        if not line and tunnel_proc.poll() is not None:
            break
        match = url_pattern.search(line)
        if match:
            tunnel_url = match.group(0)
            break

    if tunnel_url:
        # 保存公网 URL 到本地文件
        try:
            (ROOT_DIR / "tunnel_url.txt").write_text(tunnel_url, encoding="utf-8")
        except Exception:
            pass

        print("\n" + "=" * 64, flush=True)
        print(" 🎉 穿透成功！手机与外网均可直接访问：", flush=True)
        print("=" * 64, flush=True)
        print(f"  📱 手机/公网访问地址:  {tunnel_url}", flush=True)
        print(f"  💻 电脑本地访问地址:    {LOCAL_URL}", flush=True)
        print(f"  📖 API 接口文档:        {tunnel_url}/docs", flush=True)
        print("=" * 64, flush=True)
        print("说明：手机使用任意网络打开上方链接，即可拍照/上传相册化验单并进行预测。", flush=True)
        print("=" * 64 + "\n", flush=True)

        # 尝试输出终端二维码
        try:
            import qrcode
            qr = qrcode.QRCode(border=1)
            qr.add_data(tunnel_url)
            qr.print_ascii(invert=True)
            print()
        except ImportError:
            pass

        # 自动弹出浏览器
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
