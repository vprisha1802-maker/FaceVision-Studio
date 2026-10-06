"""
Launcher Script for FaceVision Studio
Runs the Streamlit application and automatically opens it in your default web browser.

Usage in Visual Studio Code Terminal:
    python run.py
"""

import sys
import os
import subprocess
import time
import webbrowser
import socket


def is_port_in_use(port: int) -> bool:
    """Check if the specified port is already open."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0


def find_available_port(start_port: int = 8501) -> int:
    """Find an available port starting from start_port."""
    port = start_port
    while is_port_in_use(port):
        port += 1
    return port


def launch_application():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "app.py")
    
    port = find_available_port(8501)
    url = f"http://localhost:{port}"

    print("=" * 65)
    print("🚀  LAUNCHING FACEVISION STUDIO...")
    print("=" * 65)
    print(f"📁 Project Directory : {script_dir}")
    print(f"🌐 Application URL   : {url}")
    print("✨ Features Loaded   : Viola-Jones, Template Matching, DeepFace, FaceNet")
    print("=" * 65)

    # Pre-open browser in a separate background thread or timer
    def open_browser():
        time.sleep(1.8)
        print(f"\n🌐 Opening {url} automatically in your default browser...")
        webbrowser.open_new(url)

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    # Launch Streamlit process
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        f"--server.port={port}",
        "--server.headless=false",
        "--browser.gatherUsageStats=false"
    ]

    try:
        subprocess.run(cmd, cwd=script_dir)
    except KeyboardInterrupt:
        print("\n👋 FaceVision Studio stopped by user. See you next time!")


if __name__ == "__main__":
    launch_application()
