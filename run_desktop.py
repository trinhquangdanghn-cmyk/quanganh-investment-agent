import sys
import time
import subprocess
import threading
import webview

def start_streamlit():
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.headless=true",
        "--server.port=8501",
        "--global.developmentMode=false"
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    # Chạy Streamlit server ở luồng ẩn
    server_thread = threading.Thread(target=start_streamlit, daemon=True)
    server_thread.start()

    # Đợi 3 giây để server khởi động xong
    time.sleep(3)

    # Khai báo cửa sổ và bật chế độ tự phóng to phù hợp màn hình (maximized=True)
    webview.create_window(
        title="QuangAnh Investment Agent",
        url="http://localhost:8501",
        maximized=True,       # Tự động mở to kín màn hình chính
        resizable=True,
        min_size=(900, 600)
    )

    # Khởi động giao diện pywebview (không truyền tham số thừa)
    webview.start()