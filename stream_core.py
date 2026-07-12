import subprocess
import time
import os
import cv2


class StreamManager:
    def __init__(self):
        self.mediamtx_path = r"D:\py_projects\mediamtx\mediamtx.exe"
        self.ffmpeg_path = r"D:\py_projects\ffmpeg\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"
        if not os.path.exists(self.ffmpeg_path):
            raise FileNotFoundError(f"FFmpeg 路径不存在，请修改 self.ffmpeg_path: {self.ffmpeg_path}")

    def start_mediamtx(self):
        self.mediamtx_process = subprocess.Popen(
            self.mediamtx_path,
            cwd=os.path.dirname(self.mediamtx_path),
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        time.sleep(1.5)
        if self.mediamtx_process.poll() is None:
            print("✅ MediaMTX 启动成功，PID为:", self.mediamtx_process.pid)
        else:
            stderr_output = self.mediamtx_process.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"MediaMTX 启动失败，错误原因:\n{stderr_output}")

    def start_push(self, video_source, stream_name="carview"):
        rtsp_url = f"rtsp://127.0.0.1:8554/{stream_name}"
        ffmpeg_cmd = [
            self.ffmpeg_path,
            "-re",
            "-i", video_source,
            "-c", "copy",
            "-f", "rtsp",
            rtsp_url
        ]
        self.ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        print(f"✅ FFmpeg 开始推流，地址: {rtsp_url}")

    def test_pull_stream(self, stream_name="test"):
        rtsp_url = f"rtsp://127.0.0.1:8554/{stream_name}"
        cap = cv2.VideoCapture(rtsp_url)
        ret, frame = cap.read()
        if ret:
            print("✅ 成功拉取到一帧画面！画面尺寸为：" + str(frame.shape))
            cap.release()
        else:
            print("❌ 拉流失败，请检查是否已有视频在推流。")


if __name__ == "__main__":
    manager = StreamManager()
    manager.start_mediamtx()
    time.sleep(2)
    manager.start_push(video_source="test.mp4", stream_name="test")
    print("🚀 推流服务已启动，按 Ctrl+C 停止...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在停止服务...")
        manager.mediamtx_process.terminate()
        manager.ffmpeg_process.terminate()
        manager.mediamtx_process.wait()
        manager.ffmpeg_process.wait()
        print("✅ 所有子进程已终止。")
