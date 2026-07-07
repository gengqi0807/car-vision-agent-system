import subprocess
import time
import os
import cv2


class StreamManager:
    def __init__(self):
        self.mediamtx_path = r"F:\programming projiect\SmallTerm\sophomore\MediaMIX\mediamtx.exe"
        self.ffmpeg_path = r"F:\programming projiect\SmallTerm\sophomore\ffmpeg\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"
        if not os.path.exists(self.ffmpeg_path):
            raise FileNotFoundError(f"FFmpeg 路径不存在，请修改 self.ffmpeg_path: {self.ffmpeg_path}")

    def start_mediamtx(self):
        process = subprocess.Popen(
            self.mediamtx_path,
            cwd=os.path.dirname(self.mediamtx_path),
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        time.sleep(1.5)
        if process.poll() is None:
            print("✅ MediaMTX 启动成功，PID为:", process.pid)
        else:
            stderr_output = process.stderr.read().decode("utf-8", errors="replace")
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
    time.sleep(3)
    manager.test_pull_stream()
