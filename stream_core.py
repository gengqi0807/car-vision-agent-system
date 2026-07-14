import os
import subprocess
import time
from pathlib import Path

import cv2


class StreamManager:
    def __init__(self, mediamtx_path: str | None = None, ffmpeg_path: str | None = None):
        self.mediamtx_path = self._resolve_tool_path(
            mediamtx_path or os.environ.get("MEDIAMTX_BIN"),
            [
                r"D:\py_projects\mediamtx\mediamtx.exe",
                r"F:\programming projiect\SmallTerm\sophomore\MediaMIX\mediamtx.exe",
            ],
        )
        self.ffmpeg_path = self._resolve_tool_path(
            ffmpeg_path or os.environ.get("FFMPEG_BIN"),
            [
                r"D:\py_projects\ffmpeg\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe",
                r"F:\programming projiect\SmallTerm\sophomore\ffmpeg\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe",
            ],
        )
        if not self.mediamtx_path:
            raise FileNotFoundError("未找到可用的 MediaMTX 可执行文件，请通过 MEDIAMTX_BIN 指定路径。")
        if not self.ffmpeg_path:
            raise FileNotFoundError("未找到可用的 FFmpeg 可执行文件，请通过 FFMPEG_BIN 指定路径。")

    def _resolve_tool_path(self, preferred: str | None, candidates: list[str]) -> str | None:
        ordered_candidates = [preferred] if preferred else []
        ordered_candidates.extend(candidates)
        for item in ordered_candidates:
            if item and Path(item).exists():
                return item
        return preferred or None

    def start_mediamtx(self):
        self.mediamtx_process = subprocess.Popen(
            self.mediamtx_path,
            cwd=os.path.dirname(self.mediamtx_path),
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW,
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
            "-i",
            video_source,
            "-c",
            "copy",
            "-f",
            "rtsp",
            rtsp_url,
        ]
        self.ffmpeg_process = subprocess.Popen(ffmpeg_cmd, creationflags=subprocess.CREATE_NO_WINDOW)
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
