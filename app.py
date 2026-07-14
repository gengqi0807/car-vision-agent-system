"""
交警手势识别 FastAPI 后端服务
启动命令: uvicorn app:app --host 0.0.0.0 --port 8000
"""
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from ctpgr_engine import CTPGREngine

# --- FastAPI 应用 ---
app = FastAPI(title="交警手势识别服务")

# CORS 跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 全局引擎（服务启动时加载一次） ---
print("[app] 正在初始化 CTPGREngine...")
engine = CTPGREngine()
print("[app] 引擎初始化完成。")


@app.post("/api/v1/police-gesture/current")
async def recognize_gesture(file: UploadFile = File(...)):
    """
    接收一帧图片，返回手势识别结果。

    - **file**: 上传的图片文件（jpg / png 等）
    """
    # 读取上传图片的字节流 → numpy 数组 → 解码为 BGR
    contents = await file.read()
    np_arr = np.frombuffer(contents, dtype=np.uint8)
    frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame_bgr is None:
        return {"error": "无法解码图片，请检查文件格式"}

    # 推理
    result = engine.predict_frame(frame_bgr)

    return {
        "gesture": result["gesture"],
        "confidence": result["confidence"],
        "keypoints": result["keypoints"],
    }


@app.post("/api/v1/owner-gesture/current")
async def owner_gesture_dummy():
    """
    虚拟车主手势接口（暂不启用模型推理）。
    立即返回占位 JSON，避免前端等待超时。
    """
    return {"gesture": "车主模式未启用", "keypoints": []}
