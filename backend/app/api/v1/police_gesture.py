import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File

from app.schemas.gesture import GestureFrameResult, GestureHistoryItem, Keypoint
from app.services.police_gesture_service import PoliceGestureService

# 将项目根目录加入 sys.path，以便导入 CTPGREngine
PROJECT_ROOT = Path(__file__).resolve().parents[4]  # backend/app/api/v1 -> 项目根
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ctpgr_engine import CTPGREngine

router = APIRouter()
service = PoliceGestureService()

# 全局引擎（启动时立即初始化）
print("[police_gesture] 正在初始化 CTPGREngine（启动时加载）...")
_engine = CTPGREngine()
print("[police_gesture] ✅ CTPGREngine 初始化完成")


def _get_engine() -> CTPGREngine:
    return _engine


@router.post("/current", response_model=GestureFrameResult)
async def current_police_gesture(file: UploadFile = File(...)) -> GestureFrameResult:
    """接收一帧图片，调用 CTPGREngine 进行交警手势识别。"""
    print("收到图片请求")

    contents = await file.read()
    np_arr = np.frombuffer(contents, dtype=np.uint8)
    frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame_bgr is None:
        print("图片解码失败")
        return GestureFrameResult(
            gesture="图片解码失败",
            confidence=0.0,
            keypoints=[],
            updated_at=datetime.now(),
        )

    print(f"图片尺寸: {frame_bgr.shape}")

    print("开始推理...")
    engine = _get_engine()
    result = engine.predict_frame(frame_bgr)

    print("========= 推理结果 =========")
    print(f"手势: {result['gesture']}")
    print(f"置信度: {result['confidence']}")
    print(f"关键点数量: {len(result.get('keypoints', []))}")
    if result.get('keypoints'):
        print(f"第一个关键点: {result['keypoints'][0]}")
    print("============================")

    return GestureFrameResult(
        gesture=result["gesture"],
        confidence=result["confidence"],
        keypoints=[Keypoint(x=kp["x"], y=kp["y"], score=1.0) for kp in result["keypoints"]],
        updated_at=datetime.now(),
    )


@router.get("/history", response_model=list[GestureHistoryItem])
async def police_gesture_history() -> list[GestureHistoryItem]:
    return service.history()
