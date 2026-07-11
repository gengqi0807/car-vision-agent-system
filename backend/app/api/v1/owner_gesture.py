import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File

from app.schemas.gesture import ControlPanelState, GestureFrameResult, Keypoint
from app.services.owner_gesture_service import OwnerGestureService

# 将项目根目录加入 sys.path，以便导入 CTPGREngine
PROJECT_ROOT = Path(__file__).resolve().parents[4]  # backend/app/api/v1 -> 项目根
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ctpgr_engine import CTPGREngine

router = APIRouter()
service = OwnerGestureService()

# 全局引擎（启动时立即初始化）
print("[owner_gesture] 正在初始化 CTPGREngine（启动时加载）...")
_engine = CTPGREngine()
print("[owner_gesture] ✅ CTPGREngine 初始化完成")


@router.get("/current", response_model=GestureFrameResult)
async def current_owner_gesture() -> GestureFrameResult:
    return service.current_result()


@router.post("/current", response_model=GestureFrameResult)
async def current_owner_gesture_post(file: UploadFile = File(...)) -> GestureFrameResult:
    """接收一帧图片，调用 CTPGREngine 进行车主手势识别。"""
    contents = await file.read()
    np_arr = np.frombuffer(contents, dtype=np.uint8)
    frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame_bgr is None:
        return GestureFrameResult(
            gesture="图片解码失败",
            confidence=0.0,
            keypoints=[],
            updated_at=datetime.now(),
        )

    result = _engine.predict_frame(frame_bgr)

    return GestureFrameResult(
        gesture=result["gesture"],
        confidence=result["confidence"],
        keypoints=[Keypoint(x=kp["x"], y=kp["y"], score=1.0) for kp in result["keypoints"]],
        updated_at=datetime.now(),
    )


@router.get("/panel", response_model=ControlPanelState)
async def owner_control_panel() -> ControlPanelState:
    return service.control_panel()
