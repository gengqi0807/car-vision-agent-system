from fastapi import APIRouter, File, UploadFile

from app.schemas.gesture import GestureFrameResult, GestureHistoryItem
from app.services.police_gesture_service import PoliceGestureService

router = APIRouter()
service = PoliceGestureService()


@router.post(
    "/current",
    response_model=GestureFrameResult,
    summary="上传图片进行交警手势识别",
    description="接收单张交警姿态图片并返回手势类别、置信度与骨骼关键点。",
    responses={200: {"description": "识别成功"}},
)
async def current_police_gesture(file: UploadFile = File(...)) -> GestureFrameResult:
    """Upload a police-pose image frame for real MediaPipe Pose inference.

    Accepts ``multipart/form-data`` with a single ``file`` field.
    """
    image_bytes = await file.read()
    return await service.process_frame(image_bytes, file.filename or "upload.jpg")


@router.get(
    "/history",
    response_model=list[GestureHistoryItem],
    summary="查询交警手势历史",
    description="返回交警手势识别历史记录或样例结果。",
    responses={200: {"description": "查询成功"}},
)
async def police_gesture_history() -> list[GestureHistoryItem]:
    return service.history()
