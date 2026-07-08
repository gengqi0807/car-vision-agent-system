from fastapi import APIRouter, File, UploadFile

from app.schemas.gesture import ControlPanelState, GestureFrameResult
from app.services.owner_gesture_service import OwnerGestureService

router = APIRouter()
service = OwnerGestureService()


@router.post(
    "/current",
    response_model=GestureFrameResult,
    summary="上传图片进行车主手势识别",
    description="接收单张手部图片并返回手势类别、置信度与关键点。",
    responses={200: {"description": "识别成功"}},
)
async def current_owner_gesture(file: UploadFile = File(...)) -> GestureFrameResult:
    """Upload a hand-gesture image frame for real MediaPipe Hands inference.

    Accepts ``multipart/form-data`` with a single ``file`` field.
    """
    image_bytes = await file.read()
    return await service.process_frame(image_bytes, file.filename or "upload.jpg")


@router.get(
    "/panel",
    response_model=ControlPanelState,
    summary="获取模拟控车面板状态",
    description="返回当前模拟车辆控制面板状态，用于前端展示手势控制效果。",
    responses={200: {"description": "查询成功"}},
)
async def owner_control_panel() -> ControlPanelState:
    return service.control_panel()
