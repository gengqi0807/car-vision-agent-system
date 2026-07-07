from fastapi import APIRouter, File, UploadFile

from app.schemas.gesture import GestureFrameResult, GestureHistoryItem
from app.services.police_gesture_service import PoliceGestureService

router = APIRouter()
service = PoliceGestureService()


@router.post("/current", response_model=GestureFrameResult)
async def current_police_gesture(file: UploadFile = File(...)) -> GestureFrameResult:
    """Upload a police-pose image frame for real MediaPipe Pose inference.

    Accepts ``multipart/form-data`` with a single ``file`` field.
    """
    image_bytes = await file.read()
    return await service.process_frame(image_bytes, file.filename or "upload.jpg")


@router.get("/history", response_model=list[GestureHistoryItem])
async def police_gesture_history() -> list[GestureHistoryItem]:
    return service.history()
