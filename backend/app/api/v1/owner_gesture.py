from fastapi import APIRouter, File, UploadFile

from app.schemas.gesture import ControlPanelState, GestureFrameResult
from app.services.owner_gesture_service import OwnerGestureService

router = APIRouter()
service = OwnerGestureService()


@router.post("/current", response_model=GestureFrameResult)
async def current_owner_gesture(file: UploadFile = File(...)) -> GestureFrameResult:
    """Upload a hand-gesture image frame for real MediaPipe Hands inference.

    Accepts ``multipart/form-data`` with a single ``file`` field.
    """
    image_bytes = await file.read()
    return await service.process_frame(image_bytes, file.filename or "upload.jpg")


@router.get("/panel", response_model=ControlPanelState)
async def owner_control_panel() -> ControlPanelState:
    return service.control_panel()
