from fastapi import APIRouter

from app.schemas.gesture import GestureFrameResult, GestureHistoryItem
from app.services.police_gesture_service import PoliceGestureService

router = APIRouter()
service = PoliceGestureService()


@router.get("/current", response_model=GestureFrameResult)
async def current_police_gesture() -> GestureFrameResult:
    return service.current_result()


@router.get("/history", response_model=list[GestureHistoryItem])
async def police_gesture_history() -> list[GestureHistoryItem]:
    return service.history()
