from fastapi import APIRouter

from app.schemas.gesture import ControlPanelState, GestureFrameResult
from app.services.owner_gesture_service import OwnerGestureService

router = APIRouter()
service = OwnerGestureService()


@router.get("/current", response_model=GestureFrameResult)
async def current_owner_gesture() -> GestureFrameResult:
    return service.current_result()


@router.get("/panel", response_model=ControlPanelState)
async def owner_control_panel() -> ControlPanelState:
    return service.control_panel()
