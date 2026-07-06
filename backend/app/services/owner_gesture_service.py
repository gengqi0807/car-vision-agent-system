from datetime import datetime

from app.schemas.gesture import ControlPanelState, GestureFrameResult, Keypoint


class OwnerGestureService:
    def current_result(self) -> GestureFrameResult:
        return GestureFrameResult(
            gesture="手掌张开",
            confidence=0.92,
            keypoints=[
                Keypoint(x=0.42, y=0.18, score=0.99),
                Keypoint(x=0.48, y=0.26, score=0.98),
            ],
            updated_at=datetime.utcnow(),
        )

    def control_panel(self) -> ControlPanelState:
        return ControlPanelState(
            volume=32,
            climate_temperature=24,
            phone_call_active=False,
            current_mode="media",
        )
