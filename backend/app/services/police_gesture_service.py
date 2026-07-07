from datetime import datetime

from app.schemas.gesture import GestureFrameResult, GestureHistoryItem, Keypoint


class PoliceGestureService:
    def current_result(self) -> GestureFrameResult:
        return GestureFrameResult(
            gesture="停止信号",
            confidence=0.88,
            keypoints=[
                Keypoint(x=0.46, y=0.22, score=0.98),
                Keypoint(x=0.51, y=0.34, score=0.97),
            ],
            updated_at=datetime.utcnow(),
        )

    def history(self) -> list[GestureHistoryItem]:
        return [
            GestureHistoryItem(
                gesture="停止信号",
                confidence=0.88,
                updated_at=datetime.utcnow(),
            )
        ]
