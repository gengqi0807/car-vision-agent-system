from datetime import datetime

from app.schemas.alert import AlertEvent, AlertOverview


class AlertService:
    def timeline(self) -> list[AlertEvent]:
        return [
            AlertEvent(
                id=1,
                level="warning",
                source="plate-recognition",
                title="车牌识别置信度持续偏低",
                summary="连续 3 次识别结果低于阈值，建议检查图像清晰度与模型状态。",
                created_at=datetime.utcnow(),
            )
        ]

    def overview(self) -> AlertOverview:
        latest = self.timeline()
        return AlertOverview(total=4, critical=1, warning=2, info=1, latest=latest)
