from datetime import datetime

from pydantic import BaseModel, ConfigDict


class AlertEvent(BaseModel):
    id: int
    level: str
    source: str
    title: str
    summary: str
    created_at: datetime


class AlertOverview(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total": 12,
                "critical": 2,
                "warning": 4,
                "info": 6,
                "latest": [
                    {
                        "id": 101,
                        "level": "critical",
                        "source": "plate-recognition",
                        "title": "连续识别失败",
                        "summary": "最近 5 分钟内车牌识别连续失败 8 次，请检查模型与输入源。",
                        "created_at": "2026-07-08T12:00:00",
                    }
                ],
            }
        }
    )

    total: int
    critical: int
    warning: int
    info: int
    latest: list[AlertEvent]
