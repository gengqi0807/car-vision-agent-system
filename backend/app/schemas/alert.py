from datetime import datetime

from pydantic import BaseModel


class AlertEvent(BaseModel):
    id: int
    level: str
    source: str
    title: str
    summary: str
    created_at: datetime


class AlertOverview(BaseModel):
    total: int
    critical: int
    warning: int
    info: int
    latest: list[AlertEvent]
