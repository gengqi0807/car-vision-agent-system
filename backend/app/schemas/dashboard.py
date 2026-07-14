from datetime import datetime

from pydantic import BaseModel, Field


class DashboardCounts(BaseModel):
    plates: int = 0
    police_gestures: int = 0
    owner_gestures: int = 0
    alerts: int = 0


class DashboardTrendPoint(BaseModel):
    date: str
    label: str
    plates: int = 0
    police_gestures: int = 0
    owner_gestures: int = 0
    total: int = 0


class DashboardAlert(BaseModel):
    id: int
    level: str
    title: str
    summary: str
    created_at: datetime


class DashboardOverview(BaseModel):
    counts: DashboardCounts
    trend: list[DashboardTrendPoint] = Field(default_factory=list)
    latest_alerts: list[DashboardAlert] = Field(default_factory=list)
