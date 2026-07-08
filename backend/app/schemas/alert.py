from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

AlertLevel = Literal["critical", "warning", "info"]


class AlertEventCreate(BaseModel):
    level: AlertLevel
    source: str = Field(min_length=2, max_length=64)
    title: str = Field(min_length=2, max_length=128)
    summary: str = Field(min_length=2, max_length=2000)
    event_type: str | None = Field(default=None, max_length=64)
    impact_scope: str | None = Field(default=None, max_length=255)
    root_cause: str | None = Field(default=None, max_length=2000)
    suggested_action: str | None = Field(default=None, max_length=2000)
    analysis: dict[str, Any] | None = None


class AlertEvent(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    level: AlertLevel
    source: str
    event_type: str | None = None
    title: str
    summary: str
    impact_scope: str | None = None
    root_cause: str | None = None
    suggested_action: str | None = None
    analysis: dict[str, Any] | None = None
    created_at: datetime


class AlertOverview(BaseModel):
    total: int
    critical: int
    warning: int
    info: int
    latest: list[AlertEvent]
    source_breakdown: list["MetricPoint"] = Field(default_factory=list)
    root_cause_breakdown: list["MetricPoint"] = Field(default_factory=list)
    notification_breakdown: list["MetricPoint"] = Field(default_factory=list)


class AlertPushRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    channel: str
    target: str
    success: bool
    created_at: datetime


class BehaviorLogRecord(BaseModel):
    id: int
    source: str
    title: str
    summary: str
    created_at: datetime


class OperationLogRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    operation_type: str
    response_status: str | None = None
    created_at: datetime


class MonitorLogRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    category: str
    source: str
    event_type: str
    level: str
    title: str
    summary: str
    status: str | None = None
    trace_id: str | None = None
    user_id: int | None = None
    alert_id: int | None = None
    confidence: float | None = None
    details: dict[str, Any] | None = None
    created_at: datetime


class MetricPoint(BaseModel):
    label: str
    value: int


class AlertReplay(BaseModel):
    alert: AlertEvent
    related_logs: list[MonitorLogRecord]
    push_logs: list[AlertPushRecord]
    reason_summary: str


class AlertDashboard(BaseModel):
    total_logs: int
    alert_overview: AlertOverview
    latest_alerts: list[AlertEvent]
    latest_logs: list[MonitorLogRecord]
    latest_operations: list[OperationLogRecord]
    top_sources: list[MetricPoint]
    top_event_types: list[MetricPoint]


AlertOverview.model_rebuild()
