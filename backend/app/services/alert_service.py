from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timedelta
from itertools import count

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agents.notifier import Notifier
from app.core.config import settings
from app.core.logger import get_logger
from app.models.alert_log import AlertLog
from app.models.alert_push_log import AlertPushLog
from app.models.monitor_log import MonitorLog
from app.models.user_operation_log import UserOperationLog
from app.schemas.alert import (
    AlertDashboard,
    AlertEvent,
    AlertEventCreate,
    AlertOverview,
    AlertPushRecord,
    AlertReplay,
    BehaviorLogRecord,
    MetricPoint,
    MonitorLogRecord,
    OperationLogRecord,
)

logger = get_logger(__name__)

_behavior_log_store: deque[BehaviorLogRecord] = deque(maxlen=24)
_behavior_log_ids = count(1)


class AlertService:
    def __init__(self, db: Session, notifier: Notifier | None = None):
        self.db = db
        self.notifier = notifier or Notifier()

    async def create_event(self, payload: AlertEventCreate) -> AlertEvent:
        alert = AlertLog(
            level=payload.level,
            source=payload.source,
            event_type=payload.event_type,
            title=payload.title,
            summary=payload.summary,
            impact_scope=payload.impact_scope,
            root_cause=payload.root_cause,
            suggested_action=payload.suggested_action,
            analysis_json=json.dumps(payload.analysis, ensure_ascii=False) if payload.analysis else None,
        )
        self.db.add(alert)
        self.db.commit()
        self.db.refresh(alert)

        event = self._to_alert_event(alert)
        await self._dispatch_notifications(event)
        return event

    def timeline(
        self,
        *,
        limit: int = 20,
        level: str | None = None,
        source: str | None = None,
    ) -> list[AlertEvent]:
        query = select(AlertLog).order_by(AlertLog.created_at.desc())
        if level:
            query = query.where(AlertLog.level == level)
        if source:
            query = query.where(AlertLog.source == source)

        alerts = self.db.scalars(query.limit(limit)).all()
        return [self._to_alert_event(alert) for alert in alerts]

    def overview(self, *, latest_limit: int = 5) -> AlertOverview:
        counts = dict(
            self.db.execute(
                select(AlertLog.level, func.count(AlertLog.id)).group_by(AlertLog.level)
            ).all()
        )
        total = sum(counts.values())
        latest = self.timeline(limit=latest_limit)
        return AlertOverview(
            total=total,
            critical=counts.get("critical", 0),
            warning=counts.get("warning", 0),
            info=counts.get("info", 0),
            latest=latest,
            source_breakdown=self._metric_points(AlertLog.source),
            root_cause_breakdown=self._root_cause_points(),
            notification_breakdown=self._notification_points(),
        )

    def list_push_logs(self, *, limit: int = 20) -> list[AlertPushRecord]:
        records = self.db.scalars(
            select(AlertPushLog).order_by(AlertPushLog.created_at.desc()).limit(limit)
        ).all()
        return [AlertPushRecord.model_validate(record) for record in records]

    def list_behavior_logs(self, *, limit: int = 12) -> list[BehaviorLogRecord]:
        records = self.db.scalars(
            select(MonitorLog)
            .where(MonitorLog.category.in_(["plate", "owner_gesture", "police_gesture"]))
            .order_by(MonitorLog.created_at.desc())
            .limit(limit)
        ).all()
        if records:
            return [
                BehaviorLogRecord(
                    id=record.id,
                    source=record.source,
                    title=record.title,
                    summary=record.summary,
                    created_at=record.created_at,
                )
                for record in records
            ]

        return list(_behavior_log_store)[:limit]

    def record_behavior(
        self,
        *,
        source: str,
        title: str,
        summary: str,
        created_at: datetime | None = None,
    ) -> BehaviorLogRecord:
        timestamp = created_at or datetime.utcnow()
        record = BehaviorLogRecord(
            id=next(_behavior_log_ids),
            source=source,
            title=title,
            summary=summary,
            created_at=timestamp,
        )
        _behavior_log_store.appendleft(record)
        self.db.add(
            MonitorLog(
                category=self._category_for_source(source),
                source=source,
                event_type="behavior_event",
                level="info",
                title=title,
                summary=summary,
                status="recorded",
                created_at=timestamp,
                updated_at=timestamp,
            )
        )
        self.db.commit()
        return record

    def list_operation_logs(
        self,
        *,
        limit: int = 20,
        user_id: int | None = None,
        operation_type: str | None = None,
    ) -> list[OperationLogRecord]:
        query = select(UserOperationLog).order_by(UserOperationLog.created_at.desc())
        if user_id is not None:
            query = query.where(UserOperationLog.user_id == user_id)
        if operation_type:
            query = query.where(UserOperationLog.operation_type == operation_type)

        records = self.db.scalars(query.limit(limit)).all()
        return [OperationLogRecord.model_validate(record) for record in records]

    def list_monitor_logs(
        self,
        *,
        limit: int = 30,
        category: str | None = None,
        source: str | None = None,
        level: str | None = None,
    ) -> list[MonitorLogRecord]:
        query = select(MonitorLog).order_by(MonitorLog.created_at.desc())
        if category:
            query = query.where(MonitorLog.category == category)
        if source:
            query = query.where(MonitorLog.source == source)
        if level:
            query = query.where(MonitorLog.level == level)
        records = self.db.scalars(query.limit(limit)).all()
        return [self._to_monitor_log(record) for record in records]

    def replay(self, alert_id: int) -> AlertReplay:
        alert = self.db.get(AlertLog, alert_id)
        if alert is None:
            raise ValueError("Alert event not found")

        since = alert.created_at - timedelta(minutes=settings.alert_replay_window_minutes)
        until = alert.created_at + timedelta(minutes=5)

        related_logs = self.db.scalars(
            select(MonitorLog)
            .where(
                MonitorLog.source == alert.source,
                MonitorLog.created_at >= since,
                MonitorLog.created_at <= until,
            )
            .order_by(MonitorLog.created_at.desc())
            .limit(20)
        ).all()

        push_logs = self.db.scalars(
            select(AlertPushLog)
            .where(AlertPushLog.created_at >= since, AlertPushLog.created_at <= until)
            .order_by(AlertPushLog.created_at.desc())
            .limit(20)
        ).all()

        reason_summary = alert.root_cause or "No root cause summary available."
        if related_logs:
            reason_summary = (
                f"{reason_summary} Related runtime evidence count: {len(related_logs)}."
            )

        return AlertReplay(
            alert=self._to_alert_event(alert),
            related_logs=[self._to_monitor_log(record) for record in related_logs],
            push_logs=[AlertPushRecord.model_validate(record) for record in push_logs],
            reason_summary=reason_summary,
        )

    def dashboard(self, *, latest_limit: int = 6, log_limit: int = 12) -> AlertDashboard:
        total_logs = int(self.db.scalar(select(func.count(MonitorLog.id))) or 0)
        return AlertDashboard(
            total_logs=total_logs,
            alert_overview=self.overview(latest_limit=latest_limit),
            latest_alerts=self.timeline(limit=latest_limit),
            latest_logs=self.list_monitor_logs(limit=log_limit),
            latest_operations=self.list_operation_logs(limit=log_limit),
            top_sources=self._metric_points(MonitorLog.source, model=MonitorLog),
            top_event_types=self._metric_points(MonitorLog.event_type, model=MonitorLog),
        )

    async def _dispatch_notifications(self, event: AlertEvent) -> None:
        try:
            results = await self.notifier.notify_alert(event.model_dump(mode="json"))
        except Exception as exc:
            logger.warning("Failed to dispatch alert event %s: %s", event.id, exc)
            results = [
                {
                    "channel": "websocket",
                    "target": "alerts:0",
                    "success": False,
                }
            ]

        for result in results:
            self.db.add(
                AlertPushLog(
                    channel=result["channel"],
                    target=result["target"],
                    success=result["success"],
                )
            )
        self.db.commit()

    def _to_alert_event(self, alert: AlertLog) -> AlertEvent:
        analysis = None
        if alert.analysis_json:
            try:
                analysis = json.loads(alert.analysis_json)
            except json.JSONDecodeError:
                analysis = {"raw": alert.analysis_json}

        return AlertEvent(
            id=alert.id,
            level=alert.level,
            source=alert.source,
            event_type=alert.event_type,
            title=alert.title,
            summary=alert.summary,
            impact_scope=alert.impact_scope,
            root_cause=alert.root_cause,
            suggested_action=alert.suggested_action,
            analysis=analysis,
            created_at=alert.created_at,
        )

    def _to_monitor_log(self, record: MonitorLog) -> MonitorLogRecord:
        details = None
        if record.details_json:
            try:
                details = json.loads(record.details_json)
            except json.JSONDecodeError:
                details = {"raw": record.details_json}

        return MonitorLogRecord(
            id=record.id,
            category=record.category,
            source=record.source,
            event_type=record.event_type,
            level=record.level,
            title=record.title,
            summary=record.summary,
            status=record.status,
            trace_id=record.trace_id,
            user_id=record.user_id,
            alert_id=record.alert_id,
            confidence=record.confidence,
            details=details,
            created_at=record.created_at,
        )

    def _metric_points(self, column, *, model=AlertLog, limit: int = 6) -> list[MetricPoint]:
        rows = self.db.execute(
            select(column, func.count(model.id))
            .where(column.is_not(None))
            .group_by(column)
            .order_by(func.count(model.id).desc())
            .limit(limit)
        ).all()
        return [MetricPoint(label=str(label), value=int(value)) for label, value in rows if label]

    def _root_cause_points(self, limit: int = 6) -> list[MetricPoint]:
        rows = self.db.execute(
            select(AlertLog.root_cause, func.count(AlertLog.id))
            .where(AlertLog.root_cause.is_not(None))
            .group_by(AlertLog.root_cause)
            .order_by(func.count(AlertLog.id).desc())
            .limit(limit)
        ).all()
        return [MetricPoint(label=str(label), value=int(value)) for label, value in rows if label]

    def _notification_points(self, limit: int = 6) -> list[MetricPoint]:
        rows = self.db.execute(
            select(AlertPushLog.channel, func.count(AlertPushLog.id))
            .group_by(AlertPushLog.channel)
            .order_by(func.count(AlertPushLog.id).desc())
            .limit(limit)
        ).all()
        return [MetricPoint(label=str(label), value=int(value)) for label, value in rows if label]

    def _category_for_source(self, source: str) -> str:
        if source == "plate-recognition":
            return "plate"
        if source == "owner-gesture":
            return "owner_gesture"
        if source == "police-gesture":
            return "police_gesture"
        if source == "auth":
            return "user_operation"
        return "system"
