from collections import deque
from datetime import datetime
from itertools import count

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agents.notifier import Notifier
from app.core.logger import get_logger
from app.models.alert_log import AlertLog
from app.models.alert_push_log import AlertPushLog
from app.models.user_operation_log import UserOperationLog
from app.schemas.alert import (
    AlertEvent,
    AlertEventCreate,
    AlertOverview,
    AlertPushRecord,
    BehaviorLogRecord,
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
            title=payload.title,
            summary=payload.summary,
        )
        self.db.add(alert)
        self.db.commit()
        self.db.refresh(alert)

        event = AlertEvent.model_validate(alert)
        await self._record_broadcast(event)
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
        return [AlertEvent.model_validate(alert) for alert in alerts]

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
        )

    def list_push_logs(self, *, limit: int = 20) -> list[AlertPushRecord]:
        records = self.db.scalars(
            select(AlertPushLog).order_by(AlertPushLog.created_at.desc()).limit(limit)
        ).all()
        return [AlertPushRecord.model_validate(record) for record in records]

    def list_behavior_logs(self, *, limit: int = 12) -> list[BehaviorLogRecord]:
        return list(_behavior_log_store)[:limit]

    def record_behavior(
        self,
        *,
        source: str,
        title: str,
        summary: str,
        created_at: datetime | None = None,
    ) -> BehaviorLogRecord:
        record = BehaviorLogRecord(
            id=next(_behavior_log_ids),
            source=source,
            title=title,
            summary=summary,
            created_at=created_at or datetime.utcnow(),
        )
        _behavior_log_store.appendleft(record)
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

    async def _record_broadcast(self, event: AlertEvent) -> None:
        success = False
        target = "alerts"

        try:
            delivered = await self.notifier.broadcast({"type": "alert.created", "data": event.model_dump(mode="json")})
            success = delivered > 0
            target = f"alerts:{delivered}"
        except Exception as exc:
            logger.warning("Failed to broadcast alert event %s: %s", event.id, exc)

        self.db.add(
            AlertPushLog(
                channel="websocket",
                target=target,
                success=success,
            )
        )
        self.db.commit()

