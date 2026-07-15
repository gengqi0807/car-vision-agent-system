from __future__ import annotations

import json
from typing import Any

from sqlalchemy.orm import Session

from app.agents.alert_agent import AlertAgent
from app.agents.notifier import Notifier
from app.models.monitor_log import MonitorLog


class MonitorService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.alert_agent = AlertAgent(db)
        self.notifier = Notifier()

    async def capture_event(
        self,
        *,
        category: str,
        source: str,
        event_type: str,
        title: str,
        summary: str,
        level: str = "info",
        status: str | None = None,
        trace_id: str | None = None,
        user_id: int | None = None,
        confidence: float | None = None,
        details: dict[str, Any] | None = None,
        trigger_alert: bool = True,
    ) -> MonitorLog:
        entry = MonitorLog(
            category=category,
            source=source,
            event_type=event_type,
            level=level,
            title=title,
            summary=summary,
            status=status,
            trace_id=trace_id,
            user_id=user_id,
            confidence=confidence,
            details_json=json.dumps(details, ensure_ascii=False) if details else None,
        )
        self.db.add(entry)
        self.db.commit()
        self.db.refresh(entry)

        self.notifier.notify_monitor_log(
            {
                "level": level,
                "source": source,
                "event_type": event_type,
                "title": title,
                "summary": summary,
                "status": status,
                "confidence": confidence,
                "details": details,
            }
        )

        if trigger_alert:
            await self.alert_agent.observe(entry, details=details)
            self.db.refresh(entry)

        return entry
