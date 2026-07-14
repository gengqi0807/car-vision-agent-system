from datetime import date, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.alert_log import AlertLog
from app.models.owner_gesture_record import OwnerGestureRecord
from app.models.plate_record import PlateRecord
from app.models.police_gesture_record import PoliceGestureRecord
from app.schemas.dashboard import DashboardAlert, DashboardCounts, DashboardOverview, DashboardTrendPoint


class DashboardService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def overview(self, days: int = 7, latest_limit: int = 5) -> DashboardOverview:
        days = max(1, min(days, 31))
        latest_limit = max(1, min(latest_limit, 20))
        counts = DashboardCounts(
            plates=self._count(PlateRecord),
            police_gestures=self._count(PoliceGestureRecord),
            owner_gestures=self._count(OwnerGestureRecord),
            alerts=self._count(AlertLog),
        )

        today = date.today()
        dates = [today - timedelta(days=offset) for offset in range(days - 1, -1, -1)]
        start_at = datetime.combine(dates[0], datetime.min.time())
        plate_counts = self._daily_counts(PlateRecord, start_at)
        police_counts = self._daily_counts(PoliceGestureRecord, start_at)
        owner_counts = self._daily_counts(OwnerGestureRecord, start_at)
        trend = []
        for item_date in dates:
            plates = plate_counts.get(item_date, 0)
            police = police_counts.get(item_date, 0)
            owner = owner_counts.get(item_date, 0)
            trend.append(
                DashboardTrendPoint(
                    date=item_date.isoformat(),
                    label=f"{item_date.month}/{item_date.day}",
                    plates=plates,
                    police_gestures=police,
                    owner_gestures=owner,
                    total=plates + police + owner,
                )
            )

        alerts = self.db.scalars(
            select(AlertLog).order_by(AlertLog.created_at.desc(), AlertLog.id.desc()).limit(latest_limit)
        ).all()
        return DashboardOverview(
            counts=counts,
            trend=trend,
            latest_alerts=[
                DashboardAlert(
                    id=item.id,
                    level=item.level,
                    title=item.title,
                    summary=item.summary,
                    created_at=item.created_at,
                )
                for item in alerts
            ],
        )

    def _count(self, model) -> int:
        return int(self.db.scalar(select(func.count()).select_from(model)) or 0)

    def _daily_counts(self, model, start_at: datetime) -> dict[date, int]:
        created_values = self.db.scalars(select(model.created_at).where(model.created_at >= start_at)).all()
        result: dict[date, int] = {}
        for created_at in created_values:
            if created_at is None:
                continue
            created_date = created_at.date()
            result[created_date] = result.get(created_date, 0) + 1
        return result
