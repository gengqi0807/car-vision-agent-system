from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.schemas.dashboard import DashboardOverview
from app.services.dashboard_service import DashboardService

router = APIRouter()


@router.get("", response_model=DashboardOverview)
async def get_dashboard(
    days: int = Query(default=7, ge=1, le=31),
    latest_limit: int = Query(default=5, ge=1, le=20),
    db: Session = Depends(get_db),
    _current_user: User = Depends(get_current_user),
) -> DashboardOverview:
    return DashboardService(db).overview(days=days, latest_limit=latest_limit)
