from fastapi import APIRouter

from app.schemas.alert import AlertEvent, AlertOverview
from app.services.alert_service import AlertService

router = APIRouter()
service = AlertService()


@router.get("/overview", response_model=AlertOverview)
async def get_alert_overview() -> AlertOverview:
    return service.overview()


@router.get("/timeline", response_model=list[AlertEvent])
async def get_alert_timeline() -> list[AlertEvent]:
    return service.timeline()
