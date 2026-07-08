from fastapi import APIRouter

from app.schemas.alert import AlertEvent, AlertOverview
from app.services.alert_service import AlertService

router = APIRouter()
service = AlertService()


@router.get(
    "/overview",
    response_model=AlertOverview,
    summary="获取告警总览",
    description="返回告警总数、严重级别分布和最新告警列表。",
    responses={200: {"description": "查询成功"}},
)
async def get_alert_overview() -> AlertOverview:
    return service.overview()


@router.get(
    "/timeline",
    response_model=list[AlertEvent],
    summary="获取告警时间线",
    description="返回按时间排序的告警事件列表，供仪表盘展示。",
    responses={200: {"description": "查询成功"}},
)
async def get_alert_timeline() -> list[AlertEvent]:
    return service.timeline()
