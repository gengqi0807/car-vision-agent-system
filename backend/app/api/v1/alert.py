from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.alert import (
    AlertDashboard,
    AlertEvent,
    AlertEventPage,
    AlertEventCreate,
    AlertOverview,
    AlertPushRecord,
    AlertReplay,
    BehaviorLogPage,
    BehaviorLogRecord,
    MonitorLogPage,
    MonitorLogRecord,
    OperationLogPage,
    OperationLogRecord,
)
from app.services.alert_service import AlertService
from app.utils.websocket_manager import websocket_manager

router = APIRouter()


@router.post("/events", response_model=AlertEvent, status_code=status.HTTP_201_CREATED)
async def create_alert_event(payload: AlertEventCreate, db: Session = Depends(get_db)) -> AlertEvent:
    service = AlertService(db)
    return await service.create_event(payload)


@router.get("/overview", response_model=AlertOverview)
async def get_alert_overview(
    latest_limit: int = Query(default=5, ge=1, le=20),
    db: Session = Depends(get_db),
) -> AlertOverview:
    service = AlertService(db)
    return service.overview(latest_limit=latest_limit)


@router.get("/timeline", response_model=list[AlertEvent])
async def get_alert_timeline(
    limit: int = Query(default=20, ge=1, le=100),
    level: str | None = Query(default=None),
    source: str | None = Query(default=None),
    keyword: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> list[AlertEvent]:
    service = AlertService(db)
    return service.timeline(limit=limit, level=level, source=source, keyword=keyword)


@router.get("/timeline-page", response_model=AlertEventPage)
async def get_alert_timeline_page(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=5, ge=1, le=10),
    level: str | None = Query(default=None),
    source: str | None = Query(default=None),
    keyword: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> AlertEventPage:
    service = AlertService(db)
    return service.timeline_page(page=page, page_size=page_size, level=level, source=source, keyword=keyword)


@router.get("/push-logs", response_model=list[AlertPushRecord])
async def get_alert_push_logs(
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[AlertPushRecord]:
    service = AlertService(db)
    return service.list_push_logs(limit=limit)


@router.get("/monitor-logs", response_model=list[MonitorLogRecord])
async def get_monitor_logs(
    limit: int = Query(default=30, ge=1, le=100),
    category: str | None = Query(default=None),
    source: str | None = Query(default=None),
    level: str | None = Query(default=None),
    keyword: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> list[MonitorLogRecord]:
    service = AlertService(db)
    return service.list_monitor_logs(limit=limit, category=category, source=source, level=level, keyword=keyword)


@router.get("/monitor-logs-page", response_model=MonitorLogPage)
async def get_monitor_logs_page(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=5, ge=1, le=10),
    category: str | None = Query(default=None),
    source: str | None = Query(default=None),
    level: str | None = Query(default=None),
    keyword: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> MonitorLogPage:
    service = AlertService(db)
    return service.list_monitor_logs_page(
        page=page,
        page_size=page_size,
        category=category,
        source=source,
        level=level,
        keyword=keyword,
    )


@router.get("/behavior-logs", response_model=list[BehaviorLogRecord])
async def get_behavior_logs(
    limit: int = Query(default=12, ge=1, le=50),
    source: str | None = Query(default=None),
    keyword: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> list[BehaviorLogRecord]:
    service = AlertService(db)
    return service.list_behavior_logs(limit=limit, source=source, keyword=keyword)


@router.get("/behavior-logs-page", response_model=BehaviorLogPage)
async def get_behavior_logs_page(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=5, ge=1, le=10),
    source: str | None = Query(default=None),
    keyword: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> BehaviorLogPage:
    service = AlertService(db)
    return service.list_behavior_logs_page(page=page, page_size=page_size, source=source, keyword=keyword)


@router.get("/operation-logs", response_model=list[OperationLogRecord])
async def get_operation_logs(
    limit: int = Query(default=20, ge=1, le=100),
    user_id: int | None = Query(default=None, ge=1),
    operation_type: str | None = Query(default=None),
    keyword: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> list[OperationLogRecord]:
    service = AlertService(db)
    return service.list_operation_logs(limit=limit, user_id=user_id, operation_type=operation_type, keyword=keyword)


@router.get("/operation-logs-page", response_model=OperationLogPage)
async def get_operation_logs_page(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=5, ge=1, le=10),
    user_id: int | None = Query(default=None, ge=1),
    operation_type: str | None = Query(default=None),
    keyword: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> OperationLogPage:
    service = AlertService(db)
    return service.list_operation_logs_page(
        page=page,
        page_size=page_size,
        user_id=user_id,
        operation_type=operation_type,
        keyword=keyword,
    )


@router.get("/replay/{alert_id}", response_model=AlertReplay)
async def get_alert_replay(alert_id: int, db: Session = Depends(get_db)) -> AlertReplay:
    service = AlertService(db)
    try:
        return service.replay(alert_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/dashboard", response_model=AlertDashboard)
async def get_alert_dashboard(
    latest_limit: int = Query(default=6, ge=1, le=20),
    log_limit: int = Query(default=12, ge=1, le=50),
    db: Session = Depends(get_db),
) -> AlertDashboard:
    service = AlertService(db)
    return service.dashboard(latest_limit=latest_limit, log_limit=log_limit)


@router.websocket("/ws")
async def alert_websocket(websocket: WebSocket) -> None:
    await websocket_manager.connect("alerts", websocket)
    await websocket.send_json({"type": "connected", "channel": "alerts"})
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_manager.disconnect("alerts", websocket)
