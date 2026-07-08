from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.alert import (
    AlertDashboard,
    AlertEvent,
    AlertEventCreate,
    AlertOverview,
    AlertPushRecord,
    AlertReplay,
    BehaviorLogRecord,
    MonitorLogRecord,
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
    db: Session = Depends(get_db),
) -> list[AlertEvent]:
    service = AlertService(db)
    return service.timeline(limit=limit, level=level, source=source)


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
    db: Session = Depends(get_db),
) -> list[MonitorLogRecord]:
    service = AlertService(db)
    return service.list_monitor_logs(limit=limit, category=category, source=source, level=level)


@router.get("/behavior-logs", response_model=list[BehaviorLogRecord])
async def get_behavior_logs(
    limit: int = Query(default=12, ge=1, le=50),
    db: Session = Depends(get_db),
) -> list[BehaviorLogRecord]:
    service = AlertService(db)
    return service.list_behavior_logs(limit=limit)


@router.get("/operation-logs", response_model=list[OperationLogRecord])
async def get_operation_logs(
    limit: int = Query(default=20, ge=1, le=100),
    user_id: int | None = Query(default=None, ge=1),
    operation_type: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> list[OperationLogRecord]:
    service = AlertService(db)
    return service.list_operation_logs(limit=limit, user_id=user_id, operation_type=operation_type)


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
