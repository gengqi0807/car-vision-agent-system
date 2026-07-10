import asyncio
import json
from typing import Any

from fastapi import APIRouter, Depends, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.database import get_db
from app.models.user import User
from app.schemas.gesture import (
    ControlPanelState,
    GestureFrameResult,
    OwnerGestureResult,
    StreamControlRequest,
    StreamState,
)
from app.services.owner_gesture_service import OwnerGestureService

router = APIRouter()
service = OwnerGestureService.instance()


@router.post(
    "/current",
    response_model=GestureFrameResult,
    summary="上传图片进行车主手势识别",
    description="接收单张手部图片并返回手势类别、置信度与关键点。",
    responses={200: {"description": "识别成功"}},
)
async def current_owner_gesture(
    file: UploadFile = File(...),
    session_id: str | None = Form(default=None),
    input_mode: str = Form(default="camera"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> GestureFrameResult:
    image_bytes = await file.read()
    return await service.process_frame(
        image_bytes,
        file.filename or "upload.jpg",
        db=db,
        user_id=current_user.id,
        session_id=session_id,
        input_mode=input_mode,
    )


@router.get("/current", response_model=OwnerGestureResult)
async def latest_owner_gesture() -> OwnerGestureResult:
    return service.current_stream_result()


@router.get("/stream", response_model=StreamState)
async def get_stream_state() -> StreamState:
    return service.stream_state


@router.post("/stream/start", response_model=StreamState)
async def start_stream(payload: StreamControlRequest) -> StreamState:
    return service.start(source=payload.source, fps=payload.fps)


@router.post("/stream/stop", response_model=StreamState)
async def stop_stream() -> StreamState:
    return service.stop()


@router.get(
    "/panel",
    response_model=ControlPanelState,
    summary="获取模拟控车面板状态",
    description="返回当前模拟车辆控制面板状态，用于前端展示手势控制效果。",
    responses={200: {"description": "查询成功"}},
)
async def owner_control_panel(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ControlPanelState:
    return service.control_panel(db, current_user.id)


@router.websocket("/ws")
async def owner_gesture_websocket(ws: WebSocket):
    await ws.accept()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def _ws_callback(payload: dict[str, Any]) -> None:
        await queue.put(payload)

    service.register_ws_callback(_ws_callback)

    async def _consumer() -> None:
        while True:
            payload = await queue.get()
            await ws.send_text(json.dumps(payload, ensure_ascii=False))

    consumer_task = asyncio.create_task(_consumer())

    try:
        while True:
            try:
                await ws.receive_text()
            except WebSocketDisconnect:
                break
    finally:
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass
        service.unregister_ws_callback(_ws_callback)
