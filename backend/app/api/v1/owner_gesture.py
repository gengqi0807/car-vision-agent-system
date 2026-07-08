"""
车主手势 API v1

端点:
  GET  /current              → 最近一次识别结果
  POST /stream/start         → 启动后端拉流
  POST /stream/stop          → 停止拉流
  WS   /ws                   → 实时 WebSocket 推送
  GET  /stream               → 流状态查询
  GET  /panel                → 控制面板状态（兼容旧接口）

接口预留:
  - POST /stream/start → source 支持 RTSP / 摄像头索引 / 视频文件
  - WS /ws → 前端实时 UI 更新
  - 控制回调 → 车机控制模块（通过 service.register_control_callback）
  - 告警回调 → 告警智能体（通过 service.register_alert_callback）
"""

import asyncio
import json
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.schemas.gesture import (
    ControlPanelState,
    OwnerGestureResult,
    StreamControlRequest,
    StreamState,
)
from app.services.owner_gesture_service import OwnerGestureService

router = APIRouter()
service = OwnerGestureService.instance()

# ----------------------------------------------------------------
# REST 端点
# ----------------------------------------------------------------


@router.get("/current", response_model=OwnerGestureResult)
async def current_owner_gesture() -> OwnerGestureResult:
    """获取最近一次识别结果（无推理时返回 unknown/idle）。"""
    return service.current_result()


@router.get("/stream", response_model=StreamState)
async def get_stream_state() -> StreamState:
    """查询当前拉流状态。"""
    return service.stream_state


@router.post("/stream/start", response_model=StreamState)
async def start_stream(payload: StreamControlRequest) -> StreamState:
    """
    启动后端拉流。

    source 示例:
      - "rtsp://127.0.0.1:8554/test"  (RTSP 流)
      - "0"                             (USB 摄像头索引)
      - "/path/to/video.mp4"            (本地视频文件)
    """
    return service.start(source=payload.source, fps=payload.fps)


@router.post("/stream/stop", response_model=StreamState)
async def stop_stream() -> StreamState:
    """停止后端拉流。"""
    return service.stop()


@router.get("/panel", response_model=ControlPanelState)
async def owner_control_panel() -> ControlPanelState:
    """控制面板状态（兼容旧接口，后续接入真实车机控制模块）。"""
    return ControlPanelState(
        volume=32,
        climate_temperature=24,
        phone_call_active=False,
        current_mode="media",
    )


# ----------------------------------------------------------------
# WebSocket
# ----------------------------------------------------------------


@router.websocket("/ws")
async def owner_gesture_websocket(ws: WebSocket):
    """实时推送识别结果 JSON。"""
    await ws.accept()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    async def _ws_callback(payload: dict[str, Any]) -> None:
        await queue.put(payload)

    service.register_ws_callback(_ws_callback)

    try:
        # 推送消费协程
        async def _consumer() -> None:
            while True:
                payload = await queue.get()
                try:
                    await ws.send_text(json.dumps(payload, ensure_ascii=False))
                except Exception:
                    break

        consumer_task = asyncio.create_task(_consumer())

        # 等待客户端关闭
        while True:
            try:
                await ws.receive_text()
            except WebSocketDisconnect:
                break
            except Exception:
                break

        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass

    finally:
        service.unregister_ws_callback(_ws_callback)
