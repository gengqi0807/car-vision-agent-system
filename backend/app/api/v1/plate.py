import asyncio

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect

from app.api.deps import get_current_user
from app.models.user import User
from app.models_infer.errors import PlateInferenceError
from app.schemas.plate import (
    PlateRecognitionResponse,
    PlateRecordSummary,
    PlateStreamControlResponse,
    PlateStreamStartRequest,
    PlateVideoRecognitionResponse,
)
from app.services.plate_push_service import PlatePushService
from app.services.plate_service import PlateService

router = APIRouter()
service = PlateService()
push_service = PlatePushService()


@router.post(
    "/image",
    response_model=PlateRecognitionResponse,
    summary="上传图片进行车牌识别",
    description="接收单张道路场景图片，返回检测到的车牌号码、颜色、置信度和检测框。",
    responses={
        200: {"description": "识别成功"},
        400: {"description": "图片格式不合法或无法解析"},
        401: {"description": "未登录或令牌失效"},
        503: {"description": "推理服务暂时不可用"},
    },
)
async def recognize_plate_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
) -> PlateRecognitionResponse:
    image_bytes = await file.read()
    try:
        return await service.recognize_image_bytes_async(
            image_bytes,
            file.filename or "unknown.jpg",
            save_history=True,
            user_id=current_user.id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PlateInferenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get(
    "/history",
    response_model=list[PlateRecordSummary],
    summary="查询车牌识别历史",
    description="返回当前登录用户的历史车牌识别记录。",
    responses={200: {"description": "查询成功"}, 401: {"description": "未登录或令牌失效"}},
)
async def get_plate_history(current_user: User = Depends(get_current_user)) -> list[PlateRecordSummary]:
    return service.list_history(current_user.id)


@router.post("/video", response_model=PlateVideoRecognitionResponse)
async def recognize_plate_video(
    request: Request,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
) -> PlateVideoRecognitionResponse:
    video_bytes = await file.read()
    try:
        response = await asyncio.to_thread(
            service.recognize_video_bytes,
            video_bytes,
            file.filename or "unknown.mp4",
            save_history=True,
            user_id=current_user.id,
        )
        processed_url = response.processed_video_url
        if processed_url.startswith("/"):
            processed_url = f"{str(request.base_url).rstrip('/')}{processed_url}"
        unread_samples = [
            f"{str(request.base_url).rstrip('/')}{item}" if item.startswith("/") else item
            for item in response.unread_samples
        ]
        return response.model_copy(update={"processed_video_url": processed_url, "unread_samples": unread_samples})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PlateInferenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/stream/start", response_model=PlateStreamControlResponse)
async def start_plate_stream(
    payload: PlateStreamStartRequest,
    _current_user: User = Depends(get_current_user),
) -> PlateStreamControlResponse:
    try:
        return push_service.start(rtsp_url=payload.rtsp_url, stream_name=payload.stream_name)
    except PlateInferenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/stream/stop", response_model=PlateStreamControlResponse)
async def stop_plate_stream(_current_user: User = Depends(get_current_user)) -> PlateStreamControlResponse:
    return push_service.stop()


@router.get("/stream/status", response_model=PlateStreamControlResponse)
async def get_plate_stream_status(_current_user: User = Depends(get_current_user)) -> PlateStreamControlResponse:
    return push_service.status()


@router.websocket("/ws/stream")
async def stream_plate_rtsp(websocket: WebSocket, rtsp_url: str = Query(..., min_length=1)) -> None:
    await websocket.accept()
    try:
        for payload in service.stream_rtsp(rtsp_url):
            await websocket.send_json(payload)
    except PlateInferenceError as exc:
        await websocket.send_json({"error": str(exc)})
    except WebSocketDisconnect:
        return
    finally:
        await websocket.close()
