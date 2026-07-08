from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect

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


@router.post("/image", response_model=PlateRecognitionResponse)
async def recognize_plate_image(file: UploadFile = File(...)) -> PlateRecognitionResponse:
    image_bytes = await file.read()
    try:
        return await service.recognize_image(file.filename or "unknown.jpg", image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PlateInferenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/history", response_model=list[PlateRecordSummary])
async def get_plate_history() -> list[PlateRecordSummary]:
    return service.list_history()


@router.post("/video", response_model=PlateVideoRecognitionResponse)
async def recognize_plate_video(request: Request, file: UploadFile = File(...)) -> PlateVideoRecognitionResponse:
    video_bytes = await file.read()
    try:
        response = service.recognize_video_bytes(video_bytes, file.filename or "unknown.mp4")
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
async def start_plate_stream(payload: PlateStreamStartRequest) -> PlateStreamControlResponse:
    try:
        return push_service.start(rtsp_url=payload.rtsp_url, stream_name=payload.stream_name)
    except PlateInferenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/stream/stop", response_model=PlateStreamControlResponse)
async def stop_plate_stream() -> PlateStreamControlResponse:
    return push_service.stop()


@router.get("/stream/status", response_model=PlateStreamControlResponse)
async def get_plate_stream_status() -> PlateStreamControlResponse:
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
