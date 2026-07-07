from fastapi import APIRouter, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect

from app.models_infer.errors import PlateInferenceError
from app.schemas.plate import PlateRecognitionResponse, PlateRecordSummary
from app.services.plate_service import PlateService

router = APIRouter()
service = PlateService()


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
