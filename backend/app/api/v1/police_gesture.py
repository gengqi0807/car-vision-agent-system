import logging
import asyncio

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse

from app.api.deps import get_current_user
from app.core.database import SessionLocal
from app.core.security import decode_access_token
from app.models.user import User
from app.schemas.gesture import (
    GestureFrameResult,
    GestureHistoryItem,
    PoliceGestureVideoJobCreateResponse,
    PoliceGestureVideoProgress,
    PoliceGestureVideoResult,
    StreamControlRequest,
    StreamState,
)
from app.services.police_gesture_service import PoliceGestureService
from app.services.police_gesture_stream_service import PoliceGestureStreamService

logger = logging.getLogger(__name__)
router = APIRouter()
service = PoliceGestureService()
stream_service = PoliceGestureStreamService.instance()


def _get_current_user_from_query_token(token: str | None) -> User:
    if not token:
        raise HTTPException(status_code=401, detail="缺少 Bearer 令牌")
    try:
        payload = decode_access_token(token)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(status_code=401, detail="令牌载荷无效")
    with SessionLocal() as session:
        user = session.get(User, int(user_id))
    if user is None:
        raise HTTPException(status_code=401, detail="用户不存在")
    return user


@router.post(
    "/current",
    response_model=GestureFrameResult,
    summary="上传图片进行交警手势识别",
    description="接收单张交警姿态图片并返回手势类别、置信度与骨骼关键点。",
    responses={200: {"description": "识别成功"}},
)
async def current_police_gesture(
    file: UploadFile = File(...),
    session_id: str | None = Form(default=None),
    input_mode: str = Form(default="image"),
    current_user: User = Depends(get_current_user),
) -> GestureFrameResult:
    image_bytes = await file.read()
    try:
        return await service.process_frame(
            image_bytes,
            file.filename or "upload.jpg",
            current_user.id,
            session_id=session_id,
            input_mode=input_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Police gesture image recognition failed for %s", file.filename or "upload.jpg")
        raise HTTPException(status_code=500, detail=f"交警手势图片识别失败：{exc}") from exc


@router.post(
    "/video/jobs",
    response_model=PoliceGestureVideoJobCreateResponse,
    summary="创建交警手势视频后台识别任务",
)
async def create_police_gesture_video_job(
    file: UploadFile = File(...),
    task_id: str | None = Form(default=None),
    current_user: User = Depends(get_current_user),
) -> PoliceGestureVideoJobCreateResponse:
    video_bytes = await file.read()
    try:
        return service.start_video_job(
            video_bytes,
            file.filename or "upload.mp4",
            current_user.id,
            task_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/video/jobs/{task_id}/cancel", response_model=PoliceGestureVideoProgress)
async def cancel_police_gesture_video_job(
    task_id: str,
    _current_user: User = Depends(get_current_user),
) -> PoliceGestureVideoProgress:
    return service.cancel_video_job(task_id)


@router.post(
    "/video",
    response_model=PoliceGestureVideoResult,
    summary="上传视频进行交警手势识别",
    description="接收单个视频文件，返回识别出的交警手势、置信度与标注后视频地址。",
    responses={200: {"description": "识别成功"}},
)
async def recognize_police_gesture_video(
    request: Request,
    file: UploadFile = File(...),
    task_id: str | None = Form(default=None),
    current_user: User = Depends(get_current_user),
) -> PoliceGestureVideoResult:
    logger.info("Received police gesture video request: %s", file.filename or "upload.mp4")
    video_bytes = await file.read()
    try:
        response = await asyncio.to_thread(
            service.process_video_bytes,
            video_bytes,
            file.filename or "upload.mp4",
            current_user.id,
            task_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Police gesture video recognition failed for %s", file.filename or "upload.mp4")
        raise HTTPException(status_code=500, detail=f"交警手势视频识别失败：{exc}") from exc

    processed_url = response.processed_video_url
    if processed_url.startswith("/"):
        processed_url = f"{str(request.base_url).rstrip('/')}{processed_url}"
    return response.model_copy(update={"processed_video_url": processed_url})


@router.get(
    "/video/progress/{task_id}",
    response_model=PoliceGestureVideoProgress,
    summary="查询交警视频识别进度",
    description="根据任务 ID 返回视频识别当前处理阶段、进度与最新消息。",
    responses={200: {"description": "查询成功"}},
)
async def police_gesture_video_progress(
    task_id: str,
    _current_user: User = Depends(get_current_user),
) -> PoliceGestureVideoProgress:
    return service.get_video_progress(task_id)


@router.get(
    "/video/preview/{task_id}",
    summary="获取交警视频实时标注预览流",
    description="返回 multipart/x-mixed-replace 的 MJPEG 预览流，持续输出后端最新标注帧。",
    responses={200: {"description": "连接成功"}},
)
async def police_gesture_video_preview(
    task_id: str,
    token: str | None = Query(default=None),
):
    _get_current_user_from_query_token(token)
    return StreamingResponse(
        service.iter_video_preview_stream(task_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/history",
    response_model=list[GestureHistoryItem],
    summary="查询交警手势历史",
    description="返回交警手势识别历史记录。",
    responses={200: {"description": "查询成功"}},
)
async def police_gesture_history(current_user: User = Depends(get_current_user)) -> list[GestureHistoryItem]:
    return service.history(current_user.id)


@router.get("/stream", response_model=StreamState)
async def police_gesture_stream_state() -> StreamState:
    return stream_service.status()


@router.post("/stream/start", response_model=StreamState)
async def start_police_gesture_stream(payload: StreamControlRequest) -> StreamState:
    try:
        return stream_service.start(source=payload.source or "0", fps=payload.fps)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/stream/stop", response_model=StreamState)
async def stop_police_gesture_stream() -> StreamState:
    return stream_service.stop()


@router.get("/stream/result", response_model=GestureFrameResult)
async def current_police_gesture_stream_result() -> GestureFrameResult:
    return stream_service.current()


@router.get("/stream/video-feed")
async def police_gesture_stream_video_feed() -> StreamingResponse:
    return StreamingResponse(
        stream_service.mjpeg_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "X-Accel-Buffering": "no"},
    )
