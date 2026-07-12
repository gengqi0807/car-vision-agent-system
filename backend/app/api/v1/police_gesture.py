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
    PoliceGestureVideoProgress,
    PoliceGestureVideoResult,
)
from app.services.police_gesture_service import PoliceGestureService

logger = logging.getLogger(__name__)
router = APIRouter()
service = PoliceGestureService()


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
    current_user: User = Depends(get_current_user),
) -> GestureFrameResult:
    filename = file.filename or "upload.jpg"
    image_bytes = await file.read()
    try:
        return await service.process_frame(image_bytes, filename, current_user.id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        await service._capture_error(
            filename=filename,
            event_type="police_gesture_image_failure",
            summary=f"Police gesture image recognition failed: {exc}",
            user_id=current_user.id,
            details={"filename": filename},
        )
        logger.exception("Police gesture image recognition failed for %s", filename)
        raise HTTPException(status_code=500, detail=f"交警手势图片识别失败：{exc}") from exc


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
    filename = file.filename or "upload.mp4"
    logger.info("Received police gesture video request: %s", filename)
    video_bytes = await file.read()
    try:
        response = await asyncio.to_thread(
            service.process_video_bytes,
            video_bytes,
            filename,
            current_user.id,
            task_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        await service._capture_error(
            filename=filename,
            event_type="police_gesture_video_failure",
            summary=f"Police gesture video recognition failed: {exc}",
            user_id=current_user.id,
            details={"filename": filename, "task_id": task_id},
        )
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        await service._capture_error(
            filename=filename,
            event_type="police_gesture_video_failure",
            summary=f"Police gesture video recognition failed: {exc}",
            user_id=current_user.id,
            details={"filename": filename, "task_id": task_id},
        )
        logger.exception("Police gesture video recognition failed for %s", filename)
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
