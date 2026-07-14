"""
自定义手势 API 路由。
"""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.models.user import User
from app.schemas.custom_gesture import (
    CustomGestureCreate,
    CustomGestureListOut,
    CustomGestureOut,
    CustomGestureSampleCreate,
    CustomGestureSampleListOut,
    CustomGestureSampleOut,
    CustomGestureTrainOut,
    CustomGestureTrainRequest,
    KeypointItem,
)
from app.services.custom_gesture_service import (
    add_sample,
    create_custom_gesture,
    delete_custom_gesture,
    delete_sample,
    extract_keypoints_from_image,
    get_custom_gesture_by_name,
    list_custom_gestures,
    list_samples,
    run_train,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/custom-gesture", tags=["custom-gesture"])


# ── 手势 CRUD ────────────────────────────────────────────────────

@router.get("/", response_model=CustomGestureListOut)
def get_gestures(
    db: Session = Depends(get_db),
    _current_user: User = Depends(get_current_user),
):
    gestures, total = list_custom_gestures(db)
    return CustomGestureListOut(gestures=gestures, total=total)


@router.post("/", response_model=CustomGestureOut)
def create_gesture(
    payload: CustomGestureCreate,
    db: Session = Depends(get_db),
    _current_user: User = Depends(get_current_user),
):
    try:
        return create_custom_gesture(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.get("/{gesture_name}", response_model=CustomGestureOut)
def get_gesture(
    gesture_name: str,
    db: Session = Depends(get_db),
    _current_user: User = Depends(get_current_user),
):
    result = get_custom_gesture_by_name(db, gesture_name)
    if not result:
        raise HTTPException(status_code=404, detail=f"手势 '{gesture_name}' 不存在")
    return result


@router.delete("/{gesture_name}")
def delete_gesture(
    gesture_name: str,
    db: Session = Depends(get_db),
    _current_user: User = Depends(get_current_user),
):
    ok = delete_custom_gesture(db, gesture_name)
    if not ok:
        raise HTTPException(status_code=404, detail=f"手势 '{gesture_name}' 不存在")
    return {"detail": f"手势 '{gesture_name}' 已删除"}


# ── 样本采集 ──────────────────────────────────────────────────────

@router.get("/{gesture_name}/samples", response_model=CustomGestureSampleListOut)
def get_samples(
    gesture_name: str,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    _current_user: User = Depends(get_current_user),
):
    samples, total = list_samples(db, gesture_name, limit=limit, offset=offset)
    return CustomGestureSampleListOut(samples=samples, total=total)


@router.post("/{gesture_name}/samples", response_model=CustomGestureSampleOut)
async def create_sample(
    gesture_name: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    _current_user: User = Depends(get_current_user),
):
    """上传手势图片，服务端自动抽取 21 个手部关键点并保存为样本。"""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传图片文件")

    try:
        image_bytes = await file.read()
        kps = extract_keypoints_from_image(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        payload = CustomGestureSampleCreate(
            keypoints=[KeypointItem(x=k["x"], y=k["y"], z=k.get("z", 0.0)) for k in kps],
            source_type="upload",
            filename=file.filename or "",
        )
        return add_sample(db, gesture_name, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/samples/{sample_id}")
def delete_sample_endpoint(
    sample_id: int,
    db: Session = Depends(get_db),
    _current_user: User = Depends(get_current_user),
):
    ok = delete_sample(db, sample_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"样本 ID {sample_id} 不存在")
    return {"detail": f"样本 {sample_id} 已删除"}


# ── 训练 ──────────────────────────────────────────────────────────

@router.post("/train", response_model=CustomGestureTrainOut)
def trigger_train(
    payload: CustomGestureTrainRequest | None = None,
    db: Session = Depends(get_db),
    _current_user: User = Depends(get_current_user),
):
    """触发自定义手势 SVM 训练。
    可选指定 gesture_names 来只训练部分手势；不指定则训练全部。
    """
    names = payload.gesture_names if payload and payload.gesture_names else None
    return run_train(gesture_names=names if names else None)
