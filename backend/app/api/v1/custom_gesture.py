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
    CustomGestureSampleBatchOut,
    CustomGestureSampleCreate,
    CustomGestureSampleListOut,
    CustomGestureSampleOut,
    CustomGestureSampleRejected,
    CustomGestureTrainOut,
    CustomGestureTrainRequest,
    KeypointItem,
)
from app.services.custom_gesture_service import (
    add_sample,
    create_custom_gesture,
    delete_custom_gesture,
    delete_sample,
    extract_keypoints_batch,
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


@router.post("/{gesture_name}/samples", response_model=CustomGestureSampleBatchOut)
async def create_sample(
    gesture_name: str,
    file: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
    _current_user: User = Depends(get_current_user),
):
    """上传手势图片（支持单张或多张），服务端自动抽取 21 个手部关键点并保存为样本。
    不合规定的图片直接丢弃并在响应中返回原因。"""
    # 确认手势存在
    gesture = get_custom_gesture_by_name(db, gesture_name)
    if not gesture:
        raise HTTPException(status_code=404, detail=f"手势 '{gesture_name}' 不存在")

    # 分离合规图片与非图片文件
    image_list: list[tuple[str, bytes]] = []
    rejected: list[CustomGestureSampleRejected] = []

    for f in file:
        filename = f.filename or "unknown"
        if not f.content_type or not f.content_type.startswith("image/"):
            rejected.append(CustomGestureSampleRejected(
                filename=filename,
                reason="非图片文件，已跳过",
            ))
            continue
        try:
            img_bytes = await f.read()
            image_list.append((filename, img_bytes))
        except Exception as exc:
            rejected.append(CustomGestureSampleRejected(
                filename=filename,
                reason=f"读取文件失败: {str(exc)}",
            ))

    # 批量提取关键点
    batch_results = extract_keypoints_batch(image_list)

    samples: list[CustomGestureSampleOut] = []
    for (filename, _), (kps, err) in zip(image_list, batch_results):
        if err is not None:
            rejected.append(CustomGestureSampleRejected(filename=filename, reason=err))
            continue
        try:
            payload = CustomGestureSampleCreate(
                keypoints=[KeypointItem(x=k["x"], y=k["y"], z=k.get("z", 0.0)) for k in kps],
                source_type="upload",
                filename=filename,
            )
            sample = add_sample(db, gesture_name, payload)
            samples.append(sample)
        except ValueError as exc:
            rejected.append(CustomGestureSampleRejected(filename=filename, reason=str(exc)))

    total_uploaded = len(file)
    return CustomGestureSampleBatchOut(
        samples=samples,
        rejected=rejected,
        total_uploaded=total_uploaded,
        total_accepted=len(samples),
        total_rejected=len(rejected),
    )


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
