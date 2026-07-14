"""
自定义手势服务层。
负责手势元数据管理、样本采集、训练触发与模型热加载。
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.core.config import settings
from app.models.custom_gesture import CustomGesture, CustomGestureSample
from app.models_infer.hand_utils import normalize_hand_landmarks_array
from app.models_infer.mediapipe_hands import MediaPipeHands
from app.schemas.custom_gesture import (
    CustomGestureCreate,
    CustomGestureOut,
    CustomGestureSampleCreate,
    CustomGestureSampleOut,
    CustomGestureTrainOut,
)

logger = logging.getLogger(__name__)


# ── 关键点提取（复用 MediaPipeHands） ─────────────────────────────

def extract_keypoints_from_image(image_bytes: bytes) -> list[dict]:
    """从图片字节中提取单手 21 个标准关键点。

    Raises:
        ValueError: 图片无法解码或未检测到手部。
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("无法解码图片，请确认上传的是有效的图片文件")

    with MediaPipeHands(
        num_hands=1,
        min_detection_confidence=0.5,
        min_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        result = hands.infer(frame)

    kps = result.get("keypoints", [])
    if not kps or result.get("num_hands_detected", 0) == 0:
        raise ValueError("未检测到手部，请确保图片中包含清晰完整的手部")

    # 取第一只手的 21 个关键点
    kps_21 = kps[:21]
    if len(kps_21) < 21:
        raise ValueError(f"检测到的手部关键点不足（{len(kps_21)}/21），请换一张更清晰的手势图片")

    return kps_21


# ── 手势元数据管理 ────────────────────────────────────────────────

def list_custom_gestures(db: Session) -> tuple[list[CustomGestureOut], int]:
    gestures = db.query(CustomGesture).order_by(CustomGesture.created_at.desc()).all()
    total = len(gestures)
    out = [CustomGestureOut.model_validate(g) for g in gestures]
    return out, total


def create_custom_gesture(db: Session, payload: CustomGestureCreate) -> CustomGestureOut:
    existing = db.query(CustomGesture).filter(CustomGesture.name == payload.name).first()
    if existing:
        raise ValueError(f"手势 '{payload.name}' 已存在")

    gesture = CustomGesture(
        name=payload.name,
        display_name=payload.display_name or payload.name,
        description=payload.description or "",
    )
    db.add(gesture)
    db.commit()
    db.refresh(gesture)
    return CustomGestureOut.model_validate(gesture)


def delete_custom_gesture(db: Session, gesture_name: str) -> bool:
    gesture = db.query(CustomGesture).filter(CustomGesture.name == gesture_name).first()
    if not gesture:
        return False
    db.delete(gesture)
    db.commit()
    return True


def get_custom_gesture_by_name(db: Session, name: str) -> CustomGestureOut | None:
    gesture = db.query(CustomGesture).filter(CustomGesture.name == name).first()
    if not gesture:
        return None
    return CustomGestureOut.model_validate(gesture)


# ── 样本采集 ──────────────────────────────────────────────────────

def add_sample(
    db: Session,
    gesture_name: str,
    payload: CustomGestureSampleCreate,
) -> CustomGestureSampleOut:
    gesture = db.query(CustomGesture).filter(CustomGesture.name == gesture_name).first()
    if not gesture:
        raise ValueError(f"手势 '{gesture_name}' 不存在，请先创建")

    # 将 KeypointItem pydantic 对象转为普通 dict 列表存 JSON
    kps_list = [kp.model_dump() for kp in payload.keypoints]
    sample = CustomGestureSample(
        gesture_id=gesture.id,
        keypoints=kps_list,
        source_type=payload.source_type,
        filename=payload.filename,
    )
    db.add(sample)
    gesture.sample_count = (gesture.sample_count or 0) + 1
    db.commit()
    db.refresh(sample)
    return CustomGestureSampleOut.model_validate(sample)


def list_samples(
    db: Session,
    gesture_name: str,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[CustomGestureSampleOut], int]:
    gesture = db.query(CustomGesture).filter(CustomGesture.name == gesture_name).first()
    if not gesture:
        return [], 0

    total = db.query(CustomGestureSample).filter(
        CustomGestureSample.gesture_id == gesture.id
    ).count()

    samples = (
        db.query(CustomGestureSample)
        .filter(CustomGestureSample.gesture_id == gesture.id)
        .order_by(CustomGestureSample.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    out = []
    for s in samples:
        item = CustomGestureSampleOut.model_validate(s)
        out.append(item)
    return out, total


def delete_sample(db: Session, sample_id: int) -> bool:
    sample = db.query(CustomGestureSample).filter(CustomGestureSample.id == sample_id).first()
    if not sample:
        return False
    gesture = sample.gesture
    gesture.sample_count = max(0, (gesture.sample_count or 1) - 1)
    db.delete(sample)
    db.commit()
    return True


# ── 训练 ──────────────────────────────────────────────────────────

def run_train(
    gesture_names: list[str] | None = None,
) -> CustomGestureTrainOut:
    """触发自定义手势 SVM 训练。返回训练结果摘要。"""
    # 先快速检查是否有足够数据
    from app.core.database import SessionLocal

    with SessionLocal() as db:
        query = db.query(CustomGesture)
        if gesture_names:
            query = query.filter(CustomGesture.name.in_(gesture_names))
        gestures = query.all()

        if not gestures:
            return CustomGestureTrainOut(
                status="no_data",
                message="没有可用的自定义手势，请先创建手势并采集样本。",
                n_samples=0,
                n_classes=0,
            )

        total_samples = sum(g.sample_count or 0 for g in gestures)
        if total_samples < 10:
            return CustomGestureTrainOut(
                status="no_data",
                message=f"当前共有 {total_samples} 个样本，不足 10 个，建议先多采集一些。",
                n_samples=total_samples,
                n_classes=len(gestures),
                class_names=[g.name for g in gestures],
            )

        if len(gestures) < 2:
            return CustomGestureTrainOut(
                status="single_class",
                message="自定义手势至少需要 2 个类别才能训练 SVM。",
                n_samples=total_samples,
                n_classes=1,
                class_names=[g.name for g in gestures],
            )

    # 调用训练脚本
    project_root = Path(__file__).resolve().parents[3]
    train_script = project_root / "scripts" / "train_custom_gesture_model.py"

    cmd = [
        sys.executable,
        str(train_script),
    ]
    if gesture_names:
        cmd.extend(["--names"] + gesture_names)

    logger.info("启动自定义手势训练: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(project_root),
        )
        stdout = result.stdout
        stderr = result.stderr

        if result.returncode != 0:
            logger.error("自定义手势训练失败: %s", stderr)
            return CustomGestureTrainOut(
                status="error",
                message=f"训练脚本退出码 {result.returncode}: {stderr[:500]}",
                n_samples=total_samples,
                n_classes=len(gestures),
                class_names=[g.name for g in gestures],
            )

        # 解析训练输出获取评估指标
        evaluation: dict = {}
        for line in stdout.splitlines():
            if "折交叉验证准确率:" in line:
                evaluation["cv_accuracy"] = line.split(":")[-1].strip()
            if "类别" in line and ":" in line:
                parts = line.split(":")
                if len(parts) >= 2:
                    evaluation["classes"] = parts[-1].strip()

        model_path = settings.resolved_custom_gesture_classifier_model_path
        model_exists = os.path.exists(model_path)

        # 热加载到分类器
        _reload_custom_classifier()

        return CustomGestureTrainOut(
            status="success" if model_exists else "error",
            message=f"训练完成，模型{'已' if model_exists else '未能'}保存。" + stdout.splitlines()[-1] if stdout else "",
            n_samples=total_samples,
            n_classes=len(gestures),
            class_names=[g.name for g in gestures],
            model_path=model_path if model_exists else "",
            evaluation=evaluation,
        )

    except subprocess.TimeoutExpired:
        return CustomGestureTrainOut(
            status="error",
            message="训练超时（5 分钟），请检查样本数据量。",
            n_samples=total_samples,
            n_classes=len(gestures),
            class_names=[g.name for g in gestures],
        )
    except Exception as exc:
        logger.exception("触发训练异常")
        return CustomGestureTrainOut(
            status="error",
            message=f"训练异常: {str(exc)}",
            n_samples=total_samples,
            n_classes=len(gestures),
            class_names=[g.name for g in gestures],
        )


def _reload_custom_classifier() -> None:
    """通知 GestureClassifier 重新加载自定义手势模型。"""
    try:
        from app.models_infer.gesture_classifier import GestureClassifier

        # 强制重建单例分类器（下次请求时自动加载新模型）
        if hasattr(GestureClassifier, "_custom_model_path"):
            # 清除已加载的模型引用，下次 classify_custom 时重新加载
            _cls = GestureClassifier
            if hasattr(_cls, "_custom_model") and _cls._custom_model is not None:
                _cls._custom_model = None
                _cls._custom_scaler = None
                _cls._custom_labels = None
                logger.info("自定义手势分类器已热卸载，下次推理时自动重新加载")
    except Exception:
        logger.debug("热加载自定义分类器时出错，忽略", exc_info=True)
