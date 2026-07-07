from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from sqlalchemy import select

from app.core.config import settings
from app.core.database import SessionLocal
from app.core.logger import get_logger
from app.models.plate_record import PlateRecord
from app.models.user_operation_log import UserOperationLog
from app.models_infer.errors import InferenceTimeoutError, PlateInferenceError
from app.models_infer.hyperlpr_recognizer import HyperLPRRecognizer
from app.schemas.plate import PlateDetection, PlateRecognitionResponse, PlateRecordSummary
from app.services.alert_service import AlertService
from app.services.monitor_service import MonitorService

logger = get_logger(__name__)


class PlateService:
    def __init__(self) -> None:
        self.recognizer = HyperLPRRecognizer()
        self._backend_dir = Path(__file__).resolve().parents[2]

    async def recognize_image(self, filename: str, image_bytes: bytes | None = None) -> PlateRecognitionResponse:
        return await self.recognize_image_bytes_async(image_bytes or b"", filename)

    async def recognize_image_bytes_async(
        self,
        image_bytes: bytes,
        filename: str = "unknown.jpg",
        *,
        save_history: bool = False,
        user_id: int | None = None,
    ) -> PlateRecognitionResponse:
        if not image_bytes:
            raise ValueError("上传文件为空。")

        image_path = self._persist_upload(image_bytes, filename) if settings.plate_save_uploads else None
        started_at = perf_counter()
        trace_id = uuid4().hex

        try:
            detections = await asyncio.wait_for(
                asyncio.to_thread(self._detect_plates, image_bytes),
                timeout=settings.plate_inference_timeout_seconds,
            )
        except TimeoutError as exc:
            elapsed_ms = int((perf_counter() - started_at) * 1000)
            await self._record_failure(
                user_id=user_id,
                filename=filename,
                elapsed_ms=elapsed_ms,
                trace_id=trace_id,
                title="车牌识别超时",
                summary=(
                    f"文件 {filename} 的车牌识别耗时超过 "
                    f"{settings.plate_inference_timeout_seconds:.1f} 秒。"
                ),
                response_status="Timeout",
            )
            raise InferenceTimeoutError("车牌识别超时。") from exc
        except PlateInferenceError as exc:
            elapsed_ms = int((perf_counter() - started_at) * 1000)
            await self._record_failure(
                user_id=user_id,
                filename=filename,
                elapsed_ms=elapsed_ms,
                trace_id=trace_id,
                title="车牌识别依赖异常",
                summary=f"文件 {filename} 的识别流程执行失败：{exc}",
                response_status="Failed",
            )
            raise
        except Exception as exc:
            elapsed_ms = int((perf_counter() - started_at) * 1000)
            await self._record_failure(
                user_id=user_id,
                filename=filename,
                elapsed_ms=elapsed_ms,
                trace_id=trace_id,
                title="车牌识别服务异常",
                summary=f"文件 {filename} 的识别流程出现未预期错误：{exc}",
                response_status="Failed",
            )
            raise PlateInferenceError("车牌识别服务异常。") from exc

        if detections and save_history and user_id is not None:
            self._save_history(detections, image_path=image_path, user_id=user_id)

        elapsed_ms = int((perf_counter() - started_at) * 1000)
        operation_status = "Success" if detections else "NoDetection"
        self._record_operation(user_id, "plate_recognition", operation_status)
        summary = self._build_behavior_summary(filename, detections, elapsed_ms)
        self._record_behavior(
            title="车牌识别完成" if detections else "车牌识别未命中结果",
            summary=summary,
        )
        await self._record_monitor_success(
            user_id=user_id,
            filename=filename,
            elapsed_ms=elapsed_ms,
            trace_id=trace_id,
            detections=detections,
        )
        return PlateRecognitionResponse(frame_id=filename, detections=detections)

    def recognize_image_bytes(
        self,
        image_bytes: bytes,
        filename: str = "unknown.jpg",
        *,
        save_history: bool = False,
        user_id: int | None = None,
    ) -> PlateRecognitionResponse:
        if not image_bytes:
            raise ValueError("上传文件为空。")

        image_path = self._persist_upload(image_bytes, filename) if settings.plate_save_uploads else None
        detections = self._detect_plates(image_bytes)

        if detections and save_history and user_id is not None:
            self._save_history(detections, image_path=image_path, user_id=user_id)

        return PlateRecognitionResponse(frame_id=filename, detections=detections)

    def list_history(self, user_id: int | None = None) -> list[PlateRecordSummary]:
        with SessionLocal() as session:
            statement = select(PlateRecord)
            if user_id is not None:
                statement = statement.where(PlateRecord.user_id == user_id)
            statement = statement.order_by(PlateRecord.created_at.desc()).limit(settings.plate_history_limit)
            records = session.scalars(statement).all()

        if not records:
            return [
                PlateRecordSummary(
                    id=1,
                    plate_number="测试123",
                    plate_color="蓝牌",
                    created_at=datetime.utcnow(),
                )
            ]

        return [
            PlateRecordSummary(
                id=record.id,
                plate_number=record.plate_number,
                plate_color=record.plate_color,
                created_at=record.created_at,
            )
            for record in records
        ]

    def _save_history(self, detections: list[PlateDetection], image_path: str | None, user_id: int) -> None:
        records = [
            PlateRecord(
                user_id=user_id,
                plate_number=detection.plate_number,
                plate_color=detection.plate_color,
                bbox=detection.bbox,
                confidence=detection.confidence,
                image_path=image_path,
            )
            for detection in detections
            if detection.plate_number
        ]
        if not records:
            return

        with SessionLocal() as session:
            session.add_all(records)
            session.commit()

    def _persist_upload(self, image_bytes: bytes, filename: str) -> str:
        suffix = Path(filename).suffix or ".jpg"
        target_dir = (self._backend_dir / settings.plate_upload_dir).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{uuid4().hex}{suffix}"
        target_path.write_bytes(image_bytes)
        return str(target_path)

    def _detect_plates(self, image_bytes: bytes) -> list[PlateDetection]:
        return [
            PlateDetection(
                plate_number=item.plate_number,
                plate_color=item.plate_color,
                confidence=item.confidence,
                bbox=item.bbox,
            )
            for item in self.recognizer.recognize_all(image_bytes)
        ]

    async def _record_failure(
        self,
        *,
        user_id: int | None,
        filename: str,
        elapsed_ms: int,
        trace_id: str,
        title: str,
        summary: str,
        response_status: str,
    ) -> None:
        full_summary = f"{summary} 处理耗时：{elapsed_ms} ms。"
        try:
            self._record_operation(user_id, "plate_recognition", response_status)
            self._record_behavior(title=title, summary=full_summary)
            with SessionLocal() as session:
                await MonitorService(session).capture_event(
                    category="plate",
                    source="plate-recognition",
                    event_type=(
                        "plate_recognition_timeout"
                        if response_status.lower() == "timeout"
                        else "plate_recognition_failure"
                    ),
                    title=title,
                    summary=full_summary,
                    level="warning",
                    status=response_status,
                    trace_id=trace_id,
                    user_id=user_id,
                    details={
                        "filename": filename,
                        "elapsed_ms": elapsed_ms,
                        "response_status": response_status,
                    },
                )
        except Exception as exc:
            logger.warning("Failed to persist plate failure log for %s: %s", filename, exc)

    async def _record_monitor_success(
        self,
        *,
        user_id: int | None,
        filename: str,
        elapsed_ms: int,
        trace_id: str,
        detections: list[PlateDetection],
    ) -> None:
        event_type = "plate_recognition_success" if detections else "plate_recognition_no_detection"
        with SessionLocal() as session:
            await MonitorService(session).capture_event(
                category="plate",
                source="plate-recognition",
                event_type=event_type,
                title="车牌识别完成" if detections else "车牌识别未命中结果",
                summary=self._build_behavior_summary(filename, detections, elapsed_ms),
                level="info" if detections else "warning",
                status="success" if detections else "no_detection",
                trace_id=trace_id,
                user_id=user_id,
                confidence=detections[0].confidence if detections else None,
                details={
                    "filename": filename,
                    "elapsed_ms": elapsed_ms,
                    "detection_count": len(detections),
                    "plates": [item.plate_number for item in detections],
                },
                trigger_alert=not detections,
            )

    def _record_operation(self, user_id: int | None, operation_type: str, response_status: str) -> None:
        if user_id is None:
            return

        with SessionLocal() as session:
            session.add(
                UserOperationLog(
                    user_id=user_id,
                    operation_type=operation_type,
                    response_status=response_status,
                )
            )
            session.commit()

    def _record_behavior(self, *, title: str, summary: str) -> None:
        with SessionLocal() as session:
            AlertService(session).record_behavior(
                source="plate-recognition",
                title=title,
                summary=summary,
            )

    def _build_behavior_summary(
        self,
        filename: str,
        detections: list[PlateDetection],
        elapsed_ms: int,
    ) -> str:
        if not detections:
            return f"{filename} 已完成识别，但未匹配到有效车牌。处理耗时：{elapsed_ms} ms。"

        first_detection = detections[0]
        return (
            f"{filename} 共识别到 {len(detections)} 个车牌结果。"
            f"首个结果为：{first_detection.plate_number}（{first_detection.plate_color}）。"
            f"处理耗时：{elapsed_ms} ms。"
        )
