from __future__ import annotations

from datetime import datetime, timedelta
from typing import Callable

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agents.llm_client import LLMClient
from app.core.config import settings
from app.models.monitor_log import MonitorLog
from app.schemas.alert import AlertEvent, AlertEventCreate
from app.services.alert_service import AlertService


class AlertAgent:
    _plate_warning_event_types = {
        "plate_recognition_no_detection",
        "plate_recognition_failure",
        "plate_recognition_timeout",
    }
    _police_image_warning_event_types = {
        "police_gesture_decode_error",
        "police_gesture_image_failure",
    }
    _owner_low_confidence_event_types = {"owner_gesture_low_confidence"}
    _ignored_streak_event_types = {"behavior_event"}

    def __init__(self, db: Session) -> None:
        self.db = db
        self.llm_client = LLMClient()

    async def observe(self, log_entry: MonitorLog, details: dict | None = None) -> AlertEvent | None:
        decision = self._decide(log_entry=log_entry, details=details or {})
        if decision is None:
            return None

        summary_payload = self.llm_client.build_summary(
            source=log_entry.source,
            payload={
                "created_at": log_entry.created_at.isoformat(),
                "event_type": log_entry.event_type,
                "level": decision["level"],
                "title": decision.get("title", log_entry.title),
                "summary": log_entry.summary,
                "impact_scope": decision["impact_scope"],
                "root_cause": decision["root_cause"],
                "suggested_action": decision["suggested_action"],
                "analysis": decision["analysis"],
            },
        )

        event = await AlertService(self.db).create_event(
            AlertEventCreate(
                level=decision["level"],
                source=log_entry.source,
                event_type=log_entry.event_type,
                title=decision.get("title", log_entry.title),
                summary=summary_payload["summary"],
                impact_scope=summary_payload["impact_scope"],
                root_cause=summary_payload["root_cause"],
                suggested_action=summary_payload["suggested_action"],
                analysis=decision["analysis"] | {"llm_mode": summary_payload.get("llm_mode", "fallback")},
            )
        )
        log_entry.alert_id = event.id
        self.db.commit()
        self.db.refresh(log_entry)
        return event

    def _decide(self, *, log_entry: MonitorLog, details: dict) -> dict | None:
        event_type = log_entry.event_type
        status = (log_entry.status or "").lower()
        confidence = log_entry.confidence

        if event_type in self._plate_warning_event_types:
            streak = self._count_consecutive_matching_logs(
                log_entry=log_entry,
                matcher=self._is_plate_warning_log,
                breaker=self._is_plate_streak_breaker,
            )
            if streak > settings.alert_consecutive_failures_threshold:
                return None

            level = "critical" if streak == settings.alert_consecutive_failures_threshold else "warning"
            return {
                "title": "车牌识别多次未命中结果" if level == "critical" else log_entry.title,
                "level": level,
                "root_cause": "车牌识别在监控时间窗口内连续出现告警，说明当前识别链路存在持续性异常。",
                "impact_scope": "影响车牌识别结果展示、上传识别流程以及后续告警联动。",
                "suggested_action": "请检查摄像头画面质量、识别模型运行状态和设备负载，并结合最近样本排查未检出、失败或超时原因。",
                "analysis": {
                    "consecutive_warning_count": streak,
                    "threshold": settings.alert_consecutive_failures_threshold,
                    "observed_status": status,
                    "observed_at": log_entry.created_at.isoformat(),
                    "details": details,
                },
            }

        if event_type in self._police_image_warning_event_types:
            streak = self._count_consecutive_matching_logs(
                log_entry=log_entry,
                matcher=self._is_police_image_warning_log,
                breaker=self._is_police_image_streak_breaker,
            )
            if streak > settings.alert_consecutive_failures_threshold:
                return None

            level = "critical" if streak == settings.alert_consecutive_failures_threshold else "warning"
            return {
                "title": "连续交警手势图片识别失败" if level == "critical" else log_entry.title,
                "level": level,
                "root_cause": "交警手势图片识别在监控时间窗口内连续失败，说明当前图片识别链路存在持续性异常。",
                "impact_scope": "影响交警手势图片识别结果展示、监控日志记录以及后续告警联动。",
                "suggested_action": "请检查上传图片是否可读、姿态模型与分类器运行状态，以及最近失败样本与后端异常日志。",
                "analysis": {
                    "consecutive_warning_count": streak,
                    "threshold": settings.alert_consecutive_failures_threshold,
                    "observed_status": status,
                    "observed_at": log_entry.created_at.isoformat(),
                    "details": details,
                },
            }

        if event_type == "unauthorized_access":
            recent_attempts = self._count_recent_logs(
                source=log_entry.source,
                event_types=["unauthorized_access"],
                statuses=["rejected"],
            )
            level = "critical" if recent_attempts >= settings.alert_consecutive_failures_threshold else "warning"
            return {
                "level": level,
                "root_cause": "受保护接口检测到重复的未授权访问尝试。",
                "impact_scope": "影响需要登录鉴权的接口以及账户安全控制链路。",
                "suggested_action": "请核验客户端令牌、排查来源 IP，并考虑限流或拦截异常来源。",
                "analysis": {
                    "recent_attempts": recent_attempts,
                    "observed_at": log_entry.created_at.isoformat(),
                    "details": details,
                },
            }

        if event_type in self._owner_low_confidence_event_types:
            streak = self._count_consecutive_matching_logs(
                log_entry=log_entry,
                matcher=self._is_owner_low_confidence_log,
                breaker=self._is_owner_streak_breaker,
            )
            if streak > settings.alert_low_confidence_window_size:
                return None

            level = "critical" if streak == settings.alert_low_confidence_window_size else "warning"
            return {
                "level": level,
                "root_cause": "手势控车识别连续多次低于置信度阈值，说明当前交互环境或模型状态存在持续波动。",
                "impact_scope": "影响车主手势控车的识别准确率与交互稳定性。",
                "suggested_action": "请检查光照、取景范围、手势姿态以及模型加载状态，必要时重新校验摄像头位置。",
                "analysis": {
                    "low_confidence_streak": streak,
                    "threshold": settings.alert_low_confidence_window_size,
                    "observed_confidence": confidence,
                    "confidence_threshold": settings.alert_low_confidence_threshold,
                    "observed_at": log_entry.created_at.isoformat(),
                    "details": details,
                },
            }

        if event_type in {"llm_api_timeout", "llm_token_exceeded"}:
            level = "critical" if event_type == "llm_token_exceeded" else "warning"
            return {
                "level": level,
                "root_cause": "用于生成告警摘要的 LLM 依赖出现超时或 Token 配额耗尽。",
                "impact_scope": "影响自然语言告警摘要生成以及后续告警增强分析。",
                "suggested_action": "请检查 API 配额、超时配置以及 LLM 服务的回退逻辑。",
                "analysis": {
                    "observed_at": log_entry.created_at.isoformat(),
                    "details": details,
                },
            }

        if log_entry.level in {"warning", "critical"} or status in {"failed", "timeout"}:
            return {
                "level": "warning" if log_entry.level == "info" else log_entry.level,
                "root_cause": "监控智能体检测到异常运行信号。",
                "impact_scope": "当前功能模块可能出现服务降级。",
                "suggested_action": "请检查该来源对应的监控日志与事件回放上下文。",
                "analysis": {
                    "observed_at": log_entry.created_at.isoformat(),
                    "details": details,
                },
            }

        return None

    def _count_recent_logs(self, *, source: str, event_types: list[str], statuses: list[str]) -> int:
        since = datetime.utcnow() - timedelta(minutes=settings.alert_replay_window_minutes)
        return int(
            self.db.scalar(
                select(func.count(MonitorLog.id)).where(
                    MonitorLog.source == source,
                    MonitorLog.event_type.in_(event_types),
                    MonitorLog.created_at >= since,
                    func.lower(func.coalesce(MonitorLog.status, "")).in_(statuses),
                )
            )
            or 0
        )

    def _count_consecutive_matching_logs(
        self,
        *,
        log_entry: MonitorLog,
        matcher: Callable[[MonitorLog], bool],
        breaker: Callable[[MonitorLog], bool],
    ) -> int:
        since = datetime.utcnow() - timedelta(minutes=settings.alert_replay_window_minutes)
        recent_logs = self.db.scalars(
            select(MonitorLog)
            .where(
                MonitorLog.source == log_entry.source,
                MonitorLog.created_at >= since,
            )
            .order_by(MonitorLog.id.desc())
        ).all()

        streak = 0
        for record in recent_logs:
            if matcher(record):
                streak += 1
                continue
            if breaker(record):
                break
        return streak

    def _is_plate_warning_log(self, record: MonitorLog) -> bool:
        return record.event_type in self._plate_warning_event_types

    def _is_plate_streak_breaker(self, record: MonitorLog) -> bool:
        if record.event_type in self._ignored_streak_event_types:
            return False
        return record.event_type == "plate_recognition_success"

    def _is_police_image_warning_log(self, record: MonitorLog) -> bool:
        return record.event_type in self._police_image_warning_event_types

    def _is_police_image_streak_breaker(self, record: MonitorLog) -> bool:
        if record.event_type in self._ignored_streak_event_types:
            return False
        return record.event_type == "police_gesture_success"

    def _is_owner_low_confidence_log(self, record: MonitorLog) -> bool:
        return record.event_type in self._owner_low_confidence_event_types

    def _is_owner_streak_breaker(self, record: MonitorLog) -> bool:
        if record.event_type in self._ignored_streak_event_types:
            return False
        return record.event_type == "owner_gesture_success"
