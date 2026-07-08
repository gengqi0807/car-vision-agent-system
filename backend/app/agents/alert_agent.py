from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agents.llm_client import LLMClient
from app.core.config import settings
from app.models.monitor_log import MonitorLog
from app.schemas.alert import AlertEvent, AlertEventCreate
from app.services.alert_service import AlertService


class AlertAgent:
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
                "title": log_entry.title,
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
                title=log_entry.title,
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

        if event_type in {"plate_recognition_failure", "plate_recognition_timeout"}:
            consecutive_failures = self._count_recent_logs(
                source=log_entry.source,
                event_types=["plate_recognition_failure", "plate_recognition_timeout"],
                statuses=["failed", "timeout"],
            )
            level = "critical" if consecutive_failures >= settings.alert_consecutive_failures_threshold else "warning"
            return {
                "level": level,
                "root_cause": "车牌识别连续失败，或多次触发超时阈值。",
                "impact_scope": "影响车牌识别上传、监控页面展示以及下游告警查看流程。",
                "suggested_action": "检查模型运行状态、图像质量和 CPU 负载，必要时使用更清晰的画面重新测试。",
                "analysis": {
                    "consecutive_failures": consecutive_failures,
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
                "suggested_action": "核验客户端令牌、排查来源 IP，并考虑限流或拦截异常来源。",
                "analysis": {
                    "recent_attempts": recent_attempts,
                    "observed_at": log_entry.created_at.isoformat(),
                    "details": details,
                },
            }

        if event_type in {"owner_gesture_low_confidence", "police_gesture_low_confidence"}:
            low_confidence_count = self._count_low_confidence_logs(log_entry.source)
            level = "critical" if low_confidence_count >= settings.alert_low_confidence_window_size else "warning"
            return {
                "level": level,
                "root_cause": "识别置信度在一段时间内持续低于系统设定阈值。",
                "impact_scope": "影响当前摄像头画面对应的手势交互识别准确率。",
                "suggested_action": "检查光照、取景范围，并确认模型文件是否已正确加载。",
                "analysis": {
                    "low_confidence_count": low_confidence_count,
                    "threshold": settings.alert_low_confidence_threshold,
                    "observed_confidence": confidence,
                    "observed_at": log_entry.created_at.isoformat(),
                    "details": details,
                },
            }

        if event_type in {"llm_api_timeout", "llm_token_exceeded"}:
            level = "critical" if event_type == "llm_token_exceeded" else "warning"
            return {
                "level": level,
                "root_cause": "用于生成告警摘要的 LLM 依赖出现超时或 Token 额度耗尽。",
                "impact_scope": "影响自然语言告警摘要生成以及后续告警增强分析。",
                "suggested_action": "检查 API 配额、超时设置以及 LLM 服务的回退逻辑。",
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
                "suggested_action": "检查该来源对应的监控日志与事件回放上下文。",
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

    def _count_low_confidence_logs(self, source: str) -> int:
        since = datetime.utcnow() - timedelta(minutes=settings.alert_replay_window_minutes)
        return int(
            self.db.scalar(
                select(func.count(MonitorLog.id)).where(
                    MonitorLog.source == source,
                    MonitorLog.created_at >= since,
                    MonitorLog.confidence.is_not(None),
                    MonitorLog.confidence < settings.alert_low_confidence_threshold,
                )
            )
            or 0
        )
