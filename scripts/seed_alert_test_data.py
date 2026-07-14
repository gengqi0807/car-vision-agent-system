from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import select

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
for path in (ROOT_DIR, BACKEND_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.core.database import SessionLocal
from app.core.security import hash_password
from app.models.alert_log import AlertLog
from app.models.alert_push_log import AlertPushLog
from app.models.monitor_log import MonitorLog
from app.models.user import User
from app.models.user_operation_log import UserOperationLog


SAMPLE_ALERTS = [
    {
        "level": "critical",
        "source": "plate-recognition",
        "event_type": "plate_recognition_timeout",
        "title": "车牌识别连续超时",
        "summary": "最近 5 分钟内，车牌识别请求连续 3 次超时。",
        "impact_scope": "影响车牌识别上传以及下游车辆审计流程。",
        "root_cause": "识别任务多次超过系统配置的超时阈值。",
        "suggested_action": "检查模型加载耗时、CPU 饱和度以及输入图像质量。",
        "minutes_ago": 3,
        "channel": "websocket",
        "target": "alerts:3",
        "success": True,
    },
    {
        "level": "warning",
        "source": "owner-gesture",
        "event_type": "owner_gesture_low_confidence",
        "title": "车主手势置信度偏低",
        "summary": "最近 1 分钟内，手势平均置信度持续低于 0.60。",
        "impact_scope": "影响车内手势控车交互。",
        "root_cause": "车内光线不足且手部存在部分遮挡，导致模型置信度下降。",
        "suggested_action": "提升车内光照，并确认手势模型资源已正确加载。",
        "minutes_ago": 9,
        "channel": "webhook",
        "target": "https://example.invalid/robot",
        "success": False,
    },
    {
        "level": "warning",
        "source": "auth",
        "event_type": "unauthorized_access",
        "title": "连续未授权访问被拦截",
        "summary": "受保护接口连续拦截了 3 次无效 Bearer 令牌请求。",
        "impact_scope": "影响需要身份认证保护的接口路由。",
        "root_cause": "无效或过期的凭据持续访问 API 网关。",
        "suggested_action": "排查来源 IP，吊销可疑令牌，并适当提高限流强度。",
        "minutes_ago": 14,
        "channel": "email",
        "target": "audit@example.com",
        "success": False,
    },
    {
        "level": "info",
        "source": "police-gesture",
        "event_type": "police_gesture_success",
        "title": "交警手势服务恢复正常",
        "summary": "最新一批姿态识别帧的置信度已恢复正常。",
        "impact_scope": "影响交警手势识别结果。",
        "root_cause": "摄像头画面在短暂抖动后已恢复稳定。",
        "suggested_action": "继续观察是否复发，当前无需人工干预。",
        "minutes_ago": 22,
        "channel": "websocket",
        "target": "alerts:1",
        "success": True,
    },
]


SAMPLE_MONITOR_LOGS = [
    {
        "category": "plate",
        "source": "plate-recognition",
        "event_type": "plate_recognition_timeout",
        "level": "warning",
        "title": "车牌识别超时",
        "summary": "test_plate_timeout_01.jpg 的推理耗时超过超时阈值。",
        "status": "Timeout",
        "confidence": None,
        "minutes_ago": 5,
        "details": {"filename": "test_plate_timeout_01.jpg", "elapsed_ms": 10320},
    },
    {
        "category": "plate",
        "source": "plate-recognition",
        "event_type": "plate_recognition_failure",
        "level": "warning",
        "title": "车牌识别依赖异常",
        "summary": "test_plate_timeout_02.jpg 的 OCR 依赖返回空结果。",
        "status": "Failed",
        "confidence": None,
        "minutes_ago": 4,
        "details": {"filename": "test_plate_timeout_02.jpg", "elapsed_ms": 8120},
    },
    {
        "category": "owner_gesture",
        "source": "owner-gesture",
        "event_type": "owner_gesture_low_confidence",
        "level": "warning",
        "title": "车主手势帧处理完成",
        "summary": "owner_hand_frame.jpg 已处理完成，置信度 0.42，检测到 1 只手。",
        "status": "processed",
        "confidence": 0.42,
        "minutes_ago": 10,
        "details": {"filename": "owner_hand_frame.jpg", "num_hands_detected": 1},
    },
    {
        "category": "police_gesture",
        "source": "police-gesture",
        "event_type": "police_gesture_success",
        "level": "info",
        "title": "交警手势帧处理完成",
        "summary": "police_pose_frame.jpg 已处理完成，置信度 0.97，检测到 1 个人体姿态。",
        "status": "processed",
        "confidence": 0.97,
        "minutes_ago": 21,
        "details": {"filename": "police_pose_frame.jpg", "num_poses_detected": 1},
    },
    {
        "category": "security",
        "source": "auth",
        "event_type": "unauthorized_access",
        "level": "warning",
        "title": "Bearer 令牌无效",
        "summary": "受保护接口被无效或过期令牌访问。",
        "status": "rejected",
        "confidence": None,
        "minutes_ago": 13,
        "details": {"reason": "Invalid or expired token"},
    },
    {
        "category": "user_operation",
        "source": "auth",
        "event_type": "login",
        "level": "info",
        "title": "用户登录成功",
        "summary": "用户 monitor_tester 登录成功。",
        "status": "Success",
        "confidence": None,
        "minutes_ago": 30,
        "details": {"username": "monitor_tester"},
    },
]


SAMPLE_OPERATIONS = [
    {"operation_type": "register", "response_status": "Success", "minutes_ago": 40},
    {"operation_type": "login", "response_status": "Success", "minutes_ago": 30},
    {"operation_type": "update_profile", "response_status": "Success", "minutes_ago": 18},
]


def ensure_user(db) -> User:
    existing = db.scalar(select(User).where(User.username == "monitor_tester"))
    if existing is not None:
        return existing

    user = User(
        username="monitor_tester",
        password_hash=hash_password("Monitor123!"),
        role="admin",
    )
    user.email = "monitor_tester@example.com"
    user.phone = "13800000000"
    db.add(user)
    db.flush()
    return user


def seed_alerts_and_logs() -> tuple[int, int, int]:
    db = SessionLocal()
    added_alerts = 0
    added_push_logs = 0
    added_monitor_logs = 0

    try:
        now = datetime.utcnow()
        user = ensure_user(db)

        alert_ids_by_source: dict[str, int] = {}

        for sample in SAMPLE_ALERTS:
            existing = db.scalar(
                select(AlertLog).where(
                    AlertLog.source == sample["source"],
                    AlertLog.title == sample["title"],
                )
            )
            if existing is None:
                created_at = now - timedelta(minutes=sample["minutes_ago"])
                alert = AlertLog(
                    level=sample["level"],
                    source=sample["source"],
                    event_type=sample["event_type"],
                    title=sample["title"],
                    summary=sample["summary"],
                    impact_scope=sample["impact_scope"],
                    root_cause=sample["root_cause"],
                    suggested_action=sample["suggested_action"],
                    analysis_json=json.dumps({"seeded": True}, ensure_ascii=False),
                    created_at=created_at,
                    updated_at=created_at,
                )
                db.add(alert)
                db.flush()
                added_alerts += 1

                push_log = AlertPushLog(
                    channel=sample["channel"],
                    target=sample["target"],
                    success=sample["success"],
                    created_at=created_at,
                    updated_at=created_at,
                )
                db.add(push_log)
                added_push_logs += 1
                alert_ids_by_source[sample["source"]] = alert.id
            else:
                alert_ids_by_source[sample["source"]] = existing.id

        for sample in SAMPLE_MONITOR_LOGS:
            existing = db.scalar(
                select(MonitorLog).where(
                    MonitorLog.source == sample["source"],
                    MonitorLog.title == sample["title"],
                    MonitorLog.event_type == sample["event_type"],
                )
            )
            if existing is not None:
                continue

            created_at = now - timedelta(minutes=sample["minutes_ago"])
            db.add(
                MonitorLog(
                    category=sample["category"],
                    source=sample["source"],
                    event_type=sample["event_type"],
                    level=sample["level"],
                    title=sample["title"],
                    summary=sample["summary"],
                    status=sample["status"],
                    user_id=user.id if sample["source"] == "auth" else None,
                    alert_id=alert_ids_by_source.get(sample["source"]),
                    confidence=sample["confidence"],
                    details_json=json.dumps(sample["details"], ensure_ascii=False),
                    created_at=created_at,
                    updated_at=created_at,
                )
            )
            added_monitor_logs += 1

        for sample in SAMPLE_OPERATIONS:
            existing = db.scalar(
                select(UserOperationLog).where(
                    UserOperationLog.user_id == user.id,
                    UserOperationLog.operation_type == sample["operation_type"],
                )
            )
            if existing is not None:
                continue
            created_at = now - timedelta(minutes=sample["minutes_ago"])
            db.add(
                UserOperationLog(
                    user_id=user.id,
                    operation_type=sample["operation_type"],
                    response_status=sample["response_status"],
                    created_at=created_at,
                    updated_at=created_at,
                )
            )

        db.commit()
        return added_alerts, added_push_logs, added_monitor_logs
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def main() -> None:
    added_alerts, added_push_logs, added_monitor_logs = seed_alerts_and_logs()
    print(
        f"Seeded {added_alerts} alert events, {added_push_logs} push logs, "
        f"and {added_monitor_logs} monitor logs."
    )


if __name__ == "__main__":
    main()
