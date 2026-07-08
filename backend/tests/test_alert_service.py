import asyncio
import os
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
TEST_DB_URL = "sqlite+pysqlite:///file:car_vision_test_alert_service?mode=memory&cache=shared&uri=true"

os.environ["DATABASE_URL"] = TEST_DB_URL

for path in (ROOT_DIR, BACKEND_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.core.database import Base, SessionLocal, engine
from app.models.alert_push_log import AlertPushLog
from app.models.user_operation_log import UserOperationLog
from app.schemas.alert import AlertEventCreate
from app.services.alert_service import AlertService


class StubNotifier:
    def __init__(self, delivered: int = 1):
        self.delivered = delivered

    async def broadcast(self, message: dict) -> int:
        _ = message
        return self.delivered


class AlertServiceTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        Base.metadata.create_all(bind=engine)

    @classmethod
    def tearDownClass(cls) -> None:
        Base.metadata.drop_all(bind=engine)
        engine.dispose()

    def setUp(self) -> None:
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        self.db = SessionLocal()

    def tearDown(self) -> None:
        self.db.close()

    def test_create_event_persists_alert_and_push_log(self) -> None:
        service = AlertService(self.db, notifier=StubNotifier(delivered=2))

        event = asyncio.run(
            service.create_event(
                AlertEventCreate(
                    level="critical",
                    source="plate-recognition",
                    title="识别服务超时",
                    summary="连续 3 次识别请求超时，建议检查模型服务和网络链路。",
                )
            )
        )

        self.assertEqual(event.level, "critical")
        self.assertEqual(event.source, "plate-recognition")

        push_logs = self.db.query(AlertPushLog).all()
        self.assertEqual(len(push_logs), 1)
        self.assertTrue(push_logs[0].success)
        self.assertEqual(push_logs[0].target, "alerts:2")

    def test_overview_and_timeline_return_database_data(self) -> None:
        service = AlertService(self.db, notifier=StubNotifier(delivered=0))

        asyncio.run(
            service.create_event(
                AlertEventCreate(
                    level="warning",
                    source="owner-gesture",
                    title="手势识别置信度偏低",
                    summary="最近一分钟平均置信度低于阈值 0.60。",
                )
            )
        )
        asyncio.run(
            service.create_event(
                AlertEventCreate(
                    level="info",
                    source="auth",
                    title="检测到新的登录设备",
                    summary="用户 admin 使用新设备登录系统。",
                )
            )
        )

        overview = service.overview(latest_limit=1)
        timeline = service.timeline(limit=10)

        self.assertEqual(overview.total, 2)
        self.assertEqual(overview.warning, 1)
        self.assertEqual(overview.info, 1)
        self.assertEqual(len(overview.latest), 1)
        self.assertEqual(len(timeline), 2)

    def test_operation_log_filters_work(self) -> None:
        self.db.add_all(
            [
                UserOperationLog(user_id=1, operation_type="login", response_status="Success"),
                UserOperationLog(user_id=2, operation_type="register", response_status="Success"),
            ]
        )
        self.db.commit()

        service = AlertService(self.db, notifier=StubNotifier())
        records = service.list_operation_logs(limit=10, user_id=1, operation_type="login")

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].user_id, 1)
        self.assertEqual(records[0].operation_type, "login")

    def test_behavior_logs_only_return_real_records(self) -> None:
        service = AlertService(self.db, notifier=StubNotifier())

        self.assertEqual(service.list_behavior_logs(limit=6), [])

        service.record_behavior(
            source="plate-recognition",
            title="车牌识别完成",
            summary="真实识别到粤B12345，颜色为蓝牌，置信度 0.94。",
        )
        records = service.list_behavior_logs(limit=6)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].source, "plate-recognition")
        self.assertEqual(records[0].title, "车牌识别完成")


if __name__ == "__main__":
    unittest.main()
