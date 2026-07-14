import os
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
TEST_DB_URL = "sqlite+pysqlite:///file:car_vision_test_alert_agent?mode=memory&cache=shared&uri=true"

os.environ["DATABASE_URL"] = TEST_DB_URL

for path in (ROOT_DIR, BACKEND_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.agents.alert_agent import AlertAgent
from app.core.database import Base, SessionLocal, engine
from app.models.monitor_log import MonitorLog


class AlertAgentTestCase(unittest.TestCase):
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
        self.agent = AlertAgent(self.db)
        self.base_time = datetime.utcnow()

    def tearDown(self) -> None:
        self.db.close()

    def test_plate_warning_streak_emits_one_critical_on_third_hit(self) -> None:
        first = self._create_monitor_log(
            source="plate-recognition",
            event_type="plate_recognition_no_detection",
            level="warning",
            status="no_detection",
            created_at=self.base_time,
        )
        self.assertEqual(self.agent._decide(log_entry=first, details={})["level"], "warning")

        self._create_monitor_log(
            source="plate-recognition",
            event_type="behavior_event",
            level="info",
            status="recorded",
            created_at=self.base_time + timedelta(milliseconds=200),
        )

        second = self._create_monitor_log(
            source="plate-recognition",
            event_type="plate_recognition_failure",
            level="warning",
            status="failed",
            created_at=self.base_time + timedelta(seconds=1),
        )
        self.assertEqual(self.agent._decide(log_entry=second, details={})["level"], "warning")

        self._create_monitor_log(
            source="plate-recognition",
            event_type="behavior_event",
            level="info",
            status="recorded",
            created_at=self.base_time + timedelta(seconds=1, milliseconds=200),
        )

        third = self._create_monitor_log(
            source="plate-recognition",
            event_type="plate_recognition_timeout",
            level="warning",
            status="timeout",
            created_at=self.base_time + timedelta(seconds=2),
        )
        third_decision = self.agent._decide(log_entry=third, details={})
        self.assertEqual(third_decision["level"], "critical")
        self.assertEqual(third_decision["title"], "车牌识别多次未命中结果")

        fourth = self._create_monitor_log(
            source="plate-recognition",
            event_type="plate_recognition_no_detection",
            level="warning",
            status="no_detection",
            created_at=self.base_time + timedelta(seconds=3),
        )
        self.assertIsNone(self.agent._decide(log_entry=fourth, details={}))

        success = self._create_monitor_log(
            source="plate-recognition",
            event_type="plate_recognition_success",
            level="info",
            status="success",
            created_at=self.base_time + timedelta(seconds=4),
        )
        recovered_warning = self._create_monitor_log(
            source="plate-recognition",
            event_type="plate_recognition_no_detection",
            level="warning",
            status="no_detection",
            created_at=self.base_time + timedelta(seconds=5),
        )

        self.assertIsNone(self.agent._decide(log_entry=success, details={}))
        self.assertEqual(self.agent._decide(log_entry=recovered_warning, details={})["level"], "warning")

    def test_owner_low_confidence_streak_emits_one_critical_on_third_hit(self) -> None:
        first = self._create_monitor_log(
            source="owner-gesture",
            event_type="owner_gesture_low_confidence",
            level="warning",
            status="processed",
            confidence=0.41,
            created_at=self.base_time,
        )
        self.assertEqual(self.agent._decide(log_entry=first, details={})["level"], "warning")

        self._create_monitor_log(
            source="owner-gesture",
            event_type="behavior_event",
            level="info",
            status="recorded",
            created_at=self.base_time + timedelta(milliseconds=200),
        )

        second = self._create_monitor_log(
            source="owner-gesture",
            event_type="owner_gesture_low_confidence",
            level="warning",
            status="processed",
            confidence=0.38,
            created_at=self.base_time + timedelta(seconds=1),
        )
        self.assertEqual(self.agent._decide(log_entry=second, details={})["level"], "warning")

        self._create_monitor_log(
            source="owner-gesture",
            event_type="behavior_event",
            level="info",
            status="recorded",
            created_at=self.base_time + timedelta(seconds=1, milliseconds=200),
        )

        third = self._create_monitor_log(
            source="owner-gesture",
            event_type="owner_gesture_low_confidence",
            level="warning",
            status="processed",
            confidence=0.35,
            created_at=self.base_time + timedelta(seconds=2),
        )
        self.assertEqual(self.agent._decide(log_entry=third, details={})["level"], "critical")

        fourth = self._create_monitor_log(
            source="owner-gesture",
            event_type="owner_gesture_low_confidence",
            level="warning",
            status="processed",
            confidence=0.33,
            created_at=self.base_time + timedelta(seconds=3),
        )
        self.assertIsNone(self.agent._decide(log_entry=fourth, details={}))

        success = self._create_monitor_log(
            source="owner-gesture",
            event_type="owner_gesture_success",
            level="info",
            status="processed",
            confidence=0.92,
            created_at=self.base_time + timedelta(seconds=4),
        )
        recovered_warning = self._create_monitor_log(
            source="owner-gesture",
            event_type="owner_gesture_low_confidence",
            level="warning",
            status="processed",
            confidence=0.44,
            created_at=self.base_time + timedelta(seconds=5),
        )

        self.assertIsNone(self.agent._decide(log_entry=success, details={}))
        self.assertEqual(self.agent._decide(log_entry=recovered_warning, details={})["level"], "warning")

    def _create_monitor_log(
        self,
        *,
        source: str,
        event_type: str,
        level: str,
        status: str,
        created_at: datetime,
        confidence: float | None = None,
    ) -> MonitorLog:
        record = MonitorLog(
            category="test",
            source=source,
            event_type=event_type,
            level=level,
            title=f"{event_type} title",
            summary=f"{event_type} summary",
            status=status,
            confidence=confidence,
            created_at=created_at,
            updated_at=created_at,
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record


if __name__ == "__main__":
    unittest.main()
