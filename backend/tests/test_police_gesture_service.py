import asyncio
import os
import sys
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
TEST_DB_URL = "sqlite+pysqlite:///file:car_vision_test_police_gesture_service?mode=memory&cache=shared&uri=true"

os.environ["DATABASE_URL"] = TEST_DB_URL

for path in (ROOT_DIR, BACKEND_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.core.database import Base
from app.models.monitor_log import MonitorLog
from app.services import police_gesture_service as police_gesture_service_module
from app.services.police_gesture_local_runtime import NO_VIDEO_GESTURE
from app.services.police_gesture_service import PoliceGestureService


def test_record_video_monitor_result_persists_no_detection_warning(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'police_monitor.db'}", future=True)
    testing_session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(police_gesture_service_module, "SessionLocal", testing_session)

    service = PoliceGestureService()

    asyncio.run(
        service._record_video_monitor_result(
            filename="no-gesture.mp4",
            user_id=1,
            gesture=NO_VIDEO_GESTURE,
            confidence=0.0,
            processed_frame_count=48,
            total_frames=48,
        )
    )

    with testing_session() as session:
        logs = session.query(MonitorLog).all()

    assert len(logs) == 1
    assert logs[0].event_type == "police_gesture_no_detection"
    assert logs[0].level == "warning"
    assert logs[0].status == "no_detection"
