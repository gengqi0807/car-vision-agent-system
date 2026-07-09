from datetime import datetime, timedelta

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.models.owner_gesture_record import OwnerGestureRecord
from app.services.owner_gesture_service import OwnerGestureService


def test_build_panel_state_uses_latest_wake_window(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'owner_gesture.db'}", future=True)
    testing_session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    service = OwnerGestureService()
    started_at = datetime(2026, 1, 1, 12, 0, 0)

    with testing_session() as db:
        db.add_all(
            [
                OwnerGestureRecord(
                    user_id=7,
                    session_id="demo",
                    gesture="open_palm",
                    confidence=0.92,
                    control_action="WakeSystem",
                    is_triggered=True,
                    created_at=started_at,
                ),
                OwnerGestureRecord(
                    user_id=7,
                    session_id="demo",
                    gesture="thumbs_up",
                    confidence=0.88,
                    control_action="AnswerCall",
                    is_triggered=True,
                    created_at=started_at + timedelta(seconds=1),
                ),
                OwnerGestureRecord(
                    user_id=7,
                    session_id="demo",
                    gesture="open_palm",
                    confidence=0.93,
                    control_action="WakeSystem",
                    is_triggered=True,
                    created_at=started_at + timedelta(seconds=2),
                ),
            ]
        )
        db.commit()

        panel_state = service.control_panel(db, 7)

    assert panel_state.system_awake is True
    assert panel_state.current_mode == "home"
    assert panel_state.phone_call_active is False
    assert panel_state.last_gesture == "open_palm"
    assert panel_state.last_command == "WakeSystem"


def test_build_panel_state_keeps_latest_triggered_mode_and_call_state(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'owner_gesture_mode.db'}", future=True)
    testing_session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    service = OwnerGestureService()
    started_at = datetime(2026, 1, 1, 12, 0, 0)

    with testing_session() as db:
        db.add_all(
            [
                OwnerGestureRecord(
                    user_id=9,
                    session_id="demo",
                    gesture="open_palm",
                    confidence=0.92,
                    control_action="WakeSystem",
                    is_triggered=True,
                    created_at=started_at,
                ),
                OwnerGestureRecord(
                    user_id=9,
                    session_id="demo",
                    gesture="fist",
                    confidence=0.90,
                    control_action="ConfirmAction",
                    is_triggered=True,
                    created_at=started_at + timedelta(seconds=1),
                ),
                OwnerGestureRecord(
                    user_id=9,
                    session_id="demo",
                    gesture="thumbs_up",
                    confidence=0.88,
                    control_action="AnswerCall",
                    is_triggered=True,
                    created_at=started_at + timedelta(seconds=2),
                ),
            ]
        )
        db.commit()

        panel_state = service.control_panel(db, 9)

    assert panel_state.system_awake is True
    assert panel_state.current_mode == "control"
    assert panel_state.phone_call_active is True
    assert panel_state.last_gesture == "thumbs_up"
    assert panel_state.last_command == "AnswerCall"


def test_prepare_frame_for_inference_downsizes_large_frames():
    service = OwnerGestureService()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    prepared = service._prepare_frame_for_inference(frame)

    assert prepared.shape[1] == 640
    assert prepared.shape[0] == 360
