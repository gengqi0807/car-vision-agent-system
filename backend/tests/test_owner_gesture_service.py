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
                    gesture="swipe_right",
                    confidence=0.86,
                    control_action="SwitchNextFeature",
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
    assert panel_state.current_mode == "call"
    assert panel_state.phone_call_active is True
    assert panel_state.last_gesture == "thumbs_up"
    assert panel_state.last_command == "AnswerCall"
    assert panel_state.focus_tile == "call"


def test_build_panel_state_isolated_by_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'owner_gesture_session.db'}", future=True)
    testing_session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    service = OwnerGestureService()
    started_at = datetime(2026, 1, 1, 12, 0, 0)

    with testing_session() as db:
        db.add_all(
            [
                OwnerGestureRecord(
                    user_id=11,
                    session_id="old-session",
                    gesture="open_palm",
                    confidence=0.92,
                    control_action="WakeSystem",
                    is_triggered=True,
                    created_at=started_at,
                ),
                OwnerGestureRecord(
                    user_id=11,
                    session_id="old-session",
                    gesture="thumbs_up",
                    confidence=0.88,
                    control_action="AnswerCall",
                    is_triggered=True,
                    created_at=started_at + timedelta(seconds=1),
                ),
                OwnerGestureRecord(
                    user_id=11,
                    session_id="new-session",
                    gesture="未检测到手部",
                    confidence=0.0,
                    control_action="None",
                    is_triggered=False,
                    created_at=started_at + timedelta(seconds=5),
                ),
            ]
        )
        db.commit()

        panel_state = service.control_panel(db, 11, session_id="new-session")

    assert panel_state.system_awake is False
    assert panel_state.current_mode == "home"
    assert panel_state.phone_call_active is False
    assert panel_state.last_gesture == "未检测到手部"
    assert panel_state.last_command is None


def test_build_panel_state_supports_function_switch_volume_confirm_and_home(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'owner_gesture_features.db'}", future=True)
    testing_session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    service = OwnerGestureService()
    started_at = datetime(2026, 1, 1, 12, 0, 0)

    with testing_session() as db:
        db.add_all(
            [
                OwnerGestureRecord(
                    user_id=13,
                    session_id="demo",
                    gesture="open_palm",
                    confidence=0.92,
                    control_action="WakeSystem",
                    is_triggered=True,
                    created_at=started_at,
                ),
                OwnerGestureRecord(
                    user_id=13,
                    session_id="demo",
                    gesture="swipe_right",
                    confidence=0.85,
                    control_action="SwitchNextFeature",
                    is_triggered=True,
                    created_at=started_at + timedelta(seconds=1),
                ),
                OwnerGestureRecord(
                    user_id=13,
                    session_id="demo",
                    gesture="index_circle",
                    confidence=0.88,
                    control_action="AdjustVolume",
                    is_triggered=True,
                    created_at=started_at + timedelta(seconds=2),
                ),
                OwnerGestureRecord(
                    user_id=13,
                    session_id="demo",
                    gesture="swipe_right",
                    confidence=0.85,
                    control_action="SwitchNextFeature",
                    is_triggered=True,
                    created_at=started_at + timedelta(seconds=3),
                ),
                OwnerGestureRecord(
                    user_id=13,
                    session_id="demo",
                    gesture="fist",
                    confidence=0.90,
                    control_action="ConfirmAction",
                    is_triggered=True,
                    created_at=started_at + timedelta(seconds=4),
                ),
                OwnerGestureRecord(
                    user_id=13,
                    session_id="demo",
                    gesture="wave",
                    confidence=0.83,
                    control_action="ReturnHome",
                    is_triggered=True,
                    created_at=started_at + timedelta(seconds=5),
                ),
            ]
        )
        db.commit()

        panel_state = service.control_panel(db, 13)

    assert panel_state.system_awake is True
    assert panel_state.current_mode == "home"
    assert panel_state.volume == 38
    assert panel_state.climate_temperature == 22
    assert panel_state.comfort_scene == "舒享"
    assert panel_state.focus_tile == "home"
    assert panel_state.last_command == "ReturnHome"
    assert panel_state.last_feedback == "已挥手返回主页。"


def test_motion_refinement_detects_wave_swipe_and_circle():
    service = OwnerGestureService()

    wave_points = [(0.34, 0.55), (0.48, 0.55), (0.36, 0.54), (0.51, 0.55), (0.39, 0.56)]
    swipe_points = [(0.22, 0.42), (0.31, 0.43), (0.44, 0.44), (0.58, 0.45)]
    circle_points = [(0.40, 0.30), (0.48, 0.38), (0.42, 0.48), (0.32, 0.42), (0.36, 0.32)]

    assert service._is_wave_motion(wave_points) is True
    assert service._classify_swipe(swipe_points) == "swipe_right"
    assert service._is_circle_motion(circle_points) is True


def test_prepare_frame_for_inference_downsizes_large_frames():
    service = OwnerGestureService()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    prepared = service._prepare_frame_for_inference(frame)

    assert prepared.shape[1] == 640
    assert prepared.shape[0] == 360


def test_build_annotated_image_returns_data_url():
    service = OwnerGestureService()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    hand = [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(21)]

    annotated = service._build_annotated_image(
        frame,
        hands=[hand],
        gesture="open_palm",
        confidence=0.92,
        control_command="WakeSystem",
    )

    assert annotated is not None
    assert annotated.startswith("data:image/jpeg;base64,")
