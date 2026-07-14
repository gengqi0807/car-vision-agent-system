import asyncio
from datetime import datetime, timedelta

import cv2
import numpy as np
import app.models_infer.gesture_classifier as gesture_classifier_module
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.models.owner_gesture_record import OwnerGestureRecord
from app.models_infer.gesture_classifier import GestureClassifier, HandGestureTracker
from app.schemas.gesture import ControlPanelState
from app.services.owner_gesture_service import OwnerGestureService, RuntimeGestureSession


def test_classify_frame_keeps_static_result_idle_until_one_second(monkeypatch):
    classifier = GestureClassifier(domain="owner")
    classifier._classify_dynamic_lstm = lambda _keypoints: None  # type: ignore[method-assign]
    classifier.tracker = None
    classifier.classify_static = lambda _keypoints: ("palm", 0.95)  # type: ignore[method-assign]
    keypoints = [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(21)]
    now = [100.0]
    monkeypatch.setattr(gesture_classifier_module.time, "time", lambda: now[0])

    assert classifier.classify_frame(keypoints) == ("idle", 0.0)
    now[0] = 100.99
    assert classifier.classify_frame(keypoints) == ("idle", 0.0)
    now[0] = 101.0
    assert classifier.classify_frame(keypoints) == ("palm", 0.95)


def test_swipe_transition_accepts_open_palm_and_short_intermediate_state():
    classifier = GestureClassifier(domain="owner")

    assert classifier._detect_swipe_transition("fist", 100.0) is None
    assert classifier._detect_swipe_transition("unknown", 100.6) is None
    assert classifier._detect_swipe_transition("open_palm", 101.2) == "swipe_left"

    classifier._swipe_from = None
    classifier._swipe_from_time = 0.0
    assert classifier._detect_swipe_transition("open_palm", 200.0) is None
    assert classifier._detect_swipe_transition("pointing", 200.5) is None
    assert classifier._detect_swipe_transition("fist", 201.1) == "swipe_right"


def test_swipe_result_overrides_locked_endpoint_gesture():
    service = OwnerGestureService()
    session = RuntimeGestureSession(
        panel_state=ControlPanelState(
            **service._default_panel_state,
            last_gesture=None,
            last_command=None,
            last_command_at=None,
            updated_at=None,
        ),
        hand_present_prev=True,
        result_locked=True,
        locked_gesture="fist",
        locked_confidence=0.91,
    )

    gesture, confidence = service._lock_camera_result(
        session,
        gesture="swipe_left",
        confidence=0.9,
        hand_present=True,
    )

    assert gesture == "swipe_left"
    assert confidence == 0.9


def test_volume_actions_can_repeat_only_after_fresh_circle_observation():
    service = OwnerGestureService()
    session = RuntimeGestureSession(
        panel_state=ControlPanelState(
            **service._default_panel_state,
            last_gesture=None,
            last_command=None,
            last_command_at=None,
            updated_at=None,
        )
    )

    assert service._should_fire_verified_camera_action(
        session,
        action="volume_up",
        control_command="AdjustVolumeUp",
        observed_gesture="circle_cw",
    ) is True
    assert service._should_fire_verified_camera_action(
        session,
        action="volume_up",
        control_command="AdjustVolumeUp",
        observed_gesture="idle",
    ) is False

    session.last_fired_at_monotonic -= 1.0
    assert service._should_fire_verified_camera_action(
        session,
        action="volume_up",
        control_command="AdjustVolumeUp",
        observed_gesture="circle_cw",
    ) is True
    assert service._should_fire_verified_camera_action(
        session,
        action="volume_down",
        control_command="AdjustVolumeDown",
        observed_gesture="circle_ccw",
    ) is True

    assert service._should_fire_verified_camera_action(
        session,
        action="confirm",
        control_command="ConfirmAction",
        observed_gesture="fist",
    ) is True
    assert service._should_fire_verified_camera_action(
        session,
        action="confirm",
        control_command="ConfirmAction",
        observed_gesture="fist",
    ) is False


def test_build_panel_state_keeps_page_on_repeated_wake(tmp_path):
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
    assert panel_state.current_mode == "call"
    assert panel_state.phone_call_active is True
    assert panel_state.last_gesture == "open_palm"
    assert panel_state.last_command == "WakeSystem"
    assert panel_state.last_feedback == "系统已唤醒。"


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
                    gesture="thunb_index",
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
    assert panel_state.last_feedback == "已捏指返回主页。"


def test_motion_refinement_detects_wave_swipe_and_circle():
    service = OwnerGestureService()

    wave_points = [(0.34, 0.55), (0.48, 0.55), (0.36, 0.54), (0.51, 0.55), (0.39, 0.56)]
    swipe_points = [(0.22, 0.42), (0.31, 0.43), (0.44, 0.44), (0.58, 0.45)]
    circle_points = [(0.40, 0.30), (0.48, 0.38), (0.42, 0.48), (0.32, 0.42), (0.36, 0.32)]

    assert service._is_wave_motion(wave_points) is True
    assert service._classify_swipe(swipe_points) == "swipe_right"
    assert service._is_circle_motion(circle_points) is True


def test_open_palm_is_not_refined_by_weak_motion_history():
    service = OwnerGestureService()

    assert service._is_wave_motion([(0.20, 0.50), (0.34, 0.50), (0.20, 0.50), (0.34, 0.50), (0.20, 0.50)]) is False


def test_hand_tracker_rejects_small_oscillation_as_wave():
    tracker = HandGestureTracker()

    def make_hand(x: float) -> list[dict]:
        return [{"x": x, "y": 0.50, "z": 0.0} for _ in range(21)]

    for x in [0.20, 0.34, 0.20, 0.34, 0.20, 0.34, 0.20, 0.34, 0.20, 0.34, 0.20, 0.34]:
        assert tracker.update(make_hand(x)) is None


def test_classify_static_prefers_high_confidence_ml_wave_over_open_palm_heuristic():
    classifier = GestureClassifier(domain="owner")
    classifier._classify_ml = lambda _keypoints: ("wave", 0.99)  # type: ignore[method-assign]
    classifier._classify_heuristic = lambda _keypoints: ("open_palm", 0.92)  # type: ignore[method-assign]

    gesture, confidence = classifier.classify_static([{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(21)])

    assert gesture == "wave"
    assert confidence == 0.99


def test_classify_static_prefers_high_confidence_ml_fist_over_open_palm_heuristic():
    classifier = GestureClassifier(domain="owner")
    classifier._classify_ml = lambda _keypoints: ("fist", 0.97)  # type: ignore[method-assign]
    classifier._classify_heuristic = lambda _keypoints: ("open_palm", 0.92)  # type: ignore[method-assign]

    gesture, confidence = classifier.classify_static([{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(21)])

    assert gesture == "fist"
    assert confidence == 0.97


def test_classify_static_can_keep_high_confidence_ml_thumbs_up_when_heuristic_is_point():
    classifier = GestureClassifier(domain="owner")
    classifier._classify_ml = lambda _keypoints: ("thumbs_up", 0.96)  # type: ignore[method-assign]
    classifier._classify_heuristic = lambda _keypoints: ("point", 0.84)  # type: ignore[method-assign]

    gesture, confidence = classifier.classify_static([{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(21)])

    assert gesture == "thumbs_up"
    assert confidence == 0.96


def test_classify_static_prefers_high_confidence_ml_fist_over_ambiguous_heuristic_thumbs_up():
    classifier = GestureClassifier(domain="owner")
    classifier._classify_ml = lambda _keypoints: ("fist", 0.91)  # type: ignore[method-assign]
    classifier._classify_heuristic = lambda _keypoints: ("thumbs_up", 0.88)  # type: ignore[method-assign]

    gesture, confidence = classifier.classify_static([{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(21)])

    assert gesture == "fist"
    assert confidence == 0.91


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


def test_should_trigger_immediately_for_image_input():
    service = OwnerGestureService()

    triggered = service._should_trigger(
        gesture="fist",
        control_command="ConfirmAction",
        recent_records=[],
        input_mode="image",
    )

    assert triggered is True


def test_thunb_index_replaces_wave_as_return_home_gesture():
    service = OwnerGestureService()

    assert service._map_gesture_to_command("thunb_index") == ("ReturnHome", True)
    assert service._map_gesture_to_command("thumb_index") == ("ReturnHome", True)
    assert service._map_gesture_to_command("wave") == (None, False)


def test_wake_system_keeps_current_page_when_already_awake():
    service = OwnerGestureService()
    state = {
        **service._default_panel_state,
        "system_awake": True,
        "phone_call_active": True,
        "current_mode": "call",
        "focus_tile": "call",
    }

    service._apply_command_to_state(state, "WakeSystem")

    assert state["system_awake"] is True
    assert state["phone_call_active"] is True
    assert state["current_mode"] == "call"
    assert state["focus_tile"] == "call"
    assert state["last_feedback"] == "系统已唤醒。"


def test_classify_upload_gesture_image_mode_uses_single_hand_static_result():
    service = OwnerGestureService()

    class StubClassifier:
        def classify_static(self, keypoints):
            assert len(keypoints) == 21
            return "fist", 0.91

        def classify(self, keypoints, domain="owner"):
            raise AssertionError("image mode should not use dynamic upload classify path")

    service._classifier = StubClassifier()

    gesture, confidence = service._classify_upload_gesture(
        raw_keypoints=[{"x": 0.1, "y": 0.2, "z": 0.0} for _ in range(21)],
        num_hands=1,
        recent_records=[],
        input_mode="image",
    )

    assert gesture == "fist"
    assert confidence == 0.91


def test_classify_upload_gesture_camera_mode_applies_motion_refinement():
    service = OwnerGestureService()

    class StubClassifier:
        def classify(self, keypoints, domain="owner"):
            assert len(keypoints) == 21
            return {"gesture": "open_palm", "confidence": 0.82}

    service._classifier = StubClassifier()
    service._refine_motion_gesture = lambda **_: "wave"  # type: ignore[method-assign]

    gesture, confidence = service._classify_upload_gesture(
        raw_keypoints=[{"x": 0.1, "y": 0.2, "z": 0.0} for _ in range(21)],
        num_hands=1,
        recent_records=[],
        input_mode="camera",
    )

    assert gesture == "wave"
    assert confidence == 0.82


def test_select_primary_hand_prefers_triggerable_image_hand():
    service = OwnerGestureService()

    left_hand = [{"x": 0.18, "y": 0.35 + index * 0.001, "z": 0.0} for index in range(21)]
    right_hand = [{"x": 0.78, "y": 0.25 + index * 0.001, "z": 0.0} for index in range(21)]

    class StubClassifier:
        def classify_static(self, keypoints):
            return ("point", 0.84) if keypoints[0]["x"] < 0.5 else ("thumbs_up", 0.88)

    service._classifier = StubClassifier()

    primary_hand, gesture, confidence = service._select_primary_hand(
        [left_hand, right_hand],
        input_mode="image",
    )

    assert primary_hand == right_hand
    assert gesture == "thumbs_up"
    assert confidence == 0.88


def test_process_frame_image_mode_matches_verify_script_first_hand_logic(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'owner_gesture_upload.db'}", future=True)
    testing_session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    service = OwnerGestureService()

    left_hand = [{"x": 0.18, "y": 0.35 + index * 0.001, "z": 0.0} for index in range(21)]
    right_hand = [{"x": 0.78, "y": 0.25 + index * 0.001, "z": 0.0} for index in range(21)]

    seen_shapes: list[tuple[int, int]] = []

    class StubHands:
        def infer(self, frame):
            seen_shapes.append((frame.shape[1], frame.shape[0]))
            return {
                "keypoints": left_hand + right_hand,
                "num_hands_detected": 2,
            }

    class StubClassifier:
        def classify_static(self, keypoints):
            return ("point", 0.84) if keypoints[0]["x"] < 0.5 else ("thumbs_up", 0.88)

    async def fake_capture_monitor_log(**_kwargs):
        return None

    service._hands = StubHands()
    service._classifier = StubClassifier()
    service._should_log_operation = lambda **_: False  # type: ignore[method-assign]
    service._should_record_behavior = lambda **_: False  # type: ignore[method-assign]
    service._capture_monitor_log = fake_capture_monitor_log  # type: ignore[method-assign]

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok is True

    with testing_session() as db:
        result = asyncio.run(
            service.process_frame(
                encoded.tobytes(),
                "upload.jpg",
                db=db,
                user_id=1,
                session_id="session-image",
                input_mode="image",
            )
        )

        stored = db.query(OwnerGestureRecord).all()

    assert seen_shapes == [(1920, 1080)]
    assert result.gesture == "idle"
    assert len(result.keypoints) == 21
    assert result.keypoints[0].x == left_hand[0]["x"]
    assert len(stored) == 1
    assert len(stored[0].hand_landmarks) == 42


def test_use_custom_primary_promotes_custom_model_to_primary_static_classifier(monkeypatch):
    custom_labels = ["fist", "palm", "thumb_up", "thumb_down", "thumb_index", "my_gesture"]

    def fake_load_custom(self):
        self._custom_model = object()
        self._custom_scaler = object()
        self._custom_labels = list(custom_labels)

    monkeypatch.setattr(GestureClassifier, "_load_custom_model", fake_load_custom)
    monkeypatch.setattr(GestureClassifier, "_load_ml_model", lambda self: None)
    monkeypatch.setattr(GestureClassifier, "_load_dynamic_lstm", lambda self: None)

    classifier = GestureClassifier(domain="owner", load_custom=True, use_custom_primary=True)

    # 自定义合并模型应晋升为主静态分类器（_ml_model / _ml_scaler / _ml_labels）
    assert classifier._ml_model is classifier._custom_model
    assert classifier._ml_scaler is classifier._custom_scaler
    assert classifier._ml_labels is not None
    promoted = classifier._ml_labels.tolist()
    assert "my_gesture" in promoted
    assert "fist" in promoted


def test_default_does_not_promote_custom_model(monkeypatch):
    def fake_load_custom(self):
        self._custom_model = object()
        self._custom_scaler = object()
        self._custom_labels = ["fist", "my_gesture"]

    monkeypatch.setattr(GestureClassifier, "_load_custom_model", fake_load_custom)
    monkeypatch.setattr(GestureClassifier, "_load_ml_model", lambda self: None)
    monkeypatch.setattr(GestureClassifier, "_load_dynamic_lstm", lambda self: None)

    classifier = GestureClassifier(domain="owner", load_custom=True, use_custom_primary=False)

    # 默认不晋升：主静态分类器保持固有模型（此处被桩跳过为 None），自定义模型独立保留
    assert classifier._ml_model is None
    assert classifier._custom_model is not None


def test_configure_stream_runtime_loads_custom_primary_classifier(monkeypatch):
    # 捕获实时流启动时创建的 GestureClassifier 构造参数，
    # 确保自定义合并模型被加载并晋升为主静态分类器（而非回退到固有模型）。
    captured = {}

    class FakeGestureClassifier:
        def __init__(self, domain="owner", load_custom=False, use_custom_primary=False):
            captured["domain"] = domain
            captured["load_custom"] = load_custom
            captured["use_custom_primary"] = use_custom_primary

    monkeypatch.setattr(
        gesture_classifier_module, "GestureClassifier", FakeGestureClassifier
    )
    monkeypatch.setattr(
        "app.models_infer.mediapipe_hands.MediaPipeHands.reset", lambda: None
    )
    monkeypatch.setattr(
        "app.models_infer.mediapipe_hands.MediaPipeHands.configure",
        lambda **kwargs: None,
    )

    service = OwnerGestureService()
    service._configure_stream_runtime()

    assert captured.get("domain") == "owner"
    assert captured.get("load_custom") is True
    assert captured.get("use_custom_primary") is True
