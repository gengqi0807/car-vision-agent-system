from types import SimpleNamespace

from app.services.police_gesture_local_runtime import (
    NO_VIDEO_GESTURE,
    PoliceGestureVideoSession,
    _select_target_person,
    _track_target_person,
)
from app.services.police_gesture_service import PoliceGestureService


def test_action_end_exposes_final_gesture_once() -> None:
    session = PoliceGestureVideoSession.__new__(PoliceGestureVideoSession)
    session.action_state = "active"
    session.action_frame_count = 12
    session.hip_both_frames = 8
    session.dl_window = [("停止信号", 0.91)]
    session.dl_filtered_gesture = "停止信号"
    session.dl_filtered_confidence = 0.91
    session.dl_gesture = "停止信号"
    session.dl_confidence = 0.91
    session.dl_engine = None
    session.action_flash_text = ""
    session.action_flash_remaining = 0
    session.action_flash_frames = 30
    session.completed_gesture = None
    session.completed_confidence = 0.0

    session._reset_action_state(show_flash=True)

    assert session._consume_completed_gesture() == "stop"
    assert session.completed_confidence == 0.91
    assert session._consume_completed_gesture() is None


def test_only_valid_final_gesture_is_recordable() -> None:
    assert PoliceGestureService._is_recordable_completed_gesture("stop", 0.91)

    for gesture in (None, "", NO_VIDEO_GESTURE, "no_pose", "unknown", "other", "其他", "其他手势", "无手势"):
        assert not PoliceGestureService._is_recordable_completed_gesture(gesture, 0.99)

    assert not PoliceGestureService._is_recordable_completed_gesture("stop", 0.50)


def _person(center_x: float, center_y: float, shoulder_width: float):
    landmarks = [SimpleNamespace(x=center_x, y=center_y) for _ in range(25)]
    landmarks[11] = SimpleNamespace(x=center_x - shoulder_width / 2.0, y=center_y - 0.1)
    landmarks[12] = SimpleNamespace(x=center_x + shoulder_width / 2.0, y=center_y - 0.1)
    landmarks[23] = SimpleNamespace(x=center_x - 0.05, y=center_y + 0.1)
    landmarks[24] = SimpleNamespace(x=center_x + 0.05, y=center_y + 0.1)
    return landmarks


def test_target_selection_prefers_main_centered_person() -> None:
    side_person = _person(0.12, 0.5, 0.16)
    main_person = _person(0.5, 0.5, 0.24)

    target_index, target_info = _select_target_person([side_person, main_person])

    assert target_index == 1
    assert target_info is not None
    assert target_info["cx"] == 0.5


def test_target_tracking_keeps_locked_person_when_another_person_moves_in() -> None:
    locked_info = {"cx": 0.3, "cy": 0.5, "sw": 0.2}
    locked_person = _person(0.32, 0.5, 0.18)
    interfering_person = _person(0.55, 0.5, 0.32)

    target_index, target_info = _track_target_person(
        [interfering_person, locked_person],
        locked_info,
        track_threshold=0.35,
    )

    assert target_index == 1
    assert target_info is not None
    assert target_info["cx"] == 0.32
