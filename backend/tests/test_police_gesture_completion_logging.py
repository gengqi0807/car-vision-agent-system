from app.services.police_gesture_local_runtime import NO_VIDEO_GESTURE, PoliceGestureVideoSession
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
