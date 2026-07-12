import asyncio
import os
import sys
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException, UploadFile

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
TEST_DB_URL = "sqlite+pysqlite:///file:car_vision_test_police_gesture_api?mode=memory&cache=shared&uri=true"

os.environ["DATABASE_URL"] = TEST_DB_URL

for path in (ROOT_DIR, BACKEND_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.api.v1 import police_gesture as police_gesture_api
from app.models.user import User


class PoliceGestureApiTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.current_user = User(
            id=1,
            username="demo_user",
            password_hash="hashed-password",
            role="user",
        )

    def test_current_police_gesture_records_failure_alert_on_unexpected_error(self) -> None:
        upload = UploadFile(file=BytesIO(b"fake-image-bytes"), filename="frame.jpg")

        with (
            patch.object(police_gesture_api.service, "process_frame", AsyncMock(side_effect=RuntimeError("boom"))),
            patch.object(police_gesture_api.service, "_capture_error", AsyncMock()) as capture_error,
        ):
            with self.assertRaises(HTTPException) as context:
                asyncio.run(
                    police_gesture_api.current_police_gesture(
                        file=upload,
                        current_user=self.current_user,
                    )
                )

        self.assertEqual(context.exception.status_code, 500)
        capture_error.assert_awaited_once()
        kwargs = capture_error.await_args.kwargs
        self.assertEqual(kwargs["filename"], "frame.jpg")
        self.assertEqual(kwargs["event_type"], "police_gesture_image_failure")
        self.assertEqual(kwargs["user_id"], 1)

    def test_recognize_police_gesture_video_records_failure_alert_on_runtime_error(self) -> None:
        upload = UploadFile(file=BytesIO(b"fake-video-bytes"), filename="clip.mp4")
        request = MagicMock()

        with (
            patch.object(police_gesture_api.service, "process_video_bytes", MagicMock(side_effect=RuntimeError("video boom"))),
            patch.object(police_gesture_api.service, "_capture_error", AsyncMock()) as capture_error,
        ):
            with self.assertRaises(HTTPException) as context:
                asyncio.run(
                    police_gesture_api.recognize_police_gesture_video(
                        request=request,
                        file=upload,
                        task_id="task-123",
                        current_user=self.current_user,
                    )
                )

        self.assertEqual(context.exception.status_code, 503)
        capture_error.assert_awaited_once()
        kwargs = capture_error.await_args.kwargs
        self.assertEqual(kwargs["filename"], "clip.mp4")
        self.assertEqual(kwargs["event_type"], "police_gesture_video_failure")
        self.assertEqual(kwargs["user_id"], 1)
        self.assertEqual(kwargs["details"]["task_id"], "task-123")


if __name__ == "__main__":
    unittest.main()
