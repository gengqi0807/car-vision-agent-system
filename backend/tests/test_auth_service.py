import os
import sys
import unittest
from pathlib import Path

from fastapi import HTTPException, status
from pydantic import ValidationError

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
TEST_DB_URL = "sqlite+pysqlite:///file:car_vision_test_auth_service?mode=memory&cache=shared&uri=true"

os.environ["DATABASE_URL"] = TEST_DB_URL

for path in (ROOT_DIR, BACKEND_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.core.database import Base, SessionLocal, engine
from app.core.security import hash_password
from app.models.alert_log import AlertLog
from app.models.monitor_log import MonitorLog
from app.models.user import User
from app.schemas.auth import LoginRequest, RegisterRequest, UpdateProfileRequest
from app.services.auth_service import AuthService
from app.utils.user_uid import USER_UID_LENGTH


class AuthServiceTestCase(unittest.TestCase):
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

    def test_register_and_login_with_password(self) -> None:
        service = AuthService(self.db)

        created = service.register(
            RegisterRequest(
                username="demo_anything",
                password="whatever",
                email="demo@example.com",
                phone="13800138000",
            )
        )
        result = service.login(LoginRequest(username="demo_anything", password="whatever"))

        self.assertTrue(result.access_token)
        self.assertEqual(len(result.user.uid), USER_UID_LENGTH)
        self.assertTrue(result.user.uid.isdigit())
        self.assertEqual(result.user.uid, created.uid)
        self.assertEqual(result.user.username, "demo_anything")
        self.assertEqual(created.username, "demo_anything")
        created_user = self.db.query(User).filter(User.username == "demo_anything").one_or_none()
        self.assertIsNotNone(created_user)
        assert created_user is not None
        self.assertIsNotNone(created_user.email_encrypted)
        self.assertIsNotNone(created_user.email_hash)
        self.assertIsNotNone(created_user.phone_encrypted)
        self.assertIsNotNone(created_user.phone_hash)
        self.assertEqual(created_user.email, "demo@example.com")
        self.assertEqual(created_user.phone, "13800138000")

    def test_login_rejects_wrong_password(self) -> None:
        self.db.add(
            User(
                username="existing_user",
                password_hash=hash_password("original-password"),
                role="user",
            )
        )
        self.db.commit()

        service = AuthService(self.db)
        with self.assertRaises(HTTPException) as context:
            service.login(LoginRequest(username="existing_user", password="wrong-password"))

        self.assertEqual(context.exception.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_register_rejects_invalid_email_and_phone(self) -> None:
        with self.assertRaises(ValidationError):
            RegisterRequest(
                username="demo_anything",
                password="whatever",
                email="invalid-email",
                phone="13800138000",
            )

        with self.assertRaises(ValidationError):
            RegisterRequest(
                username="demo_anything",
                password="whatever",
                email="demo@example.co",
                phone="13800138000",
            )

        with self.assertRaises(ValidationError):
            RegisterRequest(
                username="demo_anything",
                password="whatever",
                email="demo@example.com",
                phone="123456",
            )

    def test_update_profile_invalid_contact_writes_warning_alert(self) -> None:
        self.db.add(
            User(
                username="contact_user",
                password_hash=hash_password("original-password"),
                role="user",
            )
        )
        self.db.commit()
        user = self.db.query(User).filter(User.username == "contact_user").one()

        service = AuthService(self.db)
        with self.assertRaises(HTTPException):
            service.update_profile(
                user.id,
                UpdateProfileRequest(username="contact_user", email="demo@example.co", phone="13800138000"),
            )

        alert_log = (
            self.db.query(AlertLog)
            .filter(
                AlertLog.source == "auth",
                AlertLog.event_type == "update_profile",
                AlertLog.level == "warning",
            )
            .one_or_none()
        )
        self.assertIsNotNone(alert_log)
        assert alert_log is not None
        self.assertIn(user.uid, alert_log.summary)
        self.assertIn("邮箱格式不正确", alert_log.summary)

    def test_update_profile_failure_writes_warning_monitor_log(self) -> None:
        self.db.add_all(
            [
                User(
                    username="first_user",
                    password_hash=hash_password("original-password"),
                    role="user",
                ),
                User(
                    username="second_user",
                    password_hash=hash_password("original-password"),
                    role="user",
                ),
            ]
        )
        self.db.commit()
        first_user = self.db.query(User).filter(User.username == "first_user").one()

        service = AuthService(self.db)
        with self.assertRaises(HTTPException):
            service.update_profile(
                first_user.id,
                UpdateProfileRequest(username="second_user", email=None, phone=None),
            )

        warning_log = (
            self.db.query(MonitorLog)
            .filter(
                MonitorLog.event_type == "update_profile",
                MonitorLog.level == "warning",
                MonitorLog.status == "Failed",
            )
            .one_or_none()
        )
        self.assertIsNotNone(warning_log)
        assert warning_log is not None
        self.assertIn(first_user.uid, warning_log.summary)
        self.assertIn("用户名已存在", warning_log.summary)

        alert_log = (
            self.db.query(AlertLog)
            .filter(
                AlertLog.source == "auth",
                AlertLog.event_type == "update_profile",
                AlertLog.level == "warning",
            )
            .one_or_none()
        )
        self.assertIsNotNone(alert_log)
        assert alert_log is not None
        self.assertEqual(alert_log.title, "个人资料修改失败")
        self.assertIn(first_user.uid, alert_log.summary)


if __name__ == "__main__":
    unittest.main()
