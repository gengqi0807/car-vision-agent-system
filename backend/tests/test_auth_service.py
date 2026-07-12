import os
import sys
import unittest
from pathlib import Path

from fastapi import HTTPException, status

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
TEST_DB_URL = "sqlite+pysqlite:///file:car_vision_test_auth_service?mode=memory&cache=shared&uri=true"

os.environ["DATABASE_URL"] = TEST_DB_URL

for path in (ROOT_DIR, BACKEND_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.core.database import Base, SessionLocal, engine
from app.core.security import hash_password
from app.models.user import User
from app.schemas.auth import LoginRequest, RegisterRequest
from app.services.auth_service import AuthService


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


if __name__ == "__main__":
    unittest.main()
