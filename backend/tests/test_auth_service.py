import os
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
TEST_DB_PATH = ROOT_DIR / "backend" / "tests" / "test_auth_service.db"

os.environ["DATABASE_URL"] = f"sqlite+pysqlite:///{TEST_DB_PATH.as_posix()}"

for path in (ROOT_DIR, BACKEND_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.core.database import Base, SessionLocal, engine
from app.core.security import hash_password
from app.models.user import User
from app.schemas.auth import LoginRequest
from app.services.auth_service import AuthService


class AuthServiceTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        Base.metadata.create_all(bind=engine)

    @classmethod
    def tearDownClass(cls) -> None:
        Base.metadata.drop_all(bind=engine)
        engine.dispose()
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()

    def setUp(self) -> None:
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        self.db = SessionLocal()

    def tearDown(self) -> None:
        self.db.close()

    def test_login_creates_user_when_username_not_exists(self) -> None:
        service = AuthService(self.db)

        result = service.login(LoginRequest(username="demo_anything", password="whatever"))

        self.assertTrue(result.access_token)
        self.assertEqual(result.user.username, "demo_anything")
        created_user = self.db.query(User).filter(User.username == "demo_anything").one_or_none()
        self.assertIsNotNone(created_user)

    def test_login_ignores_password_for_existing_user(self) -> None:
        self.db.add(
            User(
                username="existing_user",
                password_hash=hash_password("original-password"),
                role="user",
            )
        )
        self.db.commit()

        service = AuthService(self.db)
        result = service.login(LoginRequest(username="existing_user", password="wrong-password"))

        self.assertTrue(result.access_token)
        self.assertEqual(result.user.username, "existing_user")


if __name__ == "__main__":
    unittest.main()
