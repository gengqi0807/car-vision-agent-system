import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
for path in (ROOT_DIR, BACKEND_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from app.core.database import init_database
except ImportError:
    from backend.app.core.database import init_database


def main() -> None:
    init_database()
    print("Database schema initialized.")


if __name__ == "__main__":
    main()
