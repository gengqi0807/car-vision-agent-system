import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
for path in (ROOT_DIR, BACKEND_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.core.database import SessionLocal
from app.services.alert_service import AlertService


def main():
    with SessionLocal() as session:
        svc = AlertService(session)
        items = svc.list_monitor_logs(limit=20)
        for item in items:
            print(item.model_dump())


if __name__ == '__main__':
    main()
