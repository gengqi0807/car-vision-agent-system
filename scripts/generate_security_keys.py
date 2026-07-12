import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    from app.utils.crypto import crypto_manager
except ImportError:
    from backend.app.utils.crypto import crypto_manager


def main() -> None:
    encryption_key, hash_key = crypto_manager.generate_env_keys()
    print(f"DATA_ENCRYPTION_KEY={encryption_key}")
    print(f"DATA_HASH_KEY={hash_key}")


if __name__ == "__main__":
    main()
