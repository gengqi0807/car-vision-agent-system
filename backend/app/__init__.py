"""Backend application package."""

import sys

# Allow imports like `from app...` to resolve whether the project is started
# from the repository root (`backend.app.main:app`) or inside `backend/`
# (`app.main:app`).
sys.modules.setdefault("app", sys.modules[__name__])
