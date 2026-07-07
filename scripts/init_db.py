import sys
from pathlib import Path

# 设置后端目录（采用 wangxiaoyan 分支的简洁写法）
BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# 尝试导入 init_database，兼容两种导入路径（保留 HEAD 分支的健壮性）
try:
    from app.core.database import init_database
except ImportError:
    from backend.app.core.database import init_database

# 导入所有模型以确保表定义被注册（来自 wangxiaoyan 分支）
from app.models import *  # noqa: F403


def main() -> None:
    init_database()
    print("Database schema initialized.")


if __name__ == "__main__":
    main()