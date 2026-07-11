from app.core.database import Base, engine
from app.models import *  # noqa: F403


def main() -> None:
    Base.metadata.create_all(bind=engine)
    print("Database tables initialized.")


if __name__ == "__main__":
    main()
