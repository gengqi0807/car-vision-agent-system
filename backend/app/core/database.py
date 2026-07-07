from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import settings


class Base(DeclarativeBase):
    pass


engine = create_engine(
    settings.sqlalchemy_database_url,
    future=True,
    echo=False,
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_database_exists() -> None:
    url = make_url(settings.sqlalchemy_database_url)
    if url.get_backend_name() != "mysql" or not url.database:
        return

    admin_engine = create_engine(url.set(database="mysql"), future=True, echo=False, pool_pre_ping=True)
    database_name = url.database.replace("`", "``")
    with admin_engine.begin() as connection:
        connection.execute(
            text(
                f"CREATE DATABASE IF NOT EXISTS `{database_name}` "
                f"CHARACTER SET {settings.mysql_charset} COLLATE {settings.mysql_charset}_unicode_ci"
            )
        )
    admin_engine.dispose()


def init_database() -> None:
    ensure_database_exists()
    Base.metadata.create_all(bind=engine)
