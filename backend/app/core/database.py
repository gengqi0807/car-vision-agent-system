from collections.abc import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import settings
from app.utils.crypto import crypto_manager, normalize_email, normalize_phone, normalize_sensitive_value


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


def _ensure_user_security_columns() -> None:
    inspector = inspect(engine)
    if "users" not in inspector.get_table_names():
        return

    columns = {column["name"] for column in inspector.get_columns("users")}
    statements: list[str] = []

    if "email_encrypted" not in columns:
        statements.append("ALTER TABLE users ADD COLUMN email_encrypted VARCHAR(512) NULL")
    if "email_hash" not in columns:
        statements.append("ALTER TABLE users ADD COLUMN email_hash VARCHAR(64) NULL")
    if "phone_encrypted" not in columns:
        statements.append("ALTER TABLE users ADD COLUMN phone_encrypted VARCHAR(512) NULL")
    if "phone_hash" not in columns:
        statements.append("ALTER TABLE users ADD COLUMN phone_hash VARCHAR(64) NULL")
    if "wechat_openid_encrypted" not in columns:
        statements.append("ALTER TABLE users ADD COLUMN wechat_openid_encrypted VARCHAR(512) NULL")
    if "wechat_openid_hash" not in columns:
        statements.append("ALTER TABLE users ADD COLUMN wechat_openid_hash VARCHAR(64) NULL")

    indexes = {index["name"] for index in inspector.get_indexes("users")}
    unique_constraints = {constraint["name"] for constraint in inspector.get_unique_constraints("users")}
    existing_names = indexes | unique_constraints

    if "uq_users_email_hash" not in existing_names:
        statements.append("CREATE UNIQUE INDEX uq_users_email_hash ON users (email_hash)")
    if "uq_users_phone_hash" not in existing_names:
        statements.append("CREATE UNIQUE INDEX uq_users_phone_hash ON users (phone_hash)")
    if "uq_users_wechat_openid_hash" not in existing_names:
        statements.append("CREATE UNIQUE INDEX uq_users_wechat_openid_hash ON users (wechat_openid_hash)")

    if not statements:
        return

    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


def _migrate_legacy_user_sensitive_data() -> None:
    inspector = inspect(engine)
    if "users" not in inspector.get_table_names():
        return

    columns = {column["name"] for column in inspector.get_columns("users")}
    required = {
        "id",
        "email",
        "phone",
        "wechat_openid",
        "email_encrypted",
        "email_hash",
        "phone_encrypted",
        "phone_hash",
        "wechat_openid_encrypted",
        "wechat_openid_hash",
    }
    if not required.issubset(columns):
        return

    with engine.begin() as connection:
        rows = connection.execute(
            text(
                """
                SELECT
                    id,
                    email,
                    phone,
                    wechat_openid,
                    email_encrypted,
                    email_hash,
                    phone_encrypted,
                    phone_hash,
                    wechat_openid_encrypted,
                    wechat_openid_hash
                FROM users
                """
            )
        ).mappings()

        for row in rows:
            email = normalize_email(row["email"])
            phone = normalize_phone(row["phone"])
            wechat_openid = normalize_sensitive_value(row["wechat_openid"])

            updates: dict[str, object] = {}

            if email and not row["email_encrypted"]:
                updates["email_encrypted"] = crypto_manager.encrypt(email)
                updates["email_hash"] = crypto_manager.fingerprint(email)
                updates["email"] = None

            if phone and not row["phone_encrypted"]:
                updates["phone_encrypted"] = crypto_manager.encrypt(phone)
                updates["phone_hash"] = crypto_manager.fingerprint(phone)
                updates["phone"] = None

            if wechat_openid and not row["wechat_openid_encrypted"]:
                updates["wechat_openid_encrypted"] = crypto_manager.encrypt(wechat_openid)
                updates["wechat_openid_hash"] = crypto_manager.fingerprint(wechat_openid)
                updates["wechat_openid"] = None

            if not updates:
                continue

            params = {"id": row["id"], **updates}
            assignments = ", ".join(f"{column} = :{column}" for column in updates)
            connection.execute(
                text(f"UPDATE users SET {assignments} WHERE id = :id"),
                params,
            )


def init_database() -> None:
    ensure_database_exists()
    Base.metadata.create_all(bind=engine)
    _ensure_user_security_columns()
    _migrate_legacy_user_sensitive_data()
