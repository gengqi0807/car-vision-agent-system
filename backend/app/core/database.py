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


def _ensure_alert_table_compatibility() -> None:
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())

    if "alert_logs" in table_names:
        _ensure_alert_logs_columns(inspector)
        _backfill_alert_logs()

    if "alert_push_logs" in table_names:
        _ensure_alert_push_logs_columns(inspector)
        _backfill_alert_push_logs()

    if "user_operation_logs" in table_names:
        _ensure_updated_at_column("user_operation_logs")
        _backfill_updated_at_from_created_at("user_operation_logs")

    if "owner_gesture_records" in table_names:
        _ensure_owner_gesture_record_columns(inspector)
    if "police_gesture_records" in table_names:
        _ensure_police_gesture_record_columns(inspector)


def _ensure_alert_logs_columns(inspector) -> None:
    columns = {column["name"]: column for column in inspector.get_columns("alert_logs")}
    statements: list[str] = []

    if "source" not in columns:
        statements.append("ALTER TABLE alert_logs ADD COLUMN source VARCHAR(64) NULL")
    if "title" not in columns:
        statements.append("ALTER TABLE alert_logs ADD COLUMN title VARCHAR(128) NULL")
    if "updated_at" not in columns:
        statements.append(
            "ALTER TABLE alert_logs ADD COLUMN updated_at DATETIME NULL DEFAULT CURRENT_TIMESTAMP"
        )
    if "event_type" not in columns:
        statements.append("ALTER TABLE alert_logs ADD COLUMN event_type VARCHAR(64) NULL")
    if "impact_scope" not in columns:
        statements.append("ALTER TABLE alert_logs ADD COLUMN impact_scope VARCHAR(255) NULL")
    if "root_cause" not in columns:
        statements.append("ALTER TABLE alert_logs ADD COLUMN root_cause TEXT NULL")
    if "suggested_action" not in columns:
        statements.append("ALTER TABLE alert_logs ADD COLUMN suggested_action TEXT NULL")
    if "analysis_json" not in columns:
        statements.append("ALTER TABLE alert_logs ADD COLUMN analysis_json TEXT NULL")

    if engine.dialect.name == "mysql":
        if "message" in columns and not columns["message"].get("nullable", True):
            statements.append("ALTER TABLE alert_logs MODIFY COLUMN message VARCHAR(255) NULL")

    if not statements:
        return

    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


def _backfill_alert_logs() -> None:
    inspector = inspect(engine)
    columns = {column["name"] for column in inspector.get_columns("alert_logs")}
    if not {"id", "created_at", "summary", "source", "title", "updated_at"}.issubset(columns):
        return

    select_columns = ["id", "created_at", "summary", "source", "title", "updated_at"]
    if "event_type" in columns:
        select_columns.append("event_type")
    if "message" in columns:
        select_columns.append("message")

    with engine.begin() as connection:
        rows = connection.execute(
            text(f"SELECT {', '.join(select_columns)} FROM alert_logs")
        ).mappings()

        for row in rows:
            updates: dict[str, object] = {}
            source = str(row.get("source") or "").strip()
            legacy_source = str(row.get("event_type") or "").strip()
            title = str(row.get("title") or "").strip()
            legacy_title = str(row.get("message") or "").strip()

            if not source:
                updates["source"] = legacy_source or "system"
            if not title:
                updates["title"] = legacy_title or f"告警事件 #{row['id']}"
            if row.get("updated_at") is None:
                updates["updated_at"] = row["created_at"]

            if not updates:
                continue

            assignments = ", ".join(f"{column} = :{column}" for column in updates)
            connection.execute(
                text(f"UPDATE alert_logs SET {assignments} WHERE id = :id"),
                {"id": row["id"], **updates},
            )


def _ensure_alert_push_logs_columns(inspector) -> None:
    columns = {column["name"]: column for column in inspector.get_columns("alert_push_logs")}
    statements: list[str] = []

    if "target" not in columns:
        statements.append("ALTER TABLE alert_push_logs ADD COLUMN target VARCHAR(128) NULL")
    if "success" not in columns:
        statements.append("ALTER TABLE alert_push_logs ADD COLUMN success BOOLEAN NULL DEFAULT 0")
    if "updated_at" not in columns:
        statements.append(
            "ALTER TABLE alert_push_logs ADD COLUMN updated_at DATETIME NULL DEFAULT CURRENT_TIMESTAMP"
        )

    if engine.dialect.name == "mysql":
        if "alert_id" in columns and not columns["alert_id"].get("nullable", True):
            statements.append("ALTER TABLE alert_push_logs MODIFY COLUMN alert_id INT NULL")
        if "push_status" in columns and not columns["push_status"].get("nullable", True):
            statements.append("ALTER TABLE alert_push_logs MODIFY COLUMN push_status VARCHAR(16) NULL")

    if not statements:
        return

    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


def _backfill_alert_push_logs() -> None:
    inspector = inspect(engine)
    columns = {column["name"] for column in inspector.get_columns("alert_push_logs")}
    if not {"id", "channel", "target", "success", "created_at", "updated_at"}.issubset(columns):
        return

    select_columns = ["id", "channel", "target", "success", "created_at", "updated_at"]
    if "push_status" in columns:
        select_columns.append("push_status")

    with engine.begin() as connection:
        rows = connection.execute(
            text(f"SELECT {', '.join(select_columns)} FROM alert_push_logs")
        ).mappings()

        for row in rows:
            updates: dict[str, object] = {}
            target = str(row.get("target") or "").strip()
            push_status = str(row.get("push_status") or "").strip().lower()

            if not target:
                updates["target"] = str(row["channel"])
            if row.get("success") is None:
                updates["success"] = push_status in {"success", "ok", "delivered"}
            if row.get("updated_at") is None:
                updates["updated_at"] = row["created_at"]

            if not updates:
                continue

            assignments = ", ".join(f"{column} = :{column}" for column in updates)
            connection.execute(
                text(f"UPDATE alert_push_logs SET {assignments} WHERE id = :id"),
                {"id": row["id"], **updates},
            )


def _ensure_updated_at_column(table_name: str) -> None:
    inspector = inspect(engine)
    columns = {column["name"] for column in inspector.get_columns(table_name)}
    if "updated_at" in columns:
        return

    with engine.begin() as connection:
        connection.execute(
            text(
                f"ALTER TABLE {table_name} "
                "ADD COLUMN updated_at DATETIME NULL DEFAULT CURRENT_TIMESTAMP"
            )
        )


def _backfill_updated_at_from_created_at(table_name: str) -> None:
    inspector = inspect(engine)
    columns = {column["name"] for column in inspector.get_columns(table_name)}
    if not {"id", "created_at", "updated_at"}.issubset(columns):
        return

    with engine.begin() as connection:
        connection.execute(
            text(
                f"UPDATE {table_name} "
                "SET updated_at = created_at "
                "WHERE updated_at IS NULL"
            )
        )


def _ensure_owner_gesture_record_columns(inspector) -> None:
    columns = {column["name"] for column in inspector.get_columns("owner_gesture_records")}
    statements: list[str] = []

    if "user_id" not in columns:
        statements.append("ALTER TABLE owner_gesture_records ADD COLUMN user_id INT NULL")
    if "session_id" not in columns:
        statements.append("ALTER TABLE owner_gesture_records ADD COLUMN session_id VARCHAR(64) NULL")
    if "hand_landmarks" not in columns:
        statements.append("ALTER TABLE owner_gesture_records ADD COLUMN hand_landmarks JSON NULL")
    if "is_triggered" not in columns:
        statements.append("ALTER TABLE owner_gesture_records ADD COLUMN is_triggered BOOLEAN NULL DEFAULT 0")
    if "processing_time_ms" not in columns:
        statements.append("ALTER TABLE owner_gesture_records ADD COLUMN processing_time_ms INT NULL")

    indexes = {index["name"] for index in inspector.get_indexes("owner_gesture_records")}
    if "ix_owner_gesture_records_user_id" not in indexes:
        statements.append("CREATE INDEX ix_owner_gesture_records_user_id ON owner_gesture_records (user_id)")
    if "ix_owner_gesture_records_session_id" not in indexes:
        statements.append("CREATE INDEX ix_owner_gesture_records_session_id ON owner_gesture_records (session_id)")

    if not statements:
        return

    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


def _ensure_police_gesture_record_columns(inspector) -> None:
    column_map = {column["name"]: column for column in inspector.get_columns("police_gesture_records")}
    columns = set(column_map)
    statements: list[str] = []

    if "user_id" in columns and not column_map["user_id"].get("nullable", True):
        statements.append("ALTER TABLE police_gesture_records MODIFY COLUMN user_id INT NULL")
    if "session_id" in columns and not column_map["session_id"].get("nullable", True):
        statements.append("ALTER TABLE police_gesture_records MODIFY COLUMN session_id VARCHAR(64) NULL")
    if "gesture" not in columns:
        statements.append("ALTER TABLE police_gesture_records ADD COLUMN gesture VARCHAR(64) NULL")
    if "confidence" not in columns:
        statements.append("ALTER TABLE police_gesture_records ADD COLUMN confidence FLOAT NULL DEFAULT 0")
    if "keypoints" not in columns:
        statements.append("ALTER TABLE police_gesture_records ADD COLUMN keypoints JSON NULL")
    if "processing_time_ms" not in columns:
        statements.append("ALTER TABLE police_gesture_records ADD COLUMN processing_time_ms INT NULL")
    if "created_at" not in columns:
        statements.append(
            "ALTER TABLE police_gesture_records "
            "ADD COLUMN created_at DATETIME NULL DEFAULT CURRENT_TIMESTAMP"
        )
    if "updated_at" not in columns:
        statements.append(
            "ALTER TABLE police_gesture_records "
            "ADD COLUMN updated_at DATETIME NULL DEFAULT CURRENT_TIMESTAMP"
        )
    if "source_path" not in columns:
        statements.append("ALTER TABLE police_gesture_records ADD COLUMN source_path VARCHAR(255) NULL")

    if not statements:
        return

    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))
        if "user_id" in columns and not column_map["user_id"].get("nullable", True):
            connection.execute(
                text(
                    "UPDATE police_gesture_records "
                    "SET user_id = 0 "
                    "WHERE user_id IS NULL"
                )
            )
        if "session_id" in columns and not column_map["session_id"].get("nullable", True):
            connection.execute(
                text(
                    "UPDATE police_gesture_records "
                    "SET session_id = 'police' "
                    "WHERE session_id IS NULL OR session_id = ''"
                )
            )
        if "gesture" not in columns:
            connection.execute(
                text(
                    "UPDATE police_gesture_records "
                    "SET gesture = 'unknown' "
                    "WHERE gesture IS NULL OR gesture = ''"
                )
            )
        if "confidence" not in columns:
            connection.execute(
                text(
                    "UPDATE police_gesture_records "
                    "SET confidence = 0 "
                    "WHERE confidence IS NULL"
                )
            )
        if "updated_at" not in columns and "created_at" in columns:
            connection.execute(
                text(
                    "UPDATE police_gesture_records "
                    "SET updated_at = created_at "
                    "WHERE updated_at IS NULL"
                )
            )


def init_database() -> None:
    ensure_database_exists()
    Base.metadata.create_all(bind=engine)
    _ensure_user_security_columns()
    _migrate_legacy_user_sensitive_data()
    _ensure_alert_table_compatibility()
