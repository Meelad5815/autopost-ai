from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine


MigrationFn = Callable[[Engine], None]


def _ensure_migrations_table(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version VARCHAR(64) PRIMARY KEY,
                    applied_at TIMESTAMP NOT NULL
                )
                """
            )
        )


def _is_applied(engine: Engine, version: str) -> bool:
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT version FROM schema_migrations WHERE version = :v"),
            {"v": version},
        ).first()
        return bool(row)


def _mark_applied(engine: Engine, version: str) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO schema_migrations(version, applied_at) VALUES (:v, :ts)"),
            {"v": version, "ts": datetime.now(timezone.utc)},
        )


def _migration_20260214_users_provider_email_verified(engine: Engine) -> None:
    with engine.begin() as conn:
        inspector = inspect(conn)
        if "users" not in inspector.get_table_names():
            return
        user_columns = {col["name"] for col in inspector.get_columns("users")}
        if "provider" not in user_columns:
            conn.execute(text("ALTER TABLE users ADD COLUMN provider VARCHAR(20) DEFAULT 'local'"))
        if "email_verified" not in user_columns:
            conn.execute(text("ALTER TABLE users ADD COLUMN email_verified BOOLEAN DEFAULT 0"))
        conn.execute(text("UPDATE users SET provider = 'local' WHERE provider IS NULL OR provider = ''"))
        conn.execute(text("UPDATE users SET email_verified = 0 WHERE email_verified IS NULL"))


MIGRATIONS: list[tuple[str, MigrationFn]] = [
    ("20260214_users_provider_email_verified", _migration_20260214_users_provider_email_verified),
]


def run_migrations(engine: Engine) -> list[str]:
    _ensure_migrations_table(engine)
    applied: list[str] = []
    for version, migration_fn in MIGRATIONS:
        if _is_applied(engine, version):
            continue
        migration_fn(engine)
        _mark_applied(engine, version)
        applied.append(version)
    return applied

