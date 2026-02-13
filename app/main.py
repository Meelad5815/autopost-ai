"""Multi-tenant SaaS API surface for autonomous WordPress automation.

Design goals:
- Multiple users, multiple websites per user
- Tenant-isolated automation jobs
- Plan-based usage controls
- Admin oversight endpoints
"""

from fastapi import FastAPI
from sqlalchemy import inspect, text
from sqlalchemy.exc import OperationalError

from app.db import Base, engine
from app.routes import admin, auth, automation, billing, health, sites, ui


app = FastAPI(title='Autopost SaaS Platform', version='1.0.0')

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(sites.router)
app.include_router(automation.router)
app.include_router(billing.router)
app.include_router(admin.router)
app.include_router(ui.router)


@app.on_event("startup")
def init_db() -> None:
    try:
        Base.metadata.create_all(bind=engine)
        with engine.begin() as conn:
            inspector = inspect(conn)
            user_columns = {col["name"] for col in inspector.get_columns("users")}
            if "provider" not in user_columns:
                conn.execute(text("ALTER TABLE users ADD COLUMN provider VARCHAR(20) DEFAULT 'local'"))
            if "email_verified" not in user_columns:
                conn.execute(text("ALTER TABLE users ADD COLUMN email_verified BOOLEAN DEFAULT 0"))
            conn.execute(text("UPDATE users SET provider = 'local' WHERE provider IS NULL OR provider = ''"))
            conn.execute(text("UPDATE users SET email_verified = 0 WHERE email_verified IS NULL"))
    except OperationalError as exc:
        print(f"DB init failed: {exc}")
