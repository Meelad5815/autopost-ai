"""Multi-tenant SaaS API surface for autonomous WordPress automation.

Design goals:
- Multiple users, multiple websites per user
- Tenant-isolated automation jobs
- Plan-based usage controls
- Admin oversight endpoints
"""

import os

from fastapi import FastAPI
from sqlalchemy.exc import OperationalError

from app.db import Base, engine
from app.core.config import settings
from app.core.secrets import _fernet
from app.db_migrations import run_migrations
from app.routes import admin, auth, automation, billing, health, sites, ui, mrk_earning, website_factory
from app.services.scheduler_daemon import start_scheduler_daemon


app = FastAPI(title='Autopost SaaS Platform', version='1.0.0')

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(sites.router)
app.include_router(automation.router)
app.include_router(billing.router)
app.include_router(admin.router)
app.include_router(ui.router)
app.include_router(mrk_earning.router)
app.include_router(website_factory.router)


@app.on_event("startup")
def init_db() -> None:
    if settings.environment.lower() in {"production", "prod"}:
        if settings.debug:
            raise RuntimeError("DEBUG must be disabled in production")
        if len(settings.jwt_secret_key) < 32 or settings.jwt_secret_key == "change-me":
            raise RuntimeError("Set a strong JWT_SECRET_KEY (at least 32 characters) in production")
        # Fail closed: production must be able to encrypt site credentials.
        _fernet()

    try:
        Base.metadata.create_all(bind=engine)
        applied = run_migrations(engine)
        if applied:
            print(f"Applied migrations: {applied}")
    except OperationalError as exc:
        print(f"DB init failed: {exc}")
    except Exception as exc:  # noqa: BLE001
        print(f"Unexpected DB init failure: {exc}")

    if os.getenv("AUTO_START_SCHEDULER", "1") == "1":
        try:
            state = start_scheduler_daemon()
            print(f"Scheduler daemon state: {state.get('status')}")
        except Exception as exc:  # noqa: BLE001
            print(f"Scheduler daemon start failed: {exc}")
