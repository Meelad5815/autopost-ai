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
from app.db_migrations import run_migrations
from app.routes import admin, auth, automation, billing, health, sites, ui
from app.services.scheduler_daemon import start_scheduler_daemon


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
