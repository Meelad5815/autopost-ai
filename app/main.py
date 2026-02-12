"""Multi-tenant SaaS API surface for autonomous WordPress automation.

Design goals:
- Multiple users, multiple websites per user
- Tenant-isolated automation jobs
- Plan-based usage controls
- Admin oversight endpoints
"""

from fastapi import FastAPI

from app.db import Base, engine
from app.routes import admin, auth, automation, billing, health, sites, ui


Base.metadata.create_all(bind=engine)

app = FastAPI(title='Autopost SaaS Platform', version='1.0.0')

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(sites.router)
app.include_router(automation.router)
app.include_router(billing.router)
app.include_router(admin.router)
app.include_router(ui.router)
