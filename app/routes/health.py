from fastapi import APIRouter
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import settings
from app.db import engine

router = APIRouter(tags=['health'])


@router.get('/healthz')
def healthz():
    db_ok = True
    db_error = ""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        db_ok = False
        db_error = str(exc)
    data = {
        "status": "ok" if db_ok else "degraded",
        "db_ok": db_ok,
        "db_backend": settings.database_url.split(":", 1)[0],
    }
    if db_error:
        data["db_error"] = db_error
    return data
