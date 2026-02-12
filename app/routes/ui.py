import json
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import get_current_user
from app.models import Site, User


router = APIRouter(tags=["ui"])


@router.get("/", response_class=HTMLResponse)
def index():
    html = Path("app/ui/index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@router.get("/ui/schedule")
def get_schedule(user: User = Depends(get_current_user)):
    _ = user
    path = Path("config/schedule.json")
    if not path.exists():
        return JSONResponse({"timezone": "Asia/Karachi", "slots": [], "topics": []})
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@router.post("/ui/schedule")
def save_schedule(payload: dict, user: User = Depends(get_current_user)):
    _ = user
    path = Path("config/schedule.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "ok"}


@router.get("/ui/seo")
def get_seo_state(user: User = Depends(get_current_user)):
    _ = user
    out = {}
    for name in ["history.json", "trends.json", "keywords.json"]:
        p = Path(name)
        if p.exists():
            out[name] = json.loads(p.read_text(encoding="utf-8"))
    return JSONResponse(out)


@router.get("/ui/metrics")
def get_metrics(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    sites = db.query(Site).filter(Site.user_id == user.id).count()
    posts = db.query(Site).filter(Site.user_id == user.id).all()
    site_ids = [s.id for s in posts]
    perf_path = Path("data/performance_history.json")
    perf = json.loads(perf_path.read_text(encoding="utf-8")) if perf_path.exists() else {"posts": []}
    history_path = Path("history.json")
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {"articles": []}
    return {
        "sites": sites,
        "site_ids": site_ids,
        "performance": perf.get("posts", [])[-20:],
        "articles": history.get("articles", [])[-20:],
    }


def tail_lines(path: Path, limit: int = 120) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return lines[-limit:]


@router.get("/ui/scheduler-status")
def get_scheduler_status(user: User = Depends(get_current_user)):
    _ = user
    history_path = Path("history.json")
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {}
    actions = history.get("actions", [])
    last_action = actions[-1] if actions else {}
    log_lines = tail_lines(Path("scheduler.log"), 80)
    return JSONResponse({
        "last_slot": last_action.get("slot"),
        "last_actions": last_action.get("actions", []),
        "log_tail": log_lines,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    })


@router.get("/ui/run-reports")
def get_run_reports(user: User = Depends(get_current_user)):
    _ = user
    reports_dir = Path("data/run_reports")
    if not reports_dir.exists():
        return JSONResponse({"reports": []})
    reports = []
    for p in sorted(reports_dir.glob("run_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {"error": "invalid json"}
        reports.append({
            "file": p.name,
            "mtime": datetime.utcfromtimestamp(p.stat().st_mtime).isoformat() + "Z",
            "summary": data.get("summary", data),
        })
    return JSONResponse({"reports": reports})


@router.post("/ui/run-slot")
def run_slot(payload: dict, user: User = Depends(get_current_user)):
    _ = user
    slot_time = str(payload.get("slot_time", "")).strip()
    if not slot_time:
        raise HTTPException(status_code=400, detail="slot_time required")
    import os
    import subprocess
    env = os.environ.copy()
    env["RUN_ONCE"] = "1"
    env["SLOT_TIME"] = slot_time
    proc = subprocess.run(["python", "scheduler.py"], env=env, check=False)
    return {"status": "ok" if proc.returncode == 0 else "failed", "code": proc.returncode}


@router.post("/ui/update-only")
def run_update_only(user: User = Depends(get_current_user)):
    _ = user
    import os
    import subprocess
    env = os.environ.copy()
    env["UPDATE_ONLY"] = "1"
    env["UPDATE_LOOP"] = "1"
    env["POSTS_PER_RUN"] = "1"
    proc = subprocess.run(["python", "autopost.py"], env=env, check=False)
    return {"status": "ok" if proc.returncode == 0 else "failed", "code": proc.returncode}
