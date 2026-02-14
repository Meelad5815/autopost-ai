import json
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import get_current_user
from app.core.security import create_access_token, hash_password, verify_password
from app.models import Site, Subscription, User
from app.services.billing import plan_limit


router = APIRouter(tags=["ui"])
CONTENT_SETTINGS_PATH = Path("config/content_settings.json")


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


def default_content_settings() -> dict:
    return {
        "content_temperature": 0.75,
        "content_similarity_max": 0.90,
        "content_rewrite_attempts": 2,
        "content_source_urls": [],
    }


def load_content_settings() -> dict:
    if not CONTENT_SETTINGS_PATH.exists():
        return default_content_settings()
    try:
        data = json.loads(CONTENT_SETTINGS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default_content_settings()
    defaults = default_content_settings()
    defaults.update({k: data.get(k, v) for k, v in defaults.items()})
    if isinstance(defaults.get("content_source_urls"), str):
        defaults["content_source_urls"] = [x.strip() for x in defaults["content_source_urls"].split(",") if x.strip()]
    return defaults


def apply_content_env(env: dict) -> None:
    settings = load_content_settings()
    env["CONTENT_TEMPERATURE"] = str(settings.get("content_temperature", 0.75))
    env["CONTENT_SIMILARITY_MAX"] = str(settings.get("content_similarity_max", 0.90))
    env["CONTENT_REWRITE_ATTEMPTS"] = str(settings.get("content_rewrite_attempts", 2))
    env["CONTENT_SOURCE_URLS"] = ",".join(settings.get("content_source_urls", []))


@router.post("/ui/run-site-audit")
def run_site_audit(user: User = Depends(get_current_user)):
    _ = user
    import subprocess

    proc = subprocess.run(["python", "site_audit.py"], capture_output=True, text=True, encoding="utf-8", errors="ignore", check=False)
    output = (proc.stdout or "").strip().splitlines()
    last_line = output[-1] if output else ""
    data = {}
    try:
        data = json.loads(last_line) if last_line else {}
    except json.JSONDecodeError:
        data = {"raw_output": last_line}
    return {"status": "ok" if proc.returncode == 0 else "failed", "code": proc.returncode, "result": data}


@router.post("/ui/run-remediation")
def run_remediation(user: User = Depends(get_current_user)):
    _ = user
    import subprocess

    proc = subprocess.run(["python", "remediate_site.py"], capture_output=True, text=True, encoding="utf-8", errors="ignore", check=False)
    output = (proc.stdout or "").strip().splitlines()
    last_line = output[-1] if output else ""
    data = {}
    try:
        data = json.loads(last_line) if last_line else {}
    except json.JSONDecodeError:
        data = {"raw_output": last_line}
    return {"status": "ok" if proc.returncode == 0 else "failed", "code": proc.returncode, "result": data}


@router.post("/ui/run-seo-normalize")
def run_seo_normalize(payload: dict | None = None, user: User = Depends(get_current_user)):
    _ = user
    import os
    import subprocess

    payload = payload or {}
    limit = int(payload.get("limit", 12) or 12)
    limit = max(1, min(limit, 50))
    env = os.environ.copy()
    env["MAX_SEO_FIX_POSTS"] = str(limit)
    proc = subprocess.run(
        ["python", "seo_normalize.py"],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        check=False,
    )
    output = (proc.stdout or "").strip().splitlines()
    last_line = output[-1] if output else ""
    data = {}
    try:
        data = json.loads(last_line) if last_line else {}
    except json.JSONDecodeError:
        data = {"raw_output": last_line}
    return {"status": "ok" if proc.returncode == 0 else "failed", "code": proc.returncode, "result": data}


@router.post("/ui/run-quick-cycle")
def run_quick_cycle(payload: dict | None = None, user: User = Depends(get_current_user)):
    _ = user
    import os
    import subprocess

    payload = payload or {}
    slot_time = str(payload.get("slot_time", "")).strip()
    env = os.environ.copy()
    env["QUICK_MODE"] = "1"
    env["POSTS_PER_RUN"] = "1"
    env["MAX_PUBLISH_RETRIES"] = "1"
    env["REQUEST_TIMEOUT"] = "35"
    if slot_time:
        env["SLOT_TIME"] = slot_time
    apply_content_env(env)
    proc = subprocess.run(["python", "run_batch.py"], env=env, check=False)
    return {"status": "ok" if proc.returncode == 0 else "failed", "code": proc.returncode}


@router.get("/ui/site-audit")
def get_site_audit(user: User = Depends(get_current_user)):
    _ = user
    audit_dir = Path("data/audits")
    if not audit_dir.exists():
        return JSONResponse({"latest": None, "reports": []})
    files = sorted(audit_dir.glob("audit_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not files:
        return JSONResponse({"latest": None, "reports": []})
    latest_file = files[0]
    latest = json.loads(latest_file.read_text(encoding="utf-8"))
    return JSONResponse({"latest": latest, "reports": [f.name for f in files[:10]]})


@router.get("/ui/scheduler-profile")
def get_scheduler_profile(user: User = Depends(get_current_user)):
    _ = user
    path = Path("scheduler.log")
    if not path.exists():
        return JSONResponse({"status": "no_log", "summary": {}, "recent_failures": []})
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()[-400:]
    run_attempts = [ln for ln in lines if "Run post attempt=" in ln]
    post_ok = [ln for ln in lines if "post_result ok=True" in ln]
    post_fail = [ln for ln in lines if "post_result ok=False" in ln]
    update_ok = [ln for ln in lines if "update_result ok=True" in ln]
    update_fail = [ln for ln in lines if "update_result ok=False" in ln]
    failures = [ln for ln in lines if "ok=False" in ln or "error" in ln.lower()][-20:]
    return JSONResponse(
        {
            "status": "ok",
            "summary": {
                "samples": len(lines),
                "run_attempts": len(run_attempts),
                "post_ok": len(post_ok),
                "post_fail": len(post_fail),
                "update_ok": len(update_ok),
                "update_fail": len(update_fail),
                "success_rate": round((len(post_ok) / max(1, len(post_ok) + len(post_fail))) * 100, 2),
            },
            "recent_failures": failures,
        }
    )


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


@router.get("/ui/content-settings")
def get_content_settings(user: User = Depends(get_current_user)):
    _ = user
    return JSONResponse(load_content_settings())


@router.post("/ui/content-settings")
def save_content_settings(payload: dict, user: User = Depends(get_current_user)):
    _ = user
    data = default_content_settings()
    data["content_temperature"] = float(payload.get("content_temperature", data["content_temperature"]))
    data["content_similarity_max"] = float(payload.get("content_similarity_max", data["content_similarity_max"]))
    data["content_rewrite_attempts"] = int(payload.get("content_rewrite_attempts", data["content_rewrite_attempts"]))
    urls = payload.get("content_source_urls", data["content_source_urls"])
    if isinstance(urls, str):
        urls = [x.strip() for x in urls.split(",") if x.strip()]
    data["content_source_urls"] = [str(x).strip() for x in urls if str(x).strip()]
    CONTENT_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONTENT_SETTINGS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "ok", "settings": data}


@router.get("/ui/uniqueness")
def get_uniqueness_metrics(user: User = Depends(get_current_user)):
    _ = user
    reports_dir = Path("data/run_reports")
    if not reports_dir.exists():
        return JSONResponse({"items": [], "avg_uniqueness": 0, "avg_similarity": 0})

    items = []
    for p in sorted(reports_dir.glob("run_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:40]:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for result in data.get("results", []):
            if result.get("action") != "created":
                continue
            items.append(
                {
                    "file": p.name,
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "uniqueness_score": int(result.get("uniqueness_score", 0) or 0),
                    "semantic_similarity": float(result.get("semantic_similarity", 0.0) or 0.0),
                }
            )
    if not items:
        return JSONResponse({"items": [], "avg_uniqueness": 0, "avg_similarity": 0})
    avg_u = round(sum(x["uniqueness_score"] for x in items) / len(items), 2)
    avg_s = round(sum(x["semantic_similarity"] for x in items) / len(items), 4)
    return JSONResponse({"items": items[:20], "avg_uniqueness": avg_u, "avg_similarity": avg_s})


@router.get("/ui/fact-check")
def get_fact_check_metrics(user: User = Depends(get_current_user)):
    _ = user
    reports_dir = Path("data/run_reports")
    if not reports_dir.exists():
        return JSONResponse(
            {
                "total_created": 0,
                "clear_count": 0,
                "controversial_count": 0,
                "triangulated_count": 0,
                "items": [],
            }
        )

    items = []
    for p in sorted(reports_dir.glob("run_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:50]:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for result in data.get("results", []):
            if result.get("action") != "created":
                continue
            label = str(result.get("conflict_label", "clear")).strip().lower()
            status = str(result.get("fact_check_status", "triangulated")).strip().lower()
            items.append(
                {
                    "file": p.name,
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "conflict_label": label or "clear",
                    "fact_check_status": status or "triangulated",
                }
            )
    clear_count = sum(1 for x in items if x["conflict_label"] == "clear")
    controversial_count = sum(1 for x in items if x["conflict_label"] == "controversial")
    triangulated_count = sum(1 for x in items if x["fact_check_status"] == "triangulated")
    return JSONResponse(
        {
            "total_created": len(items),
            "clear_count": clear_count,
            "controversial_count": controversial_count,
            "triangulated_count": triangulated_count,
            "items": items[:20],
        }
    )


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
    apply_content_env(env)
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
    apply_content_env(env)
    proc = subprocess.run(["python", "autopost.py"], env=env, check=False)
    return {"status": "ok" if proc.returncode == 0 else "failed", "code": proc.returncode}


@router.post("/ui/first-setup")
def first_setup(payload: dict, db: Session = Depends(get_db)):
    email = str(payload.get("email", "")).strip()
    password = str(payload.get("password", "")).strip()
    site = payload.get("site") or {}

    if not email or not password:
        raise HTTPException(status_code=400, detail="email and password required")

    user = db.query(User).filter(User.email == email).first()
    if not user:
        user = User(
            email=email,
            password_hash=hash_password(password),
            is_active=True,
            role="admin",
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        db.add(Subscription(user_id=user.id, plan="free", monthly_post_limit=plan_limit("free")))
        db.commit()
    else:
        if not verify_password(password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid credentials for existing user")

    created_site = None
    if site:
        wp_url = str(site.get("wp_url", "")).strip()
        wp_user = str(site.get("wp_user", "")).strip()
        wp_app_password = str(site.get("wp_app_password", "")).strip()
        name = str(site.get("name", "")).strip() or "Primary Site"
        niche = str(site.get("niche", "")).strip()
        frequency_hours = int(site.get("frequency_hours", 24))
        if not wp_url or not wp_user or not wp_app_password:
            raise HTTPException(status_code=400, detail="site wp_url, wp_user, wp_app_password required")

        existing = db.query(Site).filter(Site.user_id == user.id, Site.wp_url == wp_url).first()
        if not existing:
            created_site = Site(
                user_id=user.id,
                name=name,
                wp_url=wp_url,
                wp_user=wp_user,
                wp_app_password_enc=wp_app_password,
                openai_api_key_enc="",
                niche=niche,
                frequency_hours=frequency_hours,
                is_active=True,
            )
            db.add(created_site)
            db.commit()
            db.refresh(created_site)
        else:
            created_site = existing

    return {
        "status": "ok",
        "user_id": user.id,
        "site_id": created_site.id if created_site else None,
        "access_token": create_access_token(str(user.id)),
        "token_type": "bearer",
    }
