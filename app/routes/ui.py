import json
import math
import os
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import get_current_user, require_admin
from app.core.security import create_access_token, hash_password, verify_password
from app.models import Site, Subscription, User
from app.services.billing import plan_limit
from app.services.scheduler_daemon import scheduler_state, start_scheduler_daemon, stop_scheduler_daemon
from engine.config import load_config
from engine.wp_client import get_posts


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


@router.get("/ui/posting-stats")
def get_posting_stats(user: User = Depends(get_current_user)):
    _ = user
    wp_count = None
    wp_error = None
    try:
        cfg = load_config()
        wp_count = len(get_posts(cfg.wp_url, cfg.wp_user, cfg.wp_app_password, cfg.request_timeout, per_page=100))
    except Exception as exc:  # noqa: BLE001
        wp_error = str(exc)

    history_path = Path("history.json")
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {}
    actions = history.get("actions", [])
    created_actions = sum(
        1 for item in actions for act in item.get("actions", []) if act.get("action") == "created" and act.get("ok") is True
    )

    reports_dir = Path("data/run_reports")
    created_reports = 0
    if reports_dir.exists():
        for p in sorted(reports_dir.glob("run_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:120]:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            for result in data.get("results", []):
                if result.get("action") == "created":
                    created_reports += 1

    return JSONResponse(
        {
            "wordpress_published_posts": wp_count,
            "history_created_actions": created_actions,
            "run_reports_created_posts": created_reports,
            "wp_error": wp_error,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
    )


@router.post("/ui/publish-now-bulk")
def publish_now_bulk(payload: dict | None = None, user: User = Depends(get_current_user)):
    _ = user
    import os
    import subprocess

    payload = payload or {}
    count = max(1, min(int(payload.get("count", 10) or 10), 30))
    language = str(payload.get("language", "en")).strip().lower() or "en"
    env = os.environ.copy()
    env["POSTS_PER_RUN"] = str(count)
    env["LOCAL_AI_LANGUAGE"] = language
    env["UPDATE_ONLY"] = "0"
    env["UPDATE_LOOP"] = "0"
    env["MAX_PUBLISH_RETRIES"] = "1"
    apply_content_env(env)
    proc = subprocess.run(["python", "autopost.py"], env=env, check=False)
    return {"status": "ok" if proc.returncode == 0 else "failed", "code": proc.returncode, "count": count, "language": language}


@router.post("/ui/apply-high-frequency-plan")
def apply_high_frequency_plan(payload: dict | None = None, user: User = Depends(get_current_user)):
    _ = user
    payload = payload or {}
    timezone = str(payload.get("timezone", "Asia/Karachi")).strip() or "Asia/Karachi"
    start_time = str(payload.get("start_time", "08:00")).strip() or "08:00"
    end_time = str(payload.get("end_time", "20:00")).strip() or "20:00"
    posts_per_half_day = max(1, min(int(payload.get("posts_per_half_day", 20) or 20), 30))
    slots = _build_even_slots(start_time, end_time, posts_per_half_day)

    path = Path("config/schedule.json")
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {"timezone": timezone, "slots": [], "topics": []}
    data["timezone"] = timezone
    data["slots"] = slots
    if payload.get("topics"):
        topics = [str(x).strip() for x in payload.get("topics", []) if str(x).strip()]
        if topics:
            data["topics"] = topics
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "ok", "timezone": timezone, "slots_count": len(slots), "slots": slots}


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


@router.post("/ui/cloud-terminal/exec")
def cloud_terminal_exec(payload: dict | None = None, user: User = Depends(require_admin)):
    _ = user
    payload = payload or {}

    command = str(payload.get("command", "")).strip()
    if not command:
        raise HTTPException(status_code=400, detail="command is required")

    requested_cwd = str(payload.get("cwd", ".")).strip() or "."
    raw_timeout = payload.get("timeout_seconds", 20)
    try:
        timeout_seconds = int(raw_timeout)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="timeout_seconds must be an integer") from None
    timeout_seconds = max(1, min(timeout_seconds, 120))

    base_dir = Path(".").resolve()
    target_dir = (base_dir / requested_cwd).resolve() if not Path(requested_cwd).is_absolute() else Path(requested_cwd).resolve()
    if os.path.commonpath([str(base_dir), str(target_dir)]) != str(base_dir):
        raise HTTPException(status_code=400, detail="cwd must stay inside repository")
    if not target_dir.exists() or not target_dir.is_dir():
        raise HTTPException(status_code=400, detail="cwd does not exist")

    try:
        proc = subprocess.run(
            command,
            cwd=str(target_dir),
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "command": command,
            "cwd": str(target_dir.relative_to(base_dir)),
            "timeout_seconds": timeout_seconds,
            "stdout": (exc.stdout or "")[-8000:],
            "stderr": (exc.stderr or "")[-8000:],
        }

    return {
        "status": "ok" if proc.returncode == 0 else "failed",
        "command": command,
        "cwd": str(target_dir.relative_to(base_dir)),
        "timeout_seconds": timeout_seconds,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "")[-12000:],
        "stderr": (proc.stderr or "")[-12000:],
    }


def _hhmm_to_minutes(value: str) -> int:
    hh, mm = value.split(":")
    return int(hh) * 60 + int(mm)


def _minutes_to_hhmm(value: int) -> str:
    value = value % (24 * 60)
    hh = value // 60
    mm = value % 60
    return f"{hh:02d}:{mm:02d}"


def _build_even_slots(start_hhmm: str, end_hhmm: str, count: int) -> list[dict]:
    start = _hhmm_to_minutes(start_hhmm)
    end = _hhmm_to_minutes(end_hhmm)
    if end <= start:
        end += 24 * 60
    window = end - start
    count = max(1, count)
    step = max(1, math.floor(window / count))
    return [{"time": _minutes_to_hhmm(start + i * step)} for i in range(count)]


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
        "daemon": scheduler_state(),
        "updated_at": datetime.utcnow().isoformat() + "Z",
    })


@router.get("/ui/scheduler-daemon")
def get_scheduler_daemon(user: User = Depends(require_admin)):
    _ = user
    return JSONResponse(scheduler_state())


@router.post("/ui/scheduler-daemon/start")
def start_scheduler_daemon_route(user: User = Depends(require_admin)):
    _ = user
    return JSONResponse(start_scheduler_daemon())


@router.post("/ui/scheduler-daemon/stop")
def stop_scheduler_daemon_route(user: User = Depends(require_admin)):
    _ = user
    return JSONResponse(stop_scheduler_daemon())




@router.get("/ui/kali-env/status")
def get_kali_env_status(user: User = Depends(require_admin)):
    _ = user
    compose_path = Path("docker-compose.kali.yml")
    docker_bin = shutil.which("docker")
    status = {
        "configured": compose_path.exists(),
        "compose_file": str(compose_path),
        "docker_installed": bool(docker_bin),
        "docker_bin": docker_bin,
        "container_running": False,
        "container_name": "autopost-kali",
        "detail": "",
    }

    if not compose_path.exists():
        status["detail"] = "docker-compose.kali.yml not found"
        return JSONResponse(status)

    if not docker_bin:
        status["detail"] = "Docker CLI not found on host"
        return JSONResponse(status)

    try:
        proc = subprocess.run(
            [docker_bin, "ps", "--filter", "name=autopost-kali", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            check=False,
            timeout=6,
        )
        names = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
        status["container_running"] = "autopost-kali" in names
        status["detail"] = "running" if status["container_running"] else "stopped"
    except Exception as exc:  # noqa: BLE001
        status["detail"] = f"docker check failed: {exc}"

    return JSONResponse(status)

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
