import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

DATA_DIR = Path("data")
REPORTS_DIR = DATA_DIR / "run_reports"
PERF_FILE = DATA_DIR / "performance_history.json"
KEYWORD_FILE = DATA_DIR / "keyword_history.json"
CALENDAR_FILE = DATA_DIR / "content_calendar.json"
NICHE_FILE = DATA_DIR / "niche_scores.json"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_run_report(report: Dict[str, Any]) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = REPORTS_DIR / f"run_{ts}.json"
    save_json(out, report)
    return out
