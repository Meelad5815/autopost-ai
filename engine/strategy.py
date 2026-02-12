from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from .ai import openai_json


def generate_calendar(api_key: str, model: str, timeout: int, niche_keywords: List[str], days: int) -> Dict[str, Any]:
    prompt = f"""
Return strict JSON with key calendar.
Create a {days}-day content plan mixing informational, commercial, and evergreen articles.
Use these niche keywords: {', '.join(niche_keywords[:12])}
Each calendar item fields: date_iso, topic, intent_type, pillar_topic, cluster_topic.
""".strip()
    return openai_json(api_key, model, prompt, timeout, temperature=0.6)


def select_today_topics(calendar: Dict[str, Any], posts_per_run: int) -> List[Dict[str, Any]]:
    today = datetime.now(timezone.utc).date()
    items = calendar.get("calendar", [])
    due = []
    for item in items:
        try:
            d = datetime.fromisoformat(str(item.get("date_iso"))).date()
        except Exception:
            continue
        if d <= today:
            due.append(item)
    return due[:posts_per_run] if due else items[:posts_per_run]


def detect_old_posts_for_refresh(existing_posts: List[Dict[str, Any]], refresh_age_days: int) -> List[Dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=refresh_age_days)
    stale = []
    for p in existing_posts:
        try:
            dt = datetime.fromisoformat(p.get("date", "").replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if dt < cutoff:
            stale.append(p)
    return stale[:5]
