import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None


def load_schedule() -> dict:
    path = Path("config/schedule.json")
    if not path.exists():
        raise FileNotFoundError("config/schedule.json not found")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_now(timezone: str) -> datetime:
    if ZoneInfo is None:
        return datetime.utcnow()
    return datetime.now(ZoneInfo(timezone))


def pick_slot(slots: list[dict], now: datetime, override: str | None) -> dict:
    if override:
        for slot in slots:
            if slot.get("time") == override:
                return slot
    hhmm = now.strftime("%H:%M")
    for slot in slots:
        if slot.get("time") == hhmm:
            return slot
    # Fallback: match by hour
    hh = now.strftime("%H")
    for slot in slots:
        if slot.get("time", "")[:2] == hh:
            return slot
    return slots[0]


def run_post(topic: str, language: str) -> None:
    env = os.environ.copy()
    env["POST_TOPIC"] = topic
    env["LOCAL_AI_LANGUAGE"] = language
    subprocess.run(["python", "autopost.py"], env=env, check=False)


def main() -> int:
    schedule = load_schedule()
    slots = schedule.get("slots", [])
    topics = schedule.get("topics", [])
    if not slots or not topics:
        print("Empty schedule or topics")
        return 1

    tz = schedule.get("timezone", "Asia/Karachi")
    now = resolve_now(tz)
    override = os.getenv("SLOT_TIME", "").strip() or None
    slot = pick_slot(slots, now, override)
    languages = slot.get("languages", [])
    if not languages:
        print("No languages configured for slot")
        return 1

    # Deterministic topic rotation per slot/day
    seed = int(now.strftime("%Y%m%d")) + int(slot.get("time", "00:00")[:2])
    for idx, lang in enumerate(languages):
        topic = topics[(seed + idx) % len(topics)]
        run_post(topic, lang)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
