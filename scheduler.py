import json
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

from seo import load_history, save_history

LOG_PATH = Path("scheduler.log")


def load_schedule() -> dict:
    path = Path("config/schedule.json")
    if not path.exists():
        raise FileNotFoundError("config/schedule.json not found")
    return json.loads(path.read_text(encoding="utf-8"))


def log(message: str) -> None:
    line = f"{datetime.now().isoformat()} | {message}"
    print(line)
    existing = LOG_PATH.read_text(encoding="utf-8") if LOG_PATH.exists() else ""
    LOG_PATH.write_text(existing + line + "\n", encoding="utf-8")


def next_slot(now: datetime, slots: list[dict]) -> tuple[datetime, dict]:
    candidates = []
    for slot in slots:
        hh, mm = slot["time"].split(":")
        dt = datetime(now.year, now.month, now.day, int(hh), int(mm))
        if dt < now:
            dt = dt + timedelta(days=1)
        candidates.append((dt, slot))
    candidates.sort(key=lambda x: x[0])
    return candidates[0]


def build_daily_language_plan() -> list[str]:
    return ["en", "ur", "roman"] * 5


def infer_intent(topic: str) -> str:
    t = topic.lower()
    if any(k in t for k in ["best", "top", "review", "pricing", "vs"]):
        return "commercial"
    if any(k in t for k in ["how", "guide", "tips", "what is"]):
        return "informational"
    return "informational"


def run_post(topic: str, language: str, retries: int = 2, update_loop: bool = False) -> bool:
    env = os.environ.copy()
    env["POST_TOPIC"] = topic
    env["LOCAL_AI_LANGUAGE"] = language
    env["CONTENT_INTENT"] = infer_intent(topic)
    env["UPDATE_LOOP"] = "1" if update_loop else "0"
    env["UPDATE_ONLY"] = "0"
    for attempt in range(1, retries + 2):
        log(f"Run post attempt={attempt} language={language} topic={topic}")
        proc = subprocess.run(["python", "autopost.py"], env=env, check=False)
        if proc.returncode == 0:
            return True
        time.sleep(2)
    return False


def run_update_only(retries: int = 2) -> bool:
    env = os.environ.copy()
    env["UPDATE_ONLY"] = "1"
    env["UPDATE_LOOP"] = "1"
    env["POSTS_PER_RUN"] = "1"
    for attempt in range(1, retries + 2):
        log(f"Update-only attempt={attempt}")
        proc = subprocess.run(["python", "autopost.py"], env=env, check=False)
        if proc.returncode == 0:
            return True
        time.sleep(2)
    return False


def pick_slot_index(slots: list[dict], slot_time: str) -> int:
    for idx, slot in enumerate(slots):
        if slot.get("time") == slot_time:
            return idx
    return 0


def run_slot(slots: list[dict], topics: list[str], topic_cursor: int, slot_index: int, daily_plan: list[str]) -> tuple[int, list[dict]]:
    actions = []
    start = slot_index * 3
    languages = daily_plan[start : start + 3]
    for lang in languages:
        topic = topics[topic_cursor % len(topics)]
        topic_cursor += 1
        ok = run_post(topic, lang, retries=2, update_loop=False)
        actions.append({"action": "created", "ok": ok, "language": lang, "topic": topic})
        log(f"post_result ok={ok} language={lang} topic={topic}")
    update_ok = run_update_only(retries=2)
    actions.append({"action": "updated", "ok": update_ok})
    log(f"update_result ok={update_ok}")
    return topic_cursor, actions


def main() -> int:
    schedule = load_schedule()
    slots = schedule.get("slots", [])
    topics = schedule.get("topics", [])
    if not slots or not topics:
        print("Empty schedule or topics")
        return 1

    history = load_history()
    topic_cursor = int(history.get("topic_cursor", 0))

    daily_plan = build_daily_language_plan()
    plan_index = 0

    run_once = os.getenv("RUN_ONCE", "0") == "1"
    slot_override = os.getenv("SLOT_TIME", "").strip()

    if run_once:
        now = datetime.now()
        slot = next_slot(now, slots)[1] if not slot_override else {"time": slot_override}
        slot_index = pick_slot_index(slots, slot["time"])
        topic_cursor, actions = run_slot(slots, topics, topic_cursor, slot_index, daily_plan)
        history["topic_cursor"] = topic_cursor
        history.setdefault("actions", []).append({"slot": slot["time"], "actions": actions})
        save_history(history)
        return 0

    while True:
        now = datetime.now()
        when, slot = next_slot(now, slots)
        wait_seconds = max(1, int((when - now).total_seconds()))
        log(f"Sleeping until slot {slot['time']} ({wait_seconds}s)")
        time.sleep(wait_seconds)
        slot_index = pick_slot_index(slots, slot["time"])
        topic_cursor, actions = run_slot(slots, topics, topic_cursor, slot_index, daily_plan)
        history["topic_cursor"] = topic_cursor
        history.setdefault("actions", []).append({"slot": slot["time"], "actions": actions})
        save_history(history)


if __name__ == "__main__":
    raise SystemExit(main())
