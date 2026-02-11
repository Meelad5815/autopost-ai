from typing import Any, Dict, List

from .storage import PERF_FILE, load_json, save_json


def record_post_performance(post_id: int, title: str, metrics: Dict[str, Any]) -> None:
    history = load_json(PERF_FILE, {"posts": []})
    history["posts"].append({"post_id": post_id, "title": title, **metrics})
    history["posts"] = history["posts"][-500:]
    save_json(PERF_FILE, history)


def topic_success_scores() -> Dict[str, float]:
    history = load_json(PERF_FILE, {"posts": []})
    scores: Dict[str, List[float]] = {}
    for p in history.get("posts", []):
        topic = p.get("topic", "unknown")
        clicks = float(p.get("clicks", 0))
        impressions = float(p.get("impressions", 1))
        ctr = clicks / max(1.0, impressions)
        scores.setdefault(topic, []).append(ctr)
    return {k: round(sum(v) / len(v), 4) for k, v in scores.items() if v}


def choose_best_topics(candidate_topics: List[str]) -> List[str]:
    scores = topic_success_scores()
    return sorted(candidate_topics, key=lambda t: scores.get(t, 0.0), reverse=True)
