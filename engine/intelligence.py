import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import requests

from .feedback import topic_success_scores
from .ai import openai_json


def fetch_suggestions(query: str, timeout: int) -> List[str]:
    try:
        resp = requests.get(
            "https://suggestqueries.google.com/complete/search",
            params={"client": "firefox", "q": query},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return [str(x) for x in data[1]] if isinstance(data, list) and len(data) > 1 else []
    except Exception:
        return []


def fetch_trends(geo: str, timeout: int) -> List[str]:
    try:
        resp = requests.get("https://trends.google.com/trending/rss", params={"geo": geo}, timeout=timeout)
        resp.raise_for_status()
        return [t.strip() for t in re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", resp.text)[1:] if t.strip()]
    except Exception:
        return []


def score_niche(keyword: str) -> Dict[str, float]:
    length = len(keyword.split())
    demand = min(100.0, 35 + length * 10)
    competition = max(5.0, 85 - length * 8)
    profitability = (demand * 0.6) + ((100 - competition) * 0.4)
    return {"demand": round(demand, 2), "competition": round(competition, 2), "profitability": round(profitability, 2)}


def detect_profitable_niches(timeout: int, niches_per_run: int) -> List[Dict[str, Any]]:
    seeds = ["ai automation", "saas growth", "cybersecurity", "ecommerce seo", "developer tools"]
    trends = fetch_trends("US", timeout)[:8]
    candidates: List[str] = []
    for seed in seeds + trends[:3]:
        candidates.extend(fetch_suggestions(seed, timeout))
    unique = sorted(set(candidates))[:80]
    scored = []
    for kw in unique:
        s = score_niche(kw)
        scored.append({"keyword": kw, **s})
    scored.sort(key=lambda x: x["profitability"], reverse=True)
    return scored[:niches_per_run]


def _intent_strength(keyword: str) -> float:
    k = keyword.lower()
    score = 0.0
    for term in ["best", "top", "vs", "review", "buy", "how to", "guide", "pricing", "tools", "tips"]:
        if term in k:
            score += 1.5
    return score


def _competition_estimate(keyword: str) -> float:
    return max(10.0, 90.0 - len(keyword.split()) * 10.0)


def _demand_estimate(keyword: str) -> float:
    words = keyword.split()
    base = 40.0 + min(5, len(words)) * 6.0
    return min(100.0, base + _intent_strength(keyword) * 8.0)


def _ranking_probability(demand: float, competition: float) -> float:
    return max(1.0, min(99.0, (demand * 0.7) + ((100 - competition) * 0.3)))


def _load_google_research_titles() -> List[str]:
    try:
        path = Path("research.json")
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return [str(x.get("title", "")).strip() for x in data.get("items", []) if x.get("title")]
    except Exception:
        return []


def _load_search_console_rows() -> List[Dict[str, Any]]:
    try:
        path = Path("data/search_console.json")
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("query_rows", []) if isinstance(data, dict) else []
    except Exception:
        return []


def _apply_search_console_query_signals(keywords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = _load_search_console_rows()
    if not rows:
        return keywords

    signals = {}
    for row in rows:
        query = str(row.get("keys", [""])[0]).strip().lower() if row.get("keys") else ""
        if not query:
            continue
        signals[query] = {
            "clicks": float(row.get("clicks", 0)),
            "impressions": float(row.get("impressions", 0)),
            "ctr": float(row.get("ctr", 0)),
            "position": float(row.get("position", 0)),
        }

    out = []
    for item in keywords:
        enriched = dict(item)
        query = str(item.get("keyword", "")).strip().lower()
        signal = signals.get(query)
        if signal:
            # Real GSC signals influence ordering only; they do not become fake ranking predictions.
            boost = min(20.0, signal["ctr"] * 100.0 * 0.20 + min(signal["clicks"], 100.0) * 0.05)
            enriched["search_console"] = signal
            enriched["gsc_learning_boost"] = round(boost, 2)
            enriched["learned_score"] = round(float(enriched.get("learned_score", enriched.get("ranking_probability", 0))) + boost, 2)
        out.append(enriched)
    return sorted(out, key=lambda x: (x.get("learned_score", 0), x.get("demand", 0)), reverse=True)


def discover_trends_and_keywords(timeout: int, topics: List[str]) -> Dict[str, Any]:
    trends_us = fetch_trends("US", timeout)[:20]
    trends_pk = fetch_trends("PK", timeout)[:20]
    research_titles = _load_google_research_titles()[:20]
    trends = list(dict.fromkeys(trends_pk + trends_us + research_titles))[:40]
    keywords: List[Dict[str, Any]] = []
    seeds = topics + trends[:10]
    seen = set()
    for seed in seeds:
        for kw in fetch_suggestions(seed, timeout)[:20]:
            k = kw.strip()
            if not k or k.lower() in seen:
                continue
            seen.add(k.lower())
            demand = _demand_estimate(k)
            competition = _competition_estimate(k)
            ranking = _ranking_probability(demand, competition)
            intent = "commercial" if any(x in k.lower() for x in ["buy", "pricing", "review", "best"]) else "informational"
            keywords.append({
                "keyword": k,
                "intent": intent,
                "demand": round(demand, 2),
                "competition": round(competition, 2),
                "ranking_probability": round(ranking, 2),
            })

    learning = load_learning_signals()
    topic_scores = topic_success_scores()
    if topic_scores:
        learning_topics = learning.setdefault("topics", {})
        for topic, ctr in topic_scores.items():
            learning_topics[str(topic).lower()] = {"clicks": 0, "impressions": 1, "ctr": ctr}
    keywords = apply_learning_signals(keywords, learning)
    keywords = _apply_search_console_query_signals(keywords)
    return {"trends": trends, "keywords": keywords[:200]}


def load_learning_signals(path: str = "data/learning_signals.json") -> Dict[str, Any]:
    try:
        p = Path(path)
        if not p.exists():
            return {"topics": {}, "keywords": {}}
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"topics": {}, "keywords": {}}
    except Exception:
        return {"topics": {}, "keywords": {}}


def apply_learning_signals(keywords: List[Dict[str, Any]], learning: Dict[str, Any]) -> List[Dict[str, Any]]:
    topic_signals = learning.get("topics", {}) if isinstance(learning, dict) else {}
    out = []
    for item in keywords:
        k = str(item.get("keyword", "")).strip()
        signal = topic_signals.get(k.lower(), {}) if isinstance(topic_signals, dict) else {}
        try:
            clicks = max(0.0, float(signal.get("clicks", 0)))
            impressions = max(0.0, float(signal.get("impressions", 0)))
        except (TypeError, ValueError):
            clicks, impressions = 0.0, 0.0
        ctr = (clicks / impressions) if impressions else 0.0
        boost = min(25.0, (ctr * 100.0) * 0.25 + min(clicks, 100.0) * 0.05)
        enriched = dict(item)
        enriched["learning_boost"] = round(boost, 2)
        enriched["learned_score"] = round(float(item.get("ranking_probability", 0)) + boost, 2)
        out.append(enriched)
    return sorted(out, key=lambda x: (x.get("learned_score", 0), x.get("demand", 0)), reverse=True)


def cluster_keywords(keywords: List[Dict[str, Any]]) -> Dict[str, Any]:
    clusters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in keywords:
        k = item["keyword"]
        head = k.split()[0].lower() if k.split() else k.lower()
        clusters[head].append(item)

    cluster_list = []
    for head, items in clusters.items():
        items_sorted = sorted(items, key=lambda x: x["ranking_probability"], reverse=True)
        pillar = items_sorted[0]["keyword"]
        cluster_list.append({"pillar": pillar, "cluster": items_sorted})
    cluster_list.sort(key=lambda x: x["cluster"][0]["ranking_probability"], reverse=True)
    return {"clusters": cluster_list}


def save_trends_keywords(trends: List[str], keywords: List[Dict[str, Any]], clusters: Dict[str, Any]) -> None:
    Path("trends.json").write_text(json.dumps({"trends": trends}, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = {"keywords": keywords, "clusters": clusters.get("clusters", [])}
    Path("keywords.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def competitor_analysis(api_key: str, model: str, timeout: int, topic: str) -> Dict[str, Any]:
    prompt = f"""
Return strict JSON with keys: competitors, content_gaps, superior_outline.
Topic: {topic}
- competitors: array of 5 objects with fields title, angle, strengths, weaknesses (simulate SERP leaders)
- content_gaps: array of concrete missing points competitors often miss
- superior_outline: array of section headings for a better article
""".strip()
    return openai_json(api_key, model, prompt, timeout, temperature=0.5)


def serp_difficulty_simulation(api_key: str, model: str, timeout: int, topic: str, niche_score: float) -> Dict[str, Any]:
    prompt = f"""
Return strict JSON with keys: difficulty_score, recommended_word_count, depth_strategy.
Topic: {topic}
Niche profitability proxy: {niche_score}
Estimate ranking difficulty (0-100), recommend content depth and target word count.
""".strip()
    return openai_json(api_key, model, prompt, timeout, temperature=0.4)
