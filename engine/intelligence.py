import re
from typing import Any, Dict, List

import requests

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
    # heuristic proxy when no paid SEO APIs are available
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
