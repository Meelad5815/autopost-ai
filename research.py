"""Bounded zero-cost Google ecosystem research loop."""
from __future__ import annotations
import json, re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import requests

USER_AGENT = "MRK-Autopost-Research/1.0"
TIMEOUT = 15
FEEDS = {
    "google_trends_us": "https://trends.google.com/trending/rss?geo=US",
    "google_trends_pk": "https://trends.google.com/trending/rss?geo=PK",
    "google_search_central": "https://developers.google.com/search/blog/rss.xml",
    "google_developers": "https://developers.googleblog.com/feeds/posts/default",
    "google_news_ai": "https://news.google.com/rss/search?q=AI+automation&hl=en-US&gl=US&ceid=US:en",
    "google_news_seo": "https://news.google.com/rss/search?q=SEO+Google+Search&hl=en-US&gl=US&ceid=US:en",
    "google_news_wordpress": "https://news.google.com/rss/search?q=WordPress&hl=en-US&gl=US&ceid=US:en",
    "google_news_shopify": "https://news.google.com/rss/search?q=Shopify&hl=en-US&gl=US&ceid=US:en",
}

def fetch_feed(name: str, url: str) -> list[dict[str, str]]:
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
        r.raise_for_status()
        body = r.text
    except Exception:
        return []
    blocks = re.findall(r"<item>(.*?)</item>", body, flags=re.S | re.I)
    if not blocks:
        blocks = re.findall(r"<entry>(.*?)</entry>", body, flags=re.S | re.I)
    out = []
    for block in blocks[:30]:
        m = re.search(r"<title[^>]*>(.*?)</title>", block, flags=re.S | re.I)
        if not m: continue
        title = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        if title: out.append({"title": title, "source": name})
    return out

def build_research() -> dict[str, Any]:
    items = []
    for name, url in FEEDS.items(): items.extend(fetch_feed(name, url))
    seen = set(); unique = []
    for item in items:
        key = re.sub(r"\s+", " ", item["title"].lower()).strip()
        if key and key not in seen: seen.add(key); unique.append(item)
    ideas = []
    for item in unique[:120]:
        low = item["title"].lower()
        if any(k in low for k in ("search", "seo", "ranking", "core update")):
            ideas.append("Review SEO/content rules against current Google Search guidance.")
        if any(k in low for k in ("ai", "gemini", "machine learning")):
            ideas.append("Review local-AI prompts and AI-tool topics for freshness.")
        if "wordpress" in low: ideas.append("Refresh WordPress topic clusters.")
        if "shopify" in low: ideas.append("Refresh Shopify topic clusters.")
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "feed_count": len(FEEDS), "item_count": len(unique), "items": unique[:120], "improvement_queue": list(dict.fromkeys(ideas))[:20], "policy": {"public_google_feeds_only": True, "no_rate_limit_bypass": True, "no_automatic_executable_code_changes": True}}

if __name__ == "__main__":
    data = build_research()
    Path("research.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    Path("improvement_queue.json").write_text(json.dumps({"generated_at": data["generated_at"], "queue": data["improvement_queue"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Google research: {data['item_count']} items; {len(data['improvement_queue'])} ideas.")