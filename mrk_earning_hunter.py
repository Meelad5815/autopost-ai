"""Cloud-first, zero-cost opportunity scanner for MRK AI Earning OS."""
from __future__ import annotations
import html, json, re, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

OUT_JSON = Path("data/earning_opportunities.json")
OUT_MD = Path("data/earning_opportunities.md")

QUERIES = [
    "freelance web developer WordPress",
    "remote Python Django developer freelance",
    "Shopify developer freelance",
    "automation Arduino PLC freelance",
    "remote junior web developer",
    "Pakistan remote developer jobs",
]

SKILL_TERMS = {
    "wordpress": 5, "web developer": 5, "python": 4, "django": 4,
    "shopify": 4, "automation": 4, "arduino": 3, "plc": 3,
    "javascript": 3, "seo": 2, "graphic design": 2,
    "remote": 2, "freelance": 2,
}

def google_news_rss(query: str) -> list[dict]:
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}
    )
    req = urllib.request.Request(url, headers={"User-Agent": "MRK-Earning-Hunter/1.0"})
    with urllib.request.urlopen(req, timeout=20) as response:
        root = ET.fromstring(response.read())
    items = []
    for item in root.findall("./channel/item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        source = item.find("source")
        source_name = source.text.strip() if source is not None and source.text else ""
        if title and link:
            items.append({"title": html.unescape(title), "url": link,
                          "published": pub, "source": source_name, "query": query})
    return items

def score(item: dict) -> int:
    text = f"{item['title']} {item['query']}".lower()
    points = sum(weight for term, weight in SKILL_TERMS.items() if term in text)
    if any(x in text for x in ("earn", "hiring", "job", "freelance", "client", "project")):
        points += 2
    return points

def main() -> None:
    found, seen = [], set()
    for query in QUERIES:
        try:
            for item in google_news_rss(query):
                key = re.sub(r"\W+", "", item["title"].lower())
                if key in seen:
                    continue
                seen.add(key)
                item["score"] = score(item)
                found.append(item)
        except Exception as exc:
            found.append({"title": f"Feed error for: {query}", "url": "",
                          "published": "", "source": "MRK", "query": query,
                          "score": -1, "error": str(exc)})
    found.sort(key=lambda x: (x["score"], x.get("published", "")), reverse=True)
    now = datetime.now(timezone.utc).isoformat()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "generated_at": now, "mode": "public-rss",
        "count": len(found), "opportunities": found[:100]
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# MRK Earning Opportunities", "", f"Generated: {now}", ""]
    for i, item in enumerate(found[:30], 1):
        lines += [f"## {i}. {item['title']}",
                  f"- Score: **{item['score']}**",
                  f"- Source: {item.get('source', '')}",
                  f"- Published: {item.get('published', '')}",
                  f"- URL: {item.get('url', '')}", ""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__":
    main()
