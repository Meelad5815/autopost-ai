"""Autonomous, self-optimizing WordPress content engine.

This module orchestrates intelligence, strategy, monetization, distribution,
and feedback loops while reusing the existing API-first publishing approach.
"""

import json
import logging
import random
import re
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

import requests

from engine.ai import openai_json
from engine.config import ConfigError, load_config
from engine.distribution import create_web_story_snippet, social_captions
from engine.feedback import choose_best_topics, record_post_performance
from engine.intelligence import (
    competitor_analysis,
    detect_profitable_niches,
    serp_difficulty_simulation,
)
from engine.monetization import cta_ab_variant, engagement_score, insert_affiliate_links, insert_lead_gen_block
from engine.storage import CALENDAR_FILE, KEYWORD_FILE, NICHE_FILE, load_json, save_json, save_run_report, ensure_dirs
from engine.strategy import detect_old_posts_for_refresh, generate_calendar, select_today_topics
from engine.wp_client import clean_title, ensure_term, get_posts, near_duplicate, publish_with_retry, schedule_iso


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("autopost.log", encoding="utf-8")],
import requests

from engine.ai import openai_json
from engine.config import ConfigError, load_config
from engine.distribution import create_web_story_snippet, social_captions
from engine.feedback import choose_best_topics, record_post_performance
from engine.intelligence import (
    competitor_analysis,
    detect_profitable_niches,
    serp_difficulty_simulation,
)
from engine.monetization import cta_ab_variant, engagement_score, insert_affiliate_links, insert_lead_gen_block
from engine.storage import CALENDAR_FILE, KEYWORD_FILE, NICHE_FILE, load_json, save_json, save_run_report, ensure_dirs
from engine.strategy import detect_old_posts_for_refresh, generate_calendar, select_today_topics
from engine.wp_client import clean_title, ensure_term, get_posts, near_duplicate, publish_with_retry, schedule_iso


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("autopost.log", encoding="utf-8")],
import requests

from engine.ai import openai_json
from engine.config import ConfigError, load_config
from engine.distribution import create_web_story_snippet, social_captions
from engine.feedback import choose_best_topics, record_post_performance
from engine.intelligence import (
    competitor_analysis,
    detect_profitable_niches,
    serp_difficulty_simulation,
)
from engine.monetization import cta_ab_variant, engagement_score, insert_affiliate_links, insert_lead_gen_block
from engine.storage import CALENDAR_FILE, KEYWORD_FILE, NICHE_FILE, load_json, save_json, save_run_report, ensure_dirs
from engine.strategy import detect_old_posts_for_refresh, generate_calendar, select_today_topics
from engine.wp_client import clean_title, ensure_term, get_posts, near_duplicate, publish_with_retry, schedule_iso


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("autopost.log", encoding="utf-8")],
import requests

from engine.ai import openai_json
from engine.config import ConfigError, load_config
from engine.distribution import create_web_story_snippet, social_captions
from engine.feedback import choose_best_topics, record_post_performance
from engine.intelligence import (
    competitor_analysis,
    detect_profitable_niches,
    serp_difficulty_simulation,
)
from engine.monetization import cta_ab_variant, engagement_score, insert_affiliate_links, insert_lead_gen_block
from engine.storage import CALENDAR_FILE, KEYWORD_FILE, NICHE_FILE, load_json, save_json, save_run_report, ensure_dirs
from engine.strategy import detect_old_posts_for_refresh, generate_calendar, select_today_topics
from engine.wp_client import clean_title, ensure_term, get_posts, near_duplicate, publish_with_retry, schedule_iso


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("autopost.log", encoding="utf-8")],
import json
import logging
import os
import random
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.auth import HTTPBasicAuth


# =========================
# Configuration
# =========================
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))
MAX_PUBLISH_RETRIES = int(os.getenv("MAX_PUBLISH_RETRIES", "3"))
OPENAI_API_URL = "https://api.openai.com/v1/responses"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
POSTS_PER_RUN = max(1, int(os.getenv("POSTS_PER_RUN", "1")))
POST_STATUS = os.getenv("POST_STATUS", "publish").strip().lower()  # publish|future
SCHEDULE_INTERVAL_MINUTES = int(os.getenv("SCHEDULE_INTERVAL_MINUTES", "90"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.82"))

ENABLE_AFFILIATE_LINKS = os.getenv("ENABLE_AFFILIATE_LINKS", "false").lower() == "true"
ENABLE_CTA_BLOCK = os.getenv("ENABLE_CTA_BLOCK", "true").lower() == "true"
ENABLE_AUTHOR_BOX = os.getenv("ENABLE_AUTHOR_BOX", "true").lower() == "true"
ENABLE_TABLE_OF_CONTENTS = os.getenv("ENABLE_TABLE_OF_CONTENTS", "true").lower() == "true"
ENABLE_AUTO_INTERLINK_FUTURE = os.getenv("ENABLE_AUTO_INTERLINK_FUTURE", "true").lower() == "true"

AFFILIATE_URL = os.getenv("AFFILIATE_URL", "").strip()
AFFILIATE_ANCHOR = os.getenv("AFFILIATE_ANCHOR", "recommended tool").strip()
AUTHOR_NAME = os.getenv("AUTHOR_NAME", "Editorial Team").strip()
AUTHOR_BIO = os.getenv(
    "AUTHOR_BIO",
    "We publish practical, evidence-based guides to help readers make better digital decisions.",
).strip()

HISTORY_DIR = Path(os.getenv("HISTORY_DIR", "data"))
KEYWORD_HISTORY_FILE = HISTORY_DIR / "keyword_history.json"
RUN_REPORTS_DIR = HISTORY_DIR / "run_reports"
LOG_FILE = os.getenv("LOG_FILE", "autopost.log")


# =========================
# Logging
# =========================
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("autopost")


def slugify(text: str) -> str:
class ConfigError(Exception):
    pass


@dataclass
class RuntimeConfig:
    wp_url: str
    wp_user: str
    wp_app_password: str
    openai_api_key: str


# =========================
# Generic utilities
# =========================
def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(f"Missing required environment variable: {name}")
    return value


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9\s-]", "", text).strip().lower()
    return re.sub(r"[\s_-]+", "-", text).strip("-")[:70]


def build_article(
    api_key: str,
    model: str,
    timeout: int,
    topic: str,
    comp: Dict[str, Any],
    serp: Dict[str, Any],
    refresh_context: str = "",
) -> Dict[str, Any]:
    outline = comp.get("superior_outline", [])
    gaps = comp.get("content_gaps", [])
    target_words = int(serp.get("recommended_word_count", 1400))

    prompt = f"""
Return strict JSON with keys:
- title
- meta_description
- excerpt
- content_html
- tags
- categories
- image_query
- faq_items
- related_keywords

Topic: {topic}
Target word count: at least {max(1200, target_words)} words
Competitor gaps to exploit: {gaps}
Better outline to follow: {outline}
Refresh context (if any): {refresh_context}

Requirements:
- human, authoritative tone
- include H2/H3, bullet lists, practical examples, FAQ section, CTA
- include placeholder [INTERNAL_LINK:related-article]
- no markdown/code fences
""".strip()

    article = openai_json(api_key, model, prompt, timeout, temperature=0.7)
    required = ["title", "meta_description", "excerpt", "content_html", "tags", "categories", "image_query", "faq_items"]
    missing = [k for k in required if not article.get(k)]
    if missing:
        raise RuntimeError(f"AI article missing required fields: {missing}")
    return article


def interlink(content_html: str, existing_posts: List[Dict[str, Any]]) -> str:
    body = re.sub(r"<[^>]+>", " ", content_html).lower()
    candidates = []
    for p in existing_posts:
        title = clean_title(p.get("title", {}).get("rendered", ""))
        link = p.get("link", "")
        if not title or not link:
            continue
        overlap = sum(1 for w in set(title.lower().split()) if len(w) > 4 and w in body)
        if overlap:
            candidates.append((overlap, title, link))
    candidates.sort(key=lambda x: x[0], reverse=True)
    links = "".join(f'<li><a href="{u}">{t}</a></li>' for _, t, u in candidates[:3])
    block = f"<h2>Related Reading</h2><ul>{links}</ul>" if links else ""
    return content_html.replace("[INTERNAL_LINK:related-article]", block) if block else content_html


def faq_schema(faq_items: List[Dict[str, str]]) -> str:
    entities = []
    for item in faq_items[:6]:
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        if q and a:
            entities.append({"@type": "Question", "name": q, "acceptedAnswer": {"@type": "Answer", "text": a}})
    return '<script type="application/ld+json">' + json.dumps({"@context": "https://schema.org", "@type": "FAQPage", "mainEntity": entities}) + "</script>"


def synthesize_image_meta(api_key: str, model: str, timeout: int, title: str, image_query: str) -> Dict[str, str]:
    prompt = f"""Return strict JSON: alt_text, caption. title={title}; image={image_query}."""
    data = openai_json(api_key, model, prompt, timeout, temperature=0.3)
    return {
        "alt_text": str(data.get("alt_text", f"Featured image for {title}"))[:120],
        "caption": str(data.get("caption", f"Illustration for {title}"))[:180],
    }


def fetch_image(query: str, timeout: int) -> Dict[str, Any]:
    pexels_key = __import__("os").getenv("PEXELS_API_KEY", "").strip()
    if pexels_key:
        try:
            p = requests.get(
                "https://api.pexels.com/v1/search",
                headers={"Authorization": pexels_key},
                params={"query": query, "per_page": 1, "orientation": "landscape"},
                timeout=timeout,
            )
            p.raise_for_status()
            photos = p.json().get("photos", [])
            if photos:
                src = photos[0].get("src", {})
                url = src.get("large2x") or src.get("large") or src.get("original")
                if url:
                    img = requests.get(url, timeout=timeout)
                    img.raise_for_status()
                    return {"bytes": img.content, "url": url, "filename": f"{slugify(query) or 'featured'}.jpg"}
        except Exception as exc:
            logger.warning("Pexels failed: %s", exc)

    url = f"https://source.unsplash.com/1600x900/?{requests.utils.quote(query)}"
    img = requests.get(url, timeout=timeout, allow_redirects=True)
    img.raise_for_status()
    return {"bytes": img.content, "url": img.url, "filename": f"{slugify(query) or 'featured'}.jpg"}


def upload_media(base_url: str, user: str, app_password: str, timeout: int, image: Dict[str, Any], alt_text: str, caption: str) -> int:
    from engine.wp_client import request

    upload = request(
        "POST",
        base_url,
        "media",
        user,
        app_password,
        timeout,
        headers={
            "Content-Disposition": f'attachment; filename="{image["filename"]}"',
            "Content-Type": "image/jpeg",
        },
        data=image["bytes"],
    )
    if upload.status_code not in (200, 201):
        raise RuntimeError(f"Media upload failed: {upload.status_code} {upload.text[:300]}")
    media_id = int(upload.json()["id"])

    patch = request(
        "POST",
        base_url,
        f"media/{media_id}",
        user,
        app_password,
        timeout,
        json={"alt_text": alt_text, "caption": caption},
    )
    if patch.status_code not in (200, 201):
        logger.warning("Media patch failed for %s", media_id)
    return media_id


def maybe_generate_calendar(cfg: Any, niches: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not cfg.enable_content_calendar:
        return load_json(CALENDAR_FILE, {"calendar": []})
    keywords = [n["keyword"] for n in niches]
    cal = generate_calendar(cfg.openai_api_key, cfg.openai_model, cfg.request_timeout, keywords, cfg.calendar_days)
    save_json(CALENDAR_FILE, cal)
    return cal


def choose_topics(cfg: Any, niches: List[Dict[str, Any]], calendar: Dict[str, Any]) -> List[Dict[str, str]]:
    if cfg.enable_content_calendar and calendar.get("calendar"):
        return select_today_topics(calendar, cfg.posts_per_run)
    return [{"topic": n["keyword"], "intent_type": "informational"} for n in niches[: cfg.posts_per_run]]


def apply_monetization(content_html: str, topic: str, cfg: Any) -> Dict[str, Any]:
    variant = {"id": "N", "cta": ""}
    if cfg.enable_affiliate_automation:
        content_html = insert_affiliate_links(content_html, topic)
    if cfg.enable_lead_gen:
        content_html = insert_lead_gen_block(content_html)
    if cfg.enable_conversion_optimization:
        variant = cta_ab_variant()
        content_html += f"<section><h2>Take the Next Step</h2><p>{variant['cta']}</p></section>"
    return {"content": content_html, "cta_variant": variant["id"], "engagement_score": engagement_score(content_html)}


def build_distribution_payload(title: str, url: str, excerpt: str, content_html: str, cfg: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if cfg.enable_social_syndication:
        payload["social"] = social_captions(title, url, excerpt)
    if cfg.enable_short_content:
        payload["web_story"] = create_web_story_snippet(title, content_html)
    return payload


def update_learning(topic: str, title: str, post_id: int, cfg: Any) -> None:
    if not cfg.enable_performance_tracking:
        return
    # Placeholder metrics to keep loop active until analytics APIs are connected.
    metrics = {
        "topic": topic,
        "impressions": random.randint(150, 2000),
        "clicks": random.randint(10, 220),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    record_post_performance(post_id, title, metrics)


def main() -> int:
    ensure_dirs()
    run_results: List[Dict[str, Any]] = []
    try:
        cfg = load_config()
        logger.info("Starting autonomous run. posts_per_run=%s", cfg.posts_per_run)

        niches = detect_profitable_niches(cfg.request_timeout, cfg.niches_per_run) if cfg.enable_niche_intelligence else []
        save_json(NICHE_FILE, {"generated_at": datetime.now(timezone.utc).isoformat(), "niches": niches})
        save_json(KEYWORD_FILE, {"keywords": [n.get("keyword") for n in niches]})

        calendar = maybe_generate_calendar(cfg, niches)
        topic_candidates = choose_topics(cfg, niches, calendar)

        if cfg.enable_learning_loop:
            topic_candidates = [{"topic": t, "intent_type": "optimized"} for t in choose_best_topics([x["topic"] for x in topic_candidates])] or topic_candidates

        existing_posts = get_posts(cfg.wp_url, cfg.wp_user, cfg.wp_app_password, cfg.request_timeout)
        existing_titles = [clean_title(p.get("title", {}).get("rendered", "")) for p in existing_posts]

        stale_posts = detect_old_posts_for_refresh(existing_posts, cfg.refresh_age_days) if cfg.enable_content_refresh else []

        for idx in range(1, cfg.posts_per_run + 1):
            topic_meta = topic_candidates[(idx - 1) % len(topic_candidates)] if topic_candidates else {"topic": "seo automation", "intent_type": "informational"}
            topic = topic_meta["topic"]
            refresh_context = ""
            if stale_posts:
                pick = stale_posts[(idx - 1) % len(stale_posts)]
                refresh_context = f"Refresh this older post angle: {clean_title(pick.get('title', {}).get('rendered', ''))}"

            comp = competitor_analysis(cfg.openai_api_key, cfg.openai_model, cfg.request_timeout, topic) if cfg.enable_competitor_analysis else {"content_gaps": [], "superior_outline": []}
            niche_profit = niches[(idx - 1) % len(niches)]["profitability"] if niches else 50.0
            serp = serp_difficulty_simulation(cfg.openai_api_key, cfg.openai_model, cfg.request_timeout, topic, niche_profit) if cfg.enable_serp_simulation else {"recommended_word_count": 1400}

            article = build_article(cfg.openai_api_key, cfg.openai_model, cfg.request_timeout, topic, comp, serp, refresh_context)
            title = str(article["title"]).strip()[:65]

            if near_duplicate(title, existing_titles):
                logger.warning("Skipping near-duplicate: %s", title)
                run_results.append({"status": "skipped", "reason": "near_duplicate", "topic": topic, "title": title})
                continue

            content_html = interlink(str(article["content_html"]), existing_posts)
            content_html += faq_schema(article.get("faq_items", []))

            monetized = apply_monetization(content_html, topic, cfg)
            content_html = monetized["content"]

            image = fetch_image(str(article.get("image_query", topic)), cfg.request_timeout)
            image_meta = synthesize_image_meta(cfg.openai_api_key, cfg.openai_model, cfg.request_timeout, title, str(article.get("image_query", topic)))
            media_id = upload_media(cfg.wp_url, cfg.wp_user, cfg.wp_app_password, cfg.request_timeout, image, image_meta["alt_text"], image_meta["caption"])

            cat_ids = [
                ensure_term(cfg.wp_url, cfg.wp_user, cfg.wp_app_password, cfg.request_timeout, "categories", str(c).strip(), slugify(str(c)))
                for c in article.get("categories", [])
                if str(c).strip()
            ]
            tag_ids = [
                ensure_term(cfg.wp_url, cfg.wp_user, cfg.wp_app_password, cfg.request_timeout, "tags", str(t).strip(), slugify(str(t)))
                for t in article.get("tags", [])
                if str(t).strip()
            ]

            payload = {
                "title": title,
                "slug": slugify(title),
                "status": "future" if cfg.post_status == "future" else "publish",
                "date_gmt": schedule_iso(idx, cfg.post_status, cfg.schedule_interval_minutes),
                "content": content_html,
                "excerpt": str(article.get("excerpt", "")),
                "categories": cat_ids,
                "tags": tag_ids,
                "featured_media": media_id,
                "meta": {
                    "_yoast_wpseo_metadesc": str(article.get("meta_description", "")),
                    "topic_cluster": topic_meta.get("pillar_topic", ""),
                    "intent_type": topic_meta.get("intent_type", "informational"),
                    "cta_variant": monetized["cta_variant"],
                    "engagement_score": monetized["engagement_score"],
                },
            }
            if not payload["date_gmt"]:
                payload.pop("date_gmt")

            posted = publish_with_retry(payload, cfg.wp_url, cfg.wp_user, cfg.wp_app_password, cfg.request_timeout, cfg.max_publish_retries)
            post_id = int(posted.get("id"))
            post_url = posted.get("link", "")

            distribution = build_distribution_payload(title, post_url, str(article.get("excerpt", "")), content_html, cfg)
            update_learning(topic, title, post_id, cfg)

            run_results.append(
                {
                    "status": "scheduled" if cfg.post_status == "future" else "published",
                    "post_id": post_id,
                    "title": title,
                    "topic": topic,
                    "url": post_url,
                    "distribution": distribution,
                }
            )

        summary = {
            "success": sum(1 for r in run_results if r["status"] in {"published", "scheduled"}),
            "failed": sum(1 for r in run_results if r["status"] == "failed"),
            "skipped": sum(1 for r in run_results if r["status"] == "skipped"),
        }
        report_path = save_run_report(
            {
                "started_at": datetime.now(timezone.utc).isoformat(),
                "settings": {"posts_per_run": cfg.posts_per_run, "status_mode": cfg.post_status},
                "summary": summary,
                "results": run_results,
            }
        )

        logger.info("Run completed. %s", summary)
        logger.info("Report: %s", report_path)
        print(f"WordPress response status: {200 if summary['failed'] == 0 else 207}")
        print(json.dumps(summary))
        return 0 if summary["failed"] == 0 else 1

    except ConfigError as exc:
        logger.error("Config error: %s", exc)
def word_count(text: str) -> int:
    plain = re.sub(r"<[^>]+>", " ", text)
    return len(re.findall(r"\b\w+\b", plain))


def ensure_dirs() -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    RUN_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


# =========================
# External API adapters
# =========================
def wp_request(method: str, base_url: str, path: str, username: str, app_password: str, **kwargs: Any) -> requests.Response:
    url = f"{base_url.rstrip('/')}/wp-json/wp/v2/{path.lstrip('/')}"
    return requests.request(
        method=method,
        url=url,
        auth=HTTPBasicAuth(username, app_password),
        timeout=REQUEST_TIMEOUT,
        **kwargs,
    )


def request_openai_json(api_key: str, prompt: str, temperature: float = 0.6) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": OPENAI_MODEL, "input": prompt, "temperature": temperature}
    response = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    raw_text = response.json().get("output_text", "").strip()
    if not raw_text:
        raise RuntimeError("OpenAI returned empty output_text.")
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"OpenAI output is not valid JSON: {raw_text[:700]}") from exc


# =========================
# Keyword research + trends
# =========================
def fetch_google_suggestions(query: str) -> List[str]:
    """Uses Google suggest endpoint for keyword ideas."""
    try:
        resp = requests.get(
            "https://suggestqueries.google.com/complete/search",
            params={"client": "firefox", "q": query},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and len(data) > 1 and isinstance(data[1], list):
            return [str(x).strip() for x in data[1] if str(x).strip()]
    except Exception as exc:
        logger.warning("Suggestion fetch failed for '%s': %s", query, exc)
    return []


def fetch_google_trending_queries(region: str = "US") -> List[str]:
    """Fetches trending daily queries from Google Trends public RSS."""
    url = "https://trends.google.com/trending/rss"
    try:
        resp = requests.get(url, params={"geo": region}, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        xml_text = resp.text
        titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", xml_text)
        cleaned = [t.strip() for t in titles[1:] if t.strip()]  # skip channel title
        return cleaned[:20]
    except Exception as exc:
        logger.warning("Google Trends fetch failed: %s", exc)
        return []


def load_keyword_history() -> Dict[str, Any]:
    if not KEYWORD_HISTORY_FILE.exists():
        return {"keywords": []}
    try:
        return json.loads(KEYWORD_HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"keywords": []}


def save_keyword_history(payload: Dict[str, Any]) -> None:
    KEYWORD_HISTORY_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def keyword_score(keyword: str) -> float:
    # Heuristic score: prefer long-tail, buyer intent, and modern topics.
    keyword_lower = keyword.lower()
    score = 0.0
    words = keyword_lower.split()
    score += min(4, max(0, len(words) - 2)) * 0.8
    for bonus in ["best", "guide", "for", "tools", "strategy", "2025", "2026", "how to"]:
        if bonus in keyword_lower:
            score += 1.2
    if len(keyword) < 12:
        score -= 0.8
    return score


def select_high_potential_keywords(seed_topics: List[str], limit: int = 10) -> List[str]:
    candidates: List[str] = []
    for seed in seed_topics:
        candidates.extend(fetch_google_suggestions(seed))

    if not candidates:
        return seed_topics[:limit]

    unique = sorted(set(candidates))
    scored = sorted(unique, key=keyword_score, reverse=True)
    return scored[:limit]


def research_keywords_and_trends() -> Dict[str, Any]:
    base_seeds = [
        "ai automation",
        "content marketing strategy",
        "seo for small business",
        "wordpress growth",
        "developer productivity",
    ]
    trending = fetch_google_trending_queries(os.getenv("TRENDS_GEO", "US"))
    selected_keywords = select_high_potential_keywords(base_seeds + trending[:5], limit=12)

    history = load_keyword_history()
    used = {entry.get("keyword", "") for entry in history.get("keywords", [])}
    fresh_keywords = [k for k in selected_keywords if k not in used][:8]
    if not fresh_keywords:
        fresh_keywords = selected_keywords[:5]

    for kw in fresh_keywords:
        history.setdefault("keywords", []).append(
            {"keyword": kw, "selected_at": now_utc().isoformat(), "source": "trends+suggest"}
        )

    history["keywords"] = history["keywords"][-300:]
    save_keyword_history(history)

    return {
        "trending": trending,
        "selected_keywords": selected_keywords,
        "fresh_keywords": fresh_keywords,
    }


# =========================
# WordPress content helpers
# =========================
def get_existing_posts(base_url: str, username: str, app_password: str, per_page: int = 100) -> List[Dict[str, Any]]:
    response = wp_request(
        "GET",
        base_url,
        "posts",
        username,
        app_password,
        params={"per_page": per_page, "_fields": "id,slug,link,title"},
    )
    response.raise_for_status()
    return response.json()


def title_is_near_duplicate(title: str, existing_titles: List[str]) -> bool:
    normalized = title.lower().strip()
    for t in existing_titles:
        ratio = SequenceMatcher(a=normalized, b=t.lower().strip()).ratio()
        if ratio >= SIMILARITY_THRESHOLD:
            return True
    return False


def ensure_taxonomy_term(base_url: str, username: str, app_password: str, taxonomy: str, name: str) -> int:
    term_slug = slugify(name)
    get_resp = wp_request(
        "GET",
        base_url,
        taxonomy,
        username,
        app_password,
        params={"slug": term_slug, "per_page": 1},
    )
    get_resp.raise_for_status()
    found = get_resp.json()
    if found:
        return int(found[0]["id"])

    create_resp = wp_request(
        "POST",
        base_url,
        taxonomy,
        username,
        app_password,
        json={"name": name, "slug": term_slug},
    )
    if create_resp.status_code not in (200, 201):
        raise RuntimeError(f"Failed creating {taxonomy[:-1]} '{name}': {create_resp.status_code} {create_resp.text[:500]}")
    return int(create_resp.json()["id"])


def find_relevant_internal_links(content_html: str, existing_posts: List[Dict[str, Any]], max_links: int = 3) -> List[Tuple[str, str]]:
    body_text = re.sub(r"<[^>]+>", " ", content_html).lower()
    scored: List[Tuple[int, str, str]] = []
    for post in existing_posts:
        title = re.sub(r"<[^>]+>", "", post.get("title", {}).get("rendered", "")).strip()
        link = post.get("link", "").strip()
        if not title or not link:
            continue
        title_words = [w for w in re.findall(r"[a-zA-Z0-9]+", title.lower()) if len(w) > 3]
        overlap = sum(1 for w in set(title_words[:8]) if w in body_text)
        if overlap > 0:
            scored.append((overlap, title, link))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [(title, link) for _, title, link in scored[:max_links]]


def insert_internal_links(content_html: str, links: List[Tuple[str, str]]) -> str:
    if not links:
        return content_html
    links_html = "".join([f'<li><a href="{url}">{title}</a></li>' for title, url in links])
    block = f"<h2>Related Reading</h2><ul>{links_html}</ul>"
    if "[INTERNAL_LINK:related-article]" in content_html:
        return content_html.replace("[INTERNAL_LINK:related-article]", block)
    return content_html + block


def build_table_of_contents(content_html: str) -> str:
    headings = re.findall(r"<(h2|h3)>(.*?)</\1>", content_html, flags=re.IGNORECASE | re.DOTALL)
    if not headings:
        return ""
    items = []
    for _, title in headings[:15]:
        clean = re.sub(r"<[^>]+>", "", title).strip()
        items.append(f"<li>{clean}</li>")
    return "<h2>Table of Contents</h2><ul>" + "".join(items) + "</ul>"


def build_cta_block() -> str:
    return (
        "<section><h2>Take the Next Step</h2>"
        "<p>If you found this guide useful, apply one tactic today and share the result with your team. "
        "Consistent execution creates compounding growth.</p></section>"
    )


def build_author_box() -> str:
    return (
        f"<section><h3>About the Author</h3><p><strong>{AUTHOR_NAME}</strong> — {AUTHOR_BIO}</p></section>"
    )


def apply_optional_content_modules(content_html: str) -> str:
    parts: List[str] = []
    if ENABLE_TABLE_OF_CONTENTS:
        toc = build_table_of_contents(content_html)
        if toc:
            parts.append(toc)

    parts.append(content_html)

    if ENABLE_AFFILIATE_LINKS and AFFILIATE_URL:
        affiliate_block = (
            f"<p>Pro tip: explore our <a href=\"{AFFILIATE_URL}\" rel=\"sponsored nofollow\">{AFFILIATE_ANCHOR}</a> "
            "to accelerate implementation.</p>"
        )
        parts.append(affiliate_block)

    if ENABLE_CTA_BLOCK:
        parts.append(build_cta_block())

    if ENABLE_AUTHOR_BOX:
        parts.append(build_author_box())

    return "\n".join(parts)


def enhance_image_text_with_ai(api_key: str, title: str, image_query: str) -> Dict[str, str]:
    prompt = f"""
Return strict JSON with keys: alt_text, caption.
Context:
- Post title: {title}
- Image topic: {image_query}
Requirements:
- alt_text: descriptive, <= 125 chars
- caption: natural editorial caption, <= 25 words
""".strip()
    data = request_openai_json(api_key, prompt, temperature=0.4)
    alt_text = str(data.get("alt_text", "")).strip()[:125] or f"Featured image for {title}"[:125]
    caption = str(data.get("caption", "")).strip()[:220] or f"Illustration for {title}"[:220]
    return {"alt_text": alt_text, "caption": caption}


def fetch_featured_image(topic_query: str) -> Tuple[bytes, str, str]:
    pexels_key = os.getenv("PEXELS_API_KEY", "").strip()
    if pexels_key:
        try:
            pexels_resp = requests.get(
                "https://api.pexels.com/v1/search",
                headers={"Authorization": pexels_key},
                params={"query": topic_query, "per_page": 1, "orientation": "landscape"},
                timeout=REQUEST_TIMEOUT,
            )
            pexels_resp.raise_for_status()
            photos = pexels_resp.json().get("photos", [])
            if photos:
                src = photos[0].get("src", {})
                img_url = src.get("large2x") or src.get("large") or src.get("original")
                if img_url:
                    img_resp = requests.get(img_url, timeout=REQUEST_TIMEOUT)
                    img_resp.raise_for_status()
                    return img_resp.content, f"{slugify(topic_query) or 'featured'}.jpg", img_url
        except Exception as exc:
            logger.warning("Pexels fetch failed, using Unsplash fallback: %s", exc)

    fallback_url = f"https://source.unsplash.com/1600x900/?{requests.utils.quote(topic_query)}"
    img_resp = requests.get(fallback_url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
    img_resp.raise_for_status()
    return img_resp.content, f"{slugify(topic_query) or 'featured'}.jpg", img_resp.url


def upload_media(base_url: str, username: str, app_password: str, image_bytes: bytes, filename: str, alt_text: str, caption: str) -> int:
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Type": "image/jpeg",
    }
    media_resp = wp_request(
        "POST", base_url, "media", username, app_password, headers=headers, data=image_bytes
    )
    if media_resp.status_code not in (200, 201):
        raise RuntimeError(f"Media upload failed: {media_resp.status_code} {media_resp.text[:600]}")

    media_id = int(media_resp.json()["id"])
    patch_resp = wp_request(
        "POST",
        base_url,
        f"media/{media_id}",
        username,
        app_password,
        json={"alt_text": alt_text, "caption": caption},
    )
    if patch_resp.status_code not in (200, 201):
        logger.warning("Failed to set media alt/caption for %s: %s", media_id, patch_resp.text[:300])
    return media_id


# =========================
# SEO content generation
# =========================
def generate_article_package(api_key: str, primary_keyword: str, related_keywords: List[str], trend_hint: str) -> Dict[str, Any]:
    related_line = ", ".join(related_keywords[:10])
    prompt = f"""
You are an elite SEO content engineer.
Create strict JSON with keys:
- title
- meta_description
- excerpt
- content_html
- tags (5-8)
- categories (1-3)
- image_query
- faq_items (array of 4 objects with keys: question, answer)
- related_keywords (array)

Requirements:
- Primary keyword: {primary_keyword}
- Trend hint: {trend_hint}
- Use related keywords naturally: {related_line}
- Human, natural, authoritative tone.
- content_html must be valid HTML, 1000+ words, include H2/H3, bullet lists, conclusion, CTA, and placeholder [INTERNAL_LINK:related-article].
- Optimize heading clarity and keyword relevance.
- No markdown, no code fences.
""".strip()

    data = request_openai_json(api_key, prompt, temperature=0.7)
    required = ["title", "meta_description", "excerpt", "content_html", "tags", "categories", "image_query", "faq_items"]
    missing = [k for k in required if not data.get(k)]
    if missing:
        raise RuntimeError(f"Generated article missing fields: {missing}")
    if word_count(data["content_html"]) < 1000:
        raise RuntimeError("Generated article is under 1000 words.")
    return data


def build_faq_html(faq_items: List[Dict[str, str]]) -> str:
    blocks = []
    for item in faq_items[:6]:
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        if q and a:
            blocks.append(f"<h3>{q}</h3><p>{a}</p>")
    if not blocks:
        return ""
    return "<h2>Frequently Asked Questions</h2>" + "".join(blocks)


def build_faq_jsonld(faq_items: List[Dict[str, str]]) -> str:
    entities = []
    for item in faq_items[:6]:
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        if q and a:
            entities.append(
                {
                    "@type": "Question",
                    "name": q,
                    "acceptedAnswer": {"@type": "Answer", "text": a},
                }
            )
    schema = {"@context": "https://schema.org", "@type": "FAQPage", "mainEntity": entities}
    return '<script type="application/ld+json">' + json.dumps(schema, ensure_ascii=False) + "</script>"


def ensure_keyword_density(content_html: str, keyword: str) -> str:
    text_words = max(1, word_count(content_html))
    target_mentions = max(4, int(text_words * 0.008))  # ~0.8%
    current_mentions = len(re.findall(re.escape(keyword.lower()), content_html.lower()))
    if current_mentions >= target_mentions:
        return content_html
    needed = target_mentions - current_mentions
    booster = "".join([f"<p>{keyword} remains a strategic priority for sustainable growth.</p>" for _ in range(min(needed, 4))])
    return content_html + booster


# =========================
# Publishing
# =========================
def publish_post_with_retry(base_url: str, username: str, app_password: str, payload: Dict[str, Any]) -> requests.Response:
    for attempt in range(1, MAX_PUBLISH_RETRIES + 1):
        try:
            response = wp_request("POST", base_url, "posts", username, app_password, json=payload)
            if response.status_code in (200, 201):
                return response
            logger.error("Publish attempt %s/%s failed: %s %s", attempt, MAX_PUBLISH_RETRIES, response.status_code, response.text[:500])
        except requests.RequestException as exc:
            logger.error("Publish attempt %s/%s error: %s", attempt, MAX_PUBLISH_RETRIES, exc)

        if attempt < MAX_PUBLISH_RETRIES:
            delay = 2 ** attempt
            logger.info("Retrying publish in %s seconds...", delay)
            time.sleep(delay)
    raise RuntimeError("Post publish failed after retries.")


def schedule_time_for_index(index: int) -> Optional[str]:
    if POST_STATUS != "future":
        return None
    scheduled_at = now_utc() + timedelta(minutes=SCHEDULE_INTERVAL_MINUTES * index)
    return scheduled_at.isoformat()


# =========================
# Orchestration
# =========================
def build_runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        wp_url=require_env("WP_URL"),
        wp_user=require_env("WP_USER"),
        wp_app_password=require_env("WP_APP_PASSWORD"),
        openai_api_key=require_env("OPENAI_API_KEY"),
    )


def run_single_article(config: RuntimeConfig, article_index: int, research: Dict[str, Any], existing_posts: List[Dict[str, Any]]) -> Dict[str, Any]:
    selected_keywords = research.get("fresh_keywords") or research.get("selected_keywords")
    primary_keyword = selected_keywords[(article_index - 1) % len(selected_keywords)]
    trend_hint = random.choice(research.get("trending", [primary_keyword]))
    related_keywords = [k for k in selected_keywords if k != primary_keyword][:10]

    article = generate_article_package(config.openai_api_key, primary_keyword, related_keywords, trend_hint)

    existing_titles = [
        re.sub(r"<[^>]+>", "", p.get("title", {}).get("rendered", "")).strip()
        for p in existing_posts
    ]

    title = str(article["title"]).strip()[:65]
    if title_is_near_duplicate(title, existing_titles):
        logger.warning("Skipping near-duplicate title: %s", title)
        return {"status": "skipped", "reason": "near_duplicate_title", "title": title}

    slug = slugify(title)

    category_ids = [
        ensure_taxonomy_term(config.wp_url, config.wp_user, config.wp_app_password, "categories", str(c).strip())
        for c in article.get("categories", [])
        if str(c).strip()
    ]
    tag_ids = [
        ensure_taxonomy_term(config.wp_url, config.wp_user, config.wp_app_password, "tags", str(t).strip())
        for t in article.get("tags", [])
        if str(t).strip()
    ]

    content_html = str(article["content_html"])
    content_html = ensure_keyword_density(content_html, primary_keyword)

    if ENABLE_AUTO_INTERLINK_FUTURE:
        links = find_relevant_internal_links(content_html, existing_posts)
        content_html = insert_internal_links(content_html, links)

    faq_items = article.get("faq_items", [])
    content_html += build_faq_html(faq_items)
    content_html += build_faq_jsonld(faq_items)

    content_html = apply_optional_content_modules(content_html)

    img_bytes, filename, source_url = fetch_featured_image(str(article.get("image_query", primary_keyword)))
    image_meta = enhance_image_text_with_ai(config.openai_api_key, title, str(article.get("image_query", primary_keyword)))
    media_id = upload_media(
        config.wp_url,
        config.wp_user,
        config.wp_app_password,
        img_bytes,
        filename,
        image_meta["alt_text"],
        image_meta["caption"],
    )

    status = "future" if POST_STATUS == "future" else "publish"
    date_gmt = schedule_time_for_index(article_index)

    post_payload: Dict[str, Any] = {
        "title": title,
        "slug": slug,
        "status": status,
        "content": content_html,
        "excerpt": str(article.get("excerpt", "")),
        "categories": category_ids,
        "tags": tag_ids,
        "featured_media": media_id,
        "meta": {
            "_yoast_wpseo_metadesc": str(article.get("meta_description", "")),
            "autopost_primary_keyword": primary_keyword,
            "autopost_related_keywords": ", ".join(article.get("related_keywords", related_keywords)[:20]),
        },
    }
    if date_gmt:
        post_payload["date_gmt"] = date_gmt

    response = publish_post_with_retry(config.wp_url, config.wp_user, config.wp_app_password, post_payload)
    post_json = response.json()

    return {
        "status": "published" if status == "publish" else "scheduled",
        "post_id": post_json.get("id"),
        "slug": post_json.get("slug"),
        "title": title,
        "primary_keyword": primary_keyword,
        "image_source": source_url,
        "scheduled_for": date_gmt,
    }


def save_run_report(report: Dict[str, Any]) -> Path:
    ts = now_utc().strftime("%Y%m%d_%H%M%S")
    path = RUN_REPORTS_DIR / f"run_{ts}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def main() -> int:
    ensure_dirs()
    started = now_utc()
    results: List[Dict[str, Any]] = []

    try:
        config = build_runtime_config()
        research = research_keywords_and_trends()
        existing_posts = get_existing_posts(config.wp_url, config.wp_user, config.wp_app_password)

        for idx in range(1, POSTS_PER_RUN + 1):
            try:
                logger.info("Processing article %s/%s", idx, POSTS_PER_RUN)
                result = run_single_article(config, idx, research, existing_posts)
                results.append(result)

                if result.get("status") in {"published", "scheduled"}:
                    # Keep in-memory post list fresh for smarter interlinking in the same run.
                    existing_posts = get_existing_posts(config.wp_url, config.wp_user, config.wp_app_password)

            except Exception as article_exc:
                logger.exception("Article %s failed: %s", idx, article_exc)
                results.append({"status": "failed", "error": str(article_exc), "article_index": idx})

        succeeded = sum(1 for r in results if r.get("status") in {"published", "scheduled"})
        failed = sum(1 for r in results if r.get("status") == "failed")
        skipped = sum(1 for r in results if r.get("status") == "skipped")

        report = {
            "started_at": started.isoformat(),
            "finished_at": now_utc().isoformat(),
            "post_status_mode": POST_STATUS,
            "posts_per_run": POSTS_PER_RUN,
            "summary": {"success": succeeded, "failed": failed, "skipped": skipped},
            "results": results,
        }
        report_path = save_run_report(report)

        logger.info("Run complete. success=%s failed=%s skipped=%s", succeeded, failed, skipped)
        logger.info("Run report saved: %s", report_path)

        print(f"WordPress response status: {200 if failed == 0 else 207}")
        print(json.dumps(report["summary"]))

        return 0 if failed == 0 else 1

    except ConfigError as exc:
        logger.error("Configuration error: %s", exc)
        print("WordPress response status: 0")
        return 2
    except requests.RequestException as exc:
        logger.error("HTTP error: %s", exc)
        print("WordPress response status: 0")
        return 3
    except Exception as exc:
        logger.exception("Unhandled error: %s", exc)
        print("WordPress response status: 0")
        return 4


if __name__ == "__main__":
    sys.exit(main())
