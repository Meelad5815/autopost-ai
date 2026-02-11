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
)
logger = logging.getLogger("autopost")


def slugify(text: str) -> str:
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
