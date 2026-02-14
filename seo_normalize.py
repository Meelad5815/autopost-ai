import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, List

import requests

from engine.config import ConfigError, load_config
from engine.wp_client import request, update_post
from seo import build_meta_description


def clean_text(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value or "")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("????", "").replace("???", "").replace("??", "")
    return text


def trim_title(title: str, limit: int = 60) -> str:
    title = clean_text(title)
    title = re.sub(r"\s*[-:|]\s*complete guide with steps\s*\(2026\)\s*$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s*\(2026\)\s*$", " (2026)", title)
    if len(title) <= limit:
        return title
    clipped = title[:limit].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped.strip(" -:")


def build_slug(text: str, max_len: int = 90) -> str:
    text = clean_text(text).lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text).strip("-")
    text = re.sub(r"-{2,}", "-", text)
    if len(text) <= max_len:
        return text or "post"
    short = text[:max_len].rstrip("-")
    if "-" in short:
        short = short.rsplit("-", 1)[0]
    return short or "post"


def excerpt_from_content(content_html: str, max_len: int = 155) -> str:
    plain = clean_text(content_html)
    return build_meta_description("", plain, max_len=max_len)


def normalize_post(post: Dict[str, Any]) -> Dict[str, Any]:
    old_title = post.get("title", {}).get("rendered", "") or ""
    old_slug = str(post.get("slug", "") or "")
    content_html = post.get("content", {}).get("rendered", "") or ""
    old_excerpt = post.get("excerpt", {}).get("rendered", "") or ""

    new_title = trim_title(old_title, 60)
    new_slug = build_slug(new_title, 90)
    new_excerpt = build_meta_description(clean_text(old_excerpt), excerpt_from_content(content_html), max_len=155)

    payload: Dict[str, Any] = {}
    if clean_text(old_title) != new_title:
        payload["title"] = new_title
    if old_slug != new_slug:
        payload["slug"] = new_slug
    if clean_text(old_excerpt) != new_excerpt and new_excerpt:
        payload["excerpt"] = new_excerpt
    return payload


def main() -> int:
    try:
        cfg = load_config()
    except ConfigError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 2

    limit = max(1, min(int(os.getenv("MAX_SEO_FIX_POSTS", "12")), 50))
    try:
        resp = request(
            "GET",
            cfg.wp_url,
            "posts",
            cfg.wp_user,
            cfg.wp_app_password,
            cfg.request_timeout,
            params={"per_page": limit, "_fields": "id,title,slug,content,excerpt,categories,tags"},
        )
        resp.raise_for_status()
        posts: List[Dict[str, Any]] = resp.json()
    except requests.RequestException as exc:
        print(json.dumps({"status": "failed", "error": f"fetch_posts_failed: {exc}"}, ensure_ascii=False))
        return 1

    checked = 0
    updated = 0
    skipped = 0
    failures: List[Dict[str, Any]] = []
    changes: List[Dict[str, Any]] = []

    for post in posts:
        checked += 1
        payload = normalize_post(post)
        if not payload:
            skipped += 1
            continue
        try:
            update_post(
                int(post["id"]),
                payload,
                cfg.wp_url,
                cfg.wp_user,
                cfg.wp_app_password,
                cfg.request_timeout,
            )
            updated += 1
            changes.append({"id": post["id"], "changes": list(payload.keys())})
        except Exception as exc:  # noqa: BLE001
            failures.append({"id": post.get("id"), "error": str(exc)})

    result = {
        "status": "ok",
        "checked": checked,
        "updated": updated,
        "skipped": skipped,
        "failed": len(failures),
        "changes": changes[:20],
        "failures": failures[:20],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
