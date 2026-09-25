"""Platform-neutral social publishing queue for WordPress posts.

This module never stores social credentials and never uses unofficial WhatsApp
automation. It prepares platform-specific payloads for an approved publisher
such as Metricool or an official platform API integration.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

QUEUE_FILE = Path("data/social_queue.json")
SUPPORTED_PLATFORMS = (
    "facebook",
    "instagram",
    "x",
    "linkedin",
    "telegram",
    "pinterest",
    "youtube",
)


def _clean(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _hashtags(topic: str, language: str) -> list[str]:
    words = [w.strip(".,!?()[]{}:;#") for w in _clean(topic).split()]
    tags = ["#MRK", "#MRKOfficial"]
    for word in words:
        if len(word) >= 4 and word.isascii() and word.isalnum():
            tags.append("#" + word)
    return list(dict.fromkeys(tags))[:6]


def _make_id(post_id: int, platform: str, url: str) -> str:
    raw = f"{post_id}:{platform}:{url}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def build_social_variants(
    post_id: int,
    title: str,
    excerpt: str,
    post_url: str,
    image_url: str = "",
    topic: str = "",
    language: str = "en",
) -> list[dict[str, Any]]:
    title = _clean(title)
    excerpt = _clean(excerpt)
    post_url = _clean(post_url)
    image_url = _clean(image_url)
    tags = " ".join(_hashtags(topic or title, language))
    base = f"{title}\n\n{excerpt}".strip()
    return [
        {
            "id": _make_id(post_id, "facebook", post_url),
            "platform": "facebook",
            "text": f"{base}\n\n{post_url}\n\n{tags}".strip(),
            "media_url": image_url,
        },
        {
            "id": _make_id(post_id, "instagram", post_url),
            "platform": "instagram",
            "text": f"{base}\n\nLink in bio.\n\n{tags}".strip(),
            "media_url": image_url,
        },
        {
            "id": _make_id(post_id, "x", post_url),
            "platform": "x",
            "text": f"{title} — {post_url}\n{tags}".strip(),
            "media_url": image_url,
        },
        {
            "id": _make_id(post_id, "linkedin", post_url),
            "platform": "linkedin",
            "text": f"{base}\n\nRead more: {post_url}\n\n{tags}".strip(),
            "media_url": image_url,
        },
        {
            "id": _make_id(post_id, "telegram", post_url),
            "platform": "telegram",
            "text": f"{base}\n\n{post_url}\n\n{tags}".strip(),
            "media_url": image_url,
        },
        {
            "id": _make_id(post_id, "pinterest", post_url),
            "platform": "pinterest",
            "text": f"{title}\n\n{excerpt}\n\n{post_url}".strip(),
            "media_url": image_url,
        },
        {
            "id": _make_id(post_id, "youtube", post_url),
            "platform": "youtube",
            "text": f"{title}\n\n{excerpt}\n\nArticle: {post_url}\n\n{tags}".strip(),
            "media_url": image_url,
        },
    ]


def enqueue_social_posts(
    post_id: int,
    title: str,
    excerpt: str,
    post_url: str,
    image_url: str = "",
    topic: str = "",
    language: str = "en",
) -> int:
    if not post_url:
        return 0

    QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if QUEUE_FILE.exists():
        try:
            data = json.loads(QUEUE_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {"items": []}
    else:
        data = {"items": []}

    items = data.get("items", [])
    existing = {str(x.get("id")) for x in items}
    now = datetime.now(timezone.utc).isoformat()
    added = 0

    for variant in build_social_variants(
        post_id, title, excerpt, post_url, image_url, topic, language
    ):
        if variant["id"] in existing:
            continue
        items.append(
            {
                **variant,
                "post_id": post_id,
                "status": "queued",
                "retry_count": 0,
                "created_at": now,
                "updated_at": now,
                "provider": os.getenv("SOCIAL_PROVIDER", "metricool"),
            }
        )
        added += 1

    data["items"] = items[-1000:]
    QUEUE_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return added
