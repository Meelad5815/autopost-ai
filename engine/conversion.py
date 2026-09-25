"""Revenue and conversion blocks for WordPress content.

Configuration-driven CTAs with auditable UTM links. No fake earnings or
fabricated click/conversion metrics.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


MONETIZATION_FILE = Path("config/monetization.json")
SERVICES_FILE = Path("config/services.json")


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def load_monetization() -> Dict[str, Any]:
    return _load_json(MONETIZATION_FILE)


def load_services() -> Dict[str, Any]:
    return _load_json(SERVICES_FILE)


def _tracked_url(url: str, topic: str, language: str, content_type: str) -> str:
    """Add deterministic UTM parameters without replacing existing query values."""
    raw = str(url or "").strip()
    if not raw:
        return ""
    try:
        parts = urlsplit(raw)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query.setdefault("utm_source", "mrk-autopost")
        query.setdefault("utm_medium", "content")
        query.setdefault("utm_campaign", "autopost")
        query.setdefault("utm_content", content_type)
        query.setdefault("utm_term", str(topic).strip()[:80])
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))
    except Exception:
        return raw


def _service_cta(topic: str, language: str) -> tuple[str, Dict[str, Any] | None]:
    configured = os.getenv("SERVICE_CONTACT_URL", "").strip()
    service_label = "MRK Digital service"
    if not configured:
        services = load_services().get("services", [])
        if isinstance(services, list):
            for item in services:
                if not isinstance(item, dict) or not item.get("enabled", True):
                    continue
                url = str(item.get("url", "")).strip()
                if url:
                    configured = url
                    service_label = str(item.get("label", service_label)).strip() or service_label
                    break
    if not configured:
        return "", None

    url = _tracked_url(configured, topic, language, "service_cta")
    if language.startswith("ur"):
        text = f"اگر آپ کو {service_label} کے لیے پروفیشنل مدد چاہیے تو MRK Digital سے رابطہ کریں۔"
        button = "سروس کے لیے رابطہ کریں"
    elif language.startswith("roman"):
        text = f"Agar aap ko {service_label} ke liye professional help chahiye to MRK Digital se rabta karein."
        button = "Service ke liye rabta karein"
    else:
        text = f"Need professional help with {service_label}? Contact MRK Digital for a practical service solution."
        button = "Request a service"
    html = f'<div class="mrk-cta mrk-service-cta"><p><strong>{text}</strong></p><p><a href="{url}" rel="nofollow">{button}</a></p></div>'
    return html, {
        "type": "service",
        "placement": "service_mid",
        "label": service_label,
        "destination": url,
    }


def _affiliate_ctas(topic: str, language: str) -> List[tuple[str, Dict[str, Any]]]:
    data = load_monetization()
    items = data.get("affiliate", [])
    if not isinstance(items, list):
        return []
    out: List[tuple[str, Dict[str, Any]]] = []
    for item in items[:2]:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", "")).strip()
        label = str(item.get("label", "")).strip()
        if not url or not label:
            continue
        tracked = _tracked_url(url, topic, language, "affiliate_cta")
        if language.startswith("ur"):
            prefix = "متعلقہ پروڈکٹ/ٹول دیکھیں:"
        elif language.startswith("roman"):
            prefix = "Related product/tool dekhein:"
        else:
            prefix = "Related product/tool:"
        html = f'<div class="mrk-cta mrk-affiliate-cta"><p>{prefix} <a href="{tracked}" rel="sponsored nofollow">{label}</a></p></div>'
        out.append((html, {
            "type": "affiliate",
            "placement": "affiliate_end",
            "label": label,
            "destination": tracked,
        }))
    return out


def inject_conversion_blocks(content_html: str, topic: str, language: str = "en") -> Dict[str, Any]:
    """Insert limited CTAs and return auditable placement metadata."""
    if os.getenv("ENABLE_CONVERSION_OPTIMIZATION", "true").lower() != "true":
        return {"content_html": content_html, "placements": [], "enabled": False}

    service, service_meta = _service_cta(topic, language)
    affiliate_blocks = _affiliate_ctas(topic, language)
    blocks = []
    if service:
        blocks.append(("service", service))

    paragraphs = content_html.split("</p>")
    placements: List[Dict[str, Any]] = []

    if service:
        for i, part in enumerate(paragraphs):
            if len(part.strip()) > 180:
                paragraphs.insert(i + 1, service)
                if service_meta:
                    placements.append(service_meta)
                break

    for block, meta in affiliate_blocks:
        paragraphs.append(block)
        placements.append(meta)

    return {
        "content_html": "</p>".join(paragraphs) if blocks or affiliate_blocks else content_html,
        "placements": placements,
        "enabled": True,
    }
