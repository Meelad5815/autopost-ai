"""Revenue and conversion blocks for WordPress content.

Deterministic, configuration-driven CTAs. No fake earnings or fabricated metrics.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List


MONETIZATION_FILE = Path("config/monetization.json")


def load_monetization() -> Dict[str, Any]:
    if not MONETIZATION_FILE.exists():
        return {}
    try:
        return json.loads(MONETIZATION_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _service_cta(topic: str, language: str) -> str:
    url = os.getenv("SERVICE_CONTACT_URL", "").strip()
    if not url:
        return ""
    if language.startswith("ur"):
        text = "اگر آپ کو یہ کام خود کرنے کے بجائے پروفیشنل مدد چاہیے تو MRK Digital سے رابطہ کریں۔"
        button = "سروس کے لیے رابطہ کریں"
    elif language.startswith("roman"):
        text = "Agar aap ko ye kaam khud karne ke bajaye professional help chahiye to MRK Digital se rabta karein."
        button = "Service ke liye rabta karein"
    else:
        text = "Need professional help with this task? Contact MRK Digital for a practical service solution."
        button = "Request a service"
    return f'<div class="mrk-cta mrk-service-cta"><p><strong>{text}</strong></p><p><a href="{url}" rel="nofollow">{button}</a></p></div>'


def _affiliate_ctas(topic: str, language: str) -> List[str]:
    data = load_monetization()
    items = data.get("affiliate", [])
    if not isinstance(items, list):
        return []
    out = []
    for item in items[:2]:
        url = str(item.get("url", "")).strip()
        label = str(item.get("label", "")).strip()
        if not url or not label:
            continue
        if language.startswith("ur"):
            prefix = "متعلقہ پروڈکٹ/ٹول دیکھیں:"
        elif language.startswith("roman"):
            prefix = "Related product/tool dekhein:"
        else:
            prefix = "Related product/tool:"
        out.append(f'<div class="mrk-cta mrk-affiliate-cta"><p>{prefix} <a href="{url}" rel="sponsored nofollow">{label}</a></p></div>')
    return out


def inject_conversion_blocks(content_html: str, topic: str, language: str = "en") -> Dict[str, Any]:
    """Insert limited CTAs into new content and return auditable placement metadata."""
    if os.getenv("ENABLE_CONVERSION_OPTIMIZATION", "true").lower() != "true":
        return {"content_html": content_html, "placements": [], "enabled": False}

    blocks = []
    service = _service_cta(topic, language)
    if service:
        blocks.append(("service", service))

    affiliate_blocks = _affiliate_ctas(topic, language)
    for block in affiliate_blocks:
        blocks.append(("affiliate", block))

    if not blocks:
        return {"content_html": content_html, "placements": [], "enabled": True}

    paragraphs = content_html.split("</p>")
    placements = []
    # One service CTA after the first substantial paragraph.
    if service:
        for i, part in enumerate(paragraphs):
            if len(part.strip()) > 180:
                paragraphs.insert(i + 1, service)
                placements.append("service_mid")
                break

    # Affiliate CTA(s) at the end, limited to two configured entries.
    for block in affiliate_blocks:
        paragraphs.append(block)
        placements.append("affiliate_end")

    return {
        "content_html": "</p>".join(paragraphs),
        "placements": placements,
        "enabled": True,
    }
