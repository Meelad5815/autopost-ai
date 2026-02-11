import os
import random
from typing import Dict


def insert_affiliate_links(content_html: str, topic: str) -> str:
    aff_url = os.getenv("AFFILIATE_URL", "").strip()
    if not aff_url:
        return content_html
    anchor = os.getenv("AFFILIATE_ANCHOR", f"best {topic} tools").strip()
    block = f'<p><strong>Partner recommendation:</strong> <a href="{aff_url}" rel="sponsored nofollow">{anchor}</a>.</p>'
    return content_html + block


def insert_lead_gen_block(content_html: str) -> str:
    lead_magnet = os.getenv("LEAD_MAGNET_TITLE", "Free SEO Growth Checklist")
    email_url = os.getenv("EMAIL_OPTIN_URL", "#")
    block = (
        f"<section><h2>{lead_magnet}</h2>"
        f"<p>Get the actionable checklist and templates directly in your inbox.</p>"
        f"<p><a href=\"{email_url}\">Download the lead magnet</a></p></section>"
    )
    return content_html + block


def cta_ab_variant() -> Dict[str, str]:
    variants = [
        {"id": "A", "cta": "Start with one quick win today and measure results in 7 days."},
        {"id": "B", "cta": "Choose one workflow, implement it this week, and track impact."},
    ]
    return random.choice(variants)


def engagement_score(content_html: str) -> float:
    score = 0.0
    score += 25 if "<h2>" in content_html else 0
    score += 15 if "<ul>" in content_html else 0
    score += min(30, content_html.count("<a ") * 5)
    score += 20 if "Frequently Asked Questions" in content_html else 0
    score += 10 if "Take the Next Step" in content_html else 0
    return round(min(100.0, score), 2)
