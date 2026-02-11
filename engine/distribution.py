from typing import Dict


def social_captions(title: str, url: str, excerpt: str) -> Dict[str, str]:
    short_excerpt = (excerpt or "").strip()[:180]
    return {
        "x": f"{title}\n\n{short_excerpt}\n{url} #SEO #ContentMarketing",
        "linkedin": f"{title}\n\n{short_excerpt}\nRead here: {url}",
        "facebook": f"{title}\n{short_excerpt}\n{url}",
    }


def create_web_story_snippet(title: str, content_html: str) -> Dict[str, str]:
    import re

    text = re.sub(r"<[^>]+>", " ", content_html)
    chunks = [c.strip() for c in text.split(".") if c.strip()]
    slides = chunks[:5]
    return {
        "story_title": f"{title} (Quick Story)",
        "slides": " | ".join(slides),
    }
