import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Tuple

HISTORY_PATH = Path("history.json")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9\s-]", "", text.lower())).strip()


def word_count(text: str) -> int:
    plain = re.sub(r"<[^>]+>", " ", text)
    return len(re.findall(r"\b\w+\b", plain))


def build_meta_description(meta: str, excerpt: str, max_len: int = 160) -> str:
    base = (meta or "").strip() or (excerpt or "").strip()
    if not base:
        return ""
    return base[:max_len].rstrip()


def focus_keyword(topic: str) -> str:
    words = [w for w in re.findall(r"[a-zA-Z0-9]+", topic) if w]
    return " ".join(words[:6]).strip() or topic[:60]


def ensure_author_signature(content_html: str, author_name: str) -> str:
    start_block = f"<p><strong>Author:</strong> {author_name}</p>"
    end_block = f"<section><h3>About the Author</h3><p>{author_name}</p></section>"
    if start_block not in content_html:
        content_html = start_block + content_html
    if end_block not in content_html:
        content_html = content_html + end_block
    return content_html


def must_have_author(content_html: str, author_name: str) -> None:
    if f"Author:</strong> {author_name}" not in content_html:
        raise RuntimeError("Author signature missing at top")
    if f"About the Author</h3><p>{author_name}</p>" not in content_html:
        raise RuntimeError("Author signature missing at bottom")


def load_history() -> Dict[str, Any]:
    if not HISTORY_PATH.exists():
        return {"titles": [], "topics": [], "topic_cursor": 0, "last_date": "", "articles": []}
    return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))


def save_history(history: Dict[str, Any]) -> None:
    HISTORY_PATH.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def record_history(
    history: Dict[str, Any],
    title: str,
    topic: str,
    language: str,
    url: str,
    intent: str = "informational",
    pillar: str = "",
    cluster: str = "",
    category: str = "",
    word_count_val: int = 0,
    status: str = "fresh",
    published_at: str = "",
) -> None:
    history.setdefault("titles", []).append(
        {"title": title, "topic": topic, "language": language, "url": url, "intent": intent, "pillar": pillar, "cluster": cluster}
    )
    history.setdefault("topics", []).append({"topic": topic, "language": language, "intent": intent})
    history.setdefault("articles", []).append(
        {
            "title": title,
            "topic": topic,
            "language": language,
            "url": url,
            "intent": intent,
            "pillar": pillar,
            "cluster": cluster,
            "category": category,
            "word_count": word_count_val,
            "status": status,
            "published_at": published_at,
        }
    )
    history["titles"] = history["titles"][-800:]
    history["topics"] = history["topics"][-800:]
    history["articles"] = history["articles"][-800:]


def intent_key(topic: str, intent: str) -> str:
    base = _normalize(topic)
    return f"{base}:{intent}"


def is_cannibalization(topic: str, intent: str, history: Dict[str, Any]) -> bool:
    key = intent_key(topic, intent)
    for item in history.get("topics", []):
        if intent_key(str(item.get("topic", "")), str(item.get("intent", "informational"))) == key:
            return True
    return False


def is_duplicate_title(title: str, existing_titles: List[str], history: Dict[str, Any]) -> bool:
    norm = _normalize(title)
    for t in existing_titles:
        if _normalize(t) == norm:
            return True
        if SequenceMatcher(a=norm, b=_normalize(t)).ratio() >= 0.9:
            return True
    for item in history.get("titles", []):
        old = _normalize(str(item.get("title", "")))
        if old == norm:
            return True
        if SequenceMatcher(a=norm, b=old).ratio() >= 0.9:
            return True
    return False


def select_internal_links(content_html: str, existing_posts: List[Dict[str, Any]], max_links: int = 2) -> List[Tuple[str, str]]:
    body = re.sub(r"<[^>]+>", " ", content_html).lower()
    scored = []
    for post in existing_posts:
        title = re.sub(r"<[^>]+>", "", post.get("title", {}).get("rendered", "")).strip()
        link = post.get("link", "").strip()
        if not title or not link:
            continue
        words = [w for w in re.findall(r"[a-zA-Z0-9]+", title.lower()) if len(w) > 3]
        overlap = sum(1 for w in set(words) if w in body)
        if overlap:
            scored.append((overlap, title, link))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(t, u) for _, t, u in scored[:max_links]]


def insert_internal_links(content_html: str, links: List[Tuple[str, str]]) -> str:
    if not links:
        return content_html
    items = "".join([f'<li><a href="{url}">{title}</a></li>' for title, url in links])
    block = f"<h2>Related Reading</h2><ul>{items}</ul>"
    if "[INTERNAL_LINK:related-article]" in content_html:
        return content_html.replace("[INTERNAL_LINK:related-article]", block)
    return content_html + block


def infer_categories_tags(topic: str) -> Dict[str, List[str]]:
    topic_l = topic.lower()
    mapping = {
        "web": ("Web Development", ["web", "development", "frontend", "backend"]),
        "wordpress": ("WordPress", ["wordpress", "cms", "themes", "plugins"]),
        "seo": ("SEO", ["seo", "search", "ranking"]),
        "freelanc": ("Freelancing", ["freelancing", "clients", "upwork", "fiverr"]),
        "ui": ("UI/UX", ["ui", "ux", "design", "product design"]),
        "islamic": ("Islamic Insights", ["islamic", "ethics", "values"]),
        "tech": ("Tech News", ["tech", "news", "updates"]),
    }
    categories = []
    tags = []
    for key, (cat, tgs) in mapping.items():
        if key in topic_l:
            categories.append(cat)
            tags.extend(tgs)
    if not categories:
        categories.append("General")
    if not tags:
        tags = re.findall(r"[a-zA-Z0-9]+", topic_l)[:6]
    return {"categories": list(dict.fromkeys(categories)), "tags": list(dict.fromkeys(tags))}


def build_content_brief(topic: str, intent: str, related_questions: List[str], entities: List[str], links: List[Tuple[str, str]]) -> Dict[str, Any]:
    return {
        "target_keyword": topic,
        "search_intent": intent,
        "related_questions": related_questions[:6],
        "entities": entities[:12],
        "internal_links": [{"title": t, "url": u} for t, u in links],
    }


def schema_suggestions(topic: str, has_faq: bool = True) -> List[str]:
    items = ["Article"]
    if has_faq:
        items.append("FAQPage")
    if "review" in topic.lower() or "best" in topic.lower():
        items.append("ItemList")
    return items
