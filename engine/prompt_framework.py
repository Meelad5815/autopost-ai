import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List


UNIQUE_ANGLES: List[str] = [
    "how this affects Pakistan's local market and digital businesses",
    "how developers and technical teams can apply this in production",
    "how founders and freelancers can turn this into measurable growth",
    "what this changes for education, skills, and jobs in South Asia",
    "the operational and cost impact for small and mid-sized teams",
]

TONE_PROFILES: List[str] = [
    "newsroom-analytical",
    "expert-explainer",
    "practical-playbook",
    "evidence-led-commentary",
    "technical-briefing",
]

OPENING_STYLES: List[str] = [
    "start with a punchy headline-style opening",
    "open with a specific real-world scenario",
    "open with a short high-stakes problem statement",
    "open with a trend shift and why it matters right now",
]


def _stable_index(topic: str, salt: str, size: int) -> int:
    key = f"{topic}|{salt}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return int(digest[:8], 16) % size


def select_generation_profile(topic: str, language: str) -> Dict[str, str]:
    date_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lang_key = language.lower().strip()
    angle = UNIQUE_ANGLES[_stable_index(topic, f"{date_key}:{lang_key}:angle", len(UNIQUE_ANGLES))]
    tone = TONE_PROFILES[_stable_index(topic, f"{date_key}:{lang_key}:tone", len(TONE_PROFILES))]
    opening = OPENING_STYLES[_stable_index(topic, f"{date_key}:{lang_key}:opening", len(OPENING_STYLES))]
    return {"angle": angle, "tone": tone, "opening_style": opening}


def build_article_prompt(
    topic: str,
    target_words: int,
    gaps: List[str],
    outline: List[str],
    refresh_context: str,
    language: str,
    profile: Dict[str, str],
    synthesis_brief: Dict[str, Any] | None = None,
) -> str:
    brief = synthesis_brief or {}
    return f"""
Role: You are a Senior Tech Journalist and Localization Expert specializing in 2026 technology trends.
Task: Create a highly original, expert-level article in language={language}.

Strict anti-duplicate rules:
1) Zero plagiarism: never translate source text word-for-word. Re-synthesize in your own structure.
2) Unique angle: prioritize this perspective -> {profile["angle"]}.
3) Semantic variation: avoid repetitive sentence structures and repeated phrase templates.
4) Structural depth: include Introduction, Why It Matters, Technical Insights, Execution Steps, and Conclusion.
5) Anti-AI pattern: do not start with generic filler lines. {profile["opening_style"]}.

Voice and format:
- Tone profile: {profile["tone"]}
- human, authoritative, concrete
- include H2/H3, bullet lists, practical examples, FAQ section, CTA
- include placeholder [INTERNAL_LINK:related-article]
- no markdown/code fences

Input topic: {topic}
Target word count: at least {max(1200, target_words)} words
Competitor gaps to exploit: {gaps}
Better outline to follow: {outline}
Refresh context (if any): {refresh_context}
Triangulated core news: {brief.get("core_news", "")}
Technical facts/specs to preserve: {brief.get("technical_specs", [])}
Pakistan-specific impact notes: {brief.get("pakistan_context", [])}
Expert verdict hints: {brief.get("expert_verdict", "")}
SEO keyword hints: {brief.get("keyword_hints", [])}
Contradictory claims to address explicitly: {brief.get("contradictory_claims", [])}
Fact-check status: {brief.get("fact_check_status", "triangulated")}
Conflict label: {brief.get("conflict_label", "clear")}

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
""".strip()
