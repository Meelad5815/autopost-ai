import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List


def _topic_from_prompt(prompt: str, default: str = "SEO automation") -> str:
    m = re.search(r"Topic:\s*(.+)", prompt)
    if not m:
        return default
    return m.group(1).strip()


def _int_from_prompt(prompt: str, pattern: str, default: int) -> int:
    m = re.search(pattern, prompt)
    if not m:
        return default
    try:
        return int(m.group(1))
    except Exception:
        return default


def _float_from_prompt(prompt: str, pattern: str, default: float) -> float:
    m = re.search(pattern, prompt)
    if not m:
        return default
    try:
        return float(m.group(1))
    except Exception:
        return default


def _keywords_from_calendar_prompt(prompt: str) -> List[str]:
    m = re.search(r"Use these niche keywords:\s*(.+)", prompt)
    if not m:
        return ["seo automation", "content strategy", "wordpress growth"]
    raw = [x.strip() for x in m.group(1).split(",")]
    return [x for x in raw if x][:12] or ["seo automation", "content strategy", "wordpress growth"]


def _gen_calendar(prompt: str) -> Dict[str, Any]:
    days = _int_from_prompt(prompt, r"Create a\s+(\d+)-day content plan", 30)
    keywords = _keywords_from_calendar_prompt(prompt)
    intents = ["informational", "commercial", "evergreen"]
    start = datetime.now(timezone.utc).date()
    items = []
    for i in range(max(1, days)):
        kw = keywords[i % len(keywords)]
        items.append(
            {
                "date_iso": datetime.combine(start + timedelta(days=i), datetime.min.time(), tzinfo=timezone.utc).isoformat(),
                "topic": f"{kw} playbook {start.year}",
                "intent_type": intents[i % len(intents)],
                "pillar_topic": kw,
                "cluster_topic": f"{kw} advanced tactics",
            }
        )
    return {"calendar": items}


def _gen_competitor_analysis(prompt: str) -> Dict[str, Any]:
    topic = _topic_from_prompt(prompt)
    competitors = []
    for i in range(1, 6):
        competitors.append(
            {
                "title": f"{topic} guide {i}",
                "angle": "beginner overview" if i % 2 else "tool comparison",
                "strengths": ["clear structure", "simple examples", "broad coverage"],
                "weaknesses": ["thin implementation detail", "limited metrics", "weak update cadence"],
            }
        )
    return {
        "competitors": competitors,
        "content_gaps": [
            "Implementation checklist with measurable KPIs",
            "Failure scenarios and recovery steps",
            "Cost and effort estimates by team size",
            "Maintenance schedule after initial rollout",
        ],
        "superior_outline": [
            f"Why {topic} matters now",
            "Execution blueprint",
            "Common mistakes and fixes",
            "30-day action plan",
            "FAQ and next steps",
        ],
    }


def _gen_serp_simulation(prompt: str) -> Dict[str, Any]:
    topic = _topic_from_prompt(prompt)
    profitability = _float_from_prompt(prompt, r"Niche profitability proxy:\s*([0-9.]+)", 50.0)
    difficulty = max(20.0, min(90.0, 75.0 - (profitability - 50.0) * 0.35))
    recommended = 1400 if difficulty < 55 else 1700 if difficulty < 75 else 2100
    return {
        "difficulty_score": round(difficulty, 2),
        "recommended_word_count": recommended,
        "depth_strategy": f"Prioritize practical examples, concrete steps, and unique insights for {topic}.",
    }


def _title_template(topic: str, intent: str, lang: str, year: int) -> str:
    t = topic.strip()
    if lang in {"ur", "urdu"}:
        return f"{t} ? ???? ??????? ({year})"
    if lang in {"roman", "roman-ur", "roman-urdu", "ur-roman"}:
        return f"{t} - Mukammal Rehnumai ({year})"
    if intent == "commercial":
        return f"{t}: Best Options, Pricing, and Reviews ({year})"
    if intent == "informational":
        return f"{t}: Complete Guide with Steps ({year})"
    return f"{t} Strategy Guide ({year})"


def _gen_article(prompt: str) -> Dict[str, Any]:
    topic = _topic_from_prompt(prompt)
    year = datetime.now(timezone.utc).year
    lang = os.getenv("LOCAL_AI_LANGUAGE", "en").strip().lower()
    author_name = os.getenv("AUTHOR_NAME", "Hafiz Muhammad Meelad Raza Attari").strip()
    intent = os.getenv("CONTENT_INTENT", "informational").strip().lower()

    title = _title_template(topic, intent, lang, year)

    if lang in {"ur", "urdu"}:
        meta = f"{topic} ?? ???? ??? ????? ??? ???? ???????? ?????? ??? ??? ?????? ?????"
        excerpt = f"{topic} ?? ????? ?? ?????? ????? ??? ??? ??????? ?? ???? ??????"
        faq_items = [
            {"question": f"{topic} ??? ????", "answer": f"{topic} ?? ???? ??? ?????? ????? ??? ?? ???? ?? ????? ??? ???? ???"},
            {"question": "??? ????? ??? ????", "answer": "??? ????? ??? ?????? ???????? ?? ????? ??? ??? ??? ???"},
            {"question": "??? ????? ???? ????? ????", "answer": "??????? ?? ?????? ??? ?? ????? ??? ??? ?????? ?? ????? ???? ??? ???"},
            {"question": "???? ??????? ???? ?? ???? ????", "answer": "???? ??????? ?? ??? ?????? ????? ????? ??? ?????? ????? ??? ???????"},
        ]
        sections = [
            ("?????", f"{topic} ?? ???? ??? ?????? ??????? ??? ?? ?????"),
            ("??? ????", "??? ?????? ????? ??? ?????? ???????? ?? ??????"),
            ("?????", "??? ?? ????? ??? ??????? ?? ???? ??????"),
            ("??? ????", "??? ?????? ??????? ?? ?????? ????? ??? (??? ?????? ???)?"),
            ("?????", "????? ????? ??? ????? ?? ??? ????? ?????"),
        ]
        author_label = "????"
        about_label = "???? ?? ???? ???"
    elif lang in {"roman", "roman-ur", "roman-urdu", "ur-roman"}:
        meta = f"{topic} ke bare mein mukhtasar aur jaami maloomat, taaruf, kaam aur numayan nuqaat."
        excerpt = f"{topic} ke hawale se bunyadi taaruf aur aham maloomat par mabni mazmoon."
        faq_items = [
            {"question": f"{topic} kaun hain?", "answer": f"{topic} ke bare mein bunyadi taaruf aur pas-e-manzar is mazmoon mein shamil hai."},
            {"question": "Aham khidmaat kya hain?", "answer": "Aham khidmaat aur numayan sargarmiyon ka khulasa pesh kiya gaya hai."},
            {"question": "Log inhein kyun jante hain?", "answer": "Maqbooliyat ke asbab, kaam ke asraat aur aham pehluon par roshni dali gayi hai."},
            {"question": "Mazeed maloomat kahan mil sakti hain?", "answer": "Mazeed tafseelaat ke liye mutalliqah mo'tabar zarai aur sarkari hawala jaat dekhein."},
        ]
        sections = [
            ("Taaruf", f"{topic} ke bare mein bunyadi maloomat aur pas-e-manzar."),
            ("Aham Pehlu", "Aham khidmaat, kirdar aur numayan sargarmiyon ka khulasa."),
            ("Asraat", "Kaam ke asraat aur muasharti ya fikri ahmiyat."),
            ("Aham Nuqaat", "Aham haqaiq, tareekhen ya numayan hawala jaat (agar dastiyab hon)."),
            ("Khulasa", "Mukhtasar nateeja aur aainda ke liye rehnuma nuqaat."),
        ]
        author_label = "Author"
        about_label = "About the Author"
    else:
        meta = f"Learn a practical {topic} framework with steps, examples, and KPIs you can apply immediately."
        excerpt = f"A complete and actionable guide to {topic} for teams that need measurable results."
        faq_items = [
            {"question": f"What is {topic}?", "answer": f"{topic} is a structured approach to improve outcomes through repeatable processes."},
            {"question": f"How long does {topic} take to show results?", "answer": "Most teams see early signals in 2-4 weeks and stronger gains in 6-12 weeks."},
            {"question": f"What tools are required for {topic}?", "answer": "Start with your CMS, analytics stack, and a clear workflow; add tools only when needed."},
            {"question": f"How do I avoid mistakes in {topic}?", "answer": "Use a checklist, review metrics weekly, and iterate based on real performance data."},
        ]
        sections = [
            ("Why this matters", f"{topic} works best when you focus on consistency, quality, and measurable impact."),
            ("Core framework", "Define goals, map tasks, assign owners, and set deadlines with feedback loops."),
            ("Execution checklist", "Research intent, build content, optimize structure, publish, and monitor outcomes."),
            ("Measurement and optimization", "Track impressions, CTR, conversion rate, and time-on-page to guide iteration."),
            ("Common mistakes", "Avoid over-automation, vague KPIs, and publishing without quality review."),
            ("30-day plan", "Week 1 plan, Week 2 publish, Week 3 optimize, Week 4 scale winners."),
        ]
        author_label = "Author"
        about_label = "About the Author"

    blocks = []
    blocks.append(f"<p><strong>{author_label}:</strong> {author_name}</p>")
    for h2, body in sections:
        blocks.append(f"<h2>{h2}</h2><p>{body}</p>")
        blocks.append(
            "<ul>"
            "<li>Define one KPI for this stage</li>"
            "<li>Assign a clear owner</li>"
            "<li>Set review cadence and success threshold</li>"
            "</ul>"
        )
        blocks.append(f"<h3>{h2} in practice</h3><p>Apply this step to your niche and refine after each publishing cycle.</p>")
    blocks.append("<h2>Related Reading</h2><p>[INTERNAL_LINK:related-article]</p>")
    blocks.append(f"<h2>Conclusion</h2><p>Use this guide to drive consistent, measurable progress.</p>")
    blocks.append(f"<section><h3>{about_label}</h3><p>{author_name}</p></section>")
    content_html = "".join(blocks)

    if lang in {"ur", "urdu"}:
        tags = ["?????", "?????", "???????", "????"]
        categories = ["?????", "???????"]
        related = [topic, f"{topic} ?????", f"{topic} ???????"]
    elif lang in {"roman", "roman-ur", "roman-urdu", "ur-roman"}:
        tags = ["taaruf", "sawaneh", "maloomat", "roman-urdu"]
        categories = ["halaat", "maloomat"]
        related = [topic, f"{topic} taaruf", f"{topic} maloomat"]
    else:
        tags = ["seo", "automation", "content strategy", "wordpress", "growth"]
        categories = ["Marketing", "Automation"]
        related = [topic, f"{topic} checklist", f"{topic} workflow", f"{topic} metrics"]

    return {
        "title": title,
        "meta_description": meta,
        "excerpt": excerpt,
        "content_html": content_html,
        "tags": tags,
        "categories": categories,
        "image_query": topic,
        "faq_items": faq_items,
        "related_keywords": related,
    }


def _gen_image_meta(prompt: str) -> Dict[str, Any]:
    t = re.search(r"title=(.*?);", prompt)
    i = re.search(r"image=(.*?)[.;]?$", prompt)
    title = t.group(1).strip() if t else "Post"
    image = i.group(1).strip() if i else "featured image"
    return {
        "alt_text": f"{title} - {image}"[:120],
        "caption": f"Illustration for {title}"[:180],
    }


def openai_json(api_key: str, model: str, prompt: str, timeout: int, temperature: float = 0.6) -> Dict[str, Any]:
    _ = (api_key, model, timeout, temperature)
    if "keys: competitors, content_gaps, superior_outline" in prompt:
        return _gen_competitor_analysis(prompt)
    if "keys: difficulty_score, recommended_word_count, depth_strategy" in prompt:
        return _gen_serp_simulation(prompt)
    if "key calendar" in prompt:
        return _gen_calendar(prompt)
    if "keys:\n- title\n- meta_description" in prompt:
        return _gen_article(prompt)
    if "alt_text, caption" in prompt:
        return _gen_image_meta(prompt)
    return {"result": "ok"}
