"""Zero-cost local content intelligence engine.

No paid API is required. The engine supports:
1) local rule-based generation (always available);
2) optional Ollama on the user's own machine (no per-request API billing).

The public function name openai_json is kept for backward compatibility with
the existing engine, but it no longer requires an OpenAI key.
"""
import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from urllib import request as urlrequest


def _topic_from_prompt(prompt: str, default: str = "SEO automation") -> str:
    m = re.search(r"Topic:\s*(.+)", prompt)
    return m.group(1).strip() if m else default


def _language(prompt: str) -> str:
    return os.getenv("LOCAL_AI_LANGUAGE", "en").strip().lower()


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
        return ["website development", "wordpress", "freelancing", "automation"]
    return [x.strip() for x in m.group(1).split(",") if x.strip()][:12]


def _clean_topic(topic: str) -> str:
    topic = re.sub(r"\s+", " ", topic).strip(" .,:;!?")
    return topic[:140] or "technology"


def _topic_parts(topic: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"[,|/:-]", topic) if p.strip()]
    return parts[:8] or [topic]


def _ollama_json(prompt: str, model: str, timeout: int) -> Dict[str, Any] | None:
    base = os.getenv("OLLAMA_URL", "").strip().rstrip("/")
    if not base:
        return None
    try:
        payload = json.dumps({
            "model": model or os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": float(os.getenv("CONTENT_TEMPERATURE", "0.6"))},
        }).encode("utf-8")
        req = urlrequest.Request(
            f"{base}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlrequest.urlopen(req, timeout=min(timeout, 90)) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data.get("response", "")
        return json.loads(text) if text else None
    except Exception:
        return None


def _gen_calendar(prompt: str) -> Dict[str, Any]:
    days = _int_from_prompt(prompt, r"Create a\s+(\d+)-day content plan", 30)
    keywords = _keywords_from_calendar_prompt(prompt)
    intents = ["informational", "commercial", "transactional", "evergreen"]
    start = datetime.now(timezone.utc).date()
    items = []
    for i in range(max(1, days)):
        kw = keywords[i % len(keywords)]
        intent = intents[i % len(intents)]
        suffix = {
            "informational": "complete guide",
            "commercial": "cost and options",
            "transactional": "services and how to choose",
            "evergreen": "practical checklist",
        }[intent]
        items.append({
            "date_iso": datetime.combine(start + timedelta(days=i), datetime.min.time(), tzinfo=timezone.utc).isoformat(),
            "topic": f"{kw} {suffix} {start.year}",
            "intent_type": intent,
            "pillar_topic": kw,
            "cluster_topic": f"{kw} practical tips",
        })
    return {"calendar": items}


def _gen_competitor_analysis(prompt: str) -> Dict[str, Any]:
    topic = _clean_topic(_topic_from_prompt(prompt))
    return {
        "competitors": [
            {"title": f"{topic} beginner guide", "angle": "beginner", "strengths": ["clear structure", "broad coverage"], "weaknesses": ["limited local detail"]},
            {"title": f"{topic} cost and options", "angle": "commercial", "strengths": ["buyer intent"], "weaknesses": ["limited implementation detail"]},
            {"title": f"{topic} practical tutorial", "angle": "how-to", "strengths": ["steps"], "weaknesses": ["limited troubleshooting"]},
        ],
        "content_gaps": [
            "Pakistan/local context",
            "step-by-step implementation",
            "realistic cost and effort ranges",
            "common failures and recovery",
            "checklist and next actions",
        ],
        "superior_outline": [
            f"What {topic} is",
            "Who needs it and why",
            "Step-by-step implementation",
            "Cost, tools and alternatives",
            "Common mistakes",
            "Practical checklist",
            "FAQ and next steps",
        ],
    }


def _gen_serp_simulation(prompt: str) -> Dict[str, Any]:
    topic = _clean_topic(_topic_from_prompt(prompt))
    profitability = _float_from_prompt(prompt, r"Niche profitability proxy:\s*([0-9.]+)", 50.0)
    difficulty = max(25.0, min(85.0, 68.0 - (profitability - 50.0) * 0.25))
    recommended = 1200 if difficulty < 45 else 1500 if difficulty < 65 else 1800
    return {
        "difficulty_score": round(difficulty, 2),
        "recommended_word_count": recommended,
        "depth_strategy": f"Cover {topic} with local examples, implementation steps, alternatives, risks and measurable checks. This is a local heuristic, not live SERP data.",
    }


def _title(topic: str, intent: str, lang: str, year: int) -> str:
    if lang in {"ur", "urdu"}:
        return f"{topic} — مکمل رہنمائی، طریقہ اور اہم معلومات ({year})"
    if lang in {"roman", "roman-ur", "roman-urdu", "ur-roman"}:
        return f"{topic} — Mukammal Rehnumai aur Practical Guide ({year})"
    if intent == "commercial":
        return f"{topic}: Cost, Options and How to Choose ({year})"
    if intent == "transactional":
        return f"{topic}: Services, Pricing and Next Steps ({year})"
    return f"{topic}: Complete Practical Guide ({year})"


def _local_article(prompt: str) -> Dict[str, Any]:
    topic = _clean_topic(_topic_from_prompt(prompt))
    lang = _language(prompt)
    year = datetime.now(timezone.utc).year
    author = os.getenv("AUTHOR_NAME", "Hafiz Muhammad Meelad Raza Attari").strip()
    intent = os.getenv("CONTENT_INTENT", "informational").strip().lower()
    parts = _topic_parts(topic)
    focus = parts[0]

    if lang in {"ur", "urdu"}:
        title = _title(topic, intent, lang, year)
        meta = f"{topic} کے بارے میں عملی رہنمائی، اہم نکات، طریقہ کار، احتیاطیں اور اگلے اقدامات۔"
        intro = f"یہ مضمون {topic} کو آسان انداز میں سمجھنے اور عملی طور پر استعمال کرنے کے لیے تیار کیا گیا ہے۔ مقصد صرف عمومی معلومات دینا نہیں بلکہ واضح steps اور قابلِ عمل checklist فراہم کرنا ہے۔"
        sections = [
            ("یہ کیا ہے اور کیوں اہم ہے؟", f"{topic} کو سمجھنے کے لیے پہلے اس کے مقصد، بنیادی اجزا اور استعمال کی صورتِ حال واضح کرنا ضروری ہے۔ {focus} کے معاملے میں درست planning وقت اور لاگت دونوں کم کر سکتی ہے۔"),
            ("شروع کرنے کا طریقہ", "پہلے اپنا مقصد طے کریں، پھر ضروری tools اور معلومات جمع کریں۔ کام کو چھوٹے مراحل میں تقسیم کریں اور ہر مرحلے کے بعد نتیجہ چیک کریں۔"),
            ("عملی مراحل", "ضرورت کے مطابق setup، testing، implementation اور monitoring کریں۔ ہر اہم تبدیلی سے پہلے backup رکھیں تاکہ مسئلہ آنے پر واپس جانا آسان ہو۔"),
            ("عام مسائل اور حل", "جلدی میں configuration بدلنے، غیر ضروری tools استعمال کرنے اور بغیر testing کے publish کرنے سے مسائل پیدا ہو سکتے ہیں۔ پہلے چھوٹے test سے آغاز کریں۔"),
            ("لاگت اور متبادل", "ہر project میں لاگت مختلف ہو سکتی ہے۔ مفت tools سے prototype بنانا ممکن ہے، جبکہ advanced features کے لیے اضافی infrastructure درکار ہو سکتا ہے۔"),
            ("پاکستان کے لیے عملی نکات", "مقامی internet، payment، availability، language اور user needs کو شروع ہی میں شامل کریں۔ مقامی audience کے لیے واضح Urdu/Roman Urdu explanations conversion بہتر بنانے میں مدد دے سکتی ہیں۔"),
            ("Checklist", "مقصد واضح کریں، required data جمع کریں، test کریں، publish کریں، performance نوٹ کریں اور بہتر نتائج والے topics کو دوبارہ expand کریں۔"),
        ]
        faq = [
            {"question": f"{topic} کیا ہے؟", "answer": f"{topic} ایک موضوع/حل ہے جسے اس مضمون میں بنیادی تصور سے عملی استعمال تک سمجھایا گیا ہے۔"},
            {"question": "شروع کرنے کے لیے کیا چاہیے؟", "answer": "واضح مقصد، بنیادی معلومات، ضروری tools اور ایک چھوٹا test setup کافی ابتدائی نقطہ ہے۔"},
            {"question": "کیا مفت طریقہ ممکن ہے؟", "answer": "بنیادی prototype اور automation کے لیے مفت یا local tools استعمال کیے جا سکتے ہیں، مگر ہر feature کی requirements مختلف ہوتی ہیں۔"},
            {"question": "بہتری کیسے ناپیں؟", "answer": "Publish count کے بجائے quality، traffic، clicks، leads، conversions اور errors کو باقاعدگی سے track کریں۔"},
        ]
        tags = [focus, "رہنمائی", "پاکستان", "عملی طریقہ"]
        categories = ["رہنمائی", "ٹیکنالوجی"]
    elif lang in {"roman", "roman-ur", "roman-urdu", "ur-roman"}:
        title = _title(topic, intent, lang, year)
        meta = f"{topic} ke bare mein practical guide, steps, cost, common problems aur Pakistan context."
        intro = f"Yeh article {topic} ko asaan alfaaz mein samjhata hai aur practical steps, testing aur next actions deta hai."
        sections = [
            ("Yeh kya hai aur kyun zaroori hai?", f"{topic} ko samajhne ke liye iska maqsad, basic parts aur real use-case samajhna zaroori hai. {focus} mein planning se waqt aur cost dono control ho sakte hain."),
            ("Shuru karne ka tareeqa", "Goal define karein, zaroori information collect karein aur kaam ko chhote steps mein divide karein. Har step ke baad result test karein."),
            ("Practical implementation", "Setup, testing, implementation aur monitoring ko alag stages mein rakhein. Important changes se pehle backup zaroor rakhein."),
            ("Common problems", "Bina testing ke publish karna, unnecessary tools aur unclear configuration common issues hain. Pehle small test run karein."),
            ("Cost aur free alternatives", "Basic prototype local aur free tools se ban sakta hai. Advanced features ke liye extra infrastructure ki zaroorat ho sakti hai."),
            ("Pakistan context", "Local internet, availability, language, payments aur audience needs ko workflow mein include karein."),
            ("Action checklist", "Goal, data, test, publish, measure aur improve — is cycle ko repeat karein."),
        ]
        faq = [
            {"question": f"{topic} kya hai?", "answer": f"Is article mein {topic} ka basic concept aur practical use explain kiya gaya hai."},
            {"question": "Start karne ke liye kya chahiye?", "answer": "Clear goal, basic information, required tools aur ek small test setup."},
            {"question": "Kya free method mumkin hai?", "answer": "Basic automation ke liye free/local tools use kiye ja sakte hain."},
            {"question": "Result kaise measure karein?", "answer": "Quality, traffic, clicks, leads, conversions aur errors ko track karein."},
        ]
        tags = [focus, "guide", "Pakistan", "practical"]
        categories = ["Guides", "Technology"]
    else:
        title = _title(topic, intent, lang, year)
        meta = f"A practical guide to {topic} with implementation steps, costs, common problems and measurable next actions."
        intro = f"This guide explains {topic} in a practical way. It focuses on decisions, implementation, testing and measurable outcomes rather than generic filler."
        sections = [
            ("What it is and why it matters", f"Start by defining what {topic} means in your specific use case. A clear scope prevents unnecessary tools, cost and maintenance."),
            ("How to start", "Define one outcome, collect the required information, choose the smallest workable setup, and test it before scaling."),
            ("Implementation workflow", "Move through setup, validation, publishing, monitoring and iteration as separate stages. Keep a backup before important changes."),
            ("Common problems and fixes", "Typical failures come from unclear configuration, weak testing, duplicate work and publishing without review. Use logs and small test runs."),
            ("Cost and free alternatives", "A useful prototype can often be built with open-source software and local processing. Paid services are optional, not required for the core workflow."),
            ("Pakistan/local context", "Consider local availability, internet reliability, language, customer behavior and practical pricing when choosing the workflow."),
            ("Measurement and improvement", "Track quality, traffic, clicks, leads, conversions and failures. Expand topics that show real user value rather than simply increasing post volume."),
            ("30-day action plan", "Week 1: setup and test. Week 2: publish a small set. Week 3: review performance and errors. Week 4: improve winners and remove weak workflows."),
        ]
        faq = [
            {"question": f"What is {topic}?", "answer": f"{topic} is explained here as a practical workflow with clear steps and measurable outcomes."},
            {"question": "Can I start without a paid API?", "answer": "Yes. The core project can use local generation and open-source tooling; paid APIs are optional upgrades."},
            {"question": "What should I measure?", "answer": "Measure quality, traffic, clicks, leads, conversions, failures and publishing consistency."},
            {"question": "How should I scale?", "answer": "Scale only after a small test is stable. Increase topic coverage gradually and keep a quality gate."},
        ]
        tags = [focus, "automation", "guide", "Pakistan"]
        categories = ["Guides", "Technology"]

    blocks = [f"<p><strong>{'مصنف' if lang in {'ur','urdu'} else 'Author'}:</strong> {author}</p>", f"<p>{intro}</p>"]
    for heading, body in sections:
        blocks.append(f"<h2>{heading}</h2><p>{body}</p>")
        blocks.append("<ul><li>ایک واضح مقصد یا KPI مقرر کریں</li><li>چھوٹا test run کریں</li><li>نتیجہ record کرکے اگلا قدم طے کریں</li></ul>" if lang in {"ur","urdu"} else "<ul><li>Define one clear goal or KPI</li><li>Run a small test first</li><li>Record the result and improve the next cycle</li></ul>")
    blocks.append("<h2>Frequently Asked Questions</h2>")
    for item in faq:
        blocks.append(f"<h3>{item['question']}</h3><p>{item['answer']}</p>")
    blocks.append(f"<h2>{'نتیجہ' if lang in {'ur','urdu'} else 'Conclusion'}</h2><p>{'اس workflow کو چھوٹے test سے شروع کریں، نتائج record کریں اور بہتر حصوں کو آہستہ آہستہ scale کریں۔' if lang in {'ur','urdu'} else 'Start with a small test, record real results, and scale only the parts that consistently create value.'}</p>")
    blocks.append(f"<section><h3>{'مصنف کے بارے میں' if lang in {'ur','urdu'} else 'About the Author'}</h3><p>{author}</p></section>")

    return {
        "title": title,
        "meta_description": meta[:155],
        "excerpt": intro[:220],
        "content_html": "".join(blocks),
        "tags": tags,
        "categories": categories,
        "image_query": topic,
        "faq_items": faq,
        "related_keywords": [topic, f"{topic} guide", f"{topic} Pakistan"],
    }


def _gen_image_meta(prompt: str) -> Dict[str, Any]:
    t = re.search(r"title=(.*?);", prompt)
    i = re.search(r"image=(.*?)[.;]?$", prompt)
    title = t.group(1).strip() if t else "Post"
    image = i.group(1).strip() if i else "featured image"
    return {"alt_text": f"{title} - {image}"[:120], "caption": f"Illustration for {title}"[:180]}


def _gen_synthesis_brief(prompt: str) -> Dict[str, Any]:
    topic = _clean_topic(_topic_from_prompt(prompt, "technology update"))
    return {
        "core_news": f"{topic} should be evaluated using practical requirements, compatibility, reliability and local context.",
        "technical_specs": ["Check compatibility and performance before rollout.", "Use a small test before scaling.", "Keep security and reliability as explicit checks."],
        "pakistan_context": ["Consider local availability and pricing.", "Consider language and connectivity.", "Document practical SME/freelancer use cases."],
        "expert_verdict": "Use a phased rollout with explicit KPI tracking.",
        "keyword_hints": [topic, f"{topic} Pakistan", f"{topic} practical guide"],
        "contradictory_claims": [],
        "conflict_label": "local-engine-no-live-source-verification",
        "fact_check_status": "needs-source-review",
    }


def openai_json(api_key: str, model: str, prompt: str, timeout: int, temperature: float = 0.6) -> Dict[str, Any]:
    """Backward-compatible entry point.

    If AI_PROVIDER=ollama, try a self-hosted Ollama model first. If unavailable,
    always fall back to the local zero-cost engine. No paid API is required.
    """
    provider = os.getenv("AI_PROVIDER", "local").strip().lower()
    if provider == "ollama":
        remote = _ollama_json(prompt, model or os.getenv("OLLAMA_MODEL", "llama3.2:3b"), timeout)
        if remote:
            return remote

    if "keys: competitors, content_gaps, superior_outline" in prompt:
        return _gen_competitor_analysis(prompt)
    if "keys: difficulty_score, recommended_word_count, depth_strategy" in prompt:
        return _gen_serp_simulation(prompt)
    if "key calendar" in prompt:
        return _gen_calendar(prompt)
    if ("keys:\n- title\n- meta_description" in prompt) or ("Return strict JSON with keys:" in prompt and "- meta_description" in prompt):
        return _local_article(prompt)
    if "Return strict JSON with keys:\n- core_news\n- technical_specs\n- pakistan_context\n- expert_verdict\n- keyword_hints\n- contradictory_claims\n- conflict_label\n- fact_check_status" in prompt:
        return _gen_synthesis_brief(prompt)
    if "alt_text, caption" in prompt:
        return _gen_image_meta(prompt)
    return {"result": "ok"}
