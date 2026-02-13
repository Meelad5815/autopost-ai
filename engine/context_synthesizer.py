import re
from typing import Any, Dict, List

import requests

from engine.ai import openai_json


def _clean_html(raw: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", raw, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def fetch_sources(urls: List[str], timeout: int, max_chars: int = 5000) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for url in urls:
        u = (url or "").strip()
        if not u:
            continue
        try:
            resp = requests.get(u, timeout=timeout)
            if resp.status_code != 200:
                continue
            text = _clean_html(resp.text)[:max_chars]
            if text:
                out.append({"url": u, "text": text})
        except requests.RequestException:
            continue
    return out


def build_synthesis_brief(
    api_key: str,
    model: str,
    timeout: int,
    topic: str,
    language: str,
    source_payloads: List[Dict[str, str]],
) -> Dict[str, Any]:
    if not source_payloads:
        return {}

    snippets = []
    for idx, src in enumerate(source_payloads[:3], start=1):
        snippets.append(f"Source {idx} ({src.get('url', '')}): {src.get('text', '')}")
    joined = "\n".join(snippets)

    prompt = f"""
As an Advanced AI Tech Analyst, synthesize raw data from multiple sources into a structured brief.

Topic: {topic}
Language: {language}
Inputs:
{joined}

Return strict JSON with keys:
- core_news
- technical_specs
- pakistan_context
- expert_verdict
- keyword_hints
- contradictory_claims
- conflict_label
- fact_check_status

Rules:
- Do not copy source sentence structure.
- Preserve important facts and specs.
- Add Pakistan/local market framing where relevant.
- If sources disagree on a key claim, mark conflict_label as "controversial".
""".strip()

    data = openai_json(api_key, model, prompt, timeout, temperature=0.4)
    if not isinstance(data, dict):
        return {}
    return {
        "core_news": str(data.get("core_news", "")),
        "technical_specs": data.get("technical_specs", []) if isinstance(data.get("technical_specs"), list) else [],
        "pakistan_context": data.get("pakistan_context", []) if isinstance(data.get("pakistan_context"), list) else [],
        "expert_verdict": str(data.get("expert_verdict", "")),
        "keyword_hints": data.get("keyword_hints", []) if isinstance(data.get("keyword_hints"), list) else [],
        "contradictory_claims": data.get("contradictory_claims", []) if isinstance(data.get("contradictory_claims"), list) else [],
        "conflict_label": str(data.get("conflict_label", "clear")).strip().lower() or "clear",
        "fact_check_status": str(data.get("fact_check_status", "triangulated")).strip().lower() or "triangulated",
    }
