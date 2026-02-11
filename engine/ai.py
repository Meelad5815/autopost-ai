import json
from typing import Any, Dict

import requests


def openai_json(api_key: str, model: str, prompt: str, timeout: int, temperature: float = 0.6) -> Dict[str, Any]:
    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "input": prompt, "temperature": temperature},
        timeout=timeout,
    )
    response.raise_for_status()
    text = response.json().get("output_text", "").strip()
    if not text:
        raise RuntimeError("OpenAI response empty")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"OpenAI invalid JSON: {text[:500]}") from exc
