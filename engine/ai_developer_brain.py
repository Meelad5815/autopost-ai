"""Optional AI developer proposal engine.

Proposal-only by design. It never writes generated code directly to the repository.
"""
from __future__ import annotations
import json
import os
import requests

def enabled() -> bool:
    return bool(os.getenv("AI_DEV_ENDPOINT") and os.getenv("AI_DEV_MODEL"))

def propose(prompt: str) -> dict:
    if not enabled():
        return {"mode": "audit-only", "proposal": None,
                "reason": "AI developer endpoint is not configured"}
    endpoint = os.environ["AI_DEV_ENDPOINT"].rstrip("/")
    payload = {
        "model": os.environ["AI_DEV_MODEL"],
        "messages": [
            {"role": "system", "content":
             "You are a cautious software maintenance planner. Return JSON only "
             "with summary, risks, files, and proposed_patch. Never expose or "
             "request secrets. Do not propose payments or destructive production changes."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }
    response = requests.post(endpoint + "/v1/chat/completions",
                             json=payload, timeout=90)
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    try:
        proposal = json.loads(content)
    except json.JSONDecodeError:
        proposal = {"raw": content}
    return {"mode": "ai-proposal", "model": os.environ["AI_DEV_MODEL"],
            "proposal": proposal}
