"""Public research hub for topics, tools and open-source projects.

This module only collects publicly accessible metadata and stores source URLs.
It does not bypass authentication, paywalls, robots restrictions, or rate limits.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen


DEFAULT_QUERIES = [
    "wordpress website development tools",
    "web development automation tools",
    "SEO automation open source",
    "social media automation open source",
    "AI coding agents open source",
    "website analytics open source",
    "Pakistan freelancing web development",
]


def _get_json(url: str, timeout: int = 15) -> Any:
    req = Request(url, headers={"User-Agent": "MRK-AutoPost-Research/1.0"})
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def github_repositories(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search public GitHub repositories using GitHub's public API."""
    url = (
        "https://api.github.com/search/repositories?q="
        + quote_plus(query)
        + "&sort=stars&order=desc&per_page="
        + str(max(1, min(limit, 30)))
    )
    try:
        data = _get_json(url)
        results = []
        for item in data.get("items", []):
            results.append({
                "source": "github",
                "name": item.get("full_name"),
                "description": item.get("description") or "",
                "stars": item.get("stargazers_count", 0),
                "language": item.get("language"),
                "license": (item.get("license") or {}).get("spdx_id"),
                "url": item.get("html_url"),
                "updated_at": item.get("updated_at"),
            })
        return results
    except Exception:
        return []


def build_research_index(queries: list[str] | None = None) -> dict[str, Any]:
    queries = queries or DEFAULT_QUERIES
    projects: list[dict[str, Any]] = []
    seen: set[str] = set()
    for query in queries:
        for item in github_repositories(query):
            url = str(item.get("url") or "")
            if url and url not in seen:
                seen.add(url)
                projects.append({**item, "query": query})
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "queries": queries,
        "github_projects": projects[:200],
        "notes": [
            "Public metadata only.",
            "Stars are popularity signals, not quality guarantees.",
            "Repository licenses must be checked before copying code.",
        ],
    }


def main() -> int:
    raw = os.getenv("RESEARCH_QUERIES", "").strip()
    queries = [x.strip() for x in raw.split("|") if x.strip()] if raw else DEFAULT_QUERIES
    Path("data").mkdir(parents=True, exist_ok=True)
    Path("data/research_hub.json").write_text(
        json.dumps(build_research_index(queries), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
