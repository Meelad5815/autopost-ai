import json
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlsplit

GSC_FILE = Path("data/search_console.json")
WP_FILE = Path("data/content_optimizer.json")


def _load_gsc() -> Dict[str, Any]:
    if not GSC_FILE.exists():
        return {}
    try:
        data = json.loads(GSC_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _slug(url: str) -> str:
    return urlsplit(url).path.strip("/").lower()


def build_optimization_report(posts: List[Dict[str, Any]]) -> Dict[str, Any]:
    gsc = _load_gsc()
    if not gsc:
        return {
            "status": "waiting_for_search_console_data",
            "opportunities": [],
            "query_gaps": [],
            "refresh_candidates": [],
        }

    post_by_path = {}
    for post in posts:
        link = str(post.get("link", "")).strip()
        if link:
            post_by_path[_slug(link)] = post

    opportunities: List[Dict[str, Any]] = []
    for row in gsc.get("page_rows", []):
        keys = row.get("keys") or []
        if not keys:
            continue
        page = str(keys[0])
        path = _slug(page)
        post = post_by_path.get(path)
        if not post:
            continue
        impressions = float(row.get("impressions", 0))
        clicks = float(row.get("clicks", 0))
        ctr = float(row.get("ctr", 0))
        position = float(row.get("position", 0))
        # Diagnostic only: no ranking prediction and no automatic title rewriting.
        if impressions >= 20 and ctr < 0.03:
            opportunities.append({
                "post_id": post.get("id"),
                "url": page,
                "title": post.get("title", {}).get("rendered", ""),
                "impressions": impressions,
                "clicks": clicks,
                "ctr": round(ctr, 4),
                "position": round(position, 2),
                "action": "review_title_meta_description",
                "reason": "real_search_console_impressions_with_low_ctr",
            })

    query_gaps: List[Dict[str, Any]] = []
    for row in gsc.get("query_rows", []):
        keys = row.get("keys") or []
        if not keys:
            continue
        query = str(keys[0]).strip()
        impressions = float(row.get("impressions", 0))
        clicks = float(row.get("clicks", 0))
        if impressions >= 10 and clicks == 0:
            query_gaps.append({
                "query": query,
                "impressions": impressions,
                "clicks": clicks,
                "ctr": round(float(row.get("ctr", 0)), 4),
                "position": round(float(row.get("position", 0)), 2),
                "action": "consider_focused_content_or_refresh",
            })

    opportunities.sort(key=lambda x: (x["impressions"], -x["ctr"]), reverse=True)
    query_gaps.sort(key=lambda x: x["impressions"], reverse=True)

    report = {
        "status": "ready",
        "source": "Google Search Console Search Analytics",
        "period": {
            "start_date": gsc.get("start_date"),
            "end_date": gsc.get("end_date"),
        },
        "opportunities": opportunities[:100],
        "query_gaps": query_gaps[:100],
        "refresh_candidates": opportunities[:25],
    }
    WP_FILE.parent.mkdir(parents=True, exist_ok=True)
    WP_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build_optimization_report([]), ensure_ascii=False, indent=2))
