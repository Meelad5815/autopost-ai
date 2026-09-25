import json
import re
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlsplit

GSC_FILE = Path("data/search_console.json")
OUTPUT_FILE = Path("data/refresh_plans.json")


def _load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _path(url: str) -> str:
    return urlsplit(str(url)).path.strip("/").lower()


def _title_text(post: Dict[str, Any]) -> str:
    value = post.get("title", "")
    if isinstance(value, dict):
        value = value.get("rendered", "")
    return re.sub(r"<[^>]+>", " ", str(value)).strip()


def _plain(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value or "")).strip()


def _query_rows_for_page(gsc: Dict[str, Any], page: str) -> List[Dict[str, Any]]:
    target = _path(page)
    rows = []
    for row in gsc.get("page_query_rows", []):
        keys = row.get("keys") or []
        if len(keys) < 2:
            continue
        if _path(keys[0]) == target:
            item = dict(row)
            item["query"] = str(keys[1]).strip()
            rows.append(item)
    return rows


def _suggest_titles(current: str, queries: List[str]) -> List[str]:
    q = [x for x in queries if x]
    primary = q[0] if q else current
    candidates = [
        f"{primary.title()}: Complete Practical Guide",
        f"{primary.title()} — Step-by-Step Guide",
        f"{primary.title()}: Cost, Setup and Best Practices",
    ]
    return [x[:90].strip() for x in candidates if x and x.lower() != current.lower()][:3]


def _suggest_meta(queries: List[str], current: str) -> List[str]:
    primary = queries[0] if queries else ""
    secondary = queries[1] if len(queries) > 1 else ""
    base = f"Learn {primary} with practical steps, key considerations and useful examples"
    if secondary:
        base += f", including {secondary}"
    return [
        (base + ".").strip()[:160],
        (f"Practical {primary} guide covering setup, common issues, costs and next steps.").strip()[:160],
    ][:2] if primary else [current[:160]]


def build_refresh_plans(posts: List[Dict[str, Any]]) -> Dict[str, Any]:
    gsc = _load(GSC_FILE)
    if not gsc:
        return {"status": "waiting_for_search_console_data", "plans": []}

    by_path = {_path(p.get("link", "")): p for p in posts if p.get("link")}
    min_impressions = int(__import__("os").getenv("MIN_REFRESH_IMPRESSIONS", "50"))
    max_position = float(__import__("os").getenv("MAX_REFRESH_POSITION", "30"))
    max_plans = int(__import__("os").getenv("MAX_REFRESH_PLANS", "10"))

    plans = []
    for row in gsc.get("page_rows", []):
        keys = row.get("keys") or []
        if not keys:
            continue
        page = str(keys[0])
        post = by_path.get(_path(page))
        if not post:
            continue

        impressions = float(row.get("impressions", 0))
        clicks = float(row.get("clicks", 0))
        ctr = float(row.get("ctr", 0))
        position = float(row.get("position", 0))
        if impressions < min_impressions or position > max_position:
            continue

        pq = _query_rows_for_page(gsc, page)
        pq.sort(key=lambda x: float(x.get("impressions", 0)), reverse=True)
        queries = [str(x.get("query", "")).strip() for x in pq if x.get("query")]
        zero_click_queries = [
            str(x.get("query", "")).strip()
            for x in pq
            if float(x.get("impressions", 0)) >= 10 and float(x.get("clicks", 0)) == 0
        ]

        title = _title_text(post)
        current_content = _plain(post.get("content", {}).get("rendered", ""))
        headings = re.findall(r"<h[2-4]\b[^>]*>(.*?)</h[2-4]>", post.get("content", {}).get("rendered", ""), re.I | re.S)
        existing_query_text = " ".join([title, current_content]).lower()
        missing_coverage = [q for q in zero_click_queries if q.lower() not in existing_query_text][:8]

        actions = []
        if ctr < 0.03:
            actions.append("review_title_and_meta")
        if missing_coverage:
            actions.append("add_missing_query_coverage")
        if len(headings) < 3:
            actions.append("expand_heading_structure")
        actions.append("review_internal_links")

        plans.append({
            "post_id": post.get("id"),
            "url": page,
            "current_title": title,
            "metrics": {
                "impressions": impressions,
                "clicks": clicks,
                "ctr": round(ctr, 4),
                "position": round(position, 2),
            },
            "target_queries": queries[:10],
            "missing_query_coverage": missing_coverage,
            "suggested_titles": _suggest_titles(title, queries),
            "suggested_meta_descriptions": _suggest_meta(queries, ""),
            "recommended_actions": actions,
            "safe_change_policy": "report_only",
        })

    plans.sort(key=lambda x: (x["metrics"]["impressions"], -x["metrics"]["ctr"]), reverse=True)
    report = {
        "status": "ready",
        "source": "Google Search Console Search Analytics",
        "period": {"start_date": gsc.get("start_date"), "end_date": gsc.get("end_date")},
        "policy": {
            "automatic_changes": False,
            "reason": "Title/meta/content changes are generated as recommendations first; no automatic rewrite is applied."
        },
        "plans": plans[:max_plans],
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build_refresh_plans([]), ensure_ascii=False, indent=2))
