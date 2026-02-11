import re
import time
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

import requests
from requests.auth import HTTPBasicAuth


def request(method: str, base_url: str, path: str, user: str, app_password: str, timeout: int, **kwargs: Any) -> requests.Response:
    url = f"{base_url.rstrip('/')}/wp-json/wp/v2/{path.lstrip('/')}"
    return requests.request(method, url, auth=HTTPBasicAuth(user, app_password), timeout=timeout, **kwargs)


def get_posts(base_url: str, user: str, app_password: str, timeout: int, per_page: int = 100) -> List[Dict[str, Any]]:
    resp = request(
        "GET",
        base_url,
        "posts",
        user,
        app_password,
        timeout,
        params={"per_page": per_page, "_fields": "id,date,link,slug,title,content"},
    )
    resp.raise_for_status()
    return resp.json()


def ensure_term(base_url: str, user: str, app_password: str, timeout: int, taxonomy: str, name: str, slug: str) -> int:
    get_resp = request(
        "GET",
        base_url,
        taxonomy,
        user,
        app_password,
        timeout,
        params={"slug": slug, "per_page": 1},
    )
    get_resp.raise_for_status()
    found = get_resp.json()
    if found:
        return int(found[0]["id"])

    create_resp = request(
        "POST", base_url, taxonomy, user, app_password, timeout, json={"name": name, "slug": slug}
    )
    if create_resp.status_code not in (200, 201):
        raise RuntimeError(f"taxonomy create failed: {create_resp.status_code} {create_resp.text[:300]}")
    return int(create_resp.json()["id"])


def near_duplicate(title: str, existing_titles: List[str], threshold: float = 0.84) -> bool:
    for existing in existing_titles:
        if SequenceMatcher(a=title.lower().strip(), b=existing.lower().strip()).ratio() >= threshold:
            return True
    return False


def publish_with_retry(
    payload: Dict[str, Any],
    base_url: str,
    user: str,
    app_password: str,
    timeout: int,
    retries: int,
) -> Dict[str, Any]:
    for attempt in range(1, retries + 1):
        try:
            resp = request("POST", base_url, "posts", user, app_password, timeout, json=payload)
            if resp.status_code in (200, 201):
                return resp.json()
        except requests.RequestException:
            pass
        if attempt < retries:
            time.sleep(2 ** attempt)
    raise RuntimeError("WordPress publish failed after retries")


def schedule_iso(index: int, mode: str, interval_minutes: int) -> Optional[str]:
    if mode != "future":
        return None
    return (datetime.now(timezone.utc) + timedelta(minutes=index * interval_minutes)).isoformat()


def clean_title(rendered: str) -> str:
    return re.sub(r"<[^>]+>", "", rendered).strip()
