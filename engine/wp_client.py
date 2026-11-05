import ipaddress
import re
import socket
import time
from urllib.parse import urlsplit
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

import requests
from requests.auth import HTTPBasicAuth


def _validate_remote_wordpress_target(base_url: str) -> None:
    """Reject non-HTTPS, local, and non-public WordPress targets before sending credentials."""
    parsed = urlsplit(base_url)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ValueError("WordPress URL must use HTTPS and include a hostname")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("WordPress URL must not contain credentials, a query, or a fragment")

    host = parsed.hostname.rstrip(".").lower()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(
        (".localhost", ".local", ".internal", ".test", ".invalid")
    ):
        raise ValueError("WordPress URL must use a public hostname")

    try:
        literal_address = ipaddress.ip_address(host)
    except ValueError:
        literal_address = None
    if literal_address is not None:
        if not literal_address.is_global or literal_address.is_multicast or literal_address.is_unspecified:
            raise ValueError("WordPress URL must use a public IP address")
        return

    try:
        resolved = socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)
    except (OSError, ValueError):
        raise ValueError("WordPress hostname could not be resolved safely") from None
    if not resolved:
        raise ValueError("WordPress hostname could not be resolved safely")
    for item in resolved:
        address = ipaddress.ip_address(item[4][0])
        if not address.is_global or address.is_multicast or address.is_unspecified:
            raise ValueError("WordPress hostname must resolve only to public IP addresses")


def request(method: str, base_url: str, path: str, user: str, app_password: str, timeout: int, **kwargs: Any) -> requests.Response:
    _validate_remote_wordpress_target(base_url)
    url = f"{base_url.rstrip('/')}/wp-json/wp/v2/{path.lstrip('/')}"
    # Never follow redirects while carrying WordPress BasicAuth credentials.
    kwargs.setdefault("allow_redirects", False)
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


def near_duplicate(title: str, existing_titles: List[str], threshold: float = 0.95) -> bool:
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


def update_post(
    post_id: int,
    payload: Dict[str, Any],
    base_url: str,
    user: str,
    app_password: str,
    timeout: int,
) -> Dict[str, Any]:
    resp = request("POST", base_url, f"posts/{post_id}", user, app_password, timeout, json=payload)
    resp.raise_for_status()
    return resp.json()


def schedule_iso(index: int, mode: str, interval_minutes: int) -> Optional[str]:
    if mode != "future":
        return None
    return (datetime.now(timezone.utc) + timedelta(minutes=index * interval_minutes)).isoformat()


def clean_title(rendered: str) -> str:
    return re.sub(r"<[^>]+>", "", rendered).strip()
