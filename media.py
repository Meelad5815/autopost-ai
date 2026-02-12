import base64
from typing import Any, Dict

import requests

from engine.wp_client import request


def fetch_royalty_free_image(query: str, timeout: int) -> Dict[str, Any]:
    url = f"https://source.unsplash.com/1600x900/?{requests.utils.quote(query)}"
    try:
        resp = requests.get(url, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        filename = f"{query[:60].strip().replace(' ', '-')}.jpg" or "featured.jpg"
        return {
            "bytes": resp.content,
            "url": resp.url,
            "filename": filename,
            "content_type": "image/jpeg",
        }
    except Exception:
        pixel_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO5m7nQAAAAASUVORK5CYII="
        )
        return {"bytes": pixel_png, "url": "local://image/fallback", "filename": "featured.png", "content_type": "image/png"}


def upload_media(
    base_url: str,
    user: str,
    app_password: str,
    timeout: int,
    image: Dict[str, Any],
    alt_text: str,
    caption: str,
) -> int:
    upload = request(
        "POST",
        base_url,
        "media",
        user,
        app_password,
        timeout,
        headers={
            "Content-Disposition": f'attachment; filename="{image["filename"]}"',
            "Content-Type": image["content_type"],
        },
        data=image["bytes"],
    )
    if upload.status_code not in (200, 201):
        raise RuntimeError(f"Media upload failed: {upload.status_code} {upload.text[:300]}")
    media_id = int(upload.json()["id"])

    patch = request(
        "POST",
        base_url,
        f"media/{media_id}",
        user,
        app_password,
        timeout,
        json={"alt_text": alt_text, "caption": caption},
    )
    if patch.status_code not in (200, 201):
        pass
    return media_id
