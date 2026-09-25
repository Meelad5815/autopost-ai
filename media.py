"""Zero-cost featured-image generation with optional remote source.

Default mode creates a real PNG locally, so image failure can never produce a
1x1 tracking pixel. Remote image providers are optional upgrades.
"""
import struct
import zlib
from typing import Any, Dict
from engine.wp_client import request


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)


def _local_png(query: str, width: int = 1200, height: int = 675) -> bytes:
    # Lightweight RGB gradient/geometry PNG. No Pillow or external API required.
    seed = sum(ord(c) for c in query) % 256
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            r = (12 + (x * 25 // width) + seed) % 256
            g = (28 + (y * 45 // height) + seed // 2) % 256
            b = (70 + ((x + y) * 35 // (width + height))) % 256
            if abs(x - width // 2) < 5 or abs(y - height // 2) < 5:
                r, g, b = 240, 190, 60
            row.extend((r, g, b))
        rows.append(bytes(row))
    raw = b"".join(rows)
    return b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + _png_chunk(b"IDAT", zlib.compress(raw, 6)) + _png_chunk(b"IEND", b"")


def fetch_royalty_free_image(query: str, timeout: int) -> Dict[str, Any]:
    # Intentionally local by default. A paid API is never required.
    data = _local_png(query)
    safe = "".join(c if c.isalnum() else "-" for c in query)[:50].strip("-") or "featured"
    return {
        "bytes": data,
        "url": "local://generated-featured-image",
        "filename": f"{safe}.png",
        "content_type": "image/png",
    }


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
        raise RuntimeError(f"Media metadata update failed: {patch.status_code} {patch.text[:300]}")
    return media_id
