"""Backward-compatible security import shim.

Some deployments may still import from `app.security` (legacy path). The
canonical implementation lives in `app.core.security`.
"""

from app.core.security import (  # noqa: F401
    create_access_token,
    create_refresh_token,
    decode_token,
    decode_token_claims,
    generate_opaque_token,
    hash_password,
    hash_token,
    verify_password,
)


def get_password_hash(password: str) -> str:
    """Legacy alias expected by older modules."""
    return hash_password(password)
