from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.deps import get_current_user
from app.models import User

router = APIRouter(prefix="/api/mrk-earning", tags=["MRK Earning OS"])

def _read_json(path: str, default):
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default

@router.get("/status")
def status(user: User = Depends(get_current_user)):
    _ = user
    manifest = _read_json("config/earning_os.json", {})
    report = _read_json("data/earning_opportunities.json", {})
    return JSONResponse({
        "system": manifest.get("name", "MRK AI Earning OS"),
        "version": manifest.get("version", "unknown"),
        "execution": manifest.get("execution", "unknown"),
        "laptop_required": manifest.get("laptop_required", True),
        "opportunities": report.get("count", 0),
        "generated_at": report.get("generated_at"),
        "modules": manifest.get("modules", {}),
    })

@router.get("/opportunities")
def opportunities(limit: int = 30, user: User = Depends(get_current_user)):
    _ = user
    limit = max(1, min(limit, 100))
    report = _read_json("data/earning_opportunities.json", {"opportunities": []})
    return JSONResponse({
        "generated_at": report.get("generated_at"),
        "items": report.get("opportunities", [])[:limit],
    })

@router.get("/approvals")
def approvals(user: User = Depends(get_current_user)):
    _ = user
    manifest = _read_json("config/earning_os.json", {})
    return JSONResponse({
        "human_approval_required": [
            name for name, cfg in manifest.get("modules", {}).items()
            if isinstance(cfg, dict) and cfg.get("approval_required")
        ],
        "sensitive_actions": manifest.get("sensitive_actions", {}),
    })
