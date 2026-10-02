"""Create an isolated, auditable maintenance patch from an AI proposal.

This module only prepares a patch artifact. It never changes main, credentials,
payments, or production state.
"""
from __future__ import annotations
import json
from pathlib import Path
from .safety_policy import POLICY

PROPOSAL = Path("data/ai_developer_proposal.json")
PATCH = Path("data/ai_patch_plan.json")

def build_patch_plan() -> dict:
    raw = json.loads(PROPOSAL.read_text(encoding="utf-8")) if PROPOSAL.exists() else {}
    proposal = raw.get("proposal")
    if not isinstance(proposal, dict):
        return {"status": "no-proposal", "reason": raw.get("reason", "No structured proposal")}
    files = proposal.get("files", [])
    if not isinstance(files, list):
        files = []
    files = [str(x) for x in files[:POLICY.max_files_per_patch]]
    text = json.dumps(proposal, ensure_ascii=False).lower()
    blocked = [term for term in POLICY.require_review if term in text]
    return {
        "status": "review-required" if blocked else "patch-plan-ready",
        "files": files,
        "blocked_reasons": blocked,
        "summary": proposal.get("summary", ""),
        "proposed_patch": proposal.get("proposed_patch"),
        "policy": {
            "max_files_per_patch": POLICY.max_files_per_patch,
            "auto_apply": list(POLICY.auto_apply),
        },
    }

def main() -> int:
    PATCH.parent.mkdir(parents=True, exist_ok=True)
    PATCH.write_text(json.dumps(build_patch_plan(), ensure_ascii=False, indent=2), encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
