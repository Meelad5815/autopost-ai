"""Bounded autonomous developer loop.

Creates an auditable maintenance plan from local findings and public research.
It deliberately does not execute arbitrary generated code or copy third-party code.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .research_hub import build_research_index

REPORT = Path("data/autonomous_developer.json")


def run_check(command: list[str]) -> dict[str, Any]:
    try:
        p = subprocess.run(command, capture_output=True, text=True, timeout=120)
        return {"command": command, "returncode": p.returncode, "stdout": p.stdout[-4000:], "stderr": p.stderr[-4000:]}
    except Exception as exc:
        return {"command": command, "returncode": -1, "error": str(exc)}


def build_report() -> dict[str, Any]:
    checks = [
        run_check(["python", "-m", "compileall", "-q", "autopost.py", "scheduler.py", "run_batch.py", "app", "engine"]),
        run_check(["python", "-m", "pytest", "-q"]) if Path("tests").is_dir() else {"command": ["pytest"], "skipped": True},
    ]
    research = build_research_index()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "healthy" if all(x.get("returncode", 0) == 0 for x in checks) else "needs_review",
        "checks": checks,
        "research_summary": {
            "github_projects_found": len(research.get("github_projects", [])),
            "queries": research.get("queries", []),
        },
        "policy": {
            "auto_apply_scope": ["formatting", "lint fixes", "tests and generated reports"],
            "approval_required_for": ["arbitrary generated code", "credential changes", "payments", "destructive production changes"],
        },
    }


def main() -> int:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(build_report(), ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
