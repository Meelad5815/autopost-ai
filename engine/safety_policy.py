"""Autonomous developer safety configuration.

The AI developer may propose patches, but automatic application is limited to
low-risk maintenance. High-impact changes are emitted as proposals for review.
"""
from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class SafetyPolicy:
    auto_apply: tuple[str, ...] = ("format", "lint", "tests", "generated-reports")
    require_review: tuple[str, ...] = (
        "new-dependency",
        "authentication",
        "credentials",
        "payment",
        "database-migration",
        "production-deployment",
        "third-party-code-copy",
        "destructive-change",
    )
    max_files_per_patch: int = 8

POLICY = SafetyPolicy()
