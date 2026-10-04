#!/usr/bin/env python3
"""MRK candidate change planner.

This deliberately produces a patch plan rather than silently modifying production.
The PR workflow is the review boundary.
"""
import json
from pathlib import Path

def main():
    p=Path("data/upgrade_candidates.json")
    data=json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    candidates=data.get("candidates",[])
    if not candidates:
        print("No upgrade candidate available.")
        return
    c=candidates[0]
    report={
        "candidate_id":c["candidate_id"],
        "selected_feature":c["selected_feature"],
        "implementation_plan":c["implementation_plan"],
        "acceptance_tests":c["acceptance_tests"],
        "deployment":"blocked until reviewed and merged",
        "changed_files":[],"mode":"review-only"
    }
    Path("data/upgrade_plan.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Review plan written for",c["candidate_id"])

if __name__=="__main__":
    main()
