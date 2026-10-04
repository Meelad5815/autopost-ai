#!/usr/bin/env python3
"""MRK isolated upgrade candidate runner.

Creates a candidate artifact from the ranked feature-gap data. This module is
intentionally limited to planning/validation; it does not deploy or modify
production website files.
"""
import json
from pathlib import Path

def main():
    gaps=Path("data/feature_gaps.json")
    out=Path("data/upgrade_candidate_execution.json")
    data=json.loads(gaps.read_text(encoding="utf-8")) if gaps.exists() else {}
    candidate=(data.get("feature_gaps") or [{}])[0]
    result={
        "status":"candidate_ready",
        "feature":candidate.get("candidate_feature",candidate.get("title","")),
        "source":candidate.get("url",""),
        "execution_mode":"isolated-review-only",
        "production_changed":False,
        "deployment_allowed":False,
        "next_step":"run tests and open/inspect draft PR before explicit human approval"
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Candidate execution artifact written.")

if __name__=="__main__":
    main()
