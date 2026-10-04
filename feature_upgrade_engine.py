#!/usr/bin/env python3
"""MRK Feature Upgrade Engine.

Converts the highest-ranked feature-gap signal into a reviewable upgrade
candidate. It never changes live production files and never deploys.
"""
from __future__ import annotations
import argparse, json, re
from datetime import datetime, timezone
from pathlib import Path

SAFE_FEATURES = {
    "accessibility": ["aria labels", "keyboard navigation", "semantic html"],
    "seo": ["meta description", "canonical url", "structured headings"],
    "performance": ["lazy loading", "defer noncritical scripts", "minified assets"],
    "search": ["site search", "search index", "filter controls"],
    "contact": ["contact form", "validation", "success state"],
    "dashboard": ["status dashboard", "health checks", "upgrade history"],
}

def classify(title: str) -> str:
    t=title.lower()
    for k in SAFE_FEATURES:
        if k in t: return k
    if any(x in t for x in ("seo","search engine","metadata")): return "seo"
    if any(x in t for x in ("performance","speed","slow","loading")): return "performance"
    if any(x in t for x in ("accessib","a11y","keyboard")): return "accessibility"
    if any(x in t for x in ("form","contact","lead")): return "contact"
    if any(x in t for x in ("dashboard","monitor","health")): return "dashboard"
    return "general"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--gaps",default="data/feature_gaps.json")
    ap.add_argument("--out",default="data/upgrade_candidates.json")
    a=ap.parse_args()
    gaps=json.loads(Path(a.gaps).read_text(encoding="utf-8")) if Path(a.gaps).exists() else {}
    items=gaps.get("feature_gaps",[])
    selected=items[0] if items else {
        "title":"No validated feature candidate found",
        "candidate_feature":"No-op",
        "priority":0,
    }
    category=classify(selected.get("candidate_feature",selected.get("title","")))
    title=selected.get("candidate_feature",selected.get("title",""))
    plan=SAFE_FEATURES.get(category,[
        "inspect current site architecture",
        "define acceptance criteria",
        "implement isolated change",
    ])
    candidate={
        "candidate_id":"mrk-"+datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        "status":"awaiting_human_approval",
        "requires_human_approval":True,
        "selected_feature":title,
        "source":selected.get("url",""),
        "category":category,
        "priority":selected.get("priority",0),
        "implementation_plan":plan,
        "acceptance_tests":[
            "existing smoke tests pass",
            "generated HTML contains title and meta description",
            "no broken internal links in generated package",
            "no credentials, OTP, payment or identity-verification logic is introduced",
        ],
        "performance_checks":[
            "package generation completes successfully",
            "HTML remains static and lightweight",
        ],
        "deployment":"blocked until explicit human approval",
        "generated_at":datetime.now(timezone.utc).isoformat(),
    }
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({"candidates":[candidate]},ensure_ascii=False,indent=2),encoding="utf-8")
    print("Upgrade candidate:",candidate["candidate_id"])

if __name__=="__main__": main()
