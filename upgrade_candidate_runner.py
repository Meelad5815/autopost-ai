#!/usr/bin/env python3
"""MRK isolated upgrade candidate runner.

Applies only low-risk, local website improvements to the candidate package.
Production is never touched. A candidate manifest records every change.
"""
import json, re
from pathlib import Path

def improve_html(path: Path):
    s=path.read_text(encoding="utf-8")
    changed=[]
    # Add a canonical URL placeholder that can be replaced by deployment config.
    if 'rel="canonical"' not in s:
        s=s.replace("</head>", '<link rel="canonical" href="./'+path.name+'"></head>')
        changed.append("canonical")
    # Add skip navigation for keyboard accessibility.
    if 'class="skip-link"' not in s:
        s=s.replace("<body>", '<body><a class="skip-link" href="#main-content">Skip to content</a>',1)
        s=s.replace("<main>", '<main id="main-content">',1)
        changed.append("skip-link")
    path.write_text(s,encoding="utf-8")
    return changed

def main():
    root=Path("data/website_factory/site")
    out=Path("data/upgrade_candidate_execution.json")
    gaps=Path("data/feature_gaps.json")
    data=json.loads(gaps.read_text(encoding="utf-8")) if gaps.exists() else {}
    candidate=(data.get("feature_gaps") or [{}])[0]
    files=[]; changes=[]
    if root.exists():
        for p in sorted(root.glob("*.html")):
            c=improve_html(p)
            if c: files.append(str(p)); changes.extend([f"{p.name}:{x}" for x in c])
    result={
        "status":"candidate_ready",
        "feature":candidate.get("candidate_feature",candidate.get("title","")),
        "source":candidate.get("url",""),
        "execution_mode":"isolated-review-only",
        "production_changed":False,
        "deployment_allowed":False,
        "files_changed":files,
        "changes":changes,
        "next_step":"run automated tests and inspect the draft PR before explicit human approval"
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Candidate execution artifact written:", len(files), "files")

if __name__=="__main__":
    main()
