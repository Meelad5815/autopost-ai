#!/usr/bin/env python3
"""MRK safe feature-code generator.

Applies only deterministic, reviewable improvements to the generated static site.
It never edits main directly and never deploys.
"""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path

ALLOWED={"accessibility","seo","performance","general"}

def improve_html(path: Path) -> bool:
    s=path.read_text(encoding="utf-8")
    original=s
    if 'class="skip-link"' not in s:
        s=s.replace("<body>", '<body><a class="skip-link" href="#main-content">Skip to content</a>', 1)
    if 'id="main-content"' not in s:
        s=s.replace("<main>", '<main id="main-content">', 1)
    if 'rel="canonical"' not in s:
        s=s.replace("</head>", f'<link rel="canonical" href="./{path.name}"></head>', 1)
    if 'aria-label="Primary navigation"' not in s:
        s=s.replace("<nav>", '<nav aria-label="Primary navigation">', 1)
    if s != original:
        path.write_text(s, encoding="utf-8")
        return True
    return False

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--candidate", default="data/upgrade_candidates.json")
    ap.add_argument("--site", default="data/website_factory/site")
    ap.add_argument("--out", default="data/upgrade_candidate_execution.json")
    args=ap.parse_args()
    candidate_path=Path(args.candidate)
    site=Path(args.site)
    data=json.loads(candidate_path.read_text(encoding="utf-8")) if candidate_path.exists() else {"candidates":[]}
    candidates=data.get("candidates",[])
    candidate=candidates[0] if candidates else None
    if not candidate:
        raise SystemExit("No upgrade candidate available.")
    category=candidate.get("category","general")
    if category not in ALLOWED:
        raise SystemExit(f"Feature category '{category}' is not enabled for automatic candidate code generation.")
    files=[]
    if site.exists():
        for p in sorted(site.glob("*.html")):
            if improve_html(p):
                files.append(str(p))
    result={
        "candidate_id":candidate["candidate_id"],
        "generated_at":datetime.now(timezone.utc).isoformat(),
        "category":category,
        "production_changed":False,
        "deployment_allowed":False,
        "requires_human_review":True,
        "changed_files":files,
        "rollback_required_before_merge":True,
        "note":"Candidate-only change. Production deployment remains blocked until explicit human approval."
    }
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Generated candidate changes:", len(files))

if __name__=="__main__": main()
