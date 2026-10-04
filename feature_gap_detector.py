#!/usr/bin/env python3
"""MRK Feature Gap Detector.

Turns public product/technology signals into ranked feature candidates.
This is a discovery engine, not an automatic copier of third-party code.
"""
from __future__ import annotations
import argparse, json, urllib.parse, urllib.request
from pathlib import Path
from datetime import datetime, timezone

def fetch_github_issues(query):
    url="https://api.github.com/search/issues?q="+urllib.parse.quote(query)+"&sort=comments&order=desc&per_page=10"
    try:
        req=urllib.request.Request(url,headers={"Accept":"application/vnd.github+json","User-Agent":"MRK-Feature-Gap-Detector"})
        with urllib.request.urlopen(req,timeout=10) as r:
            return json.load(r).get("items",[])
    except Exception:
        return []

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--product","default="MRK AI Website Factory")
    ap.add_argument("--purpose",default="easy-to-run websites, automation, SEO and online services")
    ap.add_argument("--out",default="data/feature_gaps.json")
    a=ap.parse_args()

    queries=[
        "website builder feature request",
        "static site generator feature request",
        "AI website builder missing feature",
        "website automation feature request",
        "SEO tool feature request",
        "developer dashboard feature request",
    ]
    candidates=[]
    for q in queries:
        for item in fetch_github_issues(q):
            title=item.get("title","").strip()
            if not title: continue
            candidates.append({
                "title":title,
                "source":"GitHub public issue search",
                "url":item.get("html_url",""),
                "comments":item.get("comments",0),
                "candidate_feature":title,
                "implementation_status":"candidate",
                "requires_human_review":True,
            })

    # Deterministic ranking: discussion volume + recency signal.
    seen=set(); ranked=[]
    for x in sorted(candidates,key=lambda z:z["comments"],reverse=True):
        key=x["title"].lower()
        if key in seen: continue
        seen.add(key)
        x["priority"]=min(100,20+x["comments"]*4)
        ranked.append(x)

    result={
        "product":a.product,
        "purpose":a.purpose,
        "generated_at":datetime.now(timezone.utc).isoformat(),
        "method":"public issue/discussion signals; candidates require validation before implementation",
        "feature_gaps":ranked[:30],
        "next_action":"validate candidate, design improvement, test, then request approval before publishing"
    }
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Feature candidates:",len(result["feature_gaps"]))

if __name__=="__main__": main()
