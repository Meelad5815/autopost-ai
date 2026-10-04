#!/usr/bin/env python3
"""MRK Programming Language Researcher.

Researches public ecosystem signals for programming languages and produces a
ranked decision report. It does not claim to benchmark every language
exhaustively; it builds a repeatable shortlist from public data sources.
"""
from __future__ import annotations
import argparse, json, re, urllib.request
from pathlib import Path

LANGUAGES = [
    ("Python","general, web, AI, automation, data"),
    ("JavaScript","web, full-stack, automation"),
    ("TypeScript","web, full-stack, large applications"),
    ("Go","backend, cloud, networking"),
    ("Rust","systems, performance, safety"),
    ("Java","enterprise, Android, backend"),
    ("C#","enterprise, desktop, games, backend"),
    ("C++","systems, performance, embedded, games"),
    ("PHP","web, WordPress, server-side"),
    ("Ruby","web, automation"),
    ("Kotlin","Android, backend"),
    ("Swift","Apple platforms"),
    ("Dart","Flutter, cross-platform apps"),
    ("R","statistics, data science, research"),
    ("C","systems, embedded"),
    ("Scala","JVM, data, backend"),
    ("Elixir","distributed systems, web"),
    ("Lua","embedded scripting, games"),
    ("Julia","scientific computing, numerical work"),
    ("Haskell","functional, research, high assurance"),
]

def github_count(language):
    url="https://api.github.com/search/repositories?q=language:"+urllib.parse.quote(language)+"&sort=stars&order=desc&per_page=1"
    try:
        req=urllib.request.Request(url,headers={"Accept":"application/vnd.github+json","User-Agent":"MRK-Language-Researcher"})
        with urllib.request.urlopen(req,timeout=10) as r:
            return int(json.load(r).get("total_count",0))
    except Exception:
        return None

def score(name, purpose):
    p=purpose.lower()
    base={"Python":10,"JavaScript":10,"TypeScript":10,"Go":8,"Rust":8,"Java":8,"C#":8,"PHP":8,"C++":7,"Kotlin":7,"Dart":7,"Ruby":6,"R":6,"C":6,"Swift":6,"Scala":6,"Elixir":6,"Lua":5,"Julia":5,"Haskell":4}.get(name,5)
    if any(x in p for x in ["web","website","saas","full stack"]):
        base += {"TypeScript":5,"JavaScript":5,"Python":4,"PHP":4,"Go":3,"Rust":2}.get(name,0)
    if any(x in p for x in ["ai","automation","data"]):
        base += {"Python":6,"R":3,"Julia":3,"TypeScript":2}.get(name,0)
    if any(x in p for x in ["performance","systems","embedded"]):
        base += {"Rust":6,"C++":5,"C":5,"Go":4}.get(name,0)
    return base

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--purpose",default="easy-to-run websites, AI automation and online services")
    ap.add_argument("--out",default="data/language_research.json")
    a=ap.parse_args()
    rows=[]
    for name,uses in LANGUAGES:
        rows.append({"language":name,"ecosystem_uses":uses,"score":score(name,a.purpose),"github_repository_count":github_count(name)})
    rows.sort(key=lambda x:(x["score"],x["github_repository_count"] or 0),reverse=True)
    result={"purpose":a.purpose,"method":"repeatable public-ecosystem shortlist; not an exhaustive benchmark","ranked_languages":rows,"recommended":rows[0]["language"],"research_time_utc":__import__("datetime").datetime.utcnow().isoformat()+"Z"}
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Recommended language:",result["recommended"])

if __name__=="__main__": main()
