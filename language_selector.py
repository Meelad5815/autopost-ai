#!/usr/bin/env python3
"""MRK Adaptive Language Selector.

Chooses a practical implementation language from a maintained capability matrix.
It does not claim to benchmark every language in existence; the matrix is
extensible and is scored for the actual website requirements.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

LANGUAGES = {
 "TypeScript":{"web":10,"frontend":10,"backend":9,"ecosystem":10,"performance":8,"ease":8,"automation":9},
 "JavaScript":{"web":10,"frontend":10,"backend":9,"ecosystem":10,"performance":7,"ease":9,"automation":9},
 "Python":{"web":8,"frontend":5,"backend":10,"ecosystem":10,"performance":7,"ease":10,"automation":10},
 "Go":{"web":8,"frontend":3,"backend":10,"ecosystem":8,"performance":10,"ease":8,"automation":8},
 "Rust":{"web":7,"frontend":3,"backend":8,"ecosystem":8,"performance":10,"ease":5,"automation":7},
 "PHP":{"web":9,"frontend":7,"backend":9,"ecosystem":9,"performance":7,"ease":9,"automation":7},
 "Java":{"web":8,"frontend":5,"backend":9,"ecosystem":9,"performance":9,"ease":6,"automation":7},
 "C#":{"web":8,"frontend":5,"backend":9,"ecosystem":9,"performance":9,"ease":8,"automation":7},
 "Ruby":{"web":8,"frontend":5,"backend":8,"ecosystem":7,"performance":6,"ease":9,"automation":8},
 "Kotlin":{"web":7,"frontend":4,"backend":8,"ecosystem":8,"performance":8,"ease":7,"automation":7},
 "R":{"web":5,"frontend":2,"backend":5,"ecosystem":8,"performance":5,"ease":8,"automation":6},
}

DEFAULT_WEIGHTS={"web":25,"frontend":15,"backend":15,"ecosystem":15,"performance":10,"ease":10,"automation":10}

def select(requirements):
    weights={**DEFAULT_WEIGHTS, **requirements}
    ranked=[]
    for name,cap in LANGUAGES.items():
        score=sum(cap[k]*weights.get(k,0) for k in weights)/sum(weights.values())
        ranked.append({"language":name,"score":round(score,2)})
    return sorted(ranked,key=lambda x:x["score"],reverse=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--requirements",default="data/website_factory/language_requirements.json")
    ap.add_argument("--out",default="data/website_factory/language_selection.json")
    a=ap.parse_args()
    req=json.loads(Path(a.requirements).read_text()) if Path(a.requirements).exists() else {}
    result={"selected":select(req)[0],"alternatives":select(req)[1:5],"note":"Selection is based on the maintained matrix; update the matrix when project requirements or ecosystem conditions change."}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
