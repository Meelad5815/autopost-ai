"""Safely materialize a structured AI patch on an isolated branch.

Only explicit file entries are accepted. Main is never modified by this module.
"""
from __future__ import annotations
import json
import os
from pathlib import Path

from .safety_policy import POLICY

PLAN=Path("data/ai_patch_plan.json")
ROOT=Path(".")

def apply() -> dict:
    if not PLAN.exists(): return {"status":"no-plan"}
    plan=json.loads(PLAN.read_text(encoding="utf-8"))
    if plan.get("status") != "patch-plan-ready":
        return {"status":"blocked","reason":plan.get("blocked_reasons",plan.get("reason","review required"))}
    proposed=plan.get("proposed_patch")
    if not isinstance(proposed, dict):
        return {"status":"no-patch","reason":"No structured proposed_patch"}
    changed=[]
    for raw_path, content in proposed.items():
        if len(changed) >= POLICY.max_files_per_patch: break
        path=Path(str(raw_path))
        if path.is_absolute() or ".." in path.parts or str(path).startswith(".git"):
            return {"status":"blocked","reason":f"unsafe path: {raw_path}"}
        if not isinstance(content,str): return {"status":"blocked","reason":f"non-text content: {raw_path}"}
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(content,encoding="utf-8")
        changed.append(str(path))
    return {"status":"applied-to-working-tree","files":changed}

if __name__=="__main__":
    print(json.dumps(apply(),ensure_ascii=False,indent=2))
