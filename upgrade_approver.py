#!/usr/bin/env python3
"""MRK explicit approval gate.

Approval is opt-in: create data/upgrade_approval.json with:
{"candidate_id":"...","approved":true}
This tool only records the decision. Deployment remains a separate action.
"""
import argparse, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--candidate",required=True)
    ap.add_argument("--approved",action="store_true")
    ap.add_argument("--out",default="data/upgrade_approval.json")
    a=ap.parse_args()
    if not a.approved:
        raise SystemExit("Approval flag not supplied; no deployment authorization granted.")
    Path(a.out).write_text(json.dumps({
        "candidate_id":a.candidate,"approved":True,
        "note":"Human approval recorded; deployment must be handled by a separate approved workflow."
    },indent=2),encoding="utf-8")
    print("Approval recorded:",a.candidate)

if __name__=="__main__": main()
