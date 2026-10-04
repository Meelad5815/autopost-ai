#!/usr/bin/env python3
"""Create a safe deployment manifest; this tool never deploys."""
from __future__ import annotations
import hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path('data/website_factory/site')
OUT=Path('data/deployment_manifest.json')
def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()
def main():
    files=[]
    if ROOT.exists():
        for path in sorted(ROOT.rglob('*')):
            if path.is_file(): files.append({'path':str(path.relative_to(ROOT)).replace('\\\\','/'),'sha256':sha256(path),'bytes':path.stat().st_size})
    manifest={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'source_commit':os.getenv('GITHUB_SHA'),'artifact_root':str(ROOT),'file_count':len(files),'files':files,'deployment':{'production_changed':False,'deployment_allowed':False,'automatic_deployment':False,'requires_explicit_workflow_dispatch':True,'supported_target':'github-pages'},'rollback':{'strategy':'retain_previous_deployment_artifact_and_source_commit','source_commit':os.getenv('GITHUB_SHA'),'automatic_rollback':False}}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8'); print(f'Wrote {OUT} with {len(files)} files.')
if __name__=='__main__': main()