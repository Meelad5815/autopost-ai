import json
from pathlib import Path

def test_feature_gap_artifact_is_valid():
    p=Path("data/feature_gaps.json")
    if p.exists():
        data=json.loads(p.read_text(encoding="utf-8"))
        assert isinstance(data.get("feature_gaps",[]), list)

def test_upgrade_candidate_requires_approval():
    p=Path("data/upgrade_candidates.json")
    if p.exists():
        data=json.loads(p.read_text(encoding="utf-8"))
        for c in data.get("candidates",[]):
            assert c["requires_human_approval"] is True
            assert c["deployment"]=="blocked until explicit human approval"

def test_execution_artifact_never_allows_deployment():
    p=Path("data/upgrade_candidate_execution.json")
    if p.exists():
        data=json.loads(p.read_text(encoding="utf-8"))
        assert data.get("production_changed") is False
        assert data.get("deployment_allowed") is False

def test_deployment_manifest_is_explicitly_gated():
    p=Path("data/deployment_manifest.json")
    if p.exists():
        data=json.loads(p.read_text(encoding="utf-8"))
        assert data["deployment"]["production_changed"] is False
        assert data["deployment"]["automatic_deployment"] is False
        assert data["deployment"]["requires_explicit_workflow_dispatch"] is True
        assert isinstance(data.get("files",[]), list)

def test_generated_html_has_accessibility_and_canonical_markers():
    root=Path("data/website_factory/site")
    if root.exists():
        pages=list(root.glob("*.html"))
        for p in pages:
            s=p.read_text(encoding="utf-8")
            assert 'class="skip-link"' in s
            assert 'rel="canonical"' in s
            assert 'id="main-content"' in s
