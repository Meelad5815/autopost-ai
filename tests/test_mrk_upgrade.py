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
