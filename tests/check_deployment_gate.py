"""Assert that candidate workflows never deploy or modify production."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "data" / "upgrade_candidate_execution.json"


def main():
    if not ARTIFACT.is_file():
        print("No candidate execution artifact; deployment gate check skipped.")
        return
    with ARTIFACT.open(encoding="utf-8") as stream:
        data = json.load(stream)
    assert data.get("production_changed") is False, data
    assert data.get("deployment_allowed") is False, data
    print("Deployment gate verified.")


if __name__ == "__main__":
    main()
