from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_shared_website_factory_routes_are_admin_only():
    source = (ROOT / "app" / "routes" / "website_factory.py").read_text(encoding="utf-8")
    assert "from app.deps import require_admin" in source
    assert source.count("Depends(require_admin)") == 2
    assert "except Exception:" not in source
    assert "def get_current_user(): return True" not in source


def test_paid_plans_cannot_be_enabled_without_checkout():
    source = (ROOT / "app" / "routes" / "billing.py").read_text(encoding="utf-8")
    assert "if plan in {'pro', 'agency'}:" in source
    assert "status_code=503" in source
    assert "Paid plan checkout is not configured yet." in source
