import pytest
from fastapi import HTTPException

from app.routes.ui import _require_terminal_enabled


def test_admin_terminal_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ENABLE_ADMIN_TERMINAL", raising=False)
    with pytest.raises(HTTPException) as exc:
        _require_terminal_enabled()
    assert exc.value.status_code == 404


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes"])
def test_admin_terminal_requires_explicit_opt_in(monkeypatch, value):
    monkeypatch.setenv("ENABLE_ADMIN_TERMINAL", value)
    _require_terminal_enabled()
