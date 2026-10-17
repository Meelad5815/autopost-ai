import os
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from app.routes.ui import _require_terminal_enabled


class AdminTerminalGateTests(unittest.TestCase):
    def test_admin_terminal_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(HTTPException) as caught:
                _require_terminal_enabled()
        self.assertEqual(caught.exception.status_code, 404)

    def test_admin_terminal_requires_explicit_opt_in(self):
        for value in ("1", "true", "TRUE", "yes"):
            with self.subTest(value=value), patch.dict(os.environ, {"ENABLE_ADMIN_TERMINAL": value}):
                _require_terminal_enabled()


if __name__ == "__main__":
    unittest.main()
