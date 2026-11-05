import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class SecurityHeadersTests(unittest.TestCase):
    def test_middleware_sets_baseline_headers(self):
        source = (ROOT / "app" / "middleware" / "security_headers.py").read_text(encoding="utf-8")
        for header in (
            "x-content-type-options",
            "x-frame-options",
            "referrer-policy",
            "permissions-policy",
        ):
            with self.subTest(header=header):
                self.assertIn(header, source)

    def test_middleware_is_registered_on_application(self):
        source = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
        self.assertIn("from app.middleware.security_headers import SecurityHeadersMiddleware", source)
        self.assertIn("app.add_middleware(SecurityHeadersMiddleware)", source)


if __name__ == "__main__":
    unittest.main()
