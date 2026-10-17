import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class DashboardCookieSessionTests(unittest.TestCase):
    def test_dashboard_does_not_persist_auth_tokens_in_web_storage(self):
        source = (ROOT / "app" / "ui" / "index.html").read_text(encoding="utf-8")
        self.assertIn('localStorage.removeItem(tokenKey)', source)
        self.assertIn('localStorage.removeItem(refreshTokenKey)', source)
        self.assertIn('sessionStorage.removeItem(tokenKey)', source)
        self.assertIn('credentials: "same-origin"', source)
        self.assertNotIn('sessionStorage.setItem(refreshTokenKey, t)', source)

    def test_dashboard_login_and_register_use_cookie_session_endpoints(self):
        source = (ROOT / "app" / "ui" / "index.html").read_text(encoding="utf-8")
        self.assertIn('/auth/session/login', source)
        self.assertIn('/auth/session/register', source)
        self.assertIn('/auth/session/refresh', source)
        self.assertIn('/auth/session/logout', source)

    def test_auth_routes_set_httponly_secure_production_cookies(self):
        source = (ROOT / "app" / "routes" / "auth.py").read_text(encoding="utf-8")
        self.assertIn('httponly=True', source)
        self.assertIn('samesite="lax"', source)
        self.assertIn('secure=secure', source)
        self.assertIn('ACCESS_COOKIE_NAME = "autopost_access"', source)
        self.assertIn('REFRESH_COOKIE_NAME = "autopost_refresh"', source)

    def test_cookie_auth_rejects_cross_origin_state_changes(self):
        deps = (ROOT / "app" / "deps.py").read_text(encoding="utf-8")
        auth = (ROOT / "app" / "routes" / "auth.py").read_text(encoding="utf-8")
        self.assertIn('Cookie-authenticated requests must be same-origin', deps)
        self.assertIn('_require_same_origin(request)', auth)

    def test_google_oauth_sets_cookies_without_redirecting_tokens(self):
        auth = (ROOT / "app" / "routes" / "auth.py").read_text(encoding="utf-8")
        ui = (ROOT / "app" / "ui" / "index.html").read_text(encoding="utf-8")
        callback = auth[auth.index("async def google_callback"):]
        self.assertIn('_set_session_cookies(response, pair)', callback)
        self.assertNotIn('access_token={quote_plus', callback)
        self.assertIn('window.history.replaceState({}, "", clean)', ui)

    def test_long_term_risk_and_cookie_controls_are_documented(self):
        docs = (ROOT / "docs" / "DEPLOYMENT_SECURITY.md").read_text(encoding="utf-8")
        self.assertIn("HttpOnly", docs)
        self.assertIn("CSRF", docs)


if __name__ == "__main__":
    unittest.main()
