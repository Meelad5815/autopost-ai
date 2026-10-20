import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class FirstSetupSecurityTests(unittest.TestCase):
    def test_public_bootstrap_is_opt_in_and_one_time(self):
        source = (ROOT / "app" / "routes" / "ui.py").read_text(encoding="utf-8")
        start = source.index('@router.post("/ui/first-setup")')
        setup = source[start:]
        self.assertIn('os.getenv("ALLOW_INITIAL_SETUP"', setup)
        self.assertIn('raise HTTPException(status_code=404, detail="Initial setup is disabled")', setup)
        self.assertIn('db.query(User.id).first() is not None', setup)
        self.assertIn('status_code=409', setup)
        self.assertIn('use /auth/login', setup)
        self.assertIn('_require_same_origin(request)', setup)

    def test_initial_site_password_is_encrypted_before_persistence(self):
        source = (ROOT / "app" / "routes" / "ui.py").read_text(encoding="utf-8")
        start = source.index('@router.post("/ui/first-setup")')
        setup = source[start:]
        self.assertIn("encrypted_wp_password = encrypt_secret(wp_app_password)", setup)
        self.assertIn('wp_app_password_enc=site_values["wp_app_password_enc"]', setup)
        self.assertNotIn("wp_app_password_enc=wp_app_password", setup)

    def test_initial_setup_uses_http_only_cookie_session(self):
        source = (ROOT / "app" / "routes" / "ui.py").read_text(encoding="utf-8")
        start = source.index('@router.post("/ui/first-setup")')
        setup = source[start:]
        self.assertIn("_set_session_cookies(response, _issue_token_pair(db, user))", setup)
        self.assertNotIn('"access_token": create_access_token', setup)


if __name__ == "__main__":
    unittest.main()
