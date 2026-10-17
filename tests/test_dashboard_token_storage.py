import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class DashboardTokenStorageTests(unittest.TestCase):
    def test_tokens_use_tab_scoped_session_storage(self):
        source = (ROOT / "app" / "ui" / "index.html").read_text(encoding="utf-8")
        self.assertIn('const getToken = () => sessionStorage.getItem(tokenKey) || "";', source)
        self.assertIn('sessionStorage.setItem(tokenKey, t)', source)
        self.assertIn('sessionStorage.setItem(refreshTokenKey, t)', source)

    def test_legacy_persistent_tokens_are_migrated_and_removed(self):
        source = (ROOT / "app" / "ui" / "index.html").read_text(encoding="utf-8")
        self.assertIn("const migrateLegacyToken = (key) =>", source)
        self.assertIn("localStorage.removeItem(key)", source)
        self.assertIn("migrateLegacyToken(tokenKey)", source)
        self.assertIn("migrateLegacyToken(refreshTokenKey)", source)

    def test_long_term_cookie_risk_is_documented(self):
        source = (ROOT / "app" / "ui" / "index.html").read_text(encoding="utf-8")
        self.assertIn("HttpOnly cookies are the long-term fix", source)


if __name__ == "__main__":
    unittest.main()
