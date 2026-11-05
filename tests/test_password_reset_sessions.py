import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class PasswordResetSessionRevocationTests(unittest.TestCase):
    def test_password_reset_revokes_all_unused_refresh_tokens(self):
        source = (ROOT / "app" / "routes" / "auth.py").read_text(encoding="utf-8")
        reset = source[source.index("@router.post('/reset-password')"):source.index("@router.post('/verify-email/request')")]
        self.assertIn("AuthToken.user_id == user.id", reset)
        self.assertIn("AuthToken.token_type == 'refresh'", reset)
        self.assertIn("AuthToken.used_at.is_(None)", reset)
        self.assertIn("AuthToken.used_at: now", reset)


if __name__ == "__main__":
    unittest.main()
