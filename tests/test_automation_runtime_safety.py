import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class AutomationRuntimeSafetyTests(unittest.TestCase):
    def test_publishing_subprocess_has_timeout_and_redacts_secrets(self):
        source = (ROOT / "app" / "services" / "automation.py").read_text(encoding="utf-8")
        self.assertIn("timeout=900", source)
        self.assertIn("subprocess.TimeoutExpired", source)
        self.assertIn("def safe_output(value) -> str:", source)
        self.assertIn("value.replace(secret, '[REDACTED]')", source)
        self.assertIn("Unable to start publishing process", source)

    def test_job_errors_are_bounded_before_persistence(self):
        source = (ROOT / "app" / "services" / "automation.py").read_text(encoding="utf-8")
        self.assertIn("return str(value)[-2000:]", source)
        self.assertIn("job.finished_at = datetime.now(timezone.utc)", source)

    def test_success_requires_confirmed_wordpress_post(self):
        automation = (ROOT / "app" / "services" / "automation.py").read_text(encoding="utf-8")
        engine = (ROOT / "autopost.py").read_text(encoding="utf-8")
        self.assertIn("AUTOPOST_RESULT_JSON:", engine)
        self.assertIn("item.get('status') == 'published'", automation)
        self.assertIn("item.get('post_id') > 0", automation)
        self.assertIn("item.get('url').startswith(('https://', 'http://'))", automation)
        self.assertIn("without a confirmed WordPress post", automation)
        self.assertNotIn("wp_post_id=0, title='Generated Post', url=''", automation)


if __name__ == "__main__":
    unittest.main()
