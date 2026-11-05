import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DashboardEntryPointTests(unittest.TestCase):
    def test_static_landing_page_is_branded_for_autopost_ai(self):
        html = (ROOT / "index.html").read_text(encoding="utf-8")
        self.assertIn("<html lang=\"en\">", html)
        self.assertIn('name="viewport"', html)
        self.assertIn("AutoPost AI", html)
        self.assertIn("Meelad5815/autopost-ai", html)
        self.assertNotIn("NovaMart", html)

    def test_fastapi_dashboard_retains_core_controls_and_mobile_layout(self):
        html = (ROOT / "app" / "ui" / "index.html").read_text(encoding="utf-8")
        self.assertIn('name="viewport"', html)
        self.assertIn("AutoPost AI", html)
        for element_id in ("authState", "email", "password", "siteSelect", "jobsList", "postsList"):
            with self.subTest(element_id=element_id):
                self.assertIn(f'id="{element_id}"', html)
        self.assertIn("@media (max-width: 700px)", html)
        self.assertIn("const loadAll = async", html)


if __name__ == "__main__":
    unittest.main()
