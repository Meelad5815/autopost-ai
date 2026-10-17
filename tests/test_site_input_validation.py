import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class SiteInputValidationTests(unittest.TestCase):
    def test_site_schema_requires_https_and_public_hostname(self):
        source = (ROOT / "app" / "schemas.py").read_text(encoding="utf-8")
        self.assertIn("@field_validator('wp_url')", source)
        self.assertIn("parsed.scheme.lower() != 'https'", source)
        self.assertIn("if parsed.username or parsed.password:", source)
        self.assertIn("if parsed.query or parsed.fragment:", source)
        self.assertIn("address.is_global", source)
        self.assertIn("host.endswith(('.localhost', '.local', '.internal', '.test', '.invalid'))", source)

    def test_site_schema_bounds_fields_and_schedule_frequency(self):
        source = (ROOT / "app" / "schemas.py").read_text(encoding="utf-8")
        self.assertIn("name: str = Field(min_length=1, max_length=120)", source)
        self.assertIn("frequency_hours: int = Field(default=24, ge=1, le=168)", source)


if __name__ == "__main__":
    unittest.main()
