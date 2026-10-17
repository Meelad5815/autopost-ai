import unittest

from pydantic import ValidationError

from app.schemas import SiteCreateRequest


def valid_payload(**overrides):
    payload = {
        "name": "Primary site",
        "wp_url": "https://example.com",
        "wp_user": "publisher",
        "wp_app_password": "example-password",
        "openai_api_key": "example-key",
    }
    payload.update(overrides)
    return payload


class SiteInputValidationTests(unittest.TestCase):
    def test_accepts_public_https_wordpress_url(self):
        site = SiteCreateRequest(**valid_payload(wp_url="https://example.com/blog/"))
        self.assertEqual(site.wp_url, "https://example.com/blog/")

    def test_rejects_unsafe_wordpress_urls(self):
        urls = [
            "http://example.com",
            "https://localhost",
            "https://service.local",
            "https://127.0.0.1",
            "https://10.0.0.8",
            "https://192.168.1.20",
            "https://169.254.169.254/latest/meta-data",
            "https://user:password@example.com",
            "https://example.com/?token=secret",
            "https://example.com/#fragment",
        ]
        for url in urls:
            with self.subTest(url=url):
                with self.assertRaises(ValidationError):
                    SiteCreateRequest(**valid_payload(wp_url=url))

    def test_rejects_out_of_range_frequency(self):
        for hours in (0, -1, 169):
            with self.subTest(hours=hours):
                with self.assertRaises(ValidationError):
                    SiteCreateRequest(**valid_payload(frequency_hours=hours))

    def test_accepts_frequency_within_one_week(self):
        self.assertEqual(
            SiteCreateRequest(**valid_payload(frequency_hours=168)).frequency_hours,
            168,
        )


if __name__ == "__main__":
    unittest.main()
