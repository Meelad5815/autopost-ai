import pytest
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


def test_site_request_accepts_public_https_wordpress_url():
    site = SiteCreateRequest(**valid_payload(wp_url="https://example.com/blog/"))
    assert site.wp_url == "https://example.com/blog/"


@pytest.mark.parametrize(
    "url",
    [
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
    ],
)
def test_site_request_rejects_unsafe_wordpress_urls(url):
    with pytest.raises(ValidationError):
        SiteCreateRequest(**valid_payload(wp_url=url))


@pytest.mark.parametrize("hours", [0, -1, 169])
def test_site_request_rejects_out_of_range_frequency(hours):
    with pytest.raises(ValidationError):
        SiteCreateRequest(**valid_payload(frequency_hours=hours))


def test_site_request_accepts_frequency_within_one_week():
    assert SiteCreateRequest(**valid_payload(frequency_hours=168)).frequency_hours == 168
