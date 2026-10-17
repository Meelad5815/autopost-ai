from datetime import datetime
import ipaddress
from urllib.parse import urlsplit

from pydantic import BaseModel, EmailStr, Field, field_validator


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str | None = None
    token_type: str = 'bearer'


class UserMeResponse(BaseModel):
    id: int
    email: EmailStr
    role: str
    provider: str
    email_verified: bool
    is_active: bool


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(min_length=8)


class VerifyEmailRequest(BaseModel):
    email: EmailStr


class VerifyEmailConfirmRequest(BaseModel):
    token: str


class SiteCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    wp_url: str = Field(min_length=8, max_length=500)
    wp_user: str = Field(min_length=1, max_length=120)
    wp_app_password: str = Field(min_length=1, max_length=500)
    openai_api_key: str = Field(min_length=1, max_length=500)
    niche: str = Field(default='', max_length=255)
    frequency_hours: int = Field(default=24, ge=1, le=168)

    @field_validator('wp_url')
    @classmethod
    def validate_wordpress_url(cls, value: str) -> str:
        value = value.strip()
        parsed = urlsplit(value)
        if parsed.scheme.lower() != 'https' or not parsed.hostname:
            raise ValueError('WordPress URL must use HTTPS and include a hostname')
        if parsed.username or parsed.password:
            raise ValueError('WordPress URL must not contain embedded credentials')
        if parsed.query or parsed.fragment:
            raise ValueError('WordPress URL must not contain a query or fragment')

        host = parsed.hostname.rstrip('.').lower()
        if host in {'localhost', 'localhost.localdomain'} or host.endswith(('.localhost', '.local', '.internal', '.test', '.invalid')):
            raise ValueError('WordPress URL must use a public hostname')
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        if address is not None and (not address.is_global or address.is_multicast or address.is_unspecified):
            raise ValueError('WordPress URL must use a public IP address')
        return value


class SiteResponse(BaseModel):
    id: int
    name: str
    wp_url: str
    niche: str
    frequency_hours: int
    is_active: bool

    class Config:
        from_attributes = True


class UsageResponse(BaseModel):
    plan: str
    monthly_limit: int
    used: int


class JobResponse(BaseModel):
    id: int
    status: str
    scheduled_for: datetime | None
    started_at: datetime | None
    finished_at: datetime | None
    error: str | None
