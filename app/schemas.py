from datetime import datetime
from pydantic import BaseModel, EmailStr


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = 'bearer'


class SiteCreateRequest(BaseModel):
    name: str
    wp_url: str
    wp_user: str
    wp_app_password: str
    openai_api_key: str
    niche: str = ''
    frequency_hours: int = 24


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
