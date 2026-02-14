import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    app_name: str = 'Autopost SaaS Platform'
    environment: str = 'development'
    debug: bool = True

    database_url: str = 'sqlite:///./saas.db'
    jwt_secret_key: str = 'change-me'
    jwt_algorithm: str = 'HS256'
    jwt_expire_minutes: int = 60 * 24
    jwt_refresh_expire_minutes: int = 60 * 24 * 30
    google_client_id: str = ''
    google_client_secret: str = ''
    auth_frontend_success_url: str = ''
    auth_base_url: str = ''
    password_reset_expire_minutes: int = 30
    email_verify_expire_minutes: int = 60 * 24

    worker_poll_seconds: int = 15

    stripe_secret_key: str = ''
    stripe_webhook_secret: str = ''

    # Plan limits per month
    free_posts_limit: int = 10
    pro_posts_limit: int = 200
    agency_posts_limit: int = 2000


def _resolve_database_url() -> str:
    explicit = os.getenv("DATABASE_URL", "").strip()
    is_serverless = any(
        os.getenv(k, "").strip()
        for k in ["VERCEL", "VERCEL_ENV", "AWS_REGION", "NOW_REGION"]
    )
    if explicit:
        # In serverless runtimes, relative sqlite paths are read-only.
        if is_serverless and explicit.startswith("sqlite:///./"):
            return "sqlite:////tmp/saas.db"
        return explicit
    if is_serverless:
        return "sqlite:////tmp/saas.db"
    return "sqlite:///./saas.db"


settings = Settings(database_url=_resolve_database_url())
