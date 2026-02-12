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

    worker_poll_seconds: int = 15

    stripe_secret_key: str = ''
    stripe_webhook_secret: str = ''

    # Plan limits per month
    free_posts_limit: int = 10
    pro_posts_limit: int = 200
    agency_posts_limit: int = 2000


def _resolve_database_url() -> str:
    explicit = os.getenv("DATABASE_URL", "").strip()
    if explicit:
        return explicit
    if os.getenv("VERCEL"):
        return "sqlite:////tmp/saas.db"
    return "sqlite:///./saas.db"


settings = Settings(database_url=_resolve_database_url())
