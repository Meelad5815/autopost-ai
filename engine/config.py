import os
from dataclasses import dataclass


class ConfigError(Exception):
    pass


def env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() == "true"


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(f"Missing required environment variable: {name}")
    return value


@dataclass
class Config:
    wp_url: str
    wp_user: str
    wp_app_password: str
    openai_api_key: str
    openai_model: str
    request_timeout: int
    max_publish_retries: int
    posts_per_run: int
    post_status: str
    schedule_interval_minutes: int

    # intelligence / strategy
    niches_per_run: int
    calendar_days: int
    refresh_age_days: int

    # feature toggles
    enable_niche_intelligence: bool
    enable_competitor_analysis: bool
    enable_serp_simulation: bool
    enable_content_calendar: bool
    enable_topic_clustering: bool
    enable_content_refresh: bool
    enable_affiliate_automation: bool
    enable_lead_gen: bool
    enable_conversion_optimization: bool
    enable_social_syndication: bool
    enable_short_content: bool
    enable_performance_tracking: bool
    enable_learning_loop: bool


def load_config() -> Config:
    return Config(
        wp_url=require_env("WP_URL"),
        wp_user=require_env("WP_USER"),
        wp_app_password=require_env("WP_APP_PASSWORD"),
        openai_api_key=require_env("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        request_timeout=int(os.getenv("REQUEST_TIMEOUT", "60")),
        max_publish_retries=int(os.getenv("MAX_PUBLISH_RETRIES", "3")),
        posts_per_run=max(1, int(os.getenv("POSTS_PER_RUN", "1"))),
        post_status=os.getenv("POST_STATUS", "publish").lower().strip(),
        schedule_interval_minutes=int(os.getenv("SCHEDULE_INTERVAL_MINUTES", "120")),
        niches_per_run=max(3, int(os.getenv("NICHES_PER_RUN", "6"))),
        calendar_days=int(os.getenv("CALENDAR_DAYS", "30")),
        refresh_age_days=int(os.getenv("REFRESH_AGE_DAYS", "120")),
        enable_niche_intelligence=env_bool("ENABLE_NICHE_INTELLIGENCE", "true"),
        enable_competitor_analysis=env_bool("ENABLE_COMPETITOR_ANALYSIS", "true"),
        enable_serp_simulation=env_bool("ENABLE_SERP_SIMULATION", "true"),
        enable_content_calendar=env_bool("ENABLE_CONTENT_CALENDAR", "true"),
        enable_topic_clustering=env_bool("ENABLE_TOPIC_CLUSTERING", "true"),
        enable_content_refresh=env_bool("ENABLE_CONTENT_REFRESH", "true"),
        enable_affiliate_automation=env_bool("ENABLE_AFFILIATE_AUTOMATION", "true"),
        enable_lead_gen=env_bool("ENABLE_LEAD_GEN", "true"),
        enable_conversion_optimization=env_bool("ENABLE_CONVERSION_OPTIMIZATION", "true"),
        enable_social_syndication=env_bool("ENABLE_SOCIAL_SYNDICATION", "true"),
        enable_short_content=env_bool("ENABLE_SHORT_CONTENT", "true"),
        enable_performance_tracking=env_bool("ENABLE_PERFORMANCE_TRACKING", "true"),
        enable_learning_loop=env_bool("ENABLE_LEARNING_LOOP", "true"),
    )
