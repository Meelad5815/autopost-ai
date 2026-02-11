from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc)
    )


class User(Base, TimestampMixin):
    __tablename__ = 'users'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20), default='user')  # admin/user
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    subscription: Mapped['Subscription'] = relationship(back_populates='user', uselist=False)
    sites: Mapped[list['Site']] = relationship(back_populates='user')


class Subscription(Base, TimestampMixin):
    __tablename__ = 'subscriptions'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'), unique=True)
    plan: Mapped[str] = mapped_column(String(20), default='free')  # free/pro/agency
    stripe_customer_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    stripe_subscription_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    monthly_post_limit: Mapped[int] = mapped_column(Integer, default=10)

    user: Mapped['User'] = relationship(back_populates='subscription')


class Site(Base, TimestampMixin):
    __tablename__ = 'sites'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'), index=True)
    name: Mapped[str] = mapped_column(String(120))
    wp_url: Mapped[str] = mapped_column(String(500))
    wp_user: Mapped[str] = mapped_column(String(120))
    wp_app_password_enc: Mapped[str] = mapped_column(String(500))
    openai_api_key_enc: Mapped[str] = mapped_column(String(500))
    niche: Mapped[str] = mapped_column(String(255), default='')
    frequency_hours: Mapped[int] = mapped_column(Integer, default=24)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)

    user: Mapped['User'] = relationship(back_populates='sites')
    jobs: Mapped[list['AutomationJob']] = relationship(back_populates='site')


class AutomationJob(Base, TimestampMixin):
    __tablename__ = 'automation_jobs'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    site_id: Mapped[int] = mapped_column(ForeignKey('sites.id'), index=True)
    status: Mapped[str] = mapped_column(String(30), default='queued')  # queued/running/success/failed
    scheduled_for: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    site: Mapped['Site'] = relationship(back_populates='jobs')


class PublishedPost(Base, TimestampMixin):
    __tablename__ = 'published_posts'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    site_id: Mapped[int] = mapped_column(ForeignKey('sites.id'), index=True)
    job_id: Mapped[int | None] = mapped_column(ForeignKey('automation_jobs.id'), nullable=True)
    wp_post_id: Mapped[int] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(String(500))
    url: Mapped[str] = mapped_column(String(500))
    impressions: Mapped[int] = mapped_column(Integer, default=0)
    clicks: Mapped[int] = mapped_column(Integer, default=0)


class UsageCounter(Base):
    __tablename__ = 'usage_counters'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'), index=True)
    year_month: Mapped[str] = mapped_column(String(7), index=True)  # YYYY-MM
    posts_generated: Mapped[int] = mapped_column(Integer, default=0)
