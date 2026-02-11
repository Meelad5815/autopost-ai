from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models import Subscription, UsageCounter


def current_year_month() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m')


def get_or_create_usage(db: Session, user_id: int) -> UsageCounter:
    ym = current_year_month()
    usage = db.query(UsageCounter).filter(UsageCounter.user_id == user_id, UsageCounter.year_month == ym).first()
    if usage:
        return usage
    usage = UsageCounter(user_id=user_id, year_month=ym, posts_generated=0)
    db.add(usage)
    db.commit()
    db.refresh(usage)
    return usage


def check_limit(db: Session, user_id: int) -> tuple[bool, int, int]:
    usage = get_or_create_usage(db, user_id)
    sub = db.query(Subscription).filter(Subscription.user_id == user_id).first()
    limit = sub.monthly_post_limit if sub else 10
    return usage.posts_generated < limit, usage.posts_generated, limit


def increment_usage(db: Session, user_id: int, amount: int = 1) -> None:
    usage = get_or_create_usage(db, user_id)
    usage.posts_generated += amount
    db.commit()
