from app.core.config import settings


PLAN_LIMITS = {
    'free': settings.free_posts_limit,
    'pro': settings.pro_posts_limit,
    'agency': settings.agency_posts_limit,
}


def plan_limit(plan: str) -> int:
    return PLAN_LIMITS.get(plan, settings.free_posts_limit)


def stripe_placeholder_checkout_url(plan: str) -> str:
    # Placeholder hook. Replace with stripe.checkout.Session.create(...) in production.
    return f'https://billing.example.com/checkout?plan={plan}'
