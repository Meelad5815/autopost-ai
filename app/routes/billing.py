from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import get_current_user
from app.models import Subscription, User
from app.services.billing import plan_limit, stripe_placeholder_checkout_url
from app.services.usage import check_limit


router = APIRouter(prefix='/billing', tags=['billing'])


@router.get('/usage')
def usage(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    sub = db.query(Subscription).filter(Subscription.user_id == user.id).first()
    allowed, used, limit = check_limit(db, user.id)
    return {'plan': sub.plan if sub else 'free', 'monthly_limit': limit, 'used': used, 'remaining': max(0, limit - used), 'allowed': allowed}


@router.post('/change-plan/{plan}')
def change_plan(plan: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    if plan not in {'free', 'pro', 'agency'}:
        raise HTTPException(status_code=400, detail='Invalid plan')
    sub = db.query(Subscription).filter(Subscription.user_id == user.id).first()
    if not sub:
        sub = Subscription(user_id=user.id, plan=plan, monthly_post_limit=plan_limit(plan))
        db.add(sub)
    else:
        sub.plan = plan
        sub.monthly_post_limit = plan_limit(plan)
    db.commit()
    return {'plan': plan, 'monthly_post_limit': plan_limit(plan), 'checkout_url': stripe_placeholder_checkout_url(plan)}
