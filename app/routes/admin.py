from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import require_admin
from app.models import Site, Subscription, User
from app.services.automation import summarize_health


router = APIRouter(prefix='/admin', tags=['admin'])


@router.get('/users')
def list_users(db: Session = Depends(get_db), _admin=Depends(require_admin)):
    users = db.query(User).all()
    response = []
    for user in users:
        sub = db.query(Subscription).filter(Subscription.user_id == user.id).first()
        sites = db.query(Site).filter(Site.user_id == user.id).count()
        response.append({'id': user.id, 'email': user.email, 'role': user.role, 'is_active': user.is_active, 'plan': sub.plan if sub else 'free', 'sites': sites})
    return response


@router.post('/users/{user_id}/suspend')
def suspend_user(user_id: int, db: Session = Depends(get_db), _admin=Depends(require_admin)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail='User not found')
    user.is_active = False
    db.commit()
    return {'status': 'suspended', 'user_id': user_id}


@router.post('/users/{user_id}/limits/{limit}')
def set_limit(user_id: int, limit: int, db: Session = Depends(get_db), _admin=Depends(require_admin)):
    sub = db.query(Subscription).filter(Subscription.user_id == user_id).first()
    if not sub:
        raise HTTPException(status_code=404, detail='Subscription not found')
    sub.monthly_post_limit = limit
    db.commit()
    return {'user_id': user_id, 'new_limit': limit}


@router.get('/health')
def health(db: Session = Depends(get_db), _admin=Depends(require_admin)):
    return summarize_health(db)
