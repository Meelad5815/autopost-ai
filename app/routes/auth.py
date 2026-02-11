from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.security import create_access_token, hash_password, verify_password
from app.db import get_db
from app.models import Subscription, User
from app.schemas import LoginRequest, RegisterRequest, TokenResponse
from app.services.billing import plan_limit


router = APIRouter(prefix='/auth', tags=['auth'])


@router.post('/register', response_model=TokenResponse)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Email already exists')

    user = User(email=payload.email, password_hash=hash_password(payload.password), role='user', is_active=True)
    db.add(user)
    db.commit()
    db.refresh(user)

    db.add(Subscription(user_id=user.id, plan='free', monthly_post_limit=plan_limit('free')))
    db.commit()

    token = create_access_token(str(user.id))
    return TokenResponse(access_token=token)


@router.post('/login', response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid credentials')
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Account suspended')
    return TokenResponse(access_token=create_access_token(str(user.id)))
