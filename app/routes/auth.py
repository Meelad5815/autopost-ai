from urllib.parse import quote_plus

from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import create_access_token, hash_password, verify_password
from app.db import get_db
from app.deps import get_current_user
from app.models import Subscription, User
from app.schemas import LoginRequest, RegisterRequest, TokenResponse, UserMeResponse
from app.services.billing import plan_limit


router = APIRouter(prefix='/auth', tags=['auth'])
oauth = OAuth()

if settings.google_client_id and settings.google_client_secret:
    oauth.register(
        name='google',
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
        client_kwargs={'scope': 'openid email profile'},
    )


@router.post('/register', response_model=TokenResponse)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Email already exists')

    user = User(
        email=payload.email,
        password_hash=hash_password(payload.password),
        provider='local',
        role='user',
        is_active=True,
    )
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
    if not user or user.provider != 'local':
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid credentials')
    if not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid credentials')
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Account suspended')
    return TokenResponse(access_token=create_access_token(str(user.id)))


@router.get('/google/login')
async def google_login(request: Request):
    if 'google' not in oauth._clients:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail='Google OAuth not configured')
    redirect_uri = request.url_for('google_callback')
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get('/google/callback')
async def google_callback(request: Request, db: Session = Depends(get_db)):
    if 'google' not in oauth._clients:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail='Google OAuth not configured')

    token = await oauth.google.authorize_access_token(request)
    userinfo = token.get('userinfo')
    if not userinfo:
        userinfo = await oauth.google.userinfo(token=token)
    email = userinfo.get('email') if userinfo else None
    if not email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Google account email not available')

    user = db.query(User).filter(User.email == email).first()
    if not user:
        user = User(
            email=email,
            password_hash='',
            provider='google',
            role='user',
            is_active=True,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        db.add(Subscription(user_id=user.id, plan='free', monthly_post_limit=plan_limit('free')))
        db.commit()

    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Account suspended')

    access_token = create_access_token(str(user.id))
    if settings.auth_frontend_success_url:
        sep = '&' if '?' in settings.auth_frontend_success_url else '?'
        url = f"{settings.auth_frontend_success_url}{sep}access_token={quote_plus(access_token)}"
        return RedirectResponse(url=url, status_code=status.HTTP_302_FOUND)
    return TokenResponse(access_token=access_token)


@router.get('/me', response_model=UserMeResponse)
def me(user: User = Depends(get_current_user)):
    return UserMeResponse(
        id=user.id,
        email=user.email,
        role=user.role,
        provider=user.provider,
        is_active=user.is_active,
    )
