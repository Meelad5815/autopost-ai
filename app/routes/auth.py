from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus

from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token_claims,
    generate_opaque_token,
    hash_password,
    hash_token,
    verify_password,
)
from app.db import get_db
from app.deps import get_current_user
from app.models import AuthToken, Subscription, User
from app.schemas import (
    ForgotPasswordRequest,
    LoginRequest,
    RefreshTokenRequest,
    RegisterRequest,
    ResetPasswordRequest,
    TokenResponse,
    UserMeResponse,
    VerifyEmailConfirmRequest,
    VerifyEmailRequest,
)
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


def _issue_token_pair(db: Session, user: User) -> TokenResponse:
    access_token = create_access_token(str(user.id))
    refresh_token = create_refresh_token(str(user.id))
    db.add(
        AuthToken(
            user_id=user.id,
            token_hash=hash_token(refresh_token),
            token_type='refresh',
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_refresh_expire_minutes),
        )
    )
    db.commit()
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post('/register', response_model=TokenResponse)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Email already exists')

    user = User(
        email=payload.email,
        password_hash=hash_password(payload.password),
        provider='local',
        email_verified=False,
        role='user',
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    db.add(Subscription(user_id=user.id, plan='free', monthly_post_limit=plan_limit('free')))
    db.commit()

    verify_token = generate_opaque_token()
    db.add(
        AuthToken(
            user_id=user.id,
            token_hash=hash_token(verify_token),
            token_type='email_verify',
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=settings.email_verify_expire_minutes),
        )
    )
    db.commit()

    resp = _issue_token_pair(db, user).model_dump()
    if settings.debug:
        resp['email_verify_token_preview'] = verify_token
    return TokenResponse(**{k: v for k, v in resp.items() if k in {'access_token', 'refresh_token', 'token_type'}})


@router.post('/login', response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or user.provider != 'local':
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid credentials')
    if not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid credentials')
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Account suspended')
    return _issue_token_pair(db, user)


@router.post('/refresh', response_model=TokenResponse)
def refresh_token(payload: RefreshTokenRequest, db: Session = Depends(get_db)):
    claims = decode_token_claims(payload.refresh_token)
    if not claims or claims.get('typ') != 'refresh':
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid refresh token')

    user_id = claims.get('sub')
    token_hash_value = hash_token(payload.refresh_token)
    stored = (
        db.query(AuthToken)
        .filter(AuthToken.token_hash == token_hash_value, AuthToken.token_type == 'refresh', AuthToken.used_at.is_(None))
        .first()
    )
    if not stored or stored.expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Refresh token expired')

    user = db.query(User).filter(User.id == int(user_id)).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='User not active')

    stored.used_at = datetime.now(timezone.utc)
    db.commit()
    return _issue_token_pair(db, user)


@router.post('/forgot-password')
def forgot_password(payload: ForgotPasswordRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or user.provider != 'local':
        return {'status': 'ok'}

    raw_token = generate_opaque_token()
    db.add(
        AuthToken(
            user_id=user.id,
            token_hash=hash_token(raw_token),
            token_type='password_reset',
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=settings.password_reset_expire_minutes),
        )
    )
    db.commit()

    response = {'status': 'ok'}
    if settings.debug:
        response['reset_token_preview'] = raw_token
    return response


@router.post('/reset-password')
def reset_password(payload: ResetPasswordRequest, db: Session = Depends(get_db)):
    token_hash_value = hash_token(payload.token)
    entry = (
        db.query(AuthToken)
        .filter(
            AuthToken.token_hash == token_hash_value,
            AuthToken.token_type == 'password_reset',
            AuthToken.used_at.is_(None),
        )
        .first()
    )
    if not entry or entry.expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Invalid or expired reset token')

    user = db.query(User).filter(User.id == entry.user_id).first()
    if not user or user.provider != 'local':
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Invalid reset target')

    user.password_hash = hash_password(payload.new_password)
    entry.used_at = datetime.now(timezone.utc)
    db.commit()
    return {'status': 'ok'}


@router.post('/verify-email/request')
def verify_email_request(payload: VerifyEmailRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user:
        return {'status': 'ok'}

    raw_token = generate_opaque_token()
    db.add(
        AuthToken(
            user_id=user.id,
            token_hash=hash_token(raw_token),
            token_type='email_verify',
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=settings.email_verify_expire_minutes),
        )
    )
    db.commit()

    response = {'status': 'ok'}
    if settings.debug:
        response['email_verify_token_preview'] = raw_token
    return response


@router.post('/verify-email/confirm')
def verify_email_confirm(payload: VerifyEmailConfirmRequest, db: Session = Depends(get_db)):
    token_hash_value = hash_token(payload.token)
    entry = (
        db.query(AuthToken)
        .filter(AuthToken.token_hash == token_hash_value, AuthToken.token_type == 'email_verify', AuthToken.used_at.is_(None))
        .first()
    )
    if not entry or entry.expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Invalid or expired verification token')

    user = db.query(User).filter(User.id == entry.user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Invalid verification target')
    user.email_verified = True
    entry.used_at = datetime.now(timezone.utc)
    db.commit()
    return {'status': 'ok'}


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
            email_verified=True,
            role='user',
            is_active=True,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        db.add(Subscription(user_id=user.id, plan='free', monthly_post_limit=plan_limit('free')))
        db.commit()
    elif user.provider == 'local' and not user.email_verified:
        user.email_verified = True
        db.commit()

    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Account suspended')

    pair = _issue_token_pair(db, user)
    if settings.auth_frontend_success_url:
        sep = '&' if '?' in settings.auth_frontend_success_url else '?'
        url = (
            f"{settings.auth_frontend_success_url}{sep}"
            f"access_token={quote_plus(pair.access_token)}&refresh_token={quote_plus(pair.refresh_token or '')}"
        )
        return RedirectResponse(url=url, status_code=status.HTTP_302_FOUND)
    return pair


@router.get('/me', response_model=UserMeResponse)
def me(user: User = Depends(get_current_user)):
    return UserMeResponse(
        id=user.id,
        email=user.email,
        role=user.role,
        provider=user.provider,
        email_verified=user.email_verified,
        is_active=user.is_active,
    )
