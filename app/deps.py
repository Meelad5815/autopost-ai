from urllib.parse import urlsplit

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.security import decode_token
from app.db import get_db
from app.models import User


bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    # Explicit bearer tokens remain supported for API clients. The dashboard uses
    # HttpOnly cookies and must pass a same-origin check for state-changing requests.
    if credentials is not None:
        raw_token = credentials.credentials
    else:
        raw_token = request.cookies.get("autopost_access", "")
        if not raw_token:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
        if request.method.upper() not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            host = request.headers.get("host", "")
            parsed_origin = urlsplit(origin or "")
            if not origin or parsed_origin.scheme not in {"http", "https"} or parsed_origin.netloc.lower() != host.lower():
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cookie-authenticated requests must be same-origin")
    user_id = decode_token(raw_token)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid token')
    user = db.query(User).filter(User.id == int(user_id)).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='User not active')
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != 'admin':
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Admin access required')
    return user
