from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import get_current_user
from app.models import Site, User
from app.schemas import SiteCreateRequest, SiteResponse
from app.core.secrets import SecretConfigurationError, encrypt_secret


router = APIRouter(prefix='/sites', tags=['sites'])


@router.post('', response_model=SiteResponse)
def create_site(payload: SiteCreateRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    try:
        encrypted_wp_password = encrypt_secret(payload.wp_app_password)
        encrypted_ai_key = encrypt_secret(payload.openai_api_key)
    except SecretConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None

    site = Site(
        user_id=user.id,
        name=payload.name,
        wp_url=payload.wp_url,
        wp_user=payload.wp_user,
        wp_app_password_enc=encrypted_wp_password,
        openai_api_key_enc=encrypted_ai_key,
        niche=payload.niche,
        frequency_hours=payload.frequency_hours,
        is_active=False,
    )
    db.add(site)
    db.commit()
    db.refresh(site)
    return site


@router.get('', response_model=list[SiteResponse])
def list_sites(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return db.query(Site).filter(Site.user_id == user.id).all()


@router.post('/{site_id}/start')
def start_site(site_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    site = db.query(Site).filter(Site.id == site_id, Site.user_id == user.id).first()
    if not site:
        raise HTTPException(status_code=404, detail='Site not found')
    site.is_active = True
    db.commit()
    return {'message': 'automation_started', 'site_id': site_id}


@router.post('/{site_id}/stop')
def stop_site(site_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    site = db.query(Site).filter(Site.id == site_id, Site.user_id == user.id).first()
    if not site:
        raise HTTPException(status_code=404, detail='Site not found')
    site.is_active = False
    db.commit()
    return {'message': 'automation_stopped', 'site_id': site_id}
