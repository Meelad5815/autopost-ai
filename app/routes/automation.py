from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import get_current_user
from app.models import AutomationJob, PublishedPost, Site, User
from app.schemas import JobResponse
from app.services.automation import enqueue_now
from app.services.usage import check_limit


router = APIRouter(prefix='/automation', tags=['automation'])


@router.post('/{site_id}/run-now', response_model=JobResponse)
def run_now(site_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    site = db.query(Site).filter(Site.id == site_id, Site.user_id == user.id).first()
    if not site:
        raise HTTPException(status_code=404, detail='Site not found')

    allowed, used, limit = check_limit(db, user.id)
    if not allowed:
        raise HTTPException(status_code=403, detail=f'Plan limit reached ({used}/{limit})')

    return enqueue_now(db, site)


@router.get('/jobs', response_model=list[JobResponse])
def list_jobs(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return (
        db.query(AutomationJob)
        .join(Site, Site.id == AutomationJob.site_id)
        .filter(Site.user_id == user.id)
        .order_by(AutomationJob.created_at.desc())
        .limit(100)
        .all()
    )


@router.get('/posts')
def list_posts(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    posts = db.query(PublishedPost).join(Site, Site.id == PublishedPost.site_id).filter(Site.user_id == user.id).all()
    return [{'id': p.id, 'wp_post_id': p.wp_post_id, 'title': p.title, 'url': p.url, 'clicks': p.clicks, 'impressions': p.impressions} for p in posts]
