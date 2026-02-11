import json
import os
import subprocess
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from app.models import AutomationJob, PublishedPost, Site
from app.services.usage import check_limit, increment_usage


def decrypt_secret(value: str) -> str:
    # Placeholder for real KMS/Vault encryption
    return value


def schedule_next_job(db: Session, site: Site) -> AutomationJob:
    job = AutomationJob(
        site_id=site.id,
        status='queued',
        scheduled_for=datetime.now(timezone.utc) + timedelta(hours=site.frequency_hours),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def enqueue_now(db: Session, site: Site) -> AutomationJob:
    job = AutomationJob(site_id=site.id, status='queued', scheduled_for=datetime.now(timezone.utc))
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def run_job(db: Session, job: AutomationJob) -> AutomationJob:
    site = db.query(Site).filter(Site.id == job.site_id).first()
    if not site:
        job.status = 'failed'
        job.error = 'Site not found'
        db.commit()
        return job

    allowed, used, limit = check_limit(db, site.user_id)
    if not allowed:
        job.status = 'failed'
        job.error = f'Plan limit reached ({used}/{limit})'
        db.commit()
        return job

    job.status = 'running'
    job.started_at = datetime.now(timezone.utc)
    db.commit()

    env = os.environ.copy()
    env['WP_URL'] = site.wp_url
    env['WP_USER'] = site.wp_user
    env['WP_APP_PASSWORD'] = decrypt_secret(site.wp_app_password_enc)
    env['OPENAI_API_KEY'] = decrypt_secret(site.openai_api_key_enc)
    env['POSTS_PER_RUN'] = '1'
    env['POST_STATUS'] = 'publish'
    env['POST_TOPIC'] = site.niche or 'autonomous content strategy'

    process = subprocess.run(['python', 'autopost.py'], capture_output=True, text=True, env=env)
    stdout = process.stdout[-2000:]
    stderr = process.stderr[-2000:]

    if process.returncode == 0:
        job.status = 'success'
        job.output_summary = stdout
        # Placeholder post record (real integration can parse wp_post_id from output/report)
        post = PublishedPost(site_id=site.id, job_id=job.id, wp_post_id=0, title='Generated Post', url='')
        db.add(post)
        increment_usage(db, site.user_id, amount=1)
        if site.is_active:
            schedule_next_job(db, site)
    else:
        job.status = 'failed'
        job.error = f'{stderr or stdout}'

    job.finished_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(job)
    return job


def summarize_health(db: Session) -> dict:
    queued = db.query(AutomationJob).filter(AutomationJob.status == 'queued').count()
    running = db.query(AutomationJob).filter(AutomationJob.status == 'running').count()
    failed = db.query(AutomationJob).filter(AutomationJob.status == 'failed').count()
    success = db.query(AutomationJob).filter(AutomationJob.status == 'success').count()
    return {'queued': queued, 'running': running, 'failed': failed, 'success': success}
