"""Background worker loop.

This process polls queued jobs and runs each tenant in isolation by injecting
site-scoped credentials into the existing `autopost.py` automation process.
"""

import logging
import time

from sqlalchemy import asc

from app.core.config import settings
from app.db import SessionLocal
from app.models import AutomationJob
from app.services.automation import run_job


logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger('worker')


def process_once() -> int:
    with SessionLocal() as db:
        job = (
            db.query(AutomationJob)
            .filter(AutomationJob.status == 'queued')
            .order_by(asc(AutomationJob.scheduled_for))
            .first()
        )
        if not job:
            return 0
        run_job(db, job)
        return 1


def main():
    logger.info('Worker started. poll_seconds=%s', settings.worker_poll_seconds)
    while True:
        try:
            processed = process_once()
            if processed == 0:
                time.sleep(settings.worker_poll_seconds)
        except Exception as exc:
            logger.exception('Worker loop error: %s', exc)
            time.sleep(settings.worker_poll_seconds)


if __name__ == '__main__':
    main()
