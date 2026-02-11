# Multi-Tenant SaaS Platform Architecture

## Folder structure

- `app/main.py` - FastAPI app bootstrap and route registration.
- `app/models.py` - multi-tenant database models (users, sites, jobs, usage, subscriptions).
- `app/routes/` - API surface for auth, sites, automation, billing, admin, health.
- `app/services/` - orchestration logic for automation, usage limits, billing placeholders.
- `app/worker.py` - background queue worker that executes queued tenant jobs.
- `engine/` + `autopost.py` - existing autonomous content generation and WP publishing stack.

## Main services

1. **Auth service**
   - Registration/login
   - JWT token issuance
   - password hashing (bcrypt)

2. **Tenant website service**
   - add/manage WordPress websites per user
   - per-site schedule and niche configuration

3. **Automation service**
   - queue jobs
   - execute tenant-isolated runs by injecting site env vars into `autopost.py`
   - persist job status and outputs

4. **Billing service (Stripe-ready placeholders)**
   - plans: free/pro/agency
   - plan limits and change-plan API
   - checkout URL placeholder to swap with Stripe session creation

5. **Admin service**
   - list users
   - suspend accounts
   - override limits
   - system health view

## Database models

- `User` (role, active flag)
- `Subscription` (plan, limits, stripe ids)
- `Site` (tenant WP credentials and schedule)
- `AutomationJob` (queue status lifecycle)
- `PublishedPost` (published history + performance placeholders)
- `UsageCounter` (monthly quotas)

## API routes

- `POST /auth/register`, `POST /auth/login`
- `POST /sites`, `GET /sites`, `POST /sites/{id}/start`, `POST /sites/{id}/stop`
- `POST /automation/{site_id}/run-now`, `GET /automation/jobs`, `GET /automation/posts`
- `GET /billing/usage`, `POST /billing/change-plan/{plan}`
- `GET /admin/users`, `POST /admin/users/{id}/suspend`, `POST /admin/users/{id}/limits/{limit}`, `GET /admin/health`
- `GET /healthz`

## Environment variable design

Core:
- `DATABASE_URL`
- `JWT_SECRET_KEY`
- `JWT_ALGORITHM`
- `JWT_EXPIRE_MINUTES`
- `WORKER_POLL_SECONDS`

Plans:
- `FREE_POSTS_LIMIT`
- `PRO_POSTS_LIMIT`
- `AGENCY_POSTS_LIMIT`

Billing placeholders:
- `STRIPE_SECRET_KEY`
- `STRIPE_WEBHOOK_SECRET`

Automation execution uses per-site stored credentials and calls `autopost.py` with:
- `WP_URL`, `WP_USER`, `WP_APP_PASSWORD`, `OPENAI_API_KEY`, `POST_TOPIC`, etc.

## Deployment / GitHub Actions idea

- CI validates both platform and engine syntax.
- Same workflow can optionally execute content engine when secrets are present.
- Deploy API as container (e.g., Fly.io/Render/ECS), and run `app.worker` as a separate process.
- For scale: move queue to Redis/RQ or Celery and process workers horizontally.

## Scalability notes

- isolate tenant secrets per site
- queue-based asynchronous automation
- plan-based quota checks before execution
- stateless API layer + independent worker pool
- database indexes on tenant and scheduling fields
