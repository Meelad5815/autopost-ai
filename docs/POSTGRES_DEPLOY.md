# PostgreSQL Deployment (Vercel)

Use PostgreSQL in production. Do not rely on SQLite for serverless runtime.

## 1) Create DB
- Create a managed PostgreSQL database (Neon/Supabase/Railway/Aiven/etc.).
- Copy connection string.

## 2) Set Vercel Environment Variables
- `DATABASE_URL=postgresql://USER:PASS@HOST:5432/DBNAME?sslmode=require`
- `JWT_SECRET_KEY=<strong-random-secret>`
- `GOOGLE_CLIENT_ID=<if using oauth>`
- `GOOGLE_CLIENT_SECRET=<if using oauth>`

## 3) Redeploy
- Redeploy latest `main`.
- Verify health endpoint: `/healthz`

Expected response:
```json
{
  "status": "ok",
  "db_ok": true,
  "db_backend": "postgresql"
}
```

## 4) If DB is not reachable
- Check firewall/network allow-list.
- Ensure `sslmode=require` in `DATABASE_URL`.
- Confirm username/password/database name.

## 5) Migrations
- App startup now runs versioned migrations automatically via `app/db_migrations.py`.
- Add future schema changes as new migration entries in `MIGRATIONS`.
