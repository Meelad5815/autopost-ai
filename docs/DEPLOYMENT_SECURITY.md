# AutoPost AI deployment security configuration

## Required production environment variables

Configure these in the hosting provider's encrypted environment-variable settings. Do not commit actual values to GitHub.

- `ENVIRONMENT=production`
- `DEBUG=false`
- `JWT_SECRET_KEY`: a unique random value of at least 32 characters. Changing it invalidates existing signed access and refresh tokens.
- `APP_ENCRYPTION_KEY`: a Fernet key used to encrypt newly saved WordPress and AI provider credentials.

Generate keys locally in a trusted environment:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Use the first command's output for `JWT_SECRET_KEY`, and the second command's output for `APP_ENCRYPTION_KEY`. Store both as independent secrets. Keep the encryption key stable across deployments and backups; losing or changing it without re-encrypting stored credentials will make encrypted credentials unreadable.

## Existing credential records

Before this change, the application stored WordPress and AI provider credentials as plaintext despite the database field names ending in `_enc`. The compatibility reader continues to read those legacy values so an existing automation run does not immediately break. This means the database may still contain plaintext credentials until a controlled migration is performed.

Before declaring the migration complete:

1. Back up the database securely and restrict backup access.
2. Configure `APP_ENCRYPTION_KEY` in the environment used by the migration and runtime.
3. Run a reviewed one-time migration that encrypts every non-empty legacy `sites.wp_app_password_enc` and `sites.openai_api_key_enc` value not already prefixed with `fernet:`.
4. Verify that encrypted records can be decrypted using the same deployment key, without printing credential values to logs.
5. Restore-test the database and encryption key together.
6. Rotate provider credentials if any real credentials were exposed in logs, source control, or unprotected database backups.

Do not run a credential migration against production without a verified backup and a reviewed migration script.

## Deployment and validation

The repository's GitHub Actions workflow validates generated-site artifacts and regression tests. A successful CI run does not by itself confirm that the FastAPI backend is deployed or that the production environment variables are configured. Verify the backend preview URL, API health endpoint, authenticated dashboard, and a test site workflow separately. Do not enable bulk publishing until credentials, tenant access controls and posting limits have been verified.


## Admin terminal safety gate

The `/ui/cloud-terminal/exec` and `/ui/kali-terminal/exec` endpoints execute commands and are therefore disabled by default, including for administrators. Do not enable them on a public production instance. If a trusted isolated environment genuinely requires them, explicitly set `ENABLE_ADMIN_TERMINAL=true`; restrict network access and admin accounts, and remove the flag when finished.

## One-time administrator bootstrap

The public `/ui/first-setup` route is disabled unless `ALLOW_INITIAL_SETUP=true` (values `1` and `yes` are also accepted). Set this variable only while provisioning the first administrator on a fresh database. The route refuses to run once any user exists, and encrypts the initial WordPress application password before persistence. After creating the first administrator, immediately remove `ALLOW_INITIAL_SETUP` from the hosting environment and redeploy. Existing accounts must use `/auth/login`; do not re-enable bootstrap on an initialized database.


## Dashboard token storage

The dashboard now migrates legacy access and refresh tokens out of persistent `localStorage` and stores them in tab-scoped `sessionStorage`, reducing their persistence after the browser tab closes. Google OAuth credentials are redirected in the URL fragment rather than the query string; the dashboard consumes them and removes them from browser history. Fragments are not sent in HTTP requests or Referer headers, but the tokens are still visible to same-origin JavaScript during processing. This is interim risk reduction only: same-origin JavaScript can still read session storage, and the manual access-token field remains a sensitive input. The durable fix is an HttpOnly, Secure, SameSite cookie-based session with CSRF protection, plus removal of bearer tokens from the page and browser JavaScript. Do not treat session storage or URL fragments as an XSS defense.
