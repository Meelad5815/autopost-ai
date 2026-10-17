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
