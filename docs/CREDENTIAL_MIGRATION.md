# Legacy Site Credential Migration

This migration converts existing plaintext values in the `sites.wp_app_password_enc` and `sites.openai_api_key_enc` columns into Fernet ciphertext using the application's `APP_ENCRYPTION_KEY`.

## Before you run it

1. Configure `APP_ENCRYPTION_KEY` in the target environment using the same stable Fernet key that the application will use.
2. Verify that the key is correct and retained securely. Losing or changing it makes encrypted credentials unreadable.
3. Create and verify a database backup, and schedule a maintenance window if automation may be running.
4. Run from the repository root in the same environment and against the same database configuration as the app.

Generate a key once (do not regenerate it for each deploy):

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

## Dry run (read-only)

```bash
python scripts/migrate_site_credentials.py
```

The script reports counts only. It never prints credential values. It validates the encryption key and any already-encrypted fields before proceeding.

## Apply after backup verification

```bash
python scripts/migrate_site_credentials.py --apply --backup-confirmed
```

The script encrypts all eligible legacy fields in one database transaction. It does not modify empty values or already encrypted values. If the key is missing/invalid or an existing ciphertext cannot be decrypted, it stops before writing.

## Important

- Do not run the apply command until the backup has been verified.
- Do not rotate `APP_ENCRYPTION_KEY` without a planned re-encryption process.
- Run the migration separately in each environment/database; a Vercel preview database is not necessarily the production database.
- This migration script has not been run against production data.
