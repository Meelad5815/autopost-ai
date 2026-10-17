"""Safely encrypt legacy plaintext site credentials.

Default mode is read-only. To write, set APP_ENCRYPTION_KEY, create a verified
database backup, then pass both --apply and --backup-confirmed.
Credential values are never printed or logged.
"""
from __future__ import annotations

import argparse
import sys

from app.core.secrets import PREFIX, SecretConfigurationError, decrypt_secret, encrypt_secret
from app.db import SessionLocal
from app.models import Site


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write encrypted credentials; otherwise dry-run")
    parser.add_argument(
        "--backup-confirmed",
        action="store_true",
        help="confirm a current database backup has been created and verified",
    )
    args = parser.parse_args()
    if args.apply and not args.backup_confirmed:
        parser.error("--apply requires --backup-confirmed after a verified database backup")
    return args


def main() -> int:
    args = parse_args()
    db = SessionLocal()
    try:
        # Validate key and encryption/decryption before reading or changing rows.
        probe = encrypt_secret("credential-migration-key-check")
        if decrypt_secret(probe) != "credential-migration-key-check":
            raise SecretConfigurationError("Encryption key validation failed")

        sites = db.query(Site).order_by(Site.id).all()
        candidates: list[tuple[Site, str, str]] = []
        already_encrypted = 0
        empty = 0

        for site in sites:
            for field in ("wp_app_password_enc", "openai_api_key_enc"):
                value = getattr(site, field) or ""
                if not value:
                    empty += 1
                    continue
                if value.startswith(PREFIX):
                    # Validate existing ciphertext before any writes; wrong key fails closed.
                    decrypt_secret(value)
                    already_encrypted += 1
                    continue
                candidates.append((site, field, value))

        print(
            f"Sites inspected: {len(sites)}; legacy credentials to encrypt: {len(candidates)}; "
            f"already encrypted fields: {already_encrypted}; empty fields: {empty}."
        )
        if not args.apply:
            print("DRY RUN ONLY: no database changes made. After verifying a backup, rerun with --apply --backup-confirmed.")
            return 0

        # Encrypt all candidates in memory before assigning anything; commit once.
        replacements = [(site, field, encrypt_secret(value)) for site, field, value in candidates]
        for site, field, encrypted in replacements:
            setattr(site, field, encrypted)
        db.commit()
        print(f"Migration committed successfully: {len(replacements)} credential fields encrypted.")
        return 0
    except SecretConfigurationError as exc:
        db.rollback()
        print(f"Migration stopped safely: {exc}", file=sys.stderr)
        return 2
    except Exception:
        db.rollback()
        # Do not print exception details; drivers can sometimes include SQL parameters.
        print("Migration failed; transaction rolled back. Inspect server logs without exposing credential values.", file=sys.stderr)
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
