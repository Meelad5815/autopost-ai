import unittest
from unittest.mock import patch

from cryptography.fernet import Fernet

from app.core.config import settings
from app.core.secrets import (
    SecretConfigurationError,
    decrypt_secret,
    encrypt_secret,
)


class SecretEncryptionTests(unittest.TestCase):
    def test_round_trip_encrypts_without_exposing_plaintext(self):
        key = Fernet.generate_key().decode("ascii")
        with patch.object(settings, "app_encryption_key", key):
            encrypted = encrypt_secret("wp-secret-value")
            self.assertTrue(encrypted.startswith("fernet:"))
            self.assertNotIn("wp-secret-value", encrypted)
            self.assertEqual(decrypt_secret(encrypted), "wp-secret-value")

    def test_missing_key_fails_closed_for_new_secret(self):
        with patch.object(settings, "app_encryption_key", ""):
            with self.assertRaises(SecretConfigurationError):
                encrypt_secret("wp-secret-value")

    def test_invalid_key_fails_closed(self):
        with patch.object(settings, "app_encryption_key", "not-a-fernet-key"):
            with self.assertRaises(SecretConfigurationError):
                encrypt_secret("wp-secret-value")

    def test_legacy_plaintext_can_be_read_for_migration_compatibility(self):
        self.assertEqual(decrypt_secret("legacy-value"), "legacy-value")


if __name__ == "__main__":
    unittest.main()
