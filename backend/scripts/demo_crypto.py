"""Demonstrate AES-GCM encryption and HMAC fingerprint behavior."""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.utils.crypto import crypto_manager


def main() -> None:
    value = "demo@example.com"

    ciphertext_1 = crypto_manager.encrypt(value)
    ciphertext_2 = crypto_manager.encrypt(value)
    fingerprint_1 = crypto_manager.fingerprint(value)
    fingerprint_2 = crypto_manager.fingerprint(value)

    print(f"Original value: {value}")
    print(f"Ciphertext 1: {ciphertext_1}")
    print(f"Ciphertext 2: {ciphertext_2}")
    print(f"Ciphertexts are different: {ciphertext_1 != ciphertext_2}")
    print()
    print(f"Fingerprint 1: {fingerprint_1}")
    print(f"Fingerprint 2: {fingerprint_2}")
    print(f"Fingerprints are identical: {fingerprint_1 == fingerprint_2}")
    print()
    print(f"Decrypted ciphertext 1: {crypto_manager.decrypt(ciphertext_1)}")
    print(f"Decrypted ciphertext 2: {crypto_manager.decrypt(ciphertext_2)}")


if __name__ == "__main__":
    main()
