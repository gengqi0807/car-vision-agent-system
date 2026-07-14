from __future__ import annotations

import base64
import hashlib
import hmac
import os
from functools import cached_property

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.core.config import settings


def _derive_key(seed: str, purpose: str) -> bytes:
    return hashlib.sha256(f"{purpose}:{seed}".encode("utf-8")).digest()


def _encode_key(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _decode_or_derive_key(raw_value: str, *, purpose: str) -> bytes:
    if raw_value:
        return base64.urlsafe_b64decode(raw_value.encode("ascii"))
    return _derive_key(settings.secret_key, purpose)


def normalize_sensitive_value(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def normalize_email(value: str | None) -> str | None:
    normalized = normalize_sensitive_value(value)
    return normalized.lower() if normalized else None


def normalize_phone(value: str | None) -> str | None:
    normalized = normalize_sensitive_value(value)
    if normalized is None:
        return None
    return normalized.replace(" ", "")


class CryptoManager:
    @cached_property
    def _aes_key(self) -> bytes:
        return _decode_or_derive_key(settings.data_encryption_key, purpose="aes")

    @cached_property
    def _hash_key(self) -> bytes:
        return _decode_or_derive_key(settings.data_hash_key, purpose="hash")

    def encrypt(self, plaintext: str) -> str:
        nonce = os.urandom(12)
        aesgcm = AESGCM(self._aes_key)
        ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
        return base64.urlsafe_b64encode(nonce + ciphertext).decode("ascii")

    def decrypt(self, ciphertext: str) -> str:
        payload = base64.urlsafe_b64decode(ciphertext.encode("ascii"))
        nonce = payload[:12]
        encrypted = payload[12:]
        aesgcm = AESGCM(self._aes_key)
        plaintext = aesgcm.decrypt(nonce, encrypted, None)
        return plaintext.decode("utf-8")

    def fingerprint(self, plaintext: str) -> str:
        digest = hmac.new(self._hash_key, plaintext.encode("utf-8"), hashlib.sha256).hexdigest()
        return digest

    def generate_env_keys(self) -> tuple[str, str]:
        return _encode_key(os.urandom(32)), _encode_key(os.urandom(32))


crypto_manager = CryptoManager()
