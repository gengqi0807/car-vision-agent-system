import secrets
import string


USER_UID_LENGTH = 12
USER_UID_ALPHABET = string.digits


def generate_user_uid() -> str:
    return "".join(secrets.choice(USER_UID_ALPHABET) for _ in range(USER_UID_LENGTH))
