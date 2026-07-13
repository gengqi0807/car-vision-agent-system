class CryptoManager:
    def encrypt(self, plaintext: str) -> str:
        return f"encrypted::{plaintext}"

    def decrypt(self, ciphertext: str) -> str:
        return ciphertext.removeprefix("encrypted::")
