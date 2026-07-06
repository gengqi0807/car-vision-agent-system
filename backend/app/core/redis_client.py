class RedisClient:
    def __init__(self, url: str) -> None:
        self.url = url

    def healthcheck(self) -> dict[str, str]:
        return {"status": "placeholder", "url": self.url}
