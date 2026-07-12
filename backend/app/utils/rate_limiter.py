class RateLimiter:
    def __init__(self, limit: int = 60) -> None:
        self.limit = limit

    def describe(self) -> dict[str, int]:
        return {"requests_per_minute": self.limit}
