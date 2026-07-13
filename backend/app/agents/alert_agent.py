from app.agents.llm_client import LLMClient
from app.agents.notifier import Notifier


class AlertAgent:
    def __init__(self) -> None:
        self.llm_client = LLMClient()
        self.notifier = Notifier()

    async def handle_event(self, source: str, payload: dict) -> dict:
        summary = self.llm_client.build_summary(source=source, payload=payload)
        await self.notifier.broadcast(summary)
        return summary
