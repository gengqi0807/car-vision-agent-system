from datetime import datetime


class LLMClient:
    def build_summary(self, source: str, payload: dict) -> dict:
        return {
            "source": source,
            "level": payload.get("level", "info"),
            "summary": payload.get("summary", "LLM summary placeholder"),
            "created_at": datetime.utcnow().isoformat(),
        }
