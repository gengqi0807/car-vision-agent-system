class YoloDetector:
    def detect(self, source: str) -> list[dict]:
        return [{"label": "license-plate", "source": source}]
