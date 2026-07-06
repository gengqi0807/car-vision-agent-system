class GestureClassifier:
    def classify(self, keypoints: list[dict], domain: str) -> dict:
        return {"domain": domain, "gesture": "placeholder", "confidence": 0.9}
