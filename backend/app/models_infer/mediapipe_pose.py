class MediaPipePose:
    def infer(self, source: str) -> dict:
        return {"source": source, "keypoints": 33}
