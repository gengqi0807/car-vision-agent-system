class OCRRecognizer:
    def recognize(self, cropped_plate_path: str) -> dict:
        return {"plate_number": "浙A12345", "source": cropped_plate_path}
