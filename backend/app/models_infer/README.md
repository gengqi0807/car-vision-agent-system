# Inference Layer

This folder hosts wrappers for pre-trained computer-vision models.

## Implemented

| Module                 | Class            | Interface                     | Status       |
|------------------------|------------------|-------------------------------|--------------|
| `mediapipe_hands.py`   | `MediaPipeHands` | `infer(source) -> dict`       | **Real**     |
| `mediapipe_pose.py`    | `MediaPipePose`  | `infer(source) -> dict`       | **Real**     |
| `yolo_detector.py`     | `YoloDetector`   | `detect(source) -> list[dict]`| Placeholder  |
| `ocr_recognizer.py`    | `OCRRecognizer`  | `recognize(path) -> dict`     | Placeholder  |
| `gesture_classifier.py`| `GestureClassifier` | `classify(kps, domain) -> dict` | Placeholder |

## Model Files

Download required `.task` models to `backend/models/`:

```bash
# From project root
python scripts/download_models.py               # all models
python scripts/download_models.py --model hand_landmarker
python scripts/download_models.py --model pose_landmarker
```

Expected files after download:

- `backend/models/hand_landmarker.task` – 21 keypoints per hand
- `backend/models/pose_landmarker_lite.task` – 33 body keypoints

## Usage

```python
from app.models_infer import MediaPipeHands, MediaPipePose

# Hands
with MediaPipeHands() as hands:
    result = hands.infer("frame.jpg")
    print(result["num_hands_detected"], result["keypoints"])

# Pose
with MediaPipePose() as pose:
    result = pose.infer("frame.jpg")
    print(result["num_poses_detected"], result["keypoints"])
```
