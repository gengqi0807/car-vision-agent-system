"""
诊断脚本：测试 circle_cw / circle_ccw 采集视频的 LSTM 识别置信度。

用法:
  python scripts/diag_circle.py

输出:
  - 每个视频的预测结果和置信度
  - 帮助判断是模型过拟合还是现场域差异
"""

import sys
from pathlib import Path

import numpy as np
import cv2

# 项目根
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.models_infer.mediapipe_hands import MediaPipeHands
from app.models_infer.hand_utils import normalize_hand_with_trajectory
from app.models_infer.dynamic_lstm import DynamicLSTMClassifier

# --- 数据集路径 ---
DATASET_DIR = PROJECT_ROOT / "owner_gesture_dataset_"

# --- 初始化 ---
MediaPipeHands.configure(model_path=str(Path(settings.models_dir) / settings.hand_landmarker_model))
MediaPipeHands.reset()

clf = DynamicLSTMClassifier()
print(f"LSTM loaded: {clf.is_loaded}")
print(f"Labels: {clf._label_names}\n")

# --- 诊断 ---
for gesture_name in ["circle_cw", "circle_ccw"]:
    gesture_dir = DATASET_DIR / gesture_name
    if not gesture_dir.exists():
        print(f"[SKIP] 目录不存在: {gesture_dir}")
        continue

    videos = sorted(gesture_dir.glob("*.mp4"))
    print(f"--- {gesture_name} ({len(videos)} videos) ---")

    correct = 0
    total = 0
    for vid in videos:
        cap = cv2.VideoCapture(str(vid))
        seq = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            hand = MediaPipeHands.infer(frame)
            if hand and len(hand[0]) == 21:
                seq.append(normalize_hand_with_trajectory(hand[0]))
        cap.release()

        total += 1
        if len(seq) < clf.MIN_SEQUENCE_LENGTH:
            print(f"  {vid.name:24s} -> TOO_SHORT  frames={len(seq)}")
            continue

        pred, conf = clf.classify_sequence(np.stack(seq))
        is_correct = "✓" if pred == gesture_name else "✗"
        correct += 1 if pred == gesture_name else 0

        # 打印所有类别的 softmax 概率
        tensor = clf._model.predict_proba(
            __import__("torch").from_numpy(np.stack(seq)).float().to(clf._device)
        )
        detail = "  ".join(
            f"{name}:{tensor[i]:.3f}" for i, name in enumerate(clf._label_names)
        )
        print(f"  {vid.name:24s} -> {pred:14s} conf={conf:.3f}  {is_correct}  [{detail}]")

    acc = correct / total * 100 if total > 0 else 0
    print(f"  >> {gesture_name} 准确率: {correct}/{total} ({acc:.1f}%)\n")
