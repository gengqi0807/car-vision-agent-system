"""
诊断脚本：分析 test_picture.jpg 手势分类过程
逐项打印每个手指的伸展判定值和最终分类结果
"""
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

backend_dir = Path(__file__).resolve().parent / "backend"
sys.path.insert(0, str(backend_dir))

from app.core.config import settings
from app.models_infer.mediapipe_hands import MediaPipeHands
from app.models_infer.gesture_classifier import GestureClassifier

# ---- 模型初始化 ----
model_path = os.path.join(settings.models_dir, settings.hand_landmarker_model)
MediaPipeHands.configure(model_path=model_path)
classifier = GestureClassifier(domain="owner")

# ---- 读取图片并推理 ----
image_path = "test_picture.jpg"
frame = cv2.imread(image_path)
if frame is None:
    print(f"[FAIL] Cannot read: {image_path}")
    sys.exit(1)

hands = MediaPipeHands.infer(frame)
if not hands:
    print("[FAIL] No hand detected!")
    sys.exit(1)

hand_kp = hands[0]
print(f"[OK] Detected {len(hands)} hand(s), keypoints: {len(hand_kp)}")

# ---- 手动计算手指伸展状态 ----
wrist = hand_kp[0]
fingers_def = [
    (4, 3, 2, "Thumb"),
    (8, 6, 5, "Index"),
    (12, 10, 9, "Middle"),
    (16, 14, 13, "Ring"),
    (20, 18, 17, "Pinky"),
]

print("\n" + "=" * 70)
print("  Finger Extension Analysis")
print("=" * 70)

extended_results = []

for tip_i, pip_i, mcp_i, name in fingers_def:
    if tip_i == 4:  # thumb: angle-based check (same as classifier)
        v1_x = hand_kp[pip_i]["x"] - hand_kp[mcp_i]["x"]
        v1_y = hand_kp[pip_i]["y"] - hand_kp[mcp_i]["y"]
        v2_x = hand_kp[tip_i]["x"] - hand_kp[pip_i]["x"]
        v2_y = hand_kp[tip_i]["y"] - hand_kp[pip_i]["y"]
        mag1 = math.hypot(v1_x, v1_y)
        mag2 = math.hypot(v2_x, v2_y)
        cos_angle = (v1_x * v2_x + v1_y * v2_y) / (mag1 * mag2) if mag1 * mag2 > 1e-16 else 0.0
        angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))
        extended = cos_angle > 0.35
        print(f"\n  {name} (angle-based: MCP->IP vs IP->TIP)")
        print(f"    MCP({mcp_i}) -> IP({pip_i}) vec = ({v1_x:.6f}, {v1_y:.6f})")
        print(f"    IP({pip_i}) -> TIP({tip_i}) vec = ({v2_x:.6f}, {v2_y:.6f})")
        print(f"    cos_angle = {cos_angle:.4f}  angle = {angle_deg:.1f} deg  (need cos > 0.35)")
    else:
        tip_wrist = math.hypot(hand_kp[tip_i]["x"] - wrist["x"],
                               hand_kp[tip_i]["y"] - wrist["y"])
        pip_wrist = math.hypot(hand_kp[pip_i]["x"] - wrist["x"],
                               hand_kp[pip_i]["y"] - wrist["y"])
        ratio = tip_wrist / pip_wrist if pip_wrist > 1e-8 else float('inf')
        extended = tip_wrist > pip_wrist * 1.2 if pip_wrist > 1e-8 else False
        print(f"\n  {name}")
        print(f"    tip({tip_i}) -> wrist dist  = {tip_wrist:.6f}")
        print(f"    PIP({pip_i}) -> wrist dist  = {pip_wrist:.6f}")
        print(f"    ratio = {ratio:.4f}  (need > 1.20)")

    print(f"    tip  coord: ({hand_kp[tip_i]['x']:.4f}, {hand_kp[tip_i]['y']:.4f})")
    print(f"    PIP  coord: ({hand_kp[pip_i]['x']:.4f}, {hand_kp[pip_i]['y']:.4f})")
    print(f"    Result: {'[EXTENDED]' if extended else '[BENT]'}")
    extended_results.append(extended)

extended_count = sum(extended_results)

# ---- 分类器判定流程 ----
print("\n" + "=" * 70)
print("  Classifier Decision Flow")
print("=" * 70)

names = ["thumb", "index", "middle", "ring", "pinky"]
stretched = [names[i] for i, v in enumerate(extended_results) if v]
print(f"\n  Extended fingers: {extended_count}/5  ->  [{', '.join(stretched)}]")
print(f"  Condition: extended==0 -> fist           : {'**** HIT' if extended_count == 0 else 'miss'}")
print(f"  Condition: extended>=4 -> palm           : {'**** HIT' if extended_count >= 4 else 'miss'}")

# thumb_up
if extended_count == 1 and extended_results[0]:
    thumb_tip = hand_kp[4]
    thumb_mcp = hand_kp[2]
    dy = thumb_tip["y"] - thumb_mcp["y"]
    print(f"  Condition: thumb-only + up/down check")
    print(f"    thumb_tip.y - thumb_mcp.y = {dy:.4f}")
    print(f"    up   (dy < -0.03): {'YES' if dy < -0.03 else 'NO'}")
    print(f"    down (dy > +0.03): {'YES' if dy > +0.03 else 'NO'}")
    print(f"  -> {'**** HIT thumb_up' if dy < -0.03 else '**** HIT thumb_down' if dy > +0.03 else 'default thumb_up'}")
else:
    hit = extended_count == 1 and extended_results[0]
    print(f"  Condition: thumb-only -> {hit}  miss")

# pointing
if extended_count == 1 and extended_results[1]:
    print(f"  Condition: index-only -> **** HIT pointing")
else:
    hit2 = extended_count == 1 and extended_results[1]
    print(f"  Condition: index-only -> {hit2}  miss")

# V gesture
if extended_count == 2 and extended_results[1] and extended_results[2]:
    print(f"  Condition: index+middle -> **** HIT pointing (V)")
else:
    print(f"  Condition: index+middle -> miss")

# ---- 最终结果 ----
gesture, conf = classifier.classify_static(hand_kp)
print(f"\n  [FINAL] gesture = {gesture}  (confidence = {conf:.2f})")

action_map = {
    "palm": "wake",
    "fist": "confirm",
    "thumb_up": "call_answer",
    "thumb_down": "call_hangup",
    "pointing": "idle",
    "unknown": "idle",
}
print(f"  [ACTION] {action_map.get(gesture, gesture)}")
print("=" * 70)

# ---- 关键点完整坐标 ----
print("\n  All 21 keypoints (x, y):")
for i, kp in enumerate(hand_kp):
    print(f"    kp[{i:2d}]: ({kp['x']:.4f}, {kp['y']:.4f})")
