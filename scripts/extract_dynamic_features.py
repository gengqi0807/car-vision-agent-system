"""
阶段 1-D：动态手势特征提取器（序列模式）。

与 extract_features.py 的区别:
  - 静态特征：逐帧 63 维 → features_static.npz（单帧样本）
  - 动态特征：整段视频 (T, 63) 序列 → features_dynamic.npz（序列样本）

数据目录结构:
  owner_gesture_dataset_/
    circle_ccw/
      subj1_r1.avi
      subj2_r1.avi
    circle_cw/
      ...
    swipe_left/
      ...
    swipe_right/
      ...
    wave/
      ...

用法:
  python scripts/extract_dynamic_features.py
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np

# 项目根
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.models_infer.hand_utils import normalize_hand_with_trajectory
from app.models_infer.mediapipe_hands import MediaPipeHands

# ----------------------------------------------------------------
# 动态手势标签映射
# ----------------------------------------------------------------
DYNAMIC_LABEL_MAP = {
    "circle_ccw":  0,
    "circle_cw":   1,
    "swipe_left":  2,
    "swipe_right": 3,
    "wave":        4,
}

DYNAMIC_LABEL_NAMES = list(DYNAMIC_LABEL_MAP.keys())

DATASET_DIR = PROJECT_ROOT / "owner_gesture_dataset_"
OUTPUT_DIR = PROJECT_ROOT / "owner_gesture_dataset_features"
OUTPUT_FILE = OUTPUT_DIR / "features_dynamic.npz"

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}


def main():
    # --- 1. 初始化 MediaPipe -------------------------------------------------
    model_path = os.path.join(settings.models_dir, settings.hand_landmarker_model)
    print(f"[INFO] 模型: {model_path}")
    if not os.path.exists(model_path):
        print(f"[FAIL] 模型文件不存在，请先下载 hand_landmarker.task")
        sys.exit(1)

    MediaPipeHands.configure(model_path=model_path)
    MediaPipeHands.reset()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 2. 遍历动态手势目录 -------------------------------------------------
    all_sequences = []   # list of np.ndarray, 每条 shape (T, 63), T 可变
    all_labels = []      # list of int
    all_file_names = []  # 各样本来源文件名
    stats = {}

    for folder_name in sorted(os.listdir(DATASET_DIR)):
        folder_path = DATASET_DIR / folder_name
        if not folder_path.is_dir():
            continue
        if folder_name not in DYNAMIC_LABEL_MAP:
            print(f"[SKIP] 非动态手势目录: {folder_name}")
            continue

        label_int = DYNAMIC_LABEL_MAP[folder_name]
        count = 0
        total_videos = 0

        print(f"\n[{label_int}] {folder_name}:")

        # 收集该目录下所有视频文件
        video_files = []
        for fpath in sorted(folder_path.rglob("*")):
            if fpath.is_file() and fpath.suffix.lower() in VIDEO_EXTS:
                video_files.append(fpath)

        total_videos = len(video_files)
        print(f"  发现 {total_videos} 个视频文件")

        for fpath in video_files:
            cap = cv2.VideoCapture(str(fpath))
            if not cap.isOpened():
                print(f"  [WARN] 无法打开: {fpath.name}")
                continue

            # 抽取所有帧的关键点
            frame_features = []
            frame_idx = 0
            valid_frames = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                hands = MediaPipeHands.infer(frame)
                if hands and len(hands[0]) == 21:
                    feat = normalize_hand_with_trajectory(hands[0])
                    frame_features.append(feat)
                    valid_frames += 1
                # 如果手丢失，插入 NaN 占位（可选：用前一帧填充）
                elif len(frame_features) > 0:
                    frame_features.append(frame_features[-1].copy())
                # else: 视频开头无手，跳过

                frame_idx += 1

            cap.release()

            if valid_frames < 8:
                print(f"  [WARN] {fpath.name}: 有效帧太少 ({valid_frames}/{total_frames})，跳过")
                continue

            seq = np.stack(frame_features, axis=0)  # (T, 63)
            all_sequences.append(seq)
            all_labels.append(label_int)
            all_file_names.append(str(fpath.relative_to(DATASET_DIR)))
            count += 1

            print(f"  [OK] {fpath.name}: {valid_frames}/{total_frames} 有效帧 → 序列长度 {seq.shape[0]}")

        stats[folder_name] = count
        print(f"  [{label_int}] {folder_name}: {count} 个序列样本")

    # --- 3. 保存 -------------------------------------------------------------
    if not all_sequences:
        print("\n[FAIL] 零有效序列样本，请先采集动态手势视频")
        sys.exit(1)

    # 用 object 数组存储变长序列
    seq_array = np.empty(len(all_sequences), dtype=object)
    for i, seq in enumerate(all_sequences):
        seq_array[i] = seq

    label_array = np.array(all_labels, dtype=np.int32)

    np.savez_compressed(
        OUTPUT_FILE,
        sequences=seq_array,
        labels=label_array,
        label_names=np.array(DYNAMIC_LABEL_NAMES),
        file_names=np.array(all_file_names),
    )

    print(f"\n[DONE] 总序列样本: {len(all_sequences)}")
    print(f"[DONE] 各类统计: {stats}")
    print(f"[DONE] 输出: {OUTPUT_FILE}")

    # 统计序列长度分布
    lengths = [s.shape[0] for s in all_sequences]
    print(f"[DONE] 序列长度 — min: {min(lengths)}, max: {max(lengths)}, "
          f"mean: {np.mean(lengths):.1f}, median: {np.median(lengths):.0f}")


if __name__ == "__main__":
    main()
