"""
阶段 1：特征提取器
- 遍历 owner_gesture_dataset_/ 下每个手势子目录（如 call/ → thumb_up）
- 每张图片用 MediaPipe Hands 提取 21 关键点
- 输出 features.npz：(X, y, label_names)
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
from app.models_infer.hand_utils import normalize_hand_landmarks
from app.models_infer.mediapipe_hands import MediaPipeHands

# ----------------------------------------------------------------
# 标签映射：子目录名 → 手势名 → 数字标签
# ----------------------------------------------------------------
# 仅静态手势（图片/视频逐帧 → 单帧 63 维 SVM 训练）
# 动态手势（circle_cw/ccw, swipe_left/right, wave）由 extract_dynamic_features.py 独立处理
LABEL_MAP = {
    "call":        "thumb_up",     # 0
    "fist":        "fist",         # 1
    "palm":        "palm",         # 2
    # 后续扩充：
    "thumb_down": "thumb_down",  # 3
    # "pointing":   "pointing",    # 4
}

FOLDER_TO_INT = {folder: i for i, folder in enumerate(LABEL_MAP.keys())}

DATASET_DIR = PROJECT_ROOT / "owner_gesture_dataset_"
OUTPUT_DIR  = PROJECT_ROOT / "owner_gesture_dataset_features"
OUTPUT_FILE = OUTPUT_DIR / "features.npz"

# 支持的图片扩展名
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# 支持的视频扩展名
VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}
# 视频抽帧间隔（每 N 帧取 1 帧，避免相邻帧高度重复）
VIDEO_FRAME_INTERVAL = 4


def main():
    # --- 1. 初始化 MediaPipe -------------------------------------------------
    model_path = os.path.join(settings.models_dir, settings.hand_landmarker_model)
    print(f"[INFO] 模型: {model_path}")
    if not os.path.exists(model_path):
        print(f"[FAIL] 模型文件不存在，请先下载 hand_landmarker.task")
        sys.exit(1)

    MediaPipeHands.configure(model_path=model_path)
    MediaPipeHands.reset()

    # --- 2. 遍历数据集 -------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_features = []   # list of (63,) 向量 [x0..x20, y0..y20, z0..z20] 已相对归一化
    all_labels   = []
    stats = {}

    for folder_name in sorted(os.listdir(DATASET_DIR)):
        folder_path = DATASET_DIR / folder_name
        if not folder_path.is_dir():
            continue
        if folder_name not in FOLDER_TO_INT:
            print(f"[SKIP] 不在 LABEL_MAP: {folder_name}")
            continue

        label_int = FOLDER_TO_INT[folder_name]
        gesture_name = LABEL_MAP[folder_name]
        count, dropped = 0, 0

        # --- 收集该手势目录下所有待处理文件 ---
        # 支持两种结构：
        #   A) 扁平：folder/*.jpg              （如 call/）
        #   B) 嵌套：folder/subjectN_rM/*.avi （如 wave/）
        all_files = []  # [(file_path, subdir_name), ...]

        subdirs = [d for d in folder_path.iterdir() if d.is_dir()]
        if subdirs:
            # 嵌套结构：递归进入每个 subject_r 子目录
            for subdir in sorted(subdirs):
                for fpath in sorted(subdir.iterdir()):
                    if fpath.is_file():
                        all_files.append((fpath, subdir.name))
        else:
            # 扁平结构
            for fpath in sorted(folder_path.iterdir()):
                if fpath.is_file():
                    all_files.append((fpath, ""))

        total_files = len(all_files)
        print(f"  [{label_int}] {gesture_name:12s} → 发现 {total_files} 个文件")

        for fpath, subdir_name in all_files:
            suffix = fpath.suffix.lower()
            frames_to_process = []

            if suffix in IMG_EXTS:
                frame = cv2.imread(str(fpath))
                if frame is None:
                    print(f"    [WARN] 读取失败: {subdir_name}/{fpath.name}")
                    continue
                frames_to_process = [frame]

            elif suffix in VIDEO_EXTS:
                cap = cv2.VideoCapture(str(fpath))
                if not cap.isOpened():
                    print(f"    [WARN] 无法打开视频: {subdir_name}/{fpath.name}")
                    continue
                frame_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_idx % VIDEO_FRAME_INTERVAL == 0:
                        frames_to_process.append(frame)
                    frame_idx += 1
                cap.release()
            else:
                continue  # 不支持的文件类型

            # 对每帧提取关键点
            for frame in frames_to_process:
                hands = MediaPipeHands.infer(frame)
                if not hands:
                    dropped += 1
                    continue

                kp = hands[0]  # 取第一只手，每点 {"x","y","z"}
                flat = normalize_hand_landmarks(kp)
                all_features.append(flat)
                all_labels.append(label_int)
                count += 1

        stats[gesture_name] = f"{count} 样本 (丢弃 {dropped})"
        print(f"  [{label_int}] {gesture_name:12s} → {count:4d} 有效 / {count + dropped:4d} 总")

    # --- 3. 保存 -------------------------------------------------------------
    if not all_features:
        print("[FAIL] 零有效样本，退出")
        sys.exit(1)

    X = np.array(all_features, dtype=np.float32)   # (N, 63)
    y = np.array(all_labels, dtype=np.int32)        # (N,)

    np.savez_compressed(
        OUTPUT_FILE,
        X=X,
        y=y,
        label_names=np.array(list(LABEL_MAP.values())),
    )

    print(f"\n[DONE] 总样本: {X.shape[0]}  特征维度: {X.shape[1]} (21点 × xyz)")
    print(f"[DONE] 各类统计: {stats}")
    print(f"[DONE] 输出: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
