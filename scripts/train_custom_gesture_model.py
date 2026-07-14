"""
自定义手势 SVM 训练脚本（合并训练版）。

输入:
    - 固有静态手势样本: owner_gesture_dataset_features/features.npz
      （由 scripts/extract_features.py 生成，5 类：thumb_up/fist/palm/thumb_down/thumb_index）
    - 自定义手势样本: 数据库 custom_gesture_samples 表

输出:
    - models/custom_gesture_classifier_svm.joblib（is_custom=True）

说明:
    合并训练把「固有静态 + 自定义」统一成一个 SVM（label = 固有5类 + 自定义N类），
    写入 custom_gesture_classifier_svm.joblib。
    该合并模型即「最新版 SVM 训练结果」，现已晋升为实时视频流的主静态分类器
    （GestureClassifier(use_custom_primary=True)），实时流不再单独使用
    gesture_classifier_svm.joblib。固有控车手势（唤醒/确认/音量…）照常工作，
    自定义手势在实时流中仅显示识别结果、不触发控车指令。
"""

import os
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.core.database import SessionLocal
from app.models.custom_gesture import CustomGesture, CustomGestureSample
from app.models_infer.hand_utils import normalize_hand_landmarks_array

# 固有静态手势离线特征集（合并训练强依赖此文件）
FEATURES_FILE = PROJECT_ROOT / "owner_gesture_dataset_features" / "features.npz"


def load_static_samples():
    """加载固有静态手势样本。

    Returns:
        (X_static, sample_names, static_label_names)
        X_static: (N, 63) float32
        sample_names: list[str] 每个样本对应的类别名
        static_label_names: list[str] 固有静态类别（保持 features.npz 中顺序）

    Raises:
        FileNotFoundError: features.npz 不存在。
    """
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"固有静态手势特征文件不存在: {FEATURES_FILE}\n"
            f"       合并训练依赖该文件，请先运行: python scripts/extract_features.py"
        )

    data = np.load(FEATURES_FILE)
    X_static = data["X"].astype(np.float32)          # (N, 63)
    y_static = data["y"]                              # (N,) 整数索引
    static_label_names = [str(n) for n in data["label_names"].tolist()]

    sample_names = [static_label_names[int(i)] for i in y_static]
    return X_static, sample_names, static_label_names


def load_custom_samples(gesture_names: list[str] | None = None):
    """从数据库加载自定义手势样本。

    Returns:
        (X_custom, sample_names, custom_label_names)
        X_custom: (M, 63) float32 或 None（无样本时）
        sample_names: list[str]
        custom_label_names: list[str] 出现过有效样本的自定义类别名
    """
    with SessionLocal() as session:
        query = session.query(CustomGesture)
        if gesture_names:
            query = query.filter(CustomGesture.name.in_(gesture_names))
        gestures = query.all()

        X_list: list[np.ndarray] = []
        sample_names: list[str] = []
        custom_label_names: list[str] = []

        for gesture in gestures:
            has_valid = False
            for sample in gesture.samples:
                kps = sample.keypoints
                if not kps or len(kps) != 21:
                    continue
                feat = normalize_hand_landmarks_array(kps)
                X_list.append(feat)
                sample_names.append(gesture.name)
                has_valid = True
            if has_valid and gesture.name not in custom_label_names:
                custom_label_names.append(gesture.name)

    if not X_list:
        return None, [], []

    X_custom = np.stack(X_list, axis=0).astype(np.float32)
    return X_custom, sample_names, custom_label_names


def load_all_samples(gesture_names: list[str] | None = None):
    """合并加载「固有静态 + 自定义」样本，返回 (X, y, label_names)。

    label 空间: 固有静态类别在前，自定义类别在后。
    """
    # 1. 固有静态样本（缺失即抛错）
    X_static, static_sample_names, static_label_names = load_static_samples()

    # 2. 自定义样本
    X_custom, custom_sample_names, custom_label_names = load_custom_samples(gesture_names)

    # 3. 统一 label 空间：固有静态在前，自定义在后（去重、避免与固有静态重名）
    label_names = list(static_label_names)
    for name in custom_label_names:
        if name not in label_names:
            label_names.append(name)
    name_to_idx = {name: i for i, name in enumerate(label_names)}

    # 4. 合并 X / y
    X_parts = [X_static]
    y_parts = [np.array([name_to_idx[n] for n in static_sample_names], dtype=np.int64)]

    if X_custom is not None:
        X_parts.append(X_custom)
        y_parts.append(np.array([name_to_idx[n] for n in custom_sample_names], dtype=np.int64))

    X = np.vstack(X_parts).astype(np.float32)
    y = np.concatenate(y_parts)
    return X, y, np.array(label_names), custom_label_names


def main(gesture_names: list[str] | None = None):
    model_path = settings.resolved_custom_gesture_classifier_model_path

    # 1. 加载数据（固有静态 + 自定义合并）
    try:
        X, y, label_names, custom_label_names = load_all_samples(gesture_names)
    except FileNotFoundError as exc:
        print(f"[FAIL] {exc}")
        sys.exit(1)

    n_samples = X.shape[0]
    n_classes = len(np.unique(y))
    print(f"[INFO] 加载样本: X={X.shape}, y={y.shape}")
    print(f"[INFO] 合并类别 ({len(label_names)}): {label_names.tolist()}")
    print(f"[INFO]   其中自定义类别 ({len(custom_label_names)}): {custom_label_names}")

    if n_classes < 2:
        # 合并了固有静态 5 类后正常不会触发，仅作兜底
        print("[FAIL] 有效类别不足 2 个，无法训练 SVM。请确认 features.npz 有效。")
        sys.exit(1)

    # 2. 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. 训练 SVM
    model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)
    print("\n[EVAL] 训练集评估:")
    print(classification_report(y, y_pred, target_names=label_names, zero_division=0))

    if n_classes >= 3 and n_samples >= n_classes * 5:
        try:
            scores = cross_val_score(model, X_scaled, y, cv=min(5, n_samples // n_classes))
            print(f"[EVAL] {len(scores)}-折交叉验证准确率: {scores.mean():.4f} +/- {scores.std():.4f}")
        except Exception as exc:
            print(f"[EVAL] 交叉验证跳过: {exc}")

    # 4. 保存模型（仅写自定义模型文件，不触碰实时流 gesture_classifier_svm.joblib）
    os.makedirs(settings.models_dir, exist_ok=True)
    bundle = {
        "model": model,
        "scaler": scaler,
        "label_names": label_names,
        "feature_order": "x0..x20_y0..y20_z0..z20",
        "is_custom": True,
    }
    joblib.dump(bundle, model_path)
    print(f"\n[DONE] 模型已保存: {model_path}")
    print(f"[DONE] 类别: {label_names.tolist()}")

    # 5. 更新 is_trained 标记（仅自定义手势行会命中，固有静态类别不在 CustomGesture 表中）
    with SessionLocal() as session:
        gestures = (
            session.query(CustomGesture)
            .filter(CustomGesture.name.in_(custom_label_names))
            .all()
        )
        for g in gestures:
            g.is_trained = True
        session.commit()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="训练自定义手势 SVM 模型（合并固有静态 + 自定义）")
    parser.add_argument("--names", nargs="*", default=None, help="指定训练的自定义手势名列表")
    args = parser.parse_args()
    main(args.names if args.names else None)
