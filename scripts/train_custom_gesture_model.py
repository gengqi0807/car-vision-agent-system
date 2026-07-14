"""
自定义手势 SVM 训练脚本。

输入: 数据库 custom_gesture_samples 表中所有样本（或指定手势）的关键点
输出: models/custom_gesture_classifier_svm.joblib
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


def load_all_samples(gesture_names: list[str] | None = None):
    """从数据库加载所有已采集的自定义手势样本，返回 (X, y, label_names)。"""
    with SessionLocal() as session:
        query = session.query(CustomGesture)
        if gesture_names:
            query = query.filter(CustomGesture.name.in_(gesture_names))
        gestures = query.all()

        if not gestures:
            return None, None, None

        X_list: list[np.ndarray] = []
        y_list: list[int] = []
        label_names: list[str] = []

        for cls_idx, gesture in enumerate(gestures):
            label_names.append(gesture.name)
            for sample in gesture.samples:
                kps = sample.keypoints
                if not kps or len(kps) != 21:
                    continue
                feat = normalize_hand_landmarks_array(kps)
                X_list.append(feat)
                y_list.append(cls_idx)

        if not X_list:
            return None, None, None

        X = np.stack(X_list, axis=0)
        y = np.array(y_list, dtype=np.int64)
        return X, y, np.array(label_names)


def main(gesture_names: list[str] | None = None):
    model_path = settings.resolved_custom_gesture_classifier_model_path

    # 1. 加载数据
    X, y, label_names = load_all_samples(gesture_names)
    if X is None or y is None:
        print("[FAIL] 没有可用的自定义手势样本，请先采集数据。")
        sys.exit(1)

    n_samples = X.shape[0]
    n_classes = len(np.unique(y))
    print(f"[INFO] 加载样本: X={X.shape}, y={y.shape}")
    print(f"[INFO] 类别 ({n_classes}): {label_names.tolist()}")

    if len(label_names) < 2:
        print("[FAIL] 自定义手势至少需要 2 个类别才能训练 SVM，当前仅有 %d 个。" % len(label_names))
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

    # 4. 保存模型
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

    # 5. 更新 is_trained 标记
    with SessionLocal() as session:
        gestures = (
            session.query(CustomGesture)
            .filter(CustomGesture.name.in_(label_names.tolist()))
            .all()
        )
        for g in gestures:
            g.is_trained = True
        session.commit()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="训练自定义手势 SVM 模型")
    parser.add_argument("--names", nargs="*", default=None, help="指定训练的手势名列表")
    args = parser.parse_args()
    main(args.names if args.names else None)
