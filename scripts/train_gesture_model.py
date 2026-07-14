"""
阶段 2：模型训练器
- 加载 features.npz 归一化特征矩阵
- StandardScaler 特征标准化
- 训练 SVM 分类器
- 保存模型 + scaler 到 models/ 目录
"""

import os
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 项目根
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.models_infer.hand_utils import SingleClassWrapper

# ----------------------------------------------------------------
# 路径
# ----------------------------------------------------------------
FEATURES_FILE = PROJECT_ROOT / "owner_gesture_dataset_features" / "features.npz"
MODEL_OUTPUT = os.path.join(settings.models_dir, "gesture_classifier_svm.joblib")


def main():
    # --- 1. 加载特征 -----------------------------------------------------------
    if not FEATURES_FILE.exists():
        print(f"[FAIL] 特征文件不存在: {FEATURES_FILE}")
        print("       请先运行: python scripts/extract_features.py")
        sys.exit(1)

    data = np.load(FEATURES_FILE)
    X: np.ndarray = data["X"]      # (N, 63)
    y: np.ndarray = data["y"]      # (N,)
    label_names: np.ndarray = data["label_names"]

    print(f"[INFO] 加载特征: X={X.shape}, y={y.shape}")
    print(f"[INFO] 类别: {label_names.tolist()}")

    n_classes = len(np.unique(y))
    print(f"[INFO] 类别数: {n_classes}")

    # --- 2. 标准化 -------------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 3. 训练 SVM -----------------------------------------------------------
    if n_classes == 1:
        print("[WARN] 仅 1 个类别，使用 OneClassSVM（临时方案），扩充数据集后将自动切换为 SVC")
        from sklearn.svm import OneClassSVM
        ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
        ocsvm.fit(X_scaled)
        model = SingleClassWrapper(ocsvm, label_names, scaler)
        print("[INFO] OneClassSVM 训练完成")
    else:
        model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
        model.fit(X_scaled, y)

        # 评估
        y_pred = model.predict(X_scaled)
        print("\n[EVAL] 训练集评估:")
        print(classification_report(y, y_pred, target_names=label_names, zero_division=0))

        # 交叉验证（类别 >= 3 时更有意义）
        if n_classes >= 3 and len(X) >= n_classes * 5:
            try:
                scores = cross_val_score(model, X_scaled, y, cv=min(5, len(X) // n_classes))
                print(f"[EVAL] {len(scores)}-折交叉验证准确率: {scores.mean():.4f} ± {scores.std():.4f}")
            except Exception as e:
                print(f"[EVAL] 交叉验证跳过: {e}")

    # --- 4. 保存模型 -----------------------------------------------------------
    os.makedirs(settings.models_dir, exist_ok=True)

    bundle = {
        "model": model,
        "scaler": scaler,
        "label_names": label_names,
        "feature_order": "x0..x20_y0..y20_z0..z20",
    }
    joblib.dump(bundle, MODEL_OUTPUT)
    print(f"\n[DONE] 模型已保存: {MODEL_OUTPUT}")
    print(f"[DONE] 类别: {label_names.tolist()}")


if __name__ == "__main__":
    main()
