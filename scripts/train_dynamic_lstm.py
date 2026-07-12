"""
阶段 2-D：动态手势 LSTM 模型训练器（PyTorch）。

- 加载 features_dynamic.npz 变长序列样本
- Pad + 打包为 DataLoader
- 训练双向 LSTM 分类器
- 保存 gesture_dynamic_lstm.pt 到 models/ 目录

用法:
  python scripts/train_dynamic_lstm.py
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset

# 项目根
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.models_infer.dynamic_lstm import BiLSTMGesture

# ----------------------------------------------------------------
# 路径
# ----------------------------------------------------------------
FEATURES_FILE = PROJECT_ROOT / "owner_gesture_dataset_features" / "features_dynamic.npz"
MODEL_OUTPUT = os.path.join(settings.models_dir, "gesture_dynamic_lstm.pt")

# ----------------------------------------------------------------
# 超参数
# ----------------------------------------------------------------
BATCH_SIZE = 8
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 200
EARLY_STOP_PATIENCE = 25
RANDOM_SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)


# ----------------------------------------------------------------
# 数据集
# ----------------------------------------------------------------

class GestureSequenceDataset(Dataset):
    def __init__(self, sequences: list[np.ndarray], labels: np.ndarray):
        self.sequences = [torch.from_numpy(s).float() for s in sequences]
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.labels[idx], len(self.sequences[idx])


def collate_fn(batch):
    """Pad 变长序列到 batch 内最大长度。"""
    seqs, labels, lengths = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    # pad to max length in batch
    max_len = max(lengths).item()
    padded = torch.zeros(len(seqs), max_len, seqs[0].shape[1])
    for i, s in enumerate(seqs):
        padded[i, :s.shape[0], :] = s

    return padded, labels, lengths


# ----------------------------------------------------------------
# 训练 / 验证
# ----------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y, lengths in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x, lengths)
        loss = criterion(logits, y)
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for x, y, lengths in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x, lengths)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(y.cpu().numpy().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


# ----------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------

def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # --- 1. 加载数据 -----------------------------------------------------------
    if not FEATURES_FILE.exists():
        print(f"[FAIL] 特征文件不存在: {FEATURES_FILE}")
        print("       请先运行: python scripts/extract_dynamic_features.py")
        sys.exit(1)

    data = np.load(FEATURES_FILE, allow_pickle=True)
    sequences: np.ndarray = data["sequences"]      # (N,) object array
    labels: np.ndarray = data["labels"]              # (N,) int
    label_names: np.ndarray = data["label_names"]    # (C,) str

    # 转为 Python list
    seq_list = [np.array(s, dtype=np.float32) for s in sequences]
    y = np.array(labels, dtype=np.int32)

    n_classes = len(np.unique(y))
    n_samples = len(seq_list)

    print(f"[INFO] 加载特征: {n_samples} 个序列样本")
    print(f"[INFO] 类别: {label_names.tolist()} ({n_classes} 类)")

    # 序列长度统计
    lengths = [s.shape[0] for s in seq_list]
    print(f"[INFO] 序列长度 — min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.1f}, median={np.median(lengths):.0f}")

    # 类别分布
    for name, count in zip(label_names, np.bincount(y, minlength=n_classes)):
        print(f"  {name}: {count}")

    if n_samples < 10:
        print("[WARN] 样本太少，训练可能不收敛")

    # --- 2. 划分训练/验证 ------------------------------------------------------
    if n_samples >= n_classes * 3:
        X_train_idx, X_val_idx = train_test_split(
            range(n_samples), test_size=0.2, random_state=RANDOM_SEED, stratify=y
        )
    else:
        X_train_idx = list(range(n_samples))
        X_val_idx = []

    seq_train = [seq_list[i] for i in X_train_idx]
    y_train = y[np.array(X_train_idx)]

    train_set = GestureSequenceDataset(seq_train, y_train)
    train_loader = DataLoader(
        train_set, batch_size=min(BATCH_SIZE, n_samples), shuffle=True, collate_fn=collate_fn
    )

    val_loader = None
    if X_val_idx:
        seq_val = [seq_list[i] for i in X_val_idx]
        y_val = y[np.array(X_val_idx)]
        val_set = GestureSequenceDataset(seq_val, y_val)
        val_loader = DataLoader(
            val_set, batch_size=min(BATCH_SIZE, len(X_val_idx)), shuffle=False, collate_fn=collate_fn
        )
        print(f"[INFO] 训练: {len(X_train_idx)}  验证: {len(X_val_idx)}")
    else:
        print("[INFO] 样本不足，跳过验证集")

    # --- 3. 构建模型 -----------------------------------------------------------
    model = BiLSTMGesture(
        input_size=67,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=n_classes,
        dropout=DROPOUT,
    ).to(DEVICE)

    # 类别不平衡加权
    class_counts = np.bincount(y_train, minlength=n_classes)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * n_classes
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    print(f"\n[INFO] 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[INFO] 设备: {DEVICE}")
    print(f"[INFO] 开始训练 (epochs={NUM_EPOCHS}, early_stop={EARLY_STOP_PATIENCE})\n")

    # --- 4. 训练循环 -----------------------------------------------------------
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)

        elapsed = time.time() - t0

        if val_loader:
            val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion)
            scheduler.step(val_loss)

            # Early stopping 基于 val_loss 而非 val_acc，避免 acc=1.0 后无法超越
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            print(
                f"  Epoch {epoch:3d}/{NUM_EPOCHS} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e} | "
                f"{elapsed:.1f}s"
            )

            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\n[INFO] Early stopping at epoch {epoch}")
                break
        else:
            print(f"  Epoch {epoch:3d}/{NUM_EPOCHS} | train_loss={train_loss:.4f} "
                  f"train_acc={train_acc:.4f} | {elapsed:.1f}s")

            if best_state is None or train_loss < best_val_loss:
                best_val_loss = train_loss
                best_val_acc = train_acc
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # --- 5. 恢复最佳权重 & 最终评估 --------------------------------------------
    if best_state:
        model.load_state_dict(best_state)
        print(f"\n[INFO] 恢复最佳模型 (epoch {best_epoch}, val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f})")

    if val_loader:
        _, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion)
        print(f"\n[EVAL] 验证集准确率: {val_acc:.4f}")
        print("\n[EVAL] 分类报告:")
        print(classification_report(val_labels, val_preds, target_names=list(label_names),
                                    zero_division=0))
        print("[EVAL] 混淆矩阵:")
        cm = confusion_matrix(val_labels, val_preds)
        print(cm)

    # --- 6. 保存模型 -----------------------------------------------------------
    os.makedirs(settings.models_dir, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "label_names": label_names.tolist(),
        "input_size": 67,
    }
    torch.save(checkpoint, MODEL_OUTPUT)
    print(f"\n[DONE] 模型已保存: {MODEL_OUTPUT}")
    print(f"[DONE] 类别: {label_names.tolist()}")


if __name__ == "__main__":
    main()
