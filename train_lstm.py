"""
============================================================
交警手势识别 LSTM 模型 — 完整训练脚本
============================================================

数据管道:
  视频(.mp4) + 标签(.csv) → 姿态估计(pose_model.pt) → 14关键点
  → 骨骼特征(25维: 13骨长 + 6对夹角sin/cos) → LSTM(25→48→9类)

使用方法:
  1. 确认 DATA_PATH 指向你的数据集根目录
  2. 运行: python train_lstm.py
  3. 模型自动保存到 checkpoints/lstm.pt
  4. 若中断，下次运行会从已有坐标缓存继续（跳过姿态估计）

参数说明（如有需要，修改下方 CONFIG）:
  - LABEL_DELAY=15: 标签延迟帧数，给RNN时间观察手势
  - RESIZE_SIZE: 姿态估计时的图片尺寸
  - NUM_EPOCHS: 训练轮数
  - LR / WEIGHT_DECAY: 学习率与正则化
  - ACCUM_STEPS: 梯度累积步数（变相增大batch size）
============================================================
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from collections import defaultdict
import time

# ---------- 添加项目根目录到 sys.path ----------
sys.path.insert(0, str(Path(__file__).parent))

from constants.enum_keys import PG
from constants.keypoints import aic_bones, aic_bone_pairs
from models.gesture_recognition_model import GestureRecognitionModel
from pgdataset.s1_skeleton import PgdSkeleton
from pgdataset.s3_handcraft import BoneLengthAngle


# ============================================================
# 配置参数（按需修改）
# ============================================================
CONFIG = {
    # 数据集根目录（包含 train/ 和 test/ 子文件夹）
    "DATA_PATH": Path(r"C:\Users\Lenovo\PoliceGestureLong"),

    # 姿态估计时图片缩放到多大
    "RESIZE_SIZE": (512, 512),

    # 标签延迟帧数（手势发生到可见之间有延迟）
    "LABEL_DELAY": 15,

    # 训练超参数
    "NUM_EPOCHS": 30,
    "LR": 0.001,
    "WEIGHT_DECAY": 1e-4,
    "ACCUM_STEPS": 4,  # 梯度累积：每 ACCUM_STEPS 个视频更新一次参数

    # 早停
    "EARLY_STOP_PATIENCE": 8,

    # 随机种子
    "SEED": 42,
}


# ============================================================
# 工具函数
# ============================================================
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def gesture_name(idx: int) -> str:
    names = {
        0: "无手势", 1: "停止", 2: "直行", 3: "左转弯",
        4: "左待转", 5: "右转弯", 6: "变道", 7: "减速", 8: "靠边停车",
    }
    return names.get(idx, f"未知({idx})")


# ============================================================
# 特征提取
# ============================================================
class FeatureExtractor:
    """
    封装姿态估计 + 手工特征提取。
    首次运行: 用 PgdSkeleton 逐帧姿态估计 → 缓存坐标 (.pkl)
    后续运行: 直接从缓存加载坐标
    """

    def __init__(self, data_path: Path, resize_size: tuple):
        self.bla = BoneLengthAngle()
        self.data_path = data_path
        self.train_dataset = PgdSkeleton(data_path, is_train=True, resize_img_size=resize_size)
        self.test_dataset = PgdSkeleton(data_path, is_train=False, resize_img_size=resize_size)

    @staticmethod
    def extract_features(coord_norm: np.ndarray, bla: BoneLengthAngle):
        """
        coord_norm: shape (F, 2, 14) — 已有的归一化坐标
        返回: features (F, C), labels (F,)
        """
        feat_dict = bla.handcrafted_features(coord_norm)  # F==1 或 F>1
        bone_len = feat_dict[PG.BONE_LENGTH]       # (F, B)
        angle_cos = feat_dict[PG.BONE_ANGLE_COS]   # (F, P)
        angle_sin = feat_dict[PG.BONE_ANGLE_SIN]   # (F, P)
        features = np.concatenate([bone_len, angle_cos, angle_sin], axis=1)  # (F, C)
        return features  # shape: (F, C), C = len(aic_bones) + 2*len(aic_bone_pairs)


def apply_label_delay(labels: np.ndarray, delay: int) -> np.ndarray:
    """标签后移 delay 帧，前面填充 0（无手势）"""
    if delay <= 0:
        return labels
    return np.concatenate([np.zeros(delay, dtype=labels.dtype), labels], axis=0)[:len(labels)]


# ============================================================
# 训练与评估
# ============================================================
def train_one_epoch(
    model, dataset, bla, config, device,
    optimizer, criterion, epoch_idx
):
    """训练一轮"""
    model.train()
    total_loss = 0.0
    total_frames = 0
    correct_frames = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    video_indices = list(range(len(dataset)))
    np.random.shuffle(video_indices)

    optimizer.zero_grad()
    n_videos = len(video_indices)

    print(f"\n{'='*60}")
    print(f"Epoch {epoch_idx + 1}/{config['NUM_EPOCHS']} — 训练集 ({n_videos} 个视频)")
    print(f"{'='*60}")

    epoch_start = time.time()

    for batch_i, idx in enumerate(video_indices):
        # 加载数据
        item = dataset[idx]
        coord_norm = item[PG.COORD_NORM]      # (F, 2, 14)
        raw_labels = item[PG.GESTURE_LABEL].astype(np.int64)  # (F,)
        video_name = item[PG.VIDEO_NAME]
        n_frames = coord_norm.shape[0]

        # 手工特征
        features = FeatureExtractor.extract_features(coord_norm, bla)  # (F, C)

        # 标签延迟
        labels = apply_label_delay(raw_labels, config["LABEL_DELAY"])

        # 转为 tensor
        # LSTM 期望输入: (seq_len, batch, input_size)
        feat_tensor = torch.from_numpy(features).float().unsqueeze(1).to(device)  # (F, 1, C)
        label_tensor = torch.from_numpy(labels).long().to(device)                 # (F,)

        # RNN 前向
        h = model.h0()
        c = model.c0()

        _, h, c, class_out = model(feat_tensor, h, c)  # class_out: (F, 1, 9)

        # 损失
        loss = criterion(class_out.squeeze(1), label_tensor)  # (F, 9) vs (F,)
        loss = loss / config["ACCUM_STEPS"]
        loss.backward()

        # 统计
        with torch.no_grad():
            preds = class_out.squeeze(1).argmax(dim=1)  # (F,)
            correct = (preds == label_tensor).sum().item()
            correct_frames += correct
            total_frames += n_frames
            total_loss += loss.item() * config["ACCUM_STEPS"] * n_frames

            for cls in range(9):
                mask = (label_tensor == cls)
                per_class_total[cls] += mask.sum().item()
                per_class_correct[cls] += ((preds == cls) & mask).sum().item()

        # 梯度累积
        if (batch_i + 1) % config["ACCUM_STEPS"] == 0 or batch_i == n_videos - 1:
            optimizer.step()
            optimizer.zero_grad()

        # 进度打印
        if (batch_i + 1) % max(1, n_videos // 10) == 0 or batch_i == n_videos - 1:
            elapsed = time.time() - epoch_start
            pct = 100 * (batch_i + 1) / n_videos
            print(f"  [{batch_i + 1:4d}/{n_videos}] {pct:5.1f}% | "
                  f"loss={total_loss / max(total_frames, 1):.4f} | "
                  f"acc={correct_frames / max(total_frames, 1):.4f} | "
                  f"{elapsed:.0f}s")

    avg_loss = total_loss / max(total_frames, 1)
    avg_acc = correct_frames / max(total_frames, 1)

    print(f"  训练完成: loss={avg_loss:.4f}  acc={avg_acc:.4f}")
    print(f"  每类准确率:")
    for cls in range(9):
        acc = per_class_correct[cls] / max(per_class_total[cls], 1)
        bar = "█" * int(acc * 20)
        print(f"    {cls} {gesture_name(cls):6s}: {acc:.3f} {bar} ({per_class_total[cls]}帧)")

    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, dataset, bla, config, device):
    """在测试集上评估"""
    model.eval()
    total_loss = 0.0
    total_frames = 0
    correct_frames = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    criterion = nn.CrossEntropyLoss(reduction="sum")

    n_videos = len(dataset)
    print(f"\n  测试集 ({n_videos} 个视频):")

    for idx in range(n_videos):
        item = dataset[idx]
        coord_norm = item[PG.COORD_NORM]
        raw_labels = item[PG.GESTURE_LABEL].astype(np.int64)
        n_frames = coord_norm.shape[0]

        features = FeatureExtractor.extract_features(coord_norm, bla)
        labels = apply_label_delay(raw_labels, config["LABEL_DELAY"])

        feat_tensor = torch.from_numpy(features).float().unsqueeze(1).to(device)
        label_tensor = torch.from_numpy(labels).long().to(device)

        h = model.h0()
        c = model.c0()
        _, h, c, class_out = model(feat_tensor, h, c)

        loss = criterion(class_out.squeeze(1), label_tensor)
        total_loss += loss.item()
        total_frames += n_frames

        preds = class_out.squeeze(1).argmax(dim=1)
        correct_frames += (preds == label_tensor).sum().item()

        for cls in range(9):
            mask = (label_tensor == cls)
            per_class_total[cls] += mask.sum().item()
            per_class_correct[cls] += ((preds == cls) & mask).sum().item()

    avg_loss = total_loss / max(total_frames, 1)
    avg_acc = correct_frames / max(total_frames, 1)

    print(f"    测试 loss={avg_loss:.4f}  acc={avg_acc:.4f}")
    for cls in range(9):
        if per_class_total[cls] > 0:
            acc = per_class_correct[cls] / max(per_class_total[cls], 1)
            print(f"      {cls} {gesture_name(cls):6s}: {acc:.3f} ({per_class_total[cls]}帧)")

    return avg_loss, avg_acc


# ============================================================
# 主函数
# ============================================================
def main():
    config = CONFIG
    set_seed(config["SEED"])
    device = get_device()
    print(f"设备: {device}")
    print(f"数据集: {config['DATA_PATH']}")

    # ---------- 1. 加载数据（首次运行会逐帧姿态估计，慢但只做一次）----------
    print("\n[1/4] 加载数据集（首次运行会缓存坐标）...")
    extractor = FeatureExtractor(config["DATA_PATH"], config["RESIZE_SIZE"])
    train_ds = extractor.train_dataset
    test_ds = extractor.test_dataset
    bla = extractor.bla

    print(f"  训练集: {len(train_ds)} 个视频")
    print(f"  测试集: {len(test_ds)} 个视频")

    # ---------- 2. 创建模型 ----------
    print("\n[2/4] 创建 GestureRecognitionModel...")
    model = GestureRecognitionModel(batch=1)
    model.load_ckpt(allow_new=True)  # 有就加载，没有就随机初始化
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  输入维度: {len(aic_bones) + 2 * len(aic_bone_pairs)} (骨长+夹角)")

    # ---------- 3. 优化器 ----------
    optimizer = optim.Adam(model.parameters(), lr=config["LR"], weight_decay=config["WEIGHT_DECAY"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )
    criterion = nn.CrossEntropyLoss(reduction="mean")

    # ---------- 4. 训练循环 ----------
    print("\n[3/4] 开始训练...")
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(config["NUM_EPOCHS"]):
        train_loss, train_acc = train_one_epoch(
            model, train_ds, bla, config, device,
            optimizer, criterion, epoch,
        )

        # 验证
        val_loss, val_acc = evaluate(model, test_ds, bla, config, device)

        scheduler.step(val_loss)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            model.save_ckpt()
            print(f"  ✅ 新的最佳模型已保存！val_acc={val_acc:.4f}")
        else:
            patience_counter += 1
            print(f"  未提升 ({patience_counter}/{config['EARLY_STOP_PATIENCE']})")

        if patience_counter >= config["EARLY_STOP_PATIENCE"]:
            print(f"\n  早停！最佳 epoch={best_epoch}, val_acc={best_val_acc:.4f}")
            break

    # ---------- 5. 最终报告 ----------
    print("\n[4/4] 训练完成！")
    print(f"  最佳测试准确率: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"  模型已保存到: checkpoints/lstm.pt")

    # 用最佳模型再测一次
    model.load_ckpt(allow_new=False)
    final_val_loss, final_val_acc = evaluate(model, test_ds, bla, config, device)
    print(f"\n  最终测试集评估:")
    print(f"    loss={final_val_loss:.4f}  acc={final_val_acc:.4f}")


if __name__ == "__main__":
    main()
