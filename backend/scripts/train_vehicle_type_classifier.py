from __future__ import annotations

import argparse
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SampleItem:
    path: Path
    label_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight vehicle type classifier.")
    parser.add_argument("--dataset-root", required=True, help="Path to the vehicleImages directory.")
    parser.add_argument(
        "--output",
        default="backend/weights/vehicle_classifier_mobilenet_v3_small.pt",
        help="Output checkpoint path.",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pretrained", choices=["auto", "never"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    try:
        import torch
        from PIL import Image
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
        from torchvision import models, transforms
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: {exc}") from exc

    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.exists():
        raise SystemExit(f"Dataset path not found: {dataset_root}")

    class_names = sorted([item.name for item in dataset_root.iterdir() if item.is_dir() and not item.name.startswith(".")])
    if not class_names:
        raise SystemExit(f"No class folders found under: {dataset_root}")
    class_to_idx = {name: index for index, name in enumerate(class_names)}

    samples_by_class: dict[str, list[SampleItem]] = defaultdict(list)
    for class_name in class_names:
        class_dir = dataset_root / class_name
        for image_path in sorted(class_dir.rglob("*")):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            samples_by_class[class_name].append(SampleItem(path=image_path, label_index=class_to_idx[class_name]))

    train_samples: list[SampleItem] = []
    val_samples: list[SampleItem] = []
    for class_name in class_names:
        items = list(samples_by_class[class_name])
        random.shuffle(items)
        split_count = max(1, int(round(len(items) * args.val_ratio)))
        if len(items) <= 3:
            split_count = 1
        val_samples.extend(items[:split_count])
        train_samples.extend(items[split_count:])

    if not train_samples or not val_samples:
        raise SystemExit("Dataset split failed: training or validation set is empty.")

    print("Classes:", class_names)
    for class_name in class_names:
        print(f"  {class_name}: {len(samples_by_class[class_name])}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")

    train_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    class VehicleDataset(Dataset):
        def __init__(self, items: list[SampleItem], transform) -> None:
            self._items = items
            self._transform = transform

        def __len__(self) -> int:
            return len(self._items)

        def __getitem__(self, index: int):
            item = self._items[index]
            with Image.open(item.path) as image:
                rgb_image = image.convert("RGB")
            return self._transform(rgb_image), item.label_index

    train_loader = DataLoader(
        VehicleDataset(train_samples, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        VehicleDataset(val_samples, val_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    pretrained_weights = None
    if args.pretrained == "auto":
        try:
            pretrained_weights = models.MobileNet_V3_Small_Weights.DEFAULT
        except Exception:
            pretrained_weights = None

    try:
        model = models.mobilenet_v3_small(weights=pretrained_weights)
        print(f"Using pretrained weights: {bool(pretrained_weights)}")
    except Exception as exc:
        print(f"Pretrained weight load failed, falling back to random init: {exc}")
        model = models.mobilenet_v3_small(weights=None)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(class_names))

    device = torch.device("cpu")
    model.to(device)

    class_counts = torch.tensor([len(samples_by_class[name]) for name in class_names], dtype=torch.float32)
    class_weights = (class_counts.sum() / torch.clamp(class_counts, min=1.0)) / len(class_names)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_val_accuracy = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_correct = 0
        train_count = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_total += float(loss.item()) * labels.size(0)
            predictions = logits.argmax(dim=1)
            train_correct += int((predictions == labels).sum().item())
            train_count += int(labels.size(0))

        model.eval()
        val_loss_total = 0.0
        val_correct = 0
        val_count = 0
        with torch.inference_mode():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss_total += float(loss.item()) * labels.size(0)
                predictions = logits.argmax(dim=1)
                val_correct += int((predictions == labels).sum().item())
                val_count += int(labels.size(0))

        scheduler.step()

        train_loss = train_loss_total / max(train_count, 1)
        train_accuracy = train_correct / max(train_count, 1)
        val_loss = val_loss_total / max(val_count, 1)
        val_accuracy = val_correct / max(val_count, 1)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is None:
        raise SystemExit("Training did not produce a checkpoint.")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "architecture": "mobilenet_v3_small",
        "classes": class_names,
        "class_to_idx": class_to_idx,
        "image_size": args.image_size,
        "best_val_accuracy": best_val_accuracy,
        "model_state": best_state,
    }
    torch.save(checkpoint, output_path)
    print(f"Saved checkpoint to: {output_path}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    main()
