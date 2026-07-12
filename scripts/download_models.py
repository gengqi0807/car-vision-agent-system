"""Download pre-trained model files for the car-vision agent system.

Usage (from project root)::

    python scripts/download_models.py                # download all models
    python scripts/download_models.py --model hand_landmarker   # single model
    python scripts/download_models.py --model pose_landmarker   # single model

Models are saved into ``backend/models/`` by default.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve

# ---------------------------------------------------------------------------
# Model registry  -----------------------------------------------------------
# ---------------------------------------------------------------------------

MODELS: dict[str, dict[str, str]] = {
    "hand_landmarker": {
        "url": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "filename": "hand_landmarker.task",
        "description": "MediaPipe Hand Landmarker (21 keypoints per hand)",
        "sha256": "",  # will be populated after download
    },
    "pose_landmarker": {
        "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        "filename": "pose_landmarker_lite.task",
        "description": "MediaPipe Pose Landmarker Lite (33 body keypoints)",
        "sha256": "",  # will be populated after download
    },
    # future entries:
    # "yolo_license_plate": {
    #     "url": "...",
    #     "filename": "yolo_license_plate.pt",
    # },
    # "paddleocr": {
    #     "url": "...",
    #     "filename": "...",
    # },
}

# ---------------------------------------------------------------------------
# Helpers  ------------------------------------------------------------------
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = PROJECT_ROOT / "backend" / "models"


def _sha256_hex(path: Path) -> str:
    """Compute SHA-256 digest of *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _progress_hook(block_count: int, block_size: int, total_size: int) -> None:
    """Simple download progress reporter (ASCII-only for cross-platform safety)."""
    downloaded = block_count * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 // total_size)
        bar_len = 50
        filled = bar_len * percent // 100
        bar = "=" * filled + "-" * (bar_len - filled)
        sys.stdout.write(f"\r  [{bar}] {percent:3d}%  {downloaded:>10,} / {total_size:>10,}")
    else:
        sys.stdout.write(f"\r  Downloaded {downloaded:,} bytes")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main  ---------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download pre-trained model files for car-vision-agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available models:\n  " + "\n  ".join(MODELS),
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS) + ["all"],
        default="all",
        help="Which model(s) to download (default: all)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Destination directory (default: {DEFAULT_DEST})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if file already exists",
    )
    return parser.parse_args()


def download_one(key: str, dest_dir: Path, *, force: bool = False) -> bool:
    """Download a single model.  Returns ``True`` on success."""
    info = MODELS[key]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / info["filename"]

    if dest_path.exists() and not force:
        print(f"[SKIP] {info['filename']} already exists at {dest_path}")
        return True

    print(f"\n[FETCH] {info['description']}")
    print(f"  URL : {info['url']}")
    print(f"  DEST: {dest_path}")

    try:
        urlretrieve(info["url"], str(dest_path), reporthook=_progress_hook)
        print()  # newline after progress bar
    except Exception as exc:
        print(f"\n[FAIL] Download error: {exc}")
        return False

    file_size = dest_path.stat().st_size
    digest = _sha256_hex(dest_path)
    info["sha256"] = digest

    print(f"[OK]   {info['filename']}  ({file_size:,} bytes)")
    print(f"       SHA-256: {digest}")
    return True


def main() -> None:
    args = _parse_args()
    keys = list(MODELS) if args.model == "all" else [args.model]

    print(f"Models will be saved to: {args.dest.resolve()}")
    ok = 0
    for key in keys:
        if download_one(key, args.dest, force=args.force):
            ok += 1
        else:
            print(f"[WARN] {key} download failed — you may need to download it manually.")

    print(f"\nDone: {ok}/{len(keys)} models ready.")


if __name__ == "__main__":
    main()
