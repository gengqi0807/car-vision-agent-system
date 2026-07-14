from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.logger import configure_logging  # noqa: E402
from app.services.plate_service import PlateService  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run plate video recognition directly from the backend CLI.")
    parser.add_argument("input", help="Local path to the input video file.")
    parser.add_argument(
        "--output",
        help="Local path to save the annotated output video. Defaults to backend/uploads/plate/cli/<name>.mp4",
    )
    parser.add_argument(
        "--preview-image",
        help="Optional local path to periodically save the latest preview frame as a JPG.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Warm up OCR and detector models before processing starts.",
    )
    return parser.parse_args()


def resolve_output_path(input_path: Path, output_arg: str | None) -> Path:
    if output_arg:
        return Path(output_arg).expanduser().resolve()
    output_dir = BACKEND_ROOT / "uploads" / "plate" / "cli"
    output_dir.mkdir(parents=True, exist_ok=True)
    return (output_dir / f"{input_path.stem}.annotated.mp4").resolve()


def main() -> int:
    configure_logging()
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists() or not input_path.is_file():
        print(f"[error] Input video not found: {input_path}")
        return 1

    output_path = resolve_output_path(input_path, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_image_path = Path(args.preview_image).expanduser().resolve() if args.preview_image else None
    if preview_image_path is not None:
        preview_image_path.parent.mkdir(parents=True, exist_ok=True)

    service = PlateService()
    if args.warmup:
        print("[info] Warming up OCR and detection models...")
        service.warmup_runtime(silent=True)
        print("[info] Warmup finished.")

    cv2 = service._require_cv2()
    started_at = time.monotonic()
    last_progress_print_at = 0.0

    def on_progress(*, processed_frame_count: int, total_frames: int, detections, annotated_frame=None) -> None:
        nonlocal last_progress_print_at
        now = time.monotonic()
        should_print = (
            processed_frame_count <= 1
            or total_frames <= 0
            or processed_frame_count >= total_frames
            or now - last_progress_print_at >= 1.0
        )
        if should_print:
            elapsed = now - started_at
            progress = (processed_frame_count / total_frames * 100.0) if total_frames > 0 else 0.0
            print(
                f"[progress] frame={processed_frame_count}/{total_frames or '?'} "
                f"progress={progress:.1f}% detections={len(detections)} elapsed={elapsed:.1f}s"
            )
            last_progress_print_at = now

        if annotated_frame is not None and preview_image_path is not None:
            try:
                cv2.imwrite(str(preview_image_path), annotated_frame)
            except Exception as exc:
                print(f"[warn] Failed to write preview image: {exc}")

    print(f"[info] Input video:  {input_path}")
    print(f"[info] Output video: {output_path}")
    if preview_image_path is not None:
        print(f"[info] Preview JPG:  {preview_image_path}")

    result = service._process_video_file(
        source_path=input_path,
        output_path=output_path,
        filename=input_path.name,
        progress_callback=on_progress,
    )

    print("[done] Video processing finished.")
    print(f"[done] Processed frames: {result.processed_frame_count}")
    print(f"[done] Duration (video): {result.duration_seconds}")
    print(f"[done] Output video: {output_path}")
    if result.detections:
        print("[done] Detections:")
        for item in result.detections:
            plate_text = item.plate_number or "unread"
            print(
                f"  - {plate_text} | {item.plate_color} | {item.vehicle_type} | "
                f"conf={item.confidence:.3f} | bbox={item.bbox}"
            )
    else:
        print("[done] No final plate detections.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
