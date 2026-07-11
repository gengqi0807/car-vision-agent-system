"""
动态手势数据采集脚本。

用摄像头或视频文件录制动态手势片段，保存为 .avi 文件。

用法:
  # 采集 swipe_up (向上滑动)
  python scripts/collect_dynamic_data.py --gesture swipe_up --count 10

  # 采集所有动态手势（逐一录制）
  python scripts/collect_dynamic_data.py --all --count 5

  # 从已有视频中截取片段保存
  python scripts/collect_dynamic_data.py --gesture wave --source test.mp4 --start 2 --duration 3

流程:
  1. 选择要采集的手势
  2. 打开摄像头，显示当前手势名 + 倒计时
  3. 录制 N 段视频，每段自动保存
  4. 按 Q 跳过当前手势 / ESC 退出

输出目录:
  owner_gesture_dataset_/
    swipe_up/
      subj_001_r1.avi
      subj_001_r2.avi
    swipe_down/
      ...
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "owner_gesture_dataset_"
DEFAULT_DURATION = 3  # 秒
DEFAULT_FPS = 20

DYNAMIC_GESTURES = ["swipe_up", "swipe_down", "swipe_left", "swipe_right", "wave"]


def get_next_filename(gesture_dir: Path) -> str:
    """为当前手势目录生成下一个文件名 subj_xxx_r1.avi。"""
    existing = list(gesture_dir.glob("*.avi"))
    max_subj = 0
    for f in existing:
        # 格式: subj_001_r1.avi
        try:
            num = int(f.stem.split("_")[1])
            if num > max_subj:
                max_subj = num
        except (IndexError, ValueError):
            pass
    return f"subj_{max_subj + 1:03d}"


def collect_gesture(
    gesture_name: str,
    count: int,
    duration: float = DEFAULT_DURATION,
    fps: int = DEFAULT_FPS,
) -> None:
    """录制指定手势的 N 段视频。"""
    gesture_dir = DATASET_DIR / gesture_name
    gesture_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  采集手势: {gesture_name}")
    print(f"  目标次数: {count}  每段时长: {duration}s  帧率: {fps}fps")
    print(f"  输出目录: {gesture_dir}")
    print(f"{'='*60}\n")

    cap = cv2.VideoCapture(0)  # 默认摄像头
    if not cap.isOpened():
        print("[FAIL] 无法打开摄像头")
        return

    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    saved_count = 0
    state = "ready"  # ready | countdown | recording | saving
    state_start = time.time()
    countdown_seconds = 3

    recorder: cv2.VideoWriter | None = None
    base_filename = get_next_filename(gesture_dir)

    print(f"  按 SPACE 开始录制 | Q 跳过 | ESC 退出")
    print(f"  当前进度: {saved_count}/{count}\n")

    while saved_count < count:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        h, w = display.shape[:2]

        # 状态栏
        cv2.rectangle(display, (0, 0), (w, 80), (0, 0, 0), -1)

        # 手势名
        cv2.putText(display, f"Gesture: {gesture_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        # 进度
        cv2.putText(display, f"Progress: {saved_count}/{count}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        if state == "ready":
            cv2.putText(display, "Press SPACE to start", (w // 2 - 130, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        elif state == "countdown":
            elapsed = time.time() - state_start
            remaining = countdown_seconds - int(elapsed)
            if remaining <= 0:
                # 开始录制
                state = "recording"
                state_start = time.time()
                filename = f"{base_filename}_r{saved_count + 1}.avi"
                filepath = str(gesture_dir / filename)
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                recorder = cv2.VideoWriter(filepath, fourcc, fps, (w, h))
                print(f"  [REC] {filename}")
            else:
                cv2.putText(display, f"Starting in {remaining}...", (w // 2 - 100, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        elif state == "recording":
            elapsed = time.time() - state_start
            remaining = duration - elapsed
            if remaining <= 0:
                # 停止录制
                if recorder:
                    recorder.release()
                    recorder = None
                saved_count += 1
                state = "saving"
                state_start = time.time()
                print(f"  [OK] 已保存 ({saved_count}/{count})")
            else:
                cv2.circle(display, (30, 60), 10, (0, 0, 255), -1)  # 录制指示器
                cv2.putText(display, f"REC {remaining:.1f}s", (50, 68),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                if recorder:
                    recorder.write(frame)

        elif state == "saving":
            # 短暂停顿后回到 ready
            if time.time() - state_start > 1.0:
                state = "ready"

        cv2.imshow("Dynamic Gesture Collection", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            print("\n[STOP] 用户退出")
            break
        elif key == ord('q'):
            print(f"\n[SKIP] 跳过 {gesture_name}")
            break
        elif key == ord(' ') and state == "ready":
            state = "countdown"
            state_start = time.time()

    if recorder:
        recorder.release()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE] {gesture_name}: 已采集 {saved_count}/{count} 段视频")


def main():
    parser = argparse.ArgumentParser(description="动态手势数据采集")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--gesture", choices=DYNAMIC_GESTURES, help="指定单个手势")
    group.add_argument("--all", action="store_true", help="采集所有动态手势")
    parser.add_argument("--count", type=int, default=10, help="录制段数 (default: 10)")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION, help=f"每段时长秒 (default: {DEFAULT_DURATION})")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help=f"帧率 (default: {DEFAULT_FPS})")
    args = parser.parse_args()

    if args.all:
        gestures = DYNAMIC_GESTURES
    else:
        gestures = [args.gesture]

    print(f"手势列表: {gestures}")
    print(f"每手势: {args.count} 段 × {args.duration}s")
    print(f"总预计录制: {len(gestures) * args.count * args.duration}s")

    for i, gesture in enumerate(gestures):
        if i > 0:
            print("\n准备好后按任意键继续下一个手势...")
            input()
        collect_gesture(gesture, args.count, args.duration, args.fps)

    print("\n[DONE] 所有手势采集完成！")
    print(f"下一步: python scripts/extract_dynamic_features.py")
    print(f"然后  : python scripts/train_dynamic_lstm.py")


if __name__ == "__main__":
    main()
