MODELS = [
    "yolo-license-plate",
    "paddleocr",
    "mediapipe-pose",
    "mediapipe-hands",
]


def main() -> None:
    for model in MODELS:
        print(f"Pending download workflow for: {model}")


if __name__ == "__main__":
    main()
