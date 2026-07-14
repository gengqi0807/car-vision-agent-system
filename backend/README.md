# Backend

## Quick Start

1. Create a virtual environment. Python `3.10` or `3.11` is recommended on Windows.
2. Install backend dependencies with `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and update the values.
4. Initialize the database with `python ..\\scripts\\init_db.py` if you want to create tables before first startup.
5. Start the server with `uvicorn app.main:app --reload`.

## Plate Recognition Setup

- Backend pipeline: the project now uses `YOLO + PaddleOCR` for Chinese license plate detection and text recognition.
- Detection weights: configure `PLATE_YOLO_MODEL_PATH` to point to your plate detector weights. The default sample is `weights/open_traffic_flow_best.pt`.
- OCR runtime: PaddleOCR downloads and caches its runtime files on first use. The Ultralytics runtime cache lives under `backend/runtime/ultralytics`.
- OCR options: tune `PLATE_OCR_CONFIDENCE_THRESHOLD`, `PADDLEOCR_USE_ANGLE_CLS`, and `PADDLEOCR_LANGUAGE` for your data.
- Large image optimization: `PLATE_MAX_IMAGE_SIDE=1600` downsizes oversized uploads before inference and maps boxes back to the original image.
- Confidence threshold: set `PLATE_CONFIDENCE_THRESHOLD` to filter weak recognition results.
- RTSP stream: the backend can pull an RTSP stream and push processed frames through `/api/v1/plate/ws/stream`.
- Stream performance: tune `PLATE_STREAM_MAX_FPS`, `PLATE_STREAM_PROCESS_EVERY_N_FRAMES`, and `PLATE_STREAM_JPEG_QUALITY` for latency and CPU usage.
- History: recognized plates are stored in `plate_records` and exposed via `/api/v1/plate/history`.
- Optional upload retention: set `PLATE_SAVE_UPLOADS=true` to keep original uploaded images under `backend/uploads/plate`.

## Source Notes

- PaddleOCR GitHub: https://github.com/PaddlePaddle/PaddleOCR
- The official PaddleOCR docs cover Python inference, Windows deployment, and the 3.x runtime API for local OCR integration.

## Current Scope

- FastAPI application skeleton
- API route placeholders for the four major business domains
- Plate recognition now uses YOLO detection plus PaddleOCR recognition for Chinese license plates
- Alert agent and websocket manager scaffolding
