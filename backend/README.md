# Backend

## Quick Start

1. Create a virtual environment. Python `3.10` or `3.11` is recommended on Windows.
2. Install backend dependencies with `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and update the values.
4. Initialize the database with `python ..\\scripts\\init_db.py` if you want to create tables before first startup.
5. Start the server with `uvicorn app.main:app --reload`.

## Plate Recognition Setup

- Backend engine: the project now uses `HyperLPR3`, a Chinese license plate recognition framework that can be installed directly with `pip`.
- Model files: `hyperlpr3` packages its own runtime assets, so you do not need to place a separate `.pt` file under `backend/weights`.
- Runtime cache: the first import writes model assets under `backend/runtime/.hyperlpr3`. You can change that with `HYPERLPR_HOME_DIR`.
- Detection level: set `HYPERLPR_DETECT_LEVEL=high` for better accuracy or `low` for faster CPU inference.
- Confidence threshold: set `PLATE_CONFIDENCE_THRESHOLD` to filter weak recognition results.
- History: recognized plates are stored in `plate_records` and exposed via `/api/v1/plate/history`.
- Optional upload retention: set `PLATE_SAVE_UPLOADS=true` to keep original uploaded images under `backend/uploads/plate`.

## Source Notes

- HyperLPR3 GitHub: https://github.com/szad670401/HyperLPR
- The repository documents `pip install hyperlpr3`, Python/Windows support, a direct Python API, and Chinese plate types supported by the packaged model.
- We evaluated OpenALPR as an alternative, but its official CLI documentation only lists `us` and `eu` country codes, so it is not a good default for Chinese license plates.

## Current Scope

- FastAPI application skeleton
- API route placeholders for the four major business domains
- Plate recognition now uses HyperLPR3 for Chinese license plate detection and recognition
- Alert agent and websocket manager scaffolding
