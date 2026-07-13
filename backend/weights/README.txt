Current default detector weight:
- `backend/weights/open_traffic_flow_best.pt`

Current plate pipeline:
- detect plate regions with YOLO
- read plate text with PaddleOCR

If you replace detector weights, also update:
- `PLATE_YOLO_MODEL_PATH` in `backend/.env`

If you want to fall back to a generic YOLO model temporarily:
- `PLATE_YOLO_MODEL_PATH=yolov8n.pt`
