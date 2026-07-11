当前默认使用这个车牌检测权重：
- `backend/weights/open_traffic_flow_best.pt`

当前主流程是：
- 先用车牌检测模型定位车牌框
- 再把车牌区域交给 OCR 识别

如果后续更换模型，请同步修改：
- `backend/.env` 中的 `PLATE_YOLO_MODEL_PATH`

如果临时没有检测模型，也可以改回官方通用模型，例如：
- `PLATE_YOLO_MODEL_PATH=yolov8n.pt`
