# 改动说明 — MediaPipe 推理层实现

> 日期：2026-07-07  
> 分支：`main`  
> 范围：`backend/` · `scripts/` · `.gitignore` · `Dockerfile`

---

## 改动概览

将 `models_infer/` 下的 MediaPipe Hands / Pose 推理类从**占位桩（stub）**提升为**真实推理实现**，配套完成模型下载脚本、依赖管理、配置项和 Docker 适配。

---

## 文件变更清单

### 新增/重写

| 文件 | 说明 |
|------|------|
| `backend/models/.gitkeep` | 模型存放目录（`.task` 文件已加入 `.gitignore`） |
| `backend/models/hand_landmarker.task` | MediaPipe 手部地标模型（7.8 MB） |
| `backend/models/pose_landmarker_lite.task` | MediaPipe 姿态地标模型（5.8 MB） |
| `scripts/download_models.py` | 重写：实际下载逻辑 + SHA-256 校验 + 进度条 |

### 修改

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `backend/requirements.txt` | 添加依赖 | 新增 `mediapipe`、`opencv-python-headless`、`numpy`、`aiofiles` |
| `backend/app/core/config.py` | 配置扩展 | 新增 `models_dir`、`hand_landmarker_model`、`pose_landmarker_model` + 分辨率属性 |
| `backend/app/models_infer/mediapipe_hands.py` | 完整重写 | 从 3 行桩 → 120+ 行真实推理（加载 .task → `detect()` → 返回 21 关键点） |
| `backend/app/models_infer/mediapipe_pose.py` | 完整重写 | 从 3 行桩 → 120+ 行真实推理（33 体关键点 + visibility 分数） |
| `backend/app/models_infer/__init__.py` | 导出更新 | 显式导出全部 5 个推理类 |
| `backend/app/models_infer/README.md` | 文档更新 | 反映当前实现状态 |
| `backend/Dockerfile` | 系统依赖 | 新增 `libgl1-mesa-glx`、`libglib2.0-0`（MediaPipe 运行时依赖） |
| `.gitignore` | 规则补充 | 忽略 `backend/models/*.task`、`*.pt`、`*.onnx`、`*.tflite` |
| `backend/app/services/owner_gesture_service.py` | 接入推理 | 新增 `process_frame()` → 调用 `MediaPipeHands.infer()`，lazy-load 模型 |
| `backend/app/services/police_gesture_service.py` | 接入推理 | 新增 `process_frame()` → 调用 `MediaPipePose.infer()`，lazy-load 模型 |
| `backend/app/api/v1/owner_gesture.py` | API 改造 | `/current` GET → POST，接收 `UploadFile`，传给 service 真实推理 |
| `backend/app/api/v1/police_gesture.py` | API 改造 | `/current` GET → POST，接收 `UploadFile`，传给 service 真实推理 |

---

## 架构说明

```
scripts/download_models.py          ← 一键下载模型到 backend/models/
        │
        ▼
backend/models/
  ├── hand_landmarker.task           ← 手部地标模型（21 关键点）
  └── pose_landmarker_lite.task      ← 姿态地标模型（33 关键点）
        │
        ▼
backend/app/core/config.py           ← Settings.models_dir · resolved_*_model_path
        │
        ▼
backend/app/models_infer/
  ├── mediapipe_hands.py             ← MediaPipeHands.infer(source) → dict
  └── mediapipe_pose.py              ← MediaPipePose.infer(source) → dict
        │
        ▼
backend/app/services/
  ├── owner_gesture_service.py       ← OwnerGestureService.process_frame() → GestureFrameResult
  └── police_gesture_service.py      ← PoliceGestureService.process_frame() → GestureFrameResult
        │
        ▼
backend/app/api/v1/
  ├── owner_gesture.py               ← POST /api/v1/owner-gesture/current   (multipart file)
  └── police_gesture.py              ← POST /api/v1/police-gesture/current  (multipart file)

        …（其余 YoloDetector / OCRRecognizer / GestureClassifier 仍为占位桩）
```

**调用链路（以手部手势为例）**：

```
POST /api/v1/owner-gesture/current  (form-data: file=hand_frame.jpg)
  → owner_gesture.py: current_owner_gesture(file: UploadFile)
    → OwnerGestureService.process_frame(image_bytes, filename)
      → MediaPipeHands._ensure_landmarker()           # lazy-load .task
      → MediaPipeHands._load_image(frame)             # bytes → np.ndarray → mp.Image
      → landmarker.detect(mp_image)                   # 21 关键点 × N 只手
      → _result_to_dict()                             # → {"num_hands_detected": 1, "keypoints": [...]}
    → GestureFrameResult(gesture="检测到 1 只手", keypoints=[...])
  → 200 JSON response
```

---

## 使用方式

### 首次运行（已由本次改动完成）

```bash
# 下载模型文件（已包含在本次 PR 中，如重建环境需重新执行）
python scripts/download_models.py

# 安装依赖
pip install -r backend/requirements.txt
```

### 验证推理

**方式一：API 调用（推荐）**

```bash
# 手部手势推理
curl -X POST http://localhost:8000/api/v1/owner-gesture/current \
  -F "file=@test_hand_frame.jpg"

# 交警姿态推理
curl -X POST http://localhost:8000/api/v1/police-gesture/current \
  -F "file=@test_pose_frame.jpg"
```

响应示例：

```json
{
  "gesture": "检测到 1 只手",
  "confidence": 0.99,
  "keypoints": [
    {"x": 0.42, "y": 0.18, "score": -0.03},
    {"x": 0.44, "y": 0.19, "score": -0.04},
    ...
  ],
  "updated_at": "2026-07-07T12:00:00.000Z"
}
```

**方式二：本地 Python 直接调用**

```python
from app.models_infer import MediaPipeHands, MediaPipePose

# 手部检测
with MediaPipeHands() as hands:
    print(hands.infer("path/to/test_frame.jpg"))

# 姿态检测
with MediaPipePose() as pose:
    print(pose.infer("path/to/test_frame.jpg"))
```

### 服务层 lazy-load 策略

`OwnerGestureService` / `PoliceGestureService` 使用 **lazy-load** 初始化 MediaPipe 模型——服务类在模块导入时即可创建实例，只有当 `POST /current` 首次被调用时才会触发 `.task` 文件的加载和校验。这意味着：

- 缺少模型文件时服务仍可正常启动，仅在推理时返回错误
- 模型仅加载一次，后续请求复用同一实例

---

## 后续待办

1. **手势分类器实现**：`GestureClassifier.classify()` 需在关键点数据联通后，实现从 21 关键点/33 关键点 → 具体手势标签的规则或模型推理（目前返回 `"检测到 N 只手/人"` 占位文案）。
2. **YOLO / OCR 推理层**：`YoloDetector`、`OCRRecognizer` 仍为占位桩，需后续实现。
3. **WebSocket 视频流模式**：当前为单帧 HTTP POST 推理；如需连续视频流可改为 `VIDEO` 模式 + WebSocket 推送以提升吞吐。
4. **history 接口**：`PoliceGestureService.history()` 目前仍为 mock，需接入持久化存储后返回真实的推理历史。
