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

---

# 改动说明 — 手势分类器 + 流媒体配置 + 前端手势页真实联调

> 日期：2026-07-07  
> 分支：`合并视频流处理`  
> 范围：`backend/` · `frontend/` · `stream_core.py` · `docker-compose.yml` · `docs/`

---

## 改动概览

1. **手势分类器规则化实现**：从占位桩 → 基于 MediaPipe 关键点的几何规则分类（手部 4 类 + 交警 5 类）
2. **流媒体服务器配置**：`stream_core.py` 路径可配置（环境变量）+ `docker-compose.yml` 加入 MediaMTX 容器
3. **前端手势页真实联调**：文件上传 → POST multipart → 关键点骨架画布叠加 → 识别置信度展示

---

## 文件变更清单

### 修改

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `backend/app/models_infer/gesture_classifier.py` | **完整重写** | 从 3 行占位 → 160+ 行基于关键点的规则分类器 |
| `backend/app/services/owner_gesture_service.py` | 接入分类器 | 新增 `GestureClassifier` lazy-load；`process_frame()` 先调用 MediaPipe 再调用规则分类 |
| `backend/app/services/police_gesture_service.py` | 接入分类器 | 同 owner，使用 `domain="police"` 调用 |
| `stream_core.py` | **重构** | 路径改为环境变量 `MEDIAMTX_PATH` / `FFMPEG_PATH` + 自动 PATH 查找；新增 `StreamConfig` 和 `stop()` |
| `docker-compose.yml` | 新增服务 | 添加 `mediamtx` 容器（bluenviron/mediamtx:latest），暴露 RTSP/RTMP/HLS/WebRTC 端口 |
| `frontend/src/api/owner_gesture.ts` | API 改造 | `/current` GET → POST multipart/form-data |
| `frontend/src/api/police_gesture.ts` | API 改造 | `/current` GET → POST multipart/form-data |
| `frontend/src/views/OwnerGesture.vue` | **重构** | 视频流占位 → 真实文件上传 + 图片预览 + canvas 手部骨架叠加 + 置信度展示 |
| `frontend/src/views/PoliceGesture.vue` | **重构** | 视频流占位 → 真实文件上传 + 图片预览 + canvas 姿态骨架叠加 + 结果列表 |
| `docs/changes-mediapipe-inference.md` | 文档 | 追加本次三项改动的说明 |

---

## 详细说明

### 一、手势分类器规则化实现

`GestureClassifier.classify(keypoints, domain)` 现已实现真实分类逻辑：

**手部（owner, 21 关键点/只手）**：

| 手势 | 规则 |
|------|------|
| `open_palm`（手掌张开）| 食指/中指/无名指/小拇指尖 y < 指根 MCP y |
| `fist`（握拳）| 四指尖 y > PIP 关节 y |
| `thumbs_up`（拇指向上）| 拇指尖 < 手腕 y - 0.05，其余手指卷曲 |
| `thumbs_down`（拇指向下）| 拇指尖 > 手腕 y + 0.05，其余手指卷曲 |
| `unknown`（未识别）| 以上都不满足 |

**交警姿态（police, 33 关键点/人）**：

| 手势 | 规则 |
|------|------|
| `stop`（停止）| 双手腕 y < 肩膀 y - 0.08 |
| `left_turn`（左转弯）| 左手腕 y < 左肩 y - 0.08（或左臂横向扩展） |
| `right_turn`（右转弯）| 右手腕 y < 右肩 y - 0.08（或右臂横向扩展） |
| `go_straight`（直行）| 双臂未举起 |
| `unknown` | 未检测到人体 |

**调用链路更新**：
```
POST /api/v1/owner-gesture/current
  → MediaPipeHands.infer(frame)                    # 21×N 关键点
  → GestureClassifier.classify(raw_kps, "owner")   # → {gesture, confidence}
  → GestureFrameResult(gesture="open_palm", confidence=0.92, ...)
```

---

### 二、流媒体配置（MediaMTX + FFmpeg）

**`stream_core.py` 重构**：
- 不再硬编码绝对路径，改用环境变量：
  - `MEDIAMTX_PATH` — 指向 `mediamtx.exe` 的绝对路径
  - `FFMPEG_PATH` — 可选，未设时自动从系统 PATH 查找 `ffmpeg`
- 新增 `StreamConfig` 数据类统一配置
- 新增 `StreamManager.stop()` 优雅关闭进程

**`docker-compose.yml` 新增 mediamtx 服务**：
```yaml
mediamtx:
  image: bluenviron/mediamtx:latest
  ports:
    - "8554:8554"     # RTSP
    - "1935:1935"     # RTMP
    - "8888:8888"     # HLS
    - "8889:8889"     # WebRTC
```

---

### 三、前端手势页真实联调

**OwnerGesture.vue 改进**：
- 文件选择 → 预览 → POST multipart → 显示 gesture/confidence
- Canvas 叠加手部 21 关键点 + 连接线（拇指/食指/中指/无名指/小指）
- 支持多只手同时渲染

**PoliceGesture.vue 改进**：
- 文件选择 → 预览 → POST multipart → 显示 gesture/confidence
- Canvas 叠加姿态 33 关键点 + 肢体连接线（手臂/躯干/腿部/面部）
- 右侧结果列表高亮当前识别手势

**手势标签中文映射**：

| 英文 key | 中文 |
|----------|------|
| `open_palm` | 手掌张开 |
| `fist` | 握拳 |
| `thumbs_up` | 拇指向上 |
| `thumbs_down` | 拇指向下 |
| `stop` | 停止信号 |
| `left_turn` | 左转弯信号 |
| `right_turn` | 右转弯信号 |
| `go_straight` | 直行信号 |

---

## 后续待办

1. **YOLO / OCR 推理层**：`YoloDetector`、`OCRRecognizer` 仍为占位桩。
2. **WebSocket 实时视频流推测**：目前为单帧 HTTP POST；如需连续推流可在此基础上接入 WebSocket。
3. **history 接口持久化**：`PoliceGestureService.history()` 仍为 mock，需接 DB。
4. **交警手势更多姿势**：当前仅 4 类规则，左待转/变道/减速/靠边需更多姿态规则或时序推理。
