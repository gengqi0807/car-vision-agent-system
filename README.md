# car-vision-agent-system

智能车载视觉感知与交互系统，聚焦以下四大能力：

- 道路车辆车牌识别
- 交警手势识别
- 车主手势控车
- 日志监控与告警智能体

## Day 1 已完成

- 后端 `FastAPI` 分层骨架
- 前端 `Vue3 + TypeScript + Vite` 页面壳与路由
- 四大业务模块的 API 占位与 mock 数据
- 告警智能体、WebSocket 管理器、文档目录骨架

## 目录

- [backend](backend/README.md)
- [frontend](frontend/README.md)
- [docs](docs/需求文档.md)

## 环境依赖

### 后端 Python 依赖

Python 版本建议 3.10+，安装方式：

```bash
cd backend
pip install -r requirements.txt
```

核心依赖与用途：

| 类别 | 包 | 用途 |
|---|---|---|
| Web 服务 | `fastapi`, `uvicorn[standard]`, `python-multipart`, `websockets` | API/WebSocket 服务、文件上传、实时推送 |
| 配置/数据 | `pydantic-settings`, `sqlalchemy` | 配置管理、ORM 模型 |
| 鉴权 | `python-jose[cryptography]`, `passlib[bcrypt]` | JWT、密码哈希 |
| 视觉/推理 | `mediapipe`, `opencv-python-headless`, `numpy` | 手部/姿态关键点、图像处理 |
| 机器学习 | `scikit-learn`, `joblib`, `torch` | 静态手势 SVM 训练/推理、动态手势 BiLSTM 训练/推理 |

> `scripts/` 下的训练脚本（`extract_features.py`、`train_gesture_model.py` 等）复用同一套后端依赖。

### 前端 Node 依赖

Node 版本建议 18+，安装方式：

```bash
cd frontend
npm install
```

| 类别 | 包 | 用途 |
|---|---|---|
| 框架 | `vue`, `vue-router`, `pinia` | Vue3 框架、路由、状态管理 |
| UI | `element-plus` | 组件库 |
| 请求/图表 | `axios`, `echarts` | HTTP 请求、数据可视化 |
| 构建 | `vite`, `@vitejs/plugin-vue`, `typescript`, `sass`, `vue-tsc` | 打包构建、类型检查 |

### 外部二进制 / 资产

1. **MediaPipe 模型文件**（存放于 `backend/models/`）：
   - `hand_landmarker.task`、`pose_landmarker_lite.task`
   - 若缺失可通过 `scripts/download_models.py` 下载。
2. **mediamtx**（流媒体服务器）：通过 `docker-compose.yml` 以 `bluenviron/mediamtx:latest` 镜像提供，用于 RTSP/WebRTC 拉流推流。

### 已知兼容性问题

- `mediapipe==0.10.21` 与 `numpy==2.2.6` 在某些平台下可能冲突（MediaPipe 旧版 wheel 常要求 `numpy<2`）。若安装或运行时出现 numpy ABI/import 错误，可尝试升级 `mediapipe` 至 `>=0.10.22`，或降级 `numpy` 至 `==1.26.x`（注意同步调整 `scikit-learn` / `torch` 版本兼容性）。

## 下一阶段建议

1. 接入数据库模型和 Alembic 迁移。
2. 接入 YOLO、OCR、MediaPipe 等预训练模型推理层。
3. 打通 WebSocket/SSE 告警推送与前端实时展示。
4. 按 Figma 原型细化页面视觉和交互。
