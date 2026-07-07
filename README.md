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

## 下一阶段建议

1. 接入数据库模型和 Alembic 迁移。
2. 接入 YOLO、OCR、MediaPipe 等预训练模型推理层。
3. 打通 WebSocket/SSE 告警推送与前端实时展示。
4. 按 Figma 原型细化页面视觉和交互。
