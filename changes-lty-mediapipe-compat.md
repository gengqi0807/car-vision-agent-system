# 改动说明 — MediaPipe 0.10.21 兼容性修复

> 日期：2026-07-14  
> 分支：`pull_from_merge_test`  
> 范围：`backend/app/models_infer/mediapipe_compat.py`

---

## 问题背景

远端分支合并了 `mediapipe_compat.py` 补丁文件，该补丁针对**新版 MediaPipe** 编写的——新版本存在 `free` 符号缺失的问题（google-ai-edge/mediapipe#6187），同时暴露了 `mediapipe.tasks.python.core.mediapipe_c_bindings` 模块。

当前环境实际使用的 **MediaPipe 0.10.21**：
- 没有 `free` bug（方案 B 已降级解决）
- **不存在** `mediapipe_c_bindings` 模块

这导致 App 启动时在 `mediapipe_compat.py` 第 8 行抛出：

```
ImportError: cannot import name 'mediapipe_c_bindings' from 'mediapipe.tasks.python.core'
```

---

## 改动内容

### `backend/app/models_infer/mediapipe_compat.py`

**改动类型**：简化（从 54 行 → 15 行）

**改动说明**：将原本针对新版 MediaPipe 的 `ctypes` 补丁逻辑替换为空操作 shim：

- 去掉顶层 `from mediapipe.tasks.python.core import mediapipe_c_bindings` —— 消除 ImportError
- 保留 `patch_windows_mediapipe_free_symbol()` 函数签名和空 `return` —— 保证 `mediapipe_hands.py` / `mediapipe_pose.py` 的调用点无报错
- 0.10.21 无 `free` bug，无需任何运行时补丁

### `police/models.py`

**改动类型**：精简（移除 ~55 行）

**改动说明**：删除 `_patch_mediapipe_windows_free()` 整段补丁函数及其顶层调用：

- 移除 `mediapipe_c_bindings` 相关导入（`ctypes`、`os`、`platform`、`resources`、`mediapipe_c_bindings`）
- 删除 `_patch_mediapipe_windows_free()` 函数定义和 `_patch_mediapipe_windows_free()` 调用
- `create_pose_detector()` / `create_hand_detector()` / `detect_pose()` / `detect_hand()` 逻辑完全不变

---

## 影响范围

- `mediapipe_hands.py` / `mediapipe_pose.py` 中现有的 import + 调用保持兼容
- `police/models.py` 推理函数 `create_pose_detector` / `create_hand_detector` / `detect_pose` / `detect_hand` 不受影响
