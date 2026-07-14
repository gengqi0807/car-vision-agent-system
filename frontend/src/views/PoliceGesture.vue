<template>
  <section class="page-shell">
    <header class="page-header compact-header">
      <h1>交警手势识别</h1>
    </header>

    <section class="two-col police-layout">
      <article class="panel monitor-panel">
        <div class="mode-panel">
          <button type="button" class="mode-btn" :class="{ active: mode === 'camera' }" @click="setMode('camera')">
            实时识别
          </button>
          <button type="button" class="mode-btn" :class="{ active: mode === 'video' }" @click="setMode('video')">
            视频识别
          </button>
        </div>

        <div v-if="mode === 'camera'" class="camera-actions">
          <label class="camera-source-control">
            <span>摄像头</span>
            <select v-model="cameraSourceChoice" :disabled="cameraActive">
              <option value="0">摄像头 0</option>
              <option value="1">摄像头 1</option>
              <option value="2">摄像头 2</option>
              <option value="3">摄像头 3</option>
              <option value="custom">其他</option>
            </select>
            <input
              v-if="cameraSourceChoice === 'custom'"
              v-model.number="customCameraSource"
              type="number"
              min="0"
              step="1"
              :disabled="cameraActive"
            />
          </label>
          <button type="button" class="mode-btn action-btn" :disabled="cameraActive" @click="startCamera">
            开启摄像头
          </button>
          <button type="button" class="mode-btn action-btn secondary" :disabled="!cameraActive" @click="stopCamera">
            关闭摄像头
          </button>
        </div>

        <label v-else class="upload-zone">
          <input
            class="hidden-input"
            accept="video/*"
            type="file"
            @change="onVideoFileChange($event)"
          />
          <div class="main">点击上传交警手势视频</div>
          <div class="sub">处理完成后会生成标注视频并保留识别结果</div>
        </label>

        <div v-if="mode === 'video' && isVideoTaskActive" class="video-task-actions">
          <span>{{ videoProgress?.message || "视频识别正在后台运行" }}</span>
          <button type="button" class="mode-btn cancel-task-btn" @click="cancelVideoTask">终止识别</button>
        </div>

        <div class="video-status" v-if="error">
          <span class="off">{{ error }}</span>
        </div>

        <div class="preview-shell">
          <div v-if="mode === 'camera'" class="preview-stage camera-stage">
            <iframe v-if="cameraPlaybackUrl" :src="cameraPlaybackUrl" class="preview-frame" allow="autoplay; fullscreen" />
            <div v-else-if="!cameraActive" class="image-placeholder police-frame">
              交警手势实时识别
              <div class="small">点击“开启摄像头”开始实时检测</div>
            </div>
          </div>

          <div v-else-if="mode === 'video' && previewStreamUrl" class="preview-stage">
            <iframe
              :src="previewStreamUrl"
              class="preview-frame"
              allow="autoplay; fullscreen"
              @load="handleVideoPreviewLoad"
            />
          </div>

          <div v-else-if="mode === 'video' && processedVideoUrl" class="preview-stage">
            <video
              :key="processedVideoUrl"
              class="preview-video"
              :src="processedVideoUrl"
              controls
              playsinline
              preload="metadata"
            />
          </div>

          <div v-else class="image-placeholder police-frame">
            交警手势视频
            <div class="small">上传视频后这里显示标注结果</div>
          </div>
        </div>

        <div class="stream-meta" v-if="mode === 'camera'">
          {{ cameraStatusText }}
        </div>
        <div class="stream-meta" v-else-if="mode === 'video' && videoProgress">
          {{ buildVideoProgressText(videoProgress) }}
        </div>
        <div class="stream-meta" v-else-if="loading">正在处理视频 ...</div>

        <template v-if="currentResult">
          <div class="summary-row">
            <span class="gesture-tag">{{ gestureLabel }}</span>
            <span class="gesture-confidence">置信度 {{ (currentResult.confidence * 100).toFixed(1) }}%</span>
          </div>
          <div class="stream-meta" v-if="mode === 'video' && videoResult">
            已处理 {{ videoResult.processed_frame_count }} 帧
            <span v-if="videoResult.duration_seconds"> · 时长 {{ videoResult.duration_seconds.toFixed(2) }} 秒</span>
          </div>
          <div class="stream-meta" v-else-if="mode === 'video' && videoProgress?.gesture">
            当前识别动作：{{ formatGesture(videoProgress.gesture) }}
            <span v-if="typeof videoProgress.confidence === 'number'">
              · {{ (videoProgress.confidence * 100).toFixed(1) }}%
            </span>
          </div>
          <div class="stream-meta" v-else-if="mode === 'camera'">
            实时模式：直接显示后端按识别模型标注后的画面
          </div>
        </template>
      </article>

      <article class="panel side-panel">
        <div class="support-label">支持手势</div>
        <div class="support-tags">停止 · 直行 · 左转弯 · 左待转 · 右转弯 · 变道 · 减速 · 靠边停车</div>

        <div class="history-head" v-if="mode === 'video'">
          <h4>识别输出</h4>
        </div>
        <div v-if="mode === 'video' && videoEvents.length === 0" class="empty-state history-empty">处理中时会在这里持续输出动作结果</div>
        <div v-for="(event, index) in videoEvents" :key="`${event.updated_at}-${index}`" class="history-row">
          <span>{{ formatGesture(event.gesture) }}</span>
          <span>{{ (event.confidence * 100).toFixed(1) }}%</span>
          <span>{{ formatVideoEventMeta(event) }}</span>
        </div>

        <div class="history-head">
          <h4>识别历史</h4>
        </div>
        <div v-if="historyItems.length === 0" class="empty-state history-empty">暂无识别记录</div>
        <div v-for="(item, index) in historyItems" :key="`${item.updated_at}-${index}`" class="history-row">
          <span>{{ formatGesture(item.gesture) }}</span>
          <span>{{ (item.confidence * 100).toFixed(1) }}%</span>
          <span>{{ formatTime(item.updated_at) }}</span>
        </div>
      </article>
    </section>
  </section>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";

import {
  cancelPoliceGestureVideoJobApi,
  createPoliceGestureVideoJobApi,
  fetchPoliceGestureHistoryApi,
  fetchPoliceGestureStreamResultApi,
  fetchPoliceGestureStreamStateApi,
  fetchPoliceGestureVideoProgressApi,
  startPoliceGestureStreamApi,
  stopPoliceGestureStreamApi,
  type PoliceGestureFrameResult,
  type PoliceGestureHistoryItem,
  type PoliceGestureVideoProgress,
  type PoliceGestureVideoResult
} from "@/api/police_gesture";

type Mode = "camera" | "video";

const mode = ref<Mode>("camera");
const loading = ref(false);
const error = ref("");
const cameraResult = ref<PoliceGestureFrameResult | null>(null);
const videoResult = ref<PoliceGestureVideoResult | null>(null);
const videoProgress = ref<PoliceGestureVideoProgress | null>(null);
const historyItems = ref<PoliceGestureHistoryItem[]>([]);

const cameraActive = ref(false);
const cameraSourceChoice = ref("0");
const customCameraSource = ref(4);
const cameraPlaybackUrl = ref("");
let streamResultTimer: number | null = null;

const GESTURE_LABELS: Record<string, string> = {
  stop: "停止信号",
  go_straight: "直行信号",
  left_turn: "左转弯信号",
  left_wait_turn: "左待转信号",
  right_turn: "右转弯信号",
  lane_change: "变道信号",
  slow_down: "减速慢行信号",
  pull_over: "靠边停车信号",
  no_gesture: "无手势",
  unknown: "未识别",
  "未检测到人体": "未检测到人体"
};

const currentResult = computed(() => {
  if (mode.value === "camera") return cameraResult.value;
  return videoResult.value;
});
const gestureLabel = computed(() => formatGesture(currentResult.value?.gesture ?? ""));
const processedVideoUrl = computed(() => normalizeMediaUrl(videoResult.value?.processed_video_url ?? ""));
const previewStreamUrl = computed(() => {
  if (mode.value !== "video") return "";
  const progress = videoProgress.value;
  if (!progress?.playback_url || progress.status === "completed" || progress.status === "failed") {
    return "";
  }
  return `${progress.playback_url}?task=${encodeURIComponent(progress.task_id)}`;
});
const videoEvents = computed(() => videoProgress.value?.events ?? []);
const videoPreviewStarted = ref(false);
const videoTaskCreated = ref(false);
const POLICE_VIDEO_TASK_STORAGE_KEY = "police-gesture-active-video-task";
const isVideoTaskActive = computed(() =>
  ["uploading", "queued", "preparing", "processing", "transcoding", "cancelling"].includes(videoProgress.value?.status ?? "")
);
const cameraStatusText = computed(() => {
  if (!cameraActive.value) return "摄像头未开启";
  if (loading.value) return "实时识别中 ... 正在等待后端返回结果";
  if (cameraResult.value) {
    return `实时识别中 · 当前动作 ${formatGesture(cameraResult.value.gesture)}`;
  }
  return "实时模式已启动，等待第一帧结果";
});
let videoProgressTimer: number | null = null;

function setMode(nextMode: Mode) {
  if (mode.value === nextMode) return;
  if (mode.value === "camera") {
    stopCamera();
  }
  if (nextMode !== "video") {
    stopVideoProgressPolling();
  }
  mode.value = nextMode;
  videoPreviewStarted.value = false;
  error.value = "";
  if (nextMode !== "camera") {
    cameraResult.value = null;
  }
  if (nextMode !== "video") {
    stopVideoProgressPolling();
  } else {
    restoreVideoTask();
  }
}

async function loadHistory() {
  try {
    const { data } = await fetchPoliceGestureHistoryApi();
    historyItems.value = data;
  } catch {
    historyItems.value = [];
  }
}

async function onVideoFileChange(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;

  videoResult.value = null;
  error.value = "";
  loading.value = true;
  videoTaskCreated.value = false;
  stopVideoProgressPolling();
  videoPreviewStarted.value = false;

  const taskId = createTaskId();
  videoProgress.value = {
    task_id: taskId,
    source_filename: file.name,
    status: "uploading",
    progress: 0.01,
    message: "视频上传中，等待后端接收任务。",
    processed_frame_count: 0,
    total_frames: null,
    gesture: null,
    confidence: null,
    annotated_frame: null,
    events: [],
    updated_at: new Date().toISOString()
  };
  startVideoProgressPolling(taskId);

  try {
    const formData = new FormData();
    formData.append("file", file);
    await createPoliceGestureVideoJobApi(formData, taskId);
    videoTaskCreated.value = true;
    window.localStorage.setItem(POLICE_VIDEO_TASK_STORAGE_KEY, taskId);
    if (videoProgress.value) {
      videoProgress.value.status = "queued";
      videoProgress.value.message = "视频已上传，后台识别任务已创建。";
    }
    startVideoProgressPolling(taskId);
  } catch (err) {
    window.localStorage.removeItem(POLICE_VIDEO_TASK_STORAGE_KEY);
    videoResult.value = null;
    stopVideoProgressPolling();
    videoProgress.value = {
      task_id: taskId,
      source_filename: file.name,
      status: "failed",
      progress: 1,
      message: extractErrorMessage(err, "视频识别失败"),
      processed_frame_count: videoProgress.value?.processed_frame_count ?? 0,
      total_frames: videoProgress.value?.total_frames ?? null,
      gesture: null,
      confidence: null,
      annotated_frame: null,
      events: videoProgress.value?.events ?? [],
      updated_at: new Date().toISOString()
    };
    error.value = extractErrorMessage(err, "视频识别失败");
  } finally {
    loading.value = false;
    input.value = "";
  }
}

async function pollVideoProgress(taskId: string) {
  try {
    const { data } = await fetchPoliceGestureVideoProgressApi(taskId);
    if (data.status === "missing") {
      if (videoTaskCreated.value) {
        window.localStorage.removeItem(POLICE_VIDEO_TASK_STORAGE_KEY);
        loading.value = false;
        error.value = "之前的视频识别任务已失效，请重新上传视频。";
        stopVideoProgressPolling();
        return;
      }
      if (loading.value) {
        videoProgress.value = {
          task_id: taskId,
          source_filename: videoProgress.value?.source_filename ?? "",
          status: "uploading",
          progress: Math.max(videoProgress.value?.progress ?? 0.01, 0.01),
          message: "视频上传中，等待后端接收任务。",
          processed_frame_count: videoProgress.value?.processed_frame_count ?? 0,
          total_frames: videoProgress.value?.total_frames ?? null,
          gesture: videoProgress.value?.gesture ?? null,
          confidence: videoProgress.value?.confidence ?? null,
          annotated_frame: null,
          events: videoProgress.value?.events ?? [],
          updated_at: new Date().toISOString()
        };
        scheduleNextVideoProgressPoll(taskId);
        return;
      }
      videoProgress.value = data;
      stopVideoProgressPolling();
      return;
    }

    videoProgress.value = data;
    if (data.status === "completed") {
      window.localStorage.removeItem(POLICE_VIDEO_TASK_STORAGE_KEY);
      if (data.processed_video_url) {
        videoResult.value = {
          source_filename: data.source_filename,
          gesture: data.gesture ?? "no_gesture",
          confidence: data.confidence ?? 0,
          keypoints: [],
          task_id: data.task_id,
          processed_video_url: data.processed_video_url,
          processed_frame_count: data.processed_frame_count,
          duration_seconds: data.duration_seconds ?? null,
          updated_at: data.updated_at
        };
      }
      loading.value = false;
      stopVideoProgressPolling();
      await loadHistory();
      return;
    }
    if (data.status === "failed" || data.status === "cancelled") {
      window.localStorage.removeItem(POLICE_VIDEO_TASK_STORAGE_KEY);
      loading.value = false;
      stopVideoProgressPolling();
      return;
    }
    scheduleNextVideoProgressPoll(taskId);
  } catch {
    scheduleNextVideoProgressPoll(taskId, true);
  }
}

function startVideoProgressPolling(taskId: string) {
  stopVideoProgressPolling();
  void pollVideoProgress(taskId);
}

function stopVideoProgressPolling() {
  if (videoProgressTimer !== null) {
    window.clearTimeout(videoProgressTimer);
    videoProgressTimer = null;
  }
}

function scheduleNextVideoProgressPoll(taskId: string, afterFailure = false) {
  stopVideoProgressPolling();
  const status = videoProgress.value?.status ?? "";
  let nextDelay = 2500;

  if (afterFailure) {
    nextDelay = 4000;
  } else if (status === "uploading" || status === "queued" || status === "preparing") {
    nextDelay = 1500;
  } else if (status === "processing") {
    nextDelay = videoPreviewStarted.value ? 5000 : 2500;
  } else if (status === "transcoding") {
    nextDelay = 3000;
  }

  videoProgressTimer = window.setTimeout(() => {
    void pollVideoProgress(taskId);
  }, nextDelay);
}

function handleVideoPreviewLoad() {
  videoPreviewStarted.value = true;
}

function restoreVideoTask() {
  const taskId = window.localStorage.getItem(POLICE_VIDEO_TASK_STORAGE_KEY);
  if (!taskId) return;
  videoTaskCreated.value = true;
  loading.value = true;
  startVideoProgressPolling(taskId);
}

async function cancelVideoTask() {
  const taskId = videoProgress.value?.task_id;
  if (!taskId || !isVideoTaskActive.value) return;
  try {
    const { data } = await cancelPoliceGestureVideoJobApi(taskId);
    videoProgress.value = data;
    startVideoProgressPolling(taskId);
  } catch (err) {
    error.value = extractErrorMessage(err, "无法终止视频识别任务");
  }
}

function extractErrorMessage(error: unknown, fallback: string) {
  if (typeof error === "object" && error && "response" in error) {
    const response = (error as { response?: { data?: { detail?: string } } }).response;
    if (response?.data?.detail) {
      return response.data.detail;
    }
  }
  return fallback;
}

function createTaskId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `police-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function buildVideoProgressText(progress: PoliceGestureVideoProgress) {
  const percentage = `${Math.round((progress.progress || 0) * 100)}%`;
  const frameText = progress.processed_frame_count > 0 ? ` · ${progress.processed_frame_count} 帧` : "";
  const gestureText = progress.gesture ? ` · ${formatGesture(progress.gesture)}` : "";
  return `${progress.message || "正在处理视频 ..."} (${percentage}${frameText}${gestureText})`;
}

function formatVideoEventMeta(event: { frame_index: number; timestamp_seconds?: number | null }) {
  if (typeof event.timestamp_seconds === "number") {
    return `${event.timestamp_seconds.toFixed(2)}s · 第${event.frame_index}帧`;
  }
  return `第${event.frame_index}帧`;
}

function formatGesture(gesture: string) {
  return GESTURE_LABELS[gesture] || gesture || "—";
}

function formatTime(value: string) {
  return new Date(value).toLocaleString("zh-CN", { hour12: false });
}

function normalizeMediaUrl(url: string) {
  if (!url) return "";
  if (url.startsWith("/media/")) {
    return `${resolveBackendOrigin()}${url}`;
  }
  try {
    const parsed = new URL(url);
    if (parsed.pathname.startsWith("/media/")) {
      return `${resolveBackendOrigin()}${parsed.pathname}${parsed.search}${parsed.hash}`;
    }
  } catch {
    return url;
  }
  return url;
}

function resolveBackendApiBase() {
  const apiBase = (import.meta.env.VITE_API_BASE || "/api/v1").replace(/\/$/, "");
  if (apiBase.startsWith("http://") || apiBase.startsWith("https://")) {
    return apiBase;
  }
  return `${resolveBackendOrigin()}${apiBase}`;
}

function resolveBackendOrigin() {
  if (typeof window === "undefined") {
    return "";
  }
  const { protocol, hostname, host, port } = window.location;
  if ((hostname === "localhost" || hostname === "127.0.0.1") && port === "5173") {
    return `${protocol}//127.0.0.1:8000`;
  }
  return `${protocol}//${host}`;
}

async function startCamera() {
  if (cameraActive.value) return;
  mode.value = "camera";
  error.value = "";
  cameraResult.value = null;
  try {
    await stopPoliceGestureStreamApi().catch(() => undefined);
    const cameraSource = cameraSourceChoice.value === "custom" ? customCameraSource.value : cameraSourceChoice.value;
    if (!Number.isInteger(Number(cameraSource)) || Number(cameraSource) < 0) {
      throw new Error("请选择有效的摄像头编号");
    }
    const { data } = await startPoliceGestureStreamApi(String(cameraSource), 15);
    if (!data.playback_url) throw new Error("后端未返回 MediaMTX 播放地址");
    cameraActive.value = true;
    startStreamResultPolling();
    await waitForStreamPublished(data.playback_url);
  } catch (err) {
    stopCamera();
    error.value = err instanceof Error ? err.message : "无法开启后端摄像头";
  }
}

function stopCamera() {
  void stopPoliceGestureStreamApi().catch(() => undefined);
  stopStreamResultPolling();
  loading.value = false;
  cameraActive.value = false;
  cameraResult.value = null;
  cameraPlaybackUrl.value = "";
}

function startStreamResultPolling() {
  stopStreamResultPolling();
  const poll = async () => {
    if (!cameraActive.value) return;
    try {
      const { data } = await fetchPoliceGestureStreamResultApi();
      cameraResult.value = data;
    } catch {
      if (cameraActive.value) error.value = "无法获取交警手势识别结果";
    } finally {
      if (cameraActive.value) streamResultTimer = window.setTimeout(poll, 200);
    }
  };
  void poll();
}

function stopStreamResultPolling() {
  if (streamResultTimer !== null) window.clearTimeout(streamResultTimer);
  streamResultTimer = null;
}

async function waitForStreamPublished(playbackUrl: string) {
  const deadline = Date.now() + 15_000;
  while (cameraActive.value && Date.now() < deadline) {
    const { data } = await fetchPoliceGestureStreamStateApi();
    if (data.last_error) throw new Error(data.last_error);
    if (!data.running) throw new Error("交警手势推流已停止");
    if (data.published) {
      cameraPlaybackUrl.value = `${playbackUrl}?t=${Date.now()}`;
      return;
    }
    await new Promise((resolve) => window.setTimeout(resolve, 150));
  }
  throw new Error("等待 MediaMTX 交警手势流就绪超时");
}

onMounted(async () => {
  await loadHistory();
  if (window.localStorage.getItem(POLICE_VIDEO_TASK_STORAGE_KEY)) {
    mode.value = "video";
    restoreVideoTask();
  }
  try {
    const { data } = await fetchPoliceGestureStreamStateApi();
    if (data.running && data.playback_url) {
      cameraActive.value = true;
      cameraPlaybackUrl.value = `${data.playback_url}?t=${Date.now()}`;
      startStreamResultPolling();
    }
  } catch {
    // Keep the camera controls idle when no previous stream exists.
  }
});

onBeforeUnmount(() => {
  stopStreamResultPolling();
  stopVideoProgressPolling();
});
</script>

<style scoped lang="scss">
.hidden-input,
.capture-canvas {
  display: none;
}

.preview-frame {
  display: block;
  width: 100%;
  min-height: 360px;
  max-height: min(72vh, 720px);
  aspect-ratio: 16 / 9;
  border: 0;
  border-radius: 10px;
  background: #111;
}

.hidden-video {
  display: none;
}

.video-task-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  min-height: 42px;
  padding: 8px 10px;
  border-left: 3px solid var(--accent);
  color: var(--text-soft);
  background: var(--surface-muted);
}

.cancel-task-btn {
  flex: 0 0 auto;
  color: #fff;
  background: #b94a48;
}

.compact-header :deep(h1) {
  margin: 0;
}

.police-layout {
  grid-template-columns: minmax(0, 1.45fr) minmax(320px, 0.9fr);
  align-items: start;
}

.monitor-panel,
.side-panel {
  display: grid;
  gap: 12px;
}

.mode-panel,
.camera-actions {
  display: flex;
  gap: 10px;
  padding: 6px;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.03);
}

.camera-actions {
  padding-top: 0;
  background: transparent;
  align-items: center;
  flex-wrap: wrap;
}

.camera-source-control {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  color: var(--muted-soft);
  font-size: 13px;
}

.camera-source-control select,
.camera-source-control input {
  height: 38px;
  min-width: 112px;
  padding: 0 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text);
  background: var(--surface-muted);
}

.camera-source-control input {
  width: 84px;
  min-width: 84px;
}

.mode-btn {
  flex: 1;
  padding: 10px 14px;
  color: var(--text-soft);
  background: var(--surface-muted);
  border-radius: 10px;
  cursor: pointer;
}

.mode-btn.active,
.action-btn {
  color: #fff;
  background: var(--accent);
}

.action-btn {
  border: none;
  font-weight: 700;
}

.action-btn:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}

.action-btn.secondary {
  background: #d8c8b5;
  color: #49372d;
}

.upload-zone {
  display: grid;
  gap: 4px;
}

.preview-shell {
  margin-top: 2px;
}

.preview-stage {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 360px;
  padding: 10px;
  overflow: hidden;
  border-radius: 16px;
  background: linear-gradient(180deg, #f7efe5, #ecdfcf);
}

.camera-stage {
  position: relative;
  width: 100%;
}

.police-frame {
  width: 100%;
  min-height: 360px;
  margin-bottom: 0;
  color: #5b4639;
  background: linear-gradient(180deg, #f8f1e8, #efe3d4);
  border: 1px solid rgba(170, 129, 95, 0.22);
  border-radius: 16px;
}

.police-frame .small {
  color: #9a7f6b;
}

.preview-image,
.preview-video {
  display: block;
  width: 100%;
  max-height: min(72vh, 620px);
  object-fit: contain;
  border-radius: 10px;
  background: #f4eadf;
}

.summary-row {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.stream-meta {
  font-size: 13px;
  color: var(--muted-soft);
}

.history-head {
  margin-top: 8px;
}

.history-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto auto;
  gap: 10px;
  font-size: 13px;
  color: var(--text-soft);
}

.history-empty {
  margin-top: 0;
}

@media (max-width: 980px) {
  .police-layout {
    grid-template-columns: 1fr;
  }
}
</style>
