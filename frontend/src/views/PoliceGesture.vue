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
          <button type="button" class="mode-btn" :class="{ active: mode === 'image' }" @click="setMode('image')">
            图片识别
          </button>
          <button type="button" class="mode-btn" :class="{ active: mode === 'video' }" @click="setMode('video')">
            视频识别
          </button>
        </div>

        <div v-if="mode === 'camera'" class="camera-actions">
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
            :accept="mode === 'video' ? 'video/*' : 'image/*'"
            type="file"
            @change="mode === 'video' ? onVideoFileChange($event) : onImageFileChange($event)"
          />
          <div class="main">{{ mode === "video" ? "点击上传交警手势视频" : "点击上传交警手势图片" }}</div>
          <div class="sub">
            {{ mode === "video" ? "处理完成后会生成标注视频并保留识别结果" : "支持 JPG、PNG、WEBP" }}
          </div>
        </label>

        <div class="video-status" v-if="error">
          <span class="off">{{ error }}</span>
        </div>

        <div class="preview-shell">
          <div v-if="mode === 'camera'" class="preview-stage camera-stage">
            <video ref="videoRef" class="hidden-video" autoplay muted playsinline />
            <canvas ref="captureCanvasRef" class="capture-canvas" />
            <img v-if="cameraDisplayUrl" :src="cameraDisplayUrl" class="preview-image" />
            <div v-if="!cameraActive" class="image-placeholder police-frame">
              交警手势实时识别
              <div class="small">点击“开启摄像头”开始实时检测</div>
            </div>
            <div v-else-if="!cameraDisplayUrl" class="image-placeholder police-frame">
              交警手势实时识别
              <div class="small">摄像头已开启，等待后端返回首帧标注结果</div>
            </div>
          </div>

          <div v-else-if="mode === 'video' && previewStreamUrl" class="preview-stage">
            <img :src="previewStreamUrl" class="preview-image" @load="handleVideoPreviewLoad" />
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

          <div v-else-if="imageDisplayUrl" class="preview-stage">
            <div class="image-frame">
              <img :src="imageDisplayUrl" class="preview-image" @load="onImgLoad" />
              <canvas
                v-if="!imageResult?.annotated_image"
                ref="canvasRef"
                class="overlay-canvas"
                :width="canvasW"
                :height="canvasH"
              />
            </div>
          </div>

          <div v-else class="image-placeholder police-frame">
            {{ mode === "video" ? "交警手势视频" : "交警手势图片" }}
            <div class="small">{{ mode === "video" ? "上传视频后这里显示标注结果" : "上传图片后这里显示关键点标注" }}</div>
          </div>
        </div>

        <div class="stream-meta" v-if="mode === 'camera'">
          {{ cameraStatusText }}
        </div>
        <div class="stream-meta" v-else-if="mode === 'video' && videoProgress">
          {{ buildVideoProgressText(videoProgress) }}
        </div>
        <div class="stream-meta" v-else-if="loading">{{ mode === "video" ? "正在处理视频 ..." : "识别中 ..." }}</div>

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
          <div class="stream-meta" v-else>
            检测到 {{ currentResult.keypoints.length }} 个关键点
          </div>
        </template>
      </article>

      <article class="panel side-panel">
        <h4>识别结果</h4>
        <div
          class="result-item"
          v-for="item in candidateList"
          :key="item.value"
          :class="{ inactive: item.value !== currentGesture }"
        >
          <span>{{ item.label }}</span>
          <span class="val">{{ item.value === currentGesture ? `${(currentConfidence * 100).toFixed(1)}%` : "--" }}</span>
        </div>

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
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from "vue";

import {
  fetchPoliceGestureApi,
  fetchPoliceGestureHistoryApi,
  fetchPoliceGestureVideoProgressApi,
  recognizePoliceGestureVideoApi,
  type PoliceGestureFrameResult,
  type PoliceGestureHistoryItem,
  type PoliceGestureVideoProgress,
  type PoliceGestureVideoResult
} from "@/api/police_gesture";

type Mode = "camera" | "image" | "video";

const mode = ref<Mode>("camera");
const loading = ref(false);
const error = ref("");
const previewUrl = ref("");
const imageResult = ref<PoliceGestureFrameResult | null>(null);
const cameraResult = ref<PoliceGestureFrameResult | null>(null);
const videoResult = ref<PoliceGestureVideoResult | null>(null);
const videoProgress = ref<PoliceGestureVideoProgress | null>(null);
const historyItems = ref<PoliceGestureHistoryItem[]>([]);

const canvasRef = ref<HTMLCanvasElement | null>(null);
const canvasW = ref(300);
const canvasH = ref(220);

const videoRef = ref<HTMLVideoElement | null>(null);
const captureCanvasRef = ref<HTMLCanvasElement | null>(null);

const cameraActive = ref(false);
let mediaStream: MediaStream | null = null;
let captureTimer: number | null = null;
let requestInFlight = false;
let lastInferenceDurationMs = 140;
const baseFrameIntervalMs = 80;
const captureJpegQuality = 0.82;
const previewIdealWidth = 960;
const previewIdealHeight = 720;
const cameraSessionId = ref("");

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

const candidateList = [
  { value: "stop", label: "停止信号" },
  { value: "go_straight", label: "直行信号" },
  { value: "left_turn", label: "左转弯信号" },
  { value: "left_wait_turn", label: "左待转信号" },
  { value: "right_turn", label: "右转弯信号" },
  { value: "lane_change", label: "变道信号" },
  { value: "slow_down", label: "减速慢行信号" },
  { value: "pull_over", label: "靠边停车信号" }
];

const currentResult = computed(() => {
  if (mode.value === "camera") return cameraResult.value;
  if (mode.value === "video") return videoResult.value;
  return imageResult.value;
});
const currentGesture = computed(() => currentResult.value?.gesture ?? "");
const currentConfidence = computed(() => currentResult.value?.confidence ?? 0);
const gestureLabel = computed(() => formatGesture(currentGesture.value));
const imageDisplayUrl = computed(() => imageResult.value?.annotated_image || previewUrl.value);
const cameraDisplayUrl = computed(() => cameraResult.value?.annotated_image || "");
const processedVideoUrl = computed(() => normalizeMediaUrl(videoResult.value?.processed_video_url ?? ""));
const previewStreamUrl = computed(() => {
  if (mode.value !== "video") return "";
  const taskId = videoProgress.value?.task_id;
  if (!taskId || videoProgress.value?.status === "completed" || videoProgress.value?.status === "failed") {
    return "";
  }
  return buildPreviewStreamUrl(taskId);
});
const videoEvents = computed(() => videoProgress.value?.events ?? []);
const videoPreviewStarted = ref(false);
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
  resetPreview();
  if (nextMode !== "image") {
    imageResult.value = null;
    clearCanvas();
  }
  if (nextMode !== "camera") {
    cameraResult.value = null;
  }
  if (nextMode !== "video") {
    videoProgress.value = null;
    videoResult.value = null;
  }
}

function resetPreview() {
  if (previewUrl.value) {
    URL.revokeObjectURL(previewUrl.value);
  }
  previewUrl.value = "";
}

async function loadHistory() {
  try {
    const { data } = await fetchPoliceGestureHistoryApi();
    historyItems.value = data;
  } catch {
    historyItems.value = [];
  }
}

async function onImageFileChange(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;

  resetPreview();
  imageResult.value = null;
  error.value = "";
  loading.value = true;
  previewUrl.value = URL.createObjectURL(file);

  try {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("input_mode", "image");
    const { data } = await fetchPoliceGestureApi(formData);
    imageResult.value = data;
    await nextTick();
    if (!data.annotated_image) {
      drawImageKeypoints(data.keypoints);
    } else {
      clearCanvas();
    }
    await loadHistory();
  } catch (err) {
    imageResult.value = null;
    clearCanvas();
    error.value = extractErrorMessage(err, "图片识别失败");
  } finally {
    loading.value = false;
    input.value = "";
  }
}

async function onVideoFileChange(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;

  resetPreview();
  videoResult.value = null;
  error.value = "";
  loading.value = true;
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
    const { data } = await recognizePoliceGestureVideoApi(formData, taskId);
    videoResult.value = data;
    videoProgress.value = {
      task_id: data.task_id ?? taskId,
      source_filename: data.source_filename,
      status: "completed",
      progress: 1,
      message: "视频识别完成，标注视频已生成。",
      processed_frame_count: data.processed_frame_count,
      total_frames: videoProgress.value?.total_frames ?? null,
      gesture: data.gesture,
      confidence: data.confidence,
      annotated_frame: null,
      events: videoProgress.value?.events ?? [],
      updated_at: data.updated_at
    };
    stopVideoProgressPolling();
    await loadHistory();
  } catch (err) {
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
    if (data.status === "completed" || data.status === "failed") {
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

function onImgLoad(event: Event) {
  const image = event.target as HTMLImageElement;
  canvasW.value = image.clientWidth || 300;
  canvasH.value = image.clientHeight || 220;
  if (imageResult.value && !imageResult.value.annotated_image) {
    drawImageKeypoints(imageResult.value.keypoints);
  }
}

function handleVideoPreviewLoad() {
  videoPreviewStarted.value = true;
}

const POSE_CONNECTIONS: [number, number][] = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24],
  [23, 25], [25, 27], [24, 26], [26, 28],
  [0, 1], [0, 4], [1, 2], [2, 3], [4, 5], [5, 6]
];

const SIMPLE_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [3, 4], [4, 5], [13, 0], [13, 3], [0, 6], [3, 9], [6, 7], [7, 8], [9, 10], [10, 11], [12, 13]
];

function drawImageKeypoints(keypoints: Array<{ x: number; y: number }>) {
  const canvas = canvasRef.value;
  const ctx = canvas?.getContext("2d");
  if (!canvas || !ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawSkeleton(ctx, canvas.width, canvas.height, keypoints);
}

function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  points: Array<{ x: number; y: number }>
) {
  if (!points.length) return;
  const usePose = points.length >= 33;
  const connections = usePose ? POSE_CONNECTIONS : SIMPLE_CONNECTIONS;
  ctx.strokeStyle = "#2dd4bf";
  ctx.lineWidth = 2;
  for (const [start, end] of connections) {
    const startPoint = points[start];
    const endPoint = points[end];
    if (!startPoint || !endPoint) continue;
    ctx.beginPath();
    ctx.moveTo(startPoint.x * width, startPoint.y * height);
    ctx.lineTo(endPoint.x * width, endPoint.y * height);
    ctx.stroke();
  }

  ctx.fillStyle = "#c9b099";
  for (const point of points) {
    ctx.beginPath();
    ctx.arc(point.x * width, point.y * height, usePose ? 3 : 4, 0, 2 * Math.PI);
    ctx.fill();
  }
}

function clearCanvas() {
  const canvas = canvasRef.value;
  const ctx = canvas?.getContext("2d");
  if (ctx && canvas) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
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

function buildPreviewStreamUrl(taskId: string) {
  const token = localStorage.getItem("cvms_token") || "";
  const apiBase = (import.meta.env.VITE_API_BASE || "/api/v1").replace(/\/$/, "");
  const baseUrl = apiBase.startsWith("http") ? apiBase : `${window.location.origin}${apiBase}`;
  const encodedTaskId = encodeURIComponent(taskId);
  const encodedToken = encodeURIComponent(token);
  return `${baseUrl}/police-gesture/video/preview/${encodedTaskId}?token=${encodedToken}`;
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
  if (url.startsWith("/media/")) return url;
  try {
    const parsed = new URL(url);
    if (parsed.pathname.startsWith("/media/")) {
      return `${parsed.pathname}${parsed.search}${parsed.hash}`;
    }
  } catch {
    return url;
  }
  return url;
}

async function startCamera() {
  if (cameraActive.value) return;
  mode.value = "camera";
  error.value = "";
  cameraResult.value = null;
  cameraSessionId.value = createTaskId();

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: previewIdealWidth, max: 1280 },
        height: { ideal: previewIdealHeight, max: 720 },
        frameRate: { ideal: 24, max: 30 }
      },
      audio: false
    });
    mediaStream = stream;
    if (!videoRef.value) {
      throw new Error("摄像头初始化失败");
    }
    videoRef.value.srcObject = stream;
    cameraActive.value = true;
    await nextTick();
    await videoRef.value.play();
    startCaptureLoop();
  } catch (err) {
    stopCamera();
    error.value = err instanceof Error ? err.message : "无法开启摄像头";
  }
}

function stopCamera() {
  stopCaptureLoop();
  loading.value = false;
  requestInFlight = false;
  cameraActive.value = false;
  cameraResult.value = null;
  cameraSessionId.value = "";

  if (videoRef.value) {
    videoRef.value.pause();
    videoRef.value.srcObject = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
}

function startCaptureLoop() {
  stopCaptureLoop();
  const tick = async () => {
    if (!cameraActive.value) return;
    await captureFrame();
    if (!cameraActive.value) return;
    const nextDelay = Math.max(80, Math.min(180, Math.round(lastInferenceDurationMs * 0.18 + baseFrameIntervalMs)));
    captureTimer = window.setTimeout(() => {
      void tick();
    }, nextDelay);
  };
  void tick();
}

function stopCaptureLoop() {
  if (captureTimer !== null) {
    window.clearTimeout(captureTimer);
    captureTimer = null;
  }
}

async function captureFrame() {
  if (!cameraActive.value || requestInFlight) return;
  const video = videoRef.value;
  const captureCanvas = captureCanvasRef.value;
  if (!video || !captureCanvas || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;

  const width = Math.max(320, Math.min(video.videoWidth || previewIdealWidth, 640));
  const height = Math.max(240, Math.min(video.videoHeight || previewIdealHeight, 480));
  captureCanvas.width = width;
  captureCanvas.height = height;

  const ctx = captureCanvas.getContext("2d");
  if (!ctx) return;
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(video, 0, 0, width, height);

  const blob = await new Promise<Blob | null>((resolve) => {
    captureCanvas.toBlob(resolve, "image/jpeg", captureJpegQuality);
  });
  if (!blob) return;

  requestInFlight = true;
  loading.value = true;
  const activeSessionId = cameraSessionId.value || createTaskId();
  cameraSessionId.value = activeSessionId;
  const formData = new FormData();
  formData.append("file", blob, "police-gesture-camera.jpg");
  formData.append("input_mode", "camera");
  formData.append("session_id", activeSessionId);
  const startedAt = performance.now();

  try {
    const { data } = await fetchPoliceGestureApi(formData);
    lastInferenceDurationMs = performance.now() - startedAt;
    if (!cameraActive.value || cameraSessionId.value !== activeSessionId) {
      return;
    }
    cameraResult.value = data;
  } catch (err) {
    if (!cameraActive.value || cameraSessionId.value !== activeSessionId) {
      return;
    }
    error.value = extractErrorMessage(err, "实时识别失败");
  } finally {
    if (cameraSessionId.value === activeSessionId) {
      loading.value = false;
      requestInFlight = false;
    }
  }
}

onMounted(async () => {
  await loadHistory();
});

onBeforeUnmount(() => {
  stopCamera();
  stopVideoProgressPolling();
  resetPreview();
});
</script>

<style scoped lang="scss">
.hidden-input,
.capture-canvas {
  display: none;
}

.hidden-video {
  display: none;
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
  justify-content: center;
  padding: 10px;
  border-radius: 16px;
  background: linear-gradient(180deg, rgba(11, 16, 32, 0.58), rgba(11, 16, 32, 0.44));
}

.camera-stage,
.image-frame {
  position: relative;
  width: 100%;
}

.preview-image,
.preview-video {
  display: block;
  width: 100%;
  max-height: 520px;
  object-fit: contain;
  border-radius: 10px;
  background: #0b1020;
}

.overlay-canvas {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
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
