<template>
  <section class="page-shell">
    <header class="page-header compact-header">
      <h1>车牌识别</h1>
    </header>

    <section class="two-col plate-layout">
      <div class="monitor-column">
        <article class="panel mode-panel">
          <button type="button" class="mode-btn" :class="{ active: mode === 'image' }" @click="setMode('image')">
            图片识别
          </button>
          <button type="button" class="mode-btn" :class="{ active: mode === 'video' }" @click="setMode('video')">
            视频识别
          </button>
          <button type="button" class="mode-btn" :class="{ active: mode === 'stream' }" @click="setMode('stream')">
            实时推流识别
          </button>
        </article>

        <label v-if="mode === 'image'" class="upload-zone">
          <input class="hidden-input" accept="image/*" type="file" @change="handleImageFileChange" />
          <div class="main">点击上传道路场景图片</div>
          <div class="sub">支持 JPG、PNG、WEBP</div>
        </label>

        <label v-else-if="mode === 'video'" class="upload-zone">
          <input class="hidden-input" accept="video/*" type="file" @change="handleVideoFileChange" />
          <div class="main">点击上传道路场景视频</div>
          <div class="sub">处理后会生成标注视频并保留识别结果</div>
        </label>

        <article v-else class="panel stream-panel">
          <div class="stream-topbar">
            <div>
              <h4>推流控制</h4>
            </div>
            <span class="stream-badge" :class="{ live: streamStatus === 'running' }">
              {{ streamBadgeText }}
            </span>
          </div>

          <div class="video-input-row stream-input-row">
            <select v-model="selectedPreset" class="stream-select" @change="applyPreset">
              <option value="">选择沙盘摄像头</option>
              <option v-for="item in streamCameraPresets" :key="item.url" :value="item.url">
                {{ item.label }}
              </option>
            </select>
            <input v-model="streamInput" type="text" placeholder="输入 RTSP 地址..." />
            <div class="stream-actions">
              <button type="button" class="btn-video" :disabled="streamStatus === 'starting'" @click="startStream">
                {{ streamProcessFrames ? "开启识别并推流" : "开启直接推流" }}
              </button>
              <button type="button" class="btn-video secondary" @click="stopStream">停止</button>
            </div>
          </div>

          <label class="stream-toggle">
            <input v-model="streamProcessFrames" type="checkbox" />
            <span>{{ streamProcessFrames ? "识别后推流" : "直接推流，不识别" }}</span>
          </label>

          <div class="video-status">
            <span :class="{ off: streamStatus === 'idle' }">{{ streamPhaseText }}</span>
          </div>

          <div v-if="streamControl?.publish_rtsp_url" class="stream-links">
            <span>推流地址：{{ streamControl.publish_rtsp_url }}</span>
            <a
              v-if="streamControl.playback_url"
              :href="streamControl.playback_url"
              target="_blank"
              rel="noreferrer"
              class="stream-link"
            >
              新窗口打开播放页
            </a>
          </div>
        </article>

        <article class="panel monitor-panel">
          <div class="monitor-panel-head">
            <div>
              <h4>{{ monitorTitle }}</h4>
            </div>
            <span class="monitor-badge" :class="{ live: mode === 'stream' && streamStatus === 'running' }">
              {{ monitorBadge }}
            </span>
          </div>

          <div class="preview-shell">
            <div v-if="mode === 'stream' && streamPlaybackUrl" class="preview-stage stream-viewer-shell">
              <iframe :src="streamPlaybackUrl" class="stream-viewer" title="实时推流识别画面" allowfullscreen />
            </div>

            <div v-else-if="mode === 'video' && processedVideoUrl" class="preview-stage">
              <video class="preview-video" :src="processedVideoUrl" controls playsinline />
            </div>

            <div v-else-if="mode === 'video' && videoPreviewImageUrl" class="preview-stage">
              <img :src="videoPreviewImageUrl" alt="视频识别处理中预览" class="preview-video preview-image" />
            </div>

            <div v-else-if="mode === 'image' && imagePreviewUrl" class="preview-stage">
              <div class="image-frame">
                <img
                  ref="previewImageRef"
                  :src="imagePreviewUrl"
                  alt="车牌识别预览"
                  class="preview-image"
                  @load="handlePreviewImageLoad"
                />

                <div
                  v-for="(item, index) in overlayDetections"
                  :key="`${item.plate_number}-${index}`"
                  class="plate-box"
                  :style="boxStyle(item)"
                >
                  <div class="plate-tag" :class="{ compact: item.width < 18 }">
                    <span>{{ item.plate_number || "未识别" }}</span>
                    <small>{{ item.plate_color }} · {{ item.vehicle_type || "未识别" }} · {{ formatConfidence(item.confidence) }}</small>
                  </div>
                </div>
              </div>
            </div>

            <div v-else class="image-placeholder">
              {{ placeholderTitle }}
              <div class="small">{{ placeholderDescription }}</div>
            </div>
          </div>

          <div v-if="isLoading" class="status-note">{{ loadingText }}</div>
          <div
            v-else-if="mode === 'image' && imagePreviewUrl && currentDetections.length === 0 && !requestError"
            class="status-note"
          >
            当前图片未识别到车牌。
          </div>
          <div
            v-else-if="mode === 'video' && processedVideoUrl && currentDetections.length === 0 && !requestError"
            class="status-note"
          >
            视频已处理完成，但没有识别到可用车牌。
          </div>
          <div v-else-if="mode === 'stream'" class="status-note">
            {{ streamMetaText }}
          </div>

          <div v-if="mode === 'video' && processedVideoUrl" class="result-links">
            <a :href="processedVideoUrl" target="_blank" rel="noreferrer" class="stream-link">打开处理后视频</a>
            <a :href="processedVideoUrl" download class="stream-link">下载标注视频</a>
          </div>

          <div class="detection-list" v-if="currentDetections.length > 0">
            <div
              v-for="(item, index) in detectionRows"
              :key="`${item.plate}-${index}`"
              :style="{ marginTop: index === 0 ? '0' : '8px' }"
            >
              <div class="detection-box" :class="{ blue: item.color.includes('蓝') }">
                <span class="label">{{ item.plate }}</span>
                <span class="conf">{{ item.confidence }}</span>
              </div>
              <span class="detection-meta">{{ item.meta }}</span>
            </div>
          </div>

          <div class="result-meta">{{ resultMeta }}</div>
          <div v-if="requestError" class="request-error">{{ requestError }}</div>
        </article>
      </div>

      <article class="panel history-panel">
        <div class="history-head">
          <div>
            <h4>识别记录</h4>
          </div>
          <span class="history-count">{{ filteredRecords.length }} 条</span>
        </div>

        <input v-model="keyword" class="search-box" placeholder="搜索车牌号..." />

        <div class="history-table">
          <div class="history-row header">
            <span>车牌号码</span>
            <span>颜色</span>
            <span>类型</span>
            <span class="time">时间</span>
          </div>
          <div v-for="record in paginatedRecords" :key="record.id" class="history-row record">
            <span>{{ record.plate }}</span>
            <span>{{ record.color }}</span>
            <span>{{ record.vehicleType }}</span>
            <span class="time">{{ record.time }}</span>
          </div>
        </div>

        <div class="history-footer">
          <span class="history-page-info">第 {{ currentPage }} / {{ totalPages }} 页</span>
          <div class="history-actions">
            <button type="button" class="history-btn" :disabled="currentPage === 1" @click="goToPreviousPage">
              上一页
            </button>
            <button type="button" class="history-btn" :disabled="currentPage === totalPages" @click="goToNextPage">
              下一页
            </button>
          </div>
        </div>
      </article>
    </section>
  </section>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";

import {
  createPlateVideoJobApi,
  fetchPlateHistoryApi,
  fetchPlatePushStreamStatusApi,
  fetchPlateVideoJobStatusApi,
  recognizePlateImageApi,
  startPlatePushStreamApi,
  stopPlatePushStreamApi,
  type PlateDetection,
  type PlateRecordSummary,
  type PlateStreamControlResponse,
  type PlateVideoJobStatusResponse,
  type PlateVideoRecognitionResponse
} from "../api/plate";

type Mode = "image" | "video" | "stream";

interface HistoryRecordView {
  id: number;
  plate: string;
  color: string;
  vehicleType: string;
  time: string;
}

interface OverlayDetection extends PlateDetection {
  left: number;
  top: number;
  width: number;
  height: number;
}

const PLATE_MODE_STORAGE_KEY = "plate-recognition-mode";
const PLATE_STREAM_INPUT_STORAGE_KEY = "plate-stream-input";
const PLATE_STREAM_PRESET_STORAGE_KEY = "plate-stream-preset";

function readStoredMode(): Mode {
  if (typeof window === "undefined") {
    return "image";
  }
  const storedMode = window.sessionStorage.getItem(PLATE_MODE_STORAGE_KEY);
  return storedMode === "video" || storedMode === "stream" ? storedMode : "image";
}

function readStoredValue(key: string) {
  if (typeof window === "undefined") {
    return "";
  }
  return window.sessionStorage.getItem(key) ?? "";
}

const streamPresets = [
  { label: "桥面 · live1", url: "rtsp://10.126.59.120:8554/live/live1" },
  { label: "停车场出口 · live2", url: "rtsp://10.126.59.120:8554/live/live2" },
  { label: "桥出口 · live5", url: "rtsp://10.126.59.120:8554/live/live5" },
  { label: "桥入口 · live6", url: "rtsp://10.126.59.120:8554/live/live6" },
  { label: "道路2 · live7", url: "rtsp://10.126.59.120:8554/live/live7" },
  { label: "道路3 · live10", url: "rtsp://10.126.59.120:8554/live/live10" },
  { label: "停车场入口 · live11", url: "rtsp://10.126.59.120:8554/live/live11" },
  { label: "道路1 · live12", url: "rtsp://10.126.59.120:8554/live/live12" }
];

const streamCameraPresets = [
  { label: "桥面 · live1", url: "rtsp://10.126.59.120:8554/live/live1" },
  { label: "停车场出口 · live2", url: "rtsp://10.126.59.120:8554/live/live2" },
  { label: "行人检测 · live3", url: "rtsp://10.126.59.120:8554/live/live3" },
  { label: "消防车识别 · live4", url: "rtsp://10.126.59.120:8554/live/live4" },
  { label: "桥出口 · live5", url: "rtsp://10.126.59.120:8554/live/live5" },
  { label: "桥入口 · live6", url: "rtsp://10.126.59.120:8554/live/live6" },
  { label: "道路2 · live7", url: "rtsp://10.126.59.120:8554/live/live7" },
  { label: "隧道事故识别 · live8", url: "rtsp://10.126.59.120:8554/live/live8" },
  { label: "隧道车辆数量 · live9", url: "rtsp://10.126.59.120:8554/live/live9" },
  { label: "道路3 · live10", url: "rtsp://10.126.59.120:8554/live/live10" },
  { label: "停车场入口 · live11", url: "rtsp://10.126.59.120:8554/live/live11" },
  { label: "道路1 · live12", url: "rtsp://10.126.59.120:8554/live/live12" }
];

const mode = ref<Mode>(readStoredMode());
const keyword = ref("");
const currentPage = ref(1);
const requestError = ref("");
const loadingText = ref("正在识别，请稍候...");
const isLoading = ref(false);

const imagePreviewUrl = ref("");
const previewImageRef = ref<HTMLImageElement | null>(null);
const imageDetections = ref<PlateDetection[]>([]);
const sourceFrameSize = ref({ width: 0, height: 0 });

const videoResult = ref<PlateVideoRecognitionResponse | null>(null);
const videoJob = ref<PlateVideoJobStatusResponse | null>(null);
let videoJobTimer: number | null = null;
let activeVideoJobId = "";

const historyRecords = ref<PlateRecordSummary[]>([]);

const selectedPreset = ref(readStoredValue(PLATE_STREAM_PRESET_STORAGE_KEY));
const streamInput = ref(readStoredValue(PLATE_STREAM_INPUT_STORAGE_KEY));
const streamProcessFrames = ref(false);
const streamControl = ref<PlateStreamControlResponse | null>(null);
const streamStatus = ref<"idle" | "starting" | "running">("idle");
const playbackRefreshKey = ref(Date.now());
let statusTimer: number | null = null;

const historyPageSize = 6;

const fallbackRecords: HistoryRecordView[] = [
  { id: 1, plate: "沪A12345", color: "蓝牌", vehicleType: "识别记录", time: "14:23" },
  { id: 2, plate: "沪B67890", color: "绿牌", vehicleType: "识别记录", time: "14:15" }
];

const processedVideoUrl = computed(() => videoResult.value?.processed_video_url ?? "");
const videoPreviewImageUrl = computed(() => {
  if (processedVideoUrl.value) {
    return "";
  }
  return videoJob.value?.preview_image_url ?? "";
});
const videoProgressPercent = computed(() => {
  if (!videoJob.value) {
    return 0;
  }
  return Math.max(0, Math.min(Math.round((videoJob.value.progress ?? 0) * 100), 100));
});

const currentDetections = computed(() => {
  if (mode.value === "image") {
    return imageDetections.value;
  }
  if (mode.value === "video") {
    if (videoResult.value) {
      return videoResult.value.detections ?? [];
    }
    return videoJob.value?.detections ?? [];
  }
  return [];
});

const displayHistory = computed<HistoryRecordView[]>(() => {
  if (historyRecords.value.length === 0) {
    return fallbackRecords;
  }

  return historyRecords.value.map((record) => ({
    id: record.id,
    plate: record.plate_number,
    color: record.plate_color,
    vehicleType: "识别记录",
    time: new Date(record.created_at).toLocaleTimeString("zh-CN", { hour12: false })
  }));
});

const filteredRecords = computed(() => {
  const term = keyword.value.trim();
  if (!term) {
    return displayHistory.value;
  }
  return displayHistory.value.filter((record) => `${record.plate}${record.color}${record.vehicleType}`.includes(term));
});

const totalPages = computed(() => Math.max(1, Math.ceil(filteredRecords.value.length / historyPageSize)));

const paginatedRecords = computed(() => {
  const startIndex = (currentPage.value - 1) * historyPageSize;
  return filteredRecords.value.slice(startIndex, startIndex + historyPageSize);
});

const detectionRows = computed(() =>
  currentDetections.value.map((item) => ({
    plate: item.plate_number || "未识别",
    color: item.plate_color,
    confidence: formatConfidence(item.confidence),
    meta: `${item.plate_color} · ${item.vehicle_type || "未识别"} · 识别成功`
  }))
);

const overlayDetections = computed<OverlayDetection[]>(() => {
  if (mode.value !== "image") {
    return [];
  }

  const { width, height } = sourceFrameSize.value;
  if (!width || !height) {
    return [];
  }

  return imageDetections.value.map((item) => {
    const [x, y, boxWidth, boxHeight] = item.bbox;
    return {
      ...item,
      left: clampToPercent((x / width) * 100),
      top: clampToPercent((y / height) * 100),
      width: clampToPercent((boxWidth / width) * 100),
      height: clampToPercent((boxHeight / height) * 100)
    };
  });
});

const streamPlaybackUrl = computed(() => {
  if (streamStatus.value !== "running" || !streamControl.value?.published || !streamControl.value?.playback_url) {
    return "";
  }
  const separator = streamControl.value.playback_url.includes("?") ? "&" : "?";
  return `${streamControl.value.playback_url}${separator}ts=${playbackRefreshKey.value}`;
});

const monitorTitle = computed(() => {
  if (mode.value === "stream") {
    return "实时监控主画面";
  }
  if (mode.value === "video") {
    return "标注视频预览";
  }
  return "识别结果预览";
});

const monitorBadge = computed(() => {
  if (mode.value === "stream") {
    return streamStatus.value === "running" ? "PUSH LIVE" : "STREAM";
  }
  if (mode.value === "video") {
    return "VIDEO";
  }
  return "IMAGE";
});

const streamPhase = computed(() => streamControl.value?.phase ?? "idle");

const streamBadgeText = computed(() => {
  if (streamPhase.value === "running") {
    return "LIVE";
  }
  if (streamPhase.value === "source_unavailable" || streamPhase.value === "interrupted") {
    return "ERROR";
  }
  if (streamStatus.value === "starting") {
    return "STARTING";
  }
  return "IDLE";
});

const streamPhaseText = computed(() => {
  return streamControl.value?.status_message ?? (streamStatus.value === "running" ? "识别推流中" : "未启动推流");
});

const placeholderTitle = computed(() => {
  if (mode.value === "stream") {
    return "实时推流监控画面";
  }
  if (mode.value === "video") {
    return "道路场景视频";
  }
  return "道路场景图像";
});

const placeholderDescription = computed(() => {
  if (mode.value === "stream") {
    return "启动推流后，这里会显示 mediamtx 提供的实时播放页。";
  }
  if (mode.value === "video") {
    return "上传视频后，这里会显示处理完成的标注视频。";
  }
  return "上传图片后，这里会显示识别框和车牌号。";
});

const streamMetaText = computed(() => {
  if (streamPhase.value === "connecting_source") {
    return "正在连接源 RTSP，监控画面会在首帧到达后出现。";
  }
  if (streamPhase.value === "waiting_publish") {
    return "源 RTSP 已连接，正在等待本地播放流发布。";
  }
  if (streamPhase.value === "running") {
    return "实时识别推流中，主画面显示的是 mediamtx 提供的播放页。";
  }
  if (streamPhase.value === "source_unavailable") {
    return "源 RTSP 未开启、不可达，或当前没有可读视频帧。";
  }
  if (streamPhase.value === "interrupted") {
    return "实时推流中断，请检查源 RTSP、mediamtx 和 ffmpeg。";
  }
  return "选择摄像头并启动后，系统会拉取 RTSP 流、识别标注并重新推送到本地流媒体服务。";
});

const resultMeta = computed(() => {
  if (mode.value === "stream") {
    if (streamPhase.value === "connecting_source") {
      return "正在连接源 RTSP，等待首帧。";
    }
    if (streamPhase.value === "waiting_publish") {
      return "源 RTSP 已连上，正在等待本地播放流就绪。";
    }
    if (streamControl.value?.playback_url) {
      return `播放地址：${streamControl.value.playback_url}`;
    }
    return streamPhaseText.value;
  }

  if (mode.value === "video") {
    if (videoResult.value) {
      return `已处理 ${videoResult.value.processed_frame_count} 帧，识别到 ${videoResult.value.detections.length} 个唯一车牌。`;
    }
    if (videoJob.value) {
      return `正在处理第 ${videoJob.value.processed_frame_count} / ${videoJob.value.total_frames || "?"} 帧，当前进度 ${videoProgressPercent.value}%。`;
    }
    return "上传视频后，会在这里显示处理完成的标注视频和识别结果。";
  }

  if (isLoading.value) {
    return "正在处理图片...";
  }
  if (!imagePreviewUrl.value) {
    return "上传图片后，会在这里显示识别结果。";
  }
  return `共识别到 ${imageDetections.value.length} 个车牌。`;
});

function clampToPercent(value: number) {
  return Math.max(0, Math.min(value, 100));
}

function formatConfidence(confidence: number) {
  return `${(confidence * 100).toFixed(1)}%`;
}

function buildStreamName(rtspUrl: string) {
  try {
    const parsed = new URL(rtspUrl);
    const parts = parsed.pathname.split("/").filter(Boolean);
    const tail = parts.length > 0 ? parts[parts.length - 1]?.trim() : "";
    if (tail) {
      return `plate-${tail}`.replace(/[^A-Za-z0-9_-]+/g, "-");
    }
  } catch {
    // Ignore invalid URLs and let the backend fall back to its default.
  }
  return undefined;
}

function boxStyle(item: OverlayDetection) {
  return {
    left: `${item.left}%`,
    top: `${item.top}%`,
    width: `${item.width}%`,
    height: `${item.height}%`
  };
}

function resetImagePreviewUrl() {
  if (imagePreviewUrl.value) {
    URL.revokeObjectURL(imagePreviewUrl.value);
    imagePreviewUrl.value = "";
  }
}

function resetImageState() {
  resetImagePreviewUrl();
  imageDetections.value = [];
  sourceFrameSize.value = { width: 0, height: 0 };
}

function resetVideoState() {
  stopVideoJobPolling();
  videoResult.value = null;
  videoJob.value = null;
  activeVideoJobId = "";
}

function stopVideoJobPolling() {
  if (videoJobTimer !== null) {
    window.clearInterval(videoJobTimer);
    videoJobTimer = null;
  }
}

async function refreshVideoJobStatus(jobId: string) {
  const { data } = await fetchPlateVideoJobStatusApi(jobId);
  if (activeVideoJobId !== jobId) {
    return;
  }

  videoJob.value = data;
  requestError.value = "";
  if (data.status === "completed" && data.processed_video_url) {
    videoResult.value = {
      source_filename: data.source_filename,
      processed_video_url: data.processed_video_url,
      detections: data.detections,
      processed_frame_count: data.processed_frame_count,
      duration_seconds: data.duration_seconds ?? null
    };
    isLoading.value = false;
    loadingText.value = "";
    stopVideoJobPolling();
    await loadHistory();
    return;
  }

  if (data.status === "failed") {
    videoResult.value = null;
    isLoading.value = false;
    loadingText.value = "";
    requestError.value = data.error_message || "视频识别失败，请稍后重试。";
    stopVideoJobPolling();
    return;
  }

  isLoading.value = true;
  loadingText.value = `正在处理视频，已完成 ${videoProgressPercent.value}%（${data.processed_frame_count}/${data.total_frames || "?"} 帧）...`;
}

function startVideoJobPolling(jobId: string) {
  stopVideoJobPolling();
  activeVideoJobId = jobId;
  videoJobTimer = window.setInterval(async () => {
    try {
      await refreshVideoJobStatus(jobId);
    } catch (error) {
      const responseStatus =
        typeof error === "object" && error && "response" in error
          ? (error as { response?: { status?: number } }).response?.status
          : undefined;
      if (responseStatus === 404) {
        stopVideoJobPolling();
        activeVideoJobId = "";
        videoJob.value = null;
        videoResult.value = null;
        isLoading.value = false;
        loadingText.value = "";
        requestError.value = "之前的视频处理任务已失效。后端重启后旧任务不会保留，请重新上传视频。";
        return;
      }
      requestError.value = extractErrorMessage(error, "无法获取视频处理进度，请确认后端仍在运行。");
    }
  }, 1200);
}

function setMode(nextMode: Mode) {
  mode.value = nextMode;
  if (typeof window !== "undefined") {
    window.sessionStorage.setItem(PLATE_MODE_STORAGE_KEY, nextMode);
  }
  requestError.value = "";
  if (nextMode !== "image") {
    resetImageState();
  }
  if (nextMode !== "video") {
    resetVideoState();
  }
}

function handlePreviewImageLoad() {
  const image = previewImageRef.value;
  if (!image) {
    return;
  }
  sourceFrameSize.value = {
    width: image.naturalWidth,
    height: image.naturalHeight
  };
}

function applyPreset() {
  if (selectedPreset.value) {
    streamInput.value = selectedPreset.value;
  }
}

function goToPreviousPage() {
  currentPage.value = Math.max(1, currentPage.value - 1);
}

function goToNextPage() {
  currentPage.value = Math.min(totalPages.value, currentPage.value + 1);
}

async function loadHistory() {
  try {
    const { data } = await fetchPlateHistoryApi();
    historyRecords.value = data;
  } catch {
    historyRecords.value = [];
  }
  currentPage.value = 1;
}

async function refreshStreamStatus() {
  try {
    const previousRunning = streamControl.value?.running ?? false;
    const previousPlaybackUrl = streamControl.value?.playback_url ?? "";
    const { data } = await fetchPlatePushStreamStatusApi();
    streamControl.value = data;
    streamStatus.value = data.running ? (data.published ? "running" : "starting") : "idle";
    if (data.running) {
      if (mode.value !== "stream") {
        mode.value = "stream";
      }
      requestError.value = "";
    }
    if (data.running && data.published && data.playback_url && (!previousRunning || previousPlaybackUrl !== data.playback_url)) {
      playbackRefreshKey.value = Date.now();
    }
    if (data.phase === "source_unavailable" || data.phase === "interrupted") {
      requestError.value = data.last_error || data.status_message || "";
    } else if (data.last_error && !data.running) {
      requestError.value = data.last_error;
    } else if (!data.running) {
      requestError.value = "";
    }
  } catch {
    if (streamStatus.value === "running" || streamStatus.value === "starting") {
      requestError.value = "无法获取推流状态，请确认后端服务仍在运行。";
    }
  }
}

async function handleImageFileChange(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];

  resetImageState();
  requestError.value = "";

  if (!file) {
    return;
  }

  imagePreviewUrl.value = URL.createObjectURL(file);
  isLoading.value = true;
  loadingText.value = "正在识别图片，请稍候...";

  try {
    await nextTick();
    const { data } = await recognizePlateImageApi(file);
    imageDetections.value = data.detections;
    isLoading.value = false;
    void loadHistory().catch(() => {
      // Recognition has already succeeded; a history refresh failure should
      // not keep the result view in a loading state.
    });
  } catch (error) {
    imageDetections.value = [];
    requestError.value = extractErrorMessage(error, "图片识别失败，请确认后端服务已启动后重试。");
  } finally {
    isLoading.value = false;
    input.value = "";
  }
}

async function handleVideoFileChange(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];

  resetVideoState();
  requestError.value = "";

  if (!file) {
    return;
  }

  isLoading.value = true;
  loadingText.value = "正在上传视频并创建处理任务，请稍候...";

  try {
    const { data } = await createPlateVideoJobApi(file);
    videoJob.value = {
      job_id: data.job_id,
      source_filename: file.name,
      status: data.status,
      progress: 0,
      processed_frame_count: 0,
      total_frames: 0,
      detections: [],
      unread_samples: []
    };
    await refreshVideoJobStatus(data.job_id);
    if (!videoResult.value) {
      startVideoJobPolling(data.job_id);
    }
  } catch (error) {
    videoResult.value = null;
    videoJob.value = null;
    requestError.value = extractErrorMessage(error, "视频识别失败，请确认视频格式可读取并稍后重试。");
  } finally {
    input.value = "";
  }
}

async function startStream() {
  const rtspUrl = streamInput.value.trim();
  if (!rtspUrl) {
    requestError.value = "请输入 RTSP 地址。";
    return;
  }

  setMode("stream");
  requestError.value = "";
  streamStatus.value = "starting";

  try {
    const { data } = await startPlatePushStreamApi(
      rtspUrl,
      buildStreamName(rtspUrl),
      streamProcessFrames.value
    );
    streamControl.value = data;
    streamStatus.value = data.running ? (data.published ? "running" : "starting") : "idle";
    playbackRefreshKey.value = Date.now();
    await loadHistory();
  } catch (error) {
    streamStatus.value = "idle";
    requestError.value = extractErrorMessage(
      error,
      "启动推流失败，请确认 mediamtx 已启动、ffmpeg 可用，并检查 RTSP 地址。"
    );
  }
}

async function stopStream() {
  try {
    const { data } = await stopPlatePushStreamApi();
    streamControl.value = data;
  } catch {
    // Ignore stop errors and just reset local state.
  } finally {
    streamStatus.value = "idle";
    requestError.value = "";
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

function startStatusPolling() {
  stopStatusPolling();
  statusTimer = window.setInterval(async () => {
    await refreshStreamStatus();
    if (mode.value === "stream") {
      await loadHistory();
    }
  }, 3000);
}

function stopStatusPolling() {
  if (statusTimer !== null) {
    window.clearInterval(statusTimer);
    statusTimer = null;
  }
}

onMounted(async () => {
  await loadHistory();
  await refreshStreamStatus();
  startStatusPolling();
});

watch(keyword, () => {
  currentPage.value = 1;
});

watch(mode, (value) => {
  if (typeof window !== "undefined") {
    window.sessionStorage.setItem(PLATE_MODE_STORAGE_KEY, value);
  }
});

watch(selectedPreset, (value) => {
  if (typeof window !== "undefined") {
    window.sessionStorage.setItem(PLATE_STREAM_PRESET_STORAGE_KEY, value);
  }
});

watch(streamInput, (value) => {
  if (typeof window !== "undefined") {
    window.sessionStorage.setItem(PLATE_STREAM_INPUT_STORAGE_KEY, value);
  }
});

watch(totalPages, (value) => {
  if (currentPage.value > value) {
    currentPage.value = value;
  }
});

onBeforeUnmount(() => {
  stopStatusPolling();
  stopVideoJobPolling();
  resetImagePreviewUrl();
});
</script>

<style scoped lang="scss">
.hidden-input {
  display: none;
}

.page-shell {
  gap: 14px;
  width: calc(100% + 36px);
  margin-left: -18px;
  margin-right: -18px;
}

.compact-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.compact-header :deep(h1) {
  margin: 0;
  font-size: 24px;
}

.compact-header :deep(p) {
  display: none;
}

.plate-layout {
  grid-template-columns: minmax(0, 1.85fr) minmax(360px, 0.92fr);
  align-items: start;
}

.monitor-column {
  display: grid;
  gap: 10px;
}

.mode-panel {
  display: flex;
  gap: 10px;
  padding: 6px;
}

.mode-btn {
  flex: 1;
  padding: 9px 14px;
  color: var(--text-soft);
  background: var(--surface-muted);
  border-radius: 10px;
  cursor: pointer;
  transition: 0.2s ease;
}

.mode-btn.active {
  color: #fff;
  background: var(--accent);
}

.stream-panel {
  display: grid;
  gap: 8px;
  padding-top: 14px;
  padding-bottom: 16px;
}

.stream-topbar,
.monitor-panel-head,
.history-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
}

.stream-badge,
.monitor-badge,
.history-count {
  flex-shrink: 0;
  padding: 6px 12px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.08em;
  color: var(--text-soft);
  background: var(--surface-muted);
}

.stream-badge.live,
.monitor-badge.live {
  color: #ffffff;
  background: linear-gradient(135deg, #1d9f8c, #127567);
}

.stream-input-row {
  flex-wrap: nowrap;
  align-items: center;
  margin-bottom: 0;
}

.stream-select {
  min-width: 190px;
  padding: 9px 12px;
  color: var(--text);
  background: var(--surface-muted);
  border: 1px solid var(--line);
  border-radius: 8px;
  outline: none;
}

.stream-input-row :deep(input) {
  min-height: 40px;
  padding-top: 9px;
  padding-bottom: 9px;
}

.stream-actions {
  display: flex;
  gap: 10px;
  flex-shrink: 0;
}

.stream-actions .btn-video {
  padding: 9px 15px;
}

.stream-toggle {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-top: 10px;
  margin-bottom: 10px;
  font-size: 13px;
  color: var(--text-soft);
}

.stream-links,
.result-links {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 14px;
  font-size: 13px;
  color: var(--text-soft);
}

.stream-link {
  color: var(--accent-strong);
}

.monitor-panel {
  display: grid;
  gap: 8px;
  padding-top: 14px;
  padding-bottom: 18px;
}

.preview-shell {
  margin-bottom: 0;
}

.preview-stage {
  display: flex;
  justify-content: center;
  padding: 10px;
  border-radius: 16px;
  background: linear-gradient(180deg, rgba(11, 16, 32, 0.58), rgba(11, 16, 32, 0.44));
}

.stream-viewer-shell {
  min-height: 720px;
}

.stream-viewer,
.preview-video {
  width: 100%;
  min-height: 720px;
  border: 0;
  border-radius: 10px;
  background: #0b1020;
}

.image-frame {
  position: relative;
  display: inline-block;
  width: 100%;
  max-width: 100%;
}

.preview-image {
  display: block;
  width: 100%;
  max-height: 500px;
  object-fit: contain;
  border-radius: 10px;
}

.plate-box {
  position: absolute;
  border: 2px solid #2dd4bf;
  border-radius: 8px;
  box-shadow: 0 0 0 1px rgba(10, 14, 26, 0.6);
  pointer-events: none;
}

.plate-tag {
  position: absolute;
  top: calc(100% + 6px);
  left: 0;
  display: grid;
  gap: 2px;
  min-width: 88px;
  max-width: 180px;
  padding: 6px 8px;
  border-radius: 8px;
  background: rgba(10, 14, 26, 0.92);
  color: #ecfeff;
  line-height: 1.1;
  white-space: nowrap;
}

.plate-tag.compact {
  left: auto;
  right: 0;
}

.plate-tag small {
  color: rgba(207, 250, 254, 0.78);
  font-size: 11px;
}

.detection-list {
  display: grid;
}

.detection-meta {
  display: inline-block;
  margin-left: 8px;
  font-size: 13px;
  color: var(--text-soft);
}

.history-panel {
  position: sticky;
  top: 0;
  display: grid;
  gap: 12px;
  max-height: calc(100vh - 110px);
  padding-top: 16px;
  padding-bottom: 18px;
  min-width: 0;
}

.history-table {
  overflow: auto;
}

.history-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-top: 2px;
}

.history-page-info {
  font-size: 13px;
  color: var(--text-soft);
}

.history-actions {
  display: flex;
  gap: 8px;
}

.history-btn {
  padding: 8px 12px;
  color: var(--text);
  background: var(--surface-muted);
  border: 1px solid var(--line);
  border-radius: 8px;
  cursor: pointer;
  transition: 0.2s ease;
}

.history-btn:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.search-box {
  margin-bottom: 0;
}

.request-error {
  margin-top: 6px;
  font-size: 13px;
  color: #b27f75;
  text-align: center;
}

.status-note {
  margin-bottom: 0;
  font-size: 12px;
  color: var(--text-soft);
  text-align: center;
}

@media (max-width: 1100px) {
  .page-shell {
    width: 100%;
    margin-left: 0;
    margin-right: 0;
  }

  .plate-layout {
    grid-template-columns: 1fr;
  }

  .history-panel {
    position: static;
    max-height: none;
  }
}

@media (max-width: 820px) {
  .compact-header {
    display: block;
  }

  .mode-panel {
    flex-wrap: wrap;
  }

  .mode-btn {
    min-width: calc(50% - 5px);
  }

  .stream-input-row {
    flex-wrap: wrap;
  }

  .stream-actions {
    width: 100%;
  }

  .stream-actions .btn-video {
    flex: 1;
  }

  .stream-viewer-shell,
  .stream-viewer,
  .preview-video {
    min-height: 480px;
  }
}

@media (max-width: 640px) {
  .stream-topbar,
  .monitor-panel-head,
  .history-head,
  .history-footer {
    flex-direction: column;
    align-items: flex-start;
  }

  .history-actions,
  .stream-actions {
    width: 100%;
  }
}
</style>
