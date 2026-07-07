<template>
  <section class="page-shell">
    <header class="page-header">
      <h1>车牌识别</h1>
      <p>以实时监控画面为主，支持道路场景图片识别与 RTSP 视频流连续识别。</p>
    </header>

    <section class="two-col plate-layout">
      <div class="monitor-column">
        <article class="panel mode-panel">
          <button type="button" class="mode-btn" :class="{ active: mode === 'image' }" @click="setMode('image')">
            图片识别
          </button>
          <button type="button" class="mode-btn" :class="{ active: mode === 'stream' }" @click="setMode('stream')">
            视频流识别
          </button>
        </article>

        <label v-if="mode === 'image'" class="upload-zone">
          <input class="hidden-input" accept="image/*" type="file" @change="handleFileChange" />
          <div class="main">点击上传图片</div>
          <div class="sub">支持 JPG、PNG、WEBP</div>
        </label>

        <article v-else class="panel stream-panel">
          <div class="stream-topbar">
            <div>
              <h4>实时视频流</h4>
              <p class="stream-subtext">优先展示监控画面，识别结果会叠加在视频帧上。</p>
            </div>
            <span class="stream-badge" :class="{ live: streamStatus === 'connected' }">
              {{ streamStatus === "connected" ? "LIVE" : "STREAM" }}
            </span>
          </div>

          <div class="video-input-row stream-input-row">
            <select v-model="selectedPreset" class="stream-select" @change="applyPreset">
              <option value="">选择沙盘摄像头</option>
              <option v-for="item in streamPresets" :key="item.url" :value="item.url">
                {{ item.label }}
              </option>
            </select>
            <input v-model="streamInput" type="text" placeholder="输入 RTSP 地址..." />
            <div class="stream-actions">
              <button type="button" class="btn-video" @click="startStream">开启识别</button>
              <button type="button" class="btn-video secondary" @click="stopStream">停止</button>
            </div>
          </div>

          <div class="video-status">
            <span v-if="streamStatus === 'idle'" class="off">未连接</span>
            <span v-else-if="streamStatus === 'connecting'">连接中...</span>
            <span v-else>已连接，实时识别中</span>
          </div>
        </article>

        <article class="panel monitor-panel">
          <div class="monitor-panel-head">
            <div>
              <h4>{{ mode === "stream" ? "监控主画面" : "识别结果预览" }}</h4>
              <p class="monitor-subtext">
                {{ mode === "stream" ? "识别框和车牌文字会实时叠加在监控画面上。" : "上传图片后会在这里显示检测框和车牌结果。" }}
              </p>
            </div>
            <span class="monitor-badge" :class="{ live: mode === 'stream' && streamStatus === 'connected' }">
              {{ mode === "stream" ? (streamStatus === "connected" ? "LIVE" : "STREAM") : "IMAGE" }}
            </span>
          </div>

          <div class="preview-shell">
            <div v-if="displayFrameUrl" class="preview-stage">
              <div class="image-frame">
                <img
                  ref="previewImageRef"
                  :src="displayFrameUrl"
                  :alt="mode === 'image' ? '车牌识别预览' : '车牌实时识别画面'"
                  :class="['preview-image', { stream: mode === 'stream' }]"
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
                    <small>{{ item.plate_color }} · {{ formatConfidence(item.confidence) }}</small>
                  </div>
                </div>
              </div>
            </div>

            <div v-else class="image-placeholder">
              {{ mode === "image" ? "道路场景图像" : "实时监控画面" }}
              <div class="small">
                {{ mode === "image" ? "检测框和车牌号会直接叠加在这里。" : "连接 RTSP 流后，这里会连续显示识别画面。" }}
              </div>
            </div>
          </div>

          <div v-if="isLoading" class="status-note">正在识别，请稍候...</div>
          <div v-else-if="mode === 'image' && detections.length === 0 && displayFrameUrl && !requestError" class="status-note">
            当前图片未识别到车牌。
          </div>
          <div v-else-if="mode === 'stream' && streamStatus === 'connected'" class="status-note">
            实时识别中，当前已检测到 {{ detections.length }} 个车牌。
          </div>

          <div class="detection-list" v-if="detectionRows.length > 0">
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
            <p class="history-subtext">保留最近识别结果，支持搜索和翻页查看。</p>
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
  buildPlateStreamWebSocketUrl,
  fetchPlateHistoryApi,
  recognizePlateImageApi,
  type PlateDetection,
  type PlateRecordSummary
} from "../api/plate";

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

interface StreamMessage {
  frame: string;
  frame_width: number;
  frame_height: number;
  detections: PlateDetection[];
  error?: string;
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

const mode = ref<"image" | "stream">("image");
const keyword = ref("");
const currentPage = ref(1);
const imagePreviewUrl = ref("");
const streamFrameUrl = ref("");
const requestError = ref("");
const isLoading = ref(false);
const detections = ref<PlateDetection[]>([]);
const historyRecords = ref<PlateRecordSummary[]>([]);
const previewImageRef = ref<HTMLImageElement | null>(null);
const sourceFrameSize = ref({ width: 0, height: 0 });
const selectedPreset = ref("");
const streamInput = ref("");
const streamStatus = ref<"idle" | "connecting" | "connected">("idle");
const streamSocket = ref<WebSocket | null>(null);
const streamOpened = ref(false);
const historyPageSize = 6;

const fallbackRecords: HistoryRecordView[] = [
  { id: 1, plate: "沪A12345", color: "蓝牌", vehicleType: "小型车", time: "14:23" },
  { id: 2, plate: "浙B67890", color: "绿牌", vehicleType: "新能源", time: "14:15" }
];

const displayFrameUrl = computed(() => (mode.value === "image" ? imagePreviewUrl.value : streamFrameUrl.value));

const displayHistory = computed<HistoryRecordView[]>(() => {
  if (historyRecords.value.length === 0) {
    return fallbackRecords;
  }

  return historyRecords.value.map((record) => ({
    id: record.id,
    plate: record.plate_number,
    color: record.plate_color,
    vehicleType: "识别记录",
    time: new Date(record.created_at).toLocaleTimeString("zh-CN", {
      hour12: false
    })
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
  detections.value.map((item) => ({
    plate: item.plate_number || "未识别",
    color: item.plate_color,
    confidence: formatConfidence(item.confidence),
    meta: `${item.plate_color} · 识别成功`
  }))
);

const overlayDetections = computed<OverlayDetection[]>(() => {
  const { width, height } = sourceFrameSize.value;
  if (!width || !height) {
    return [];
  }

  return detections.value.map((item) => {
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

const resultMeta = computed(() => {
  if (mode.value === "stream") {
    if (streamStatus.value === "connecting") {
      return "正在连接视频流...";
    }
    if (streamStatus.value === "connected") {
      return `实时识别中，当前画面检测到 ${detections.value.length} 个车牌。`;
    }
    return "输入 RTSP 地址后可开始连续识别。";
  }

  if (isLoading.value) {
    return "正在处理图片...";
  }
  if (!imagePreviewUrl.value) {
    return "上传图片后会在这里显示识别结果。";
  }
  return `共识别到 ${detections.value.length} 个车牌。`;
});

function clampToPercent(value: number) {
  return Math.max(0, Math.min(value, 100));
}

function formatConfidence(confidence: number) {
  return `${(confidence * 100).toFixed(1)}%`;
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

function clearStreamFrame() {
  streamFrameUrl.value = "";
}

function setMode(nextMode: "image" | "stream") {
  mode.value = nextMode;
  requestError.value = "";
  detections.value = [];
  sourceFrameSize.value = { width: 0, height: 0 };
  if (nextMode === "image") {
    stopStream();
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

async function handleFileChange(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];

  resetImagePreviewUrl();
  requestError.value = "";
  detections.value = [];
  sourceFrameSize.value = { width: 0, height: 0 };

  if (!file) {
    return;
  }

  imagePreviewUrl.value = URL.createObjectURL(file);
  isLoading.value = true;

  try {
    await nextTick();
    const { data } = await recognizePlateImageApi(file);
    detections.value = data.detections;
    await loadHistory();
  } catch (error) {
    detections.value = [];
    requestError.value = "识别失败，请确认后端服务已启动，或稍后重试。";
    if (typeof error === "object" && error && "response" in error) {
      const response = (error as { response?: { data?: { detail?: string } } }).response;
      if (response?.data?.detail) {
        requestError.value = response.data.detail;
      }
    }
  } finally {
    isLoading.value = false;
  }
}

function stopStream() {
  streamSocket.value?.close();
  streamSocket.value = null;
  streamOpened.value = false;
  streamStatus.value = "idle";
  clearStreamFrame();
  if (mode.value === "stream") {
    detections.value = [];
    sourceFrameSize.value = { width: 0, height: 0 };
  }
}

function startStream() {
  const rtspUrl = streamInput.value.trim();
  if (!rtspUrl) {
    requestError.value = "请输入 RTSP 地址。";
    return;
  }

  stopStream();
  mode.value = "stream";
  requestError.value = "";
  streamStatus.value = "connecting";

  const socket = new WebSocket(buildPlateStreamWebSocketUrl(rtspUrl));
  streamSocket.value = socket;
  streamOpened.value = false;

  socket.onopen = () => {
    streamOpened.value = true;
    streamStatus.value = "connected";
  };

  socket.onmessage = (event) => {
    const payload = JSON.parse(event.data) as StreamMessage;
    if (payload.error) {
      requestError.value = payload.error;
      stopStream();
      return;
    }

    streamFrameUrl.value = `data:image/jpeg;base64,${payload.frame}`;
    detections.value = payload.detections || [];
    sourceFrameSize.value = {
      width: payload.frame_width,
      height: payload.frame_height
    };
  };

  socket.onerror = () => {
    requestError.value = "视频流连接失败，请确认 RTSP 地址和后端服务。";
    streamStatus.value = "idle";
  };

  socket.onclose = () => {
    if (streamSocket.value === socket) {
      streamSocket.value = null;
      if (streamStatus.value !== "idle") {
        streamStatus.value = "idle";
      }
    }
  };
}

onMounted(() => {
  void loadHistory();
});

watch(keyword, () => {
  currentPage.value = 1;
});

watch(totalPages, (value) => {
  if (currentPage.value > value) {
    currentPage.value = value;
  }
});

onBeforeUnmount(() => {
  stopStream();
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

.page-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.page-header :deep(h1) {
  margin: 0;
  font-size: 24px;
}

.page-header :deep(p) {
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

.stream-subtext,
.monitor-subtext,
.history-subtext {
  display: none;
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

.preview-image.stream {
  max-height: 760px;
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
  .page-header {
    display: block;
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

  .preview-image.stream {
    max-height: 520px;
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
