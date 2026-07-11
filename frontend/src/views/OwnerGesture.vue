<template>
  <section class="page-shell">
    <header class="page-header">
      <h1>手势控车</h1>
      <p>支持单张图片与实时视频输入，实时模式保留摄像头原画面并叠加手部节点识别结果</p>
    </header>

    <section class="two-col">
      <article class="panel">
        <div class="upload-row">
          <div class="mode-switch">
            <button
              class="mode-chip"
              type="button"
              :class="{ active: inputMode === 'camera' }"
              @click="switchInputMode('camera')"
            >
              实时视频
            </button>
            <button
              class="mode-chip"
              type="button"
              :class="{ active: inputMode === 'image' }"
              @click="switchInputMode('image')"
            >
              图片识别
            </button>
          </div>

          <template v-if="inputMode === 'camera'">
            <button class="btn-video control-button" type="button" :disabled="cameraActive" @click="startCamera">
              开启摄像头
            </button>
            <button
              class="btn-video control-button secondary-button"
              type="button"
              :disabled="!cameraActive"
              @click="stopCamera"
            >
              停止识别
            </button>
            <select
              v-if="videoDevices.length > 1"
              v-model="selectedDeviceId"
              class="device-select"
              :disabled="cameraActive"
            >
              <option v-for="device in videoDevices" :key="device.deviceId" :value="device.deviceId">
                {{ device.label }}
              </option>
            </select>
            <button
              v-if="videoDevices.length > 1"
              class="device-refresh"
              type="button"
              :disabled="cameraActive"
              @click="refreshVideoDevices"
            >
              刷新设备
            </button>
          </template>

          <template v-else>
            <button class="btn-video control-button" type="button" @click="openImagePicker">
              选择图片
            </button>
            <button
              class="btn-video control-button secondary-button"
              type="button"
              :disabled="!sourcePreviewUrl && !result"
              @click="clearImageSelection"
            >
              清空结果
            </button>
            <input
              ref="fileInputRef"
              class="hidden-file-input"
              type="file"
              accept="image/*"
              @change="onImageSelected"
            />
          </template>

          <span class="file-name">{{ inputStatusText }}</span>
        </div>

        <div class="video-status" v-if="error">
          <span class="off">{{ error }}</span>
        </div>

        <div class="preview-canvas-wrap">
          <div v-if="inputMode === 'camera'" class="preview-stage">
            <video
              ref="videoRef"
              class="preview-video"
              autoplay
              muted
              playsinline
              v-show="cameraActive"
            />
            <canvas
              ref="overlayCanvasRef"
              v-show="cameraActive"
              class="overlay-canvas"
            />
          </div>
          <div v-else-if="sourcePreviewUrl" class="preview-stage image-preview-stage">
            <img
              ref="imagePreviewRef"
              class="preview-image image-preview-fit"
              :src="sourcePreviewUrl"
              alt="手势图片识别原图"
              @load="handleImagePreviewLoad"
            />
            <canvas
              ref="imageOverlayCanvasRef"
              class="overlay-canvas image-overlay-canvas"
            />
          </div>
          <canvas ref="captureCanvasRef" class="capture-canvas" />
          <div
            v-if="(inputMode === 'camera' && !cameraActive) || (inputMode === 'image' && !sourcePreviewUrl)"
            class="image-placeholder gesture-frame"
          >
            {{ inputMode === "camera" ? "手势实时识别" : "手势图片识别" }}
            <div class="small">
              {{ inputMode === "camera" ? "点击“开启摄像头”开始检测驾驶员手势" : "上传单张手势图片，直接查看原图与节点标注结果" }}
            </div>
          </div>
        </div>

        <div class="stream-meta">{{ previewStatusText }}</div>

        <template v-if="result">
        <div class="recognition-summary">
          <span class="gesture-tag">{{ gestureLabel }}</span>
          <span class="recognition-chip gesture-confidence">
            置信度 {{ recognitionConfidence }}
          </span>
          <span class="gesture-action-chip">{{ gestureActionLabel }}</span>
          <span class="recognition-chip">{{ gestureCommandLabel }}</span>
          <span class="recognition-chip" v-if="sessionId">会话 {{ sessionId }}</span>
        </div>
        </template>
      </article>

      <article class="panel">
        <div class="cockpit-header">
          <div>
            <h4>智能座舱面板</h4>
          </div>
        </div>

        <div class="cockpit-shell">
          <div class="cockpit-topbar">
            <div class="cockpit-brand-group">
              <div class="cockpit-brand">CMC Smart Cabin</div>
              <div class="cockpit-page-tag">{{ activePageLabel }}</div>
            </div>
            <div class="cockpit-meta">
              <span>5G</span>
              <span>{{ panelState.climate_temperature }}°C</span>
              <span>{{ cockpitClock }}</span>
            </div>
          </div>

          <div class="cockpit-display">
            <div v-if="showConfirmSpotlight" class="confirm-spotlight">
              <span class="section-label">确认执行</span>
              <strong>{{ focusTileLabel }}</strong>
              <span>{{ confirmSpotlightText }}</span>
            </div>

            <div class="cockpit-off" v-if="activeCockpitPage === 'off'">
              <div class="off-screen">
                <div class="off-glow"></div>
                <div class="off-copy">
                  <span class="section-label">待机状态</span>
                  <strong>CMC 已关闭</strong>
                  <span>识别到手掌张开后唤醒系统，进入主页待命。</span>
                </div>
              </div>
            </div>

            <div class="cockpit-home" v-else-if="activeCockpitPage === 'home'">
              <div class="home-hero">
                <div class="map-surface">
                  <div class="map-grid"></div>
                  <div class="map-route route-a"></div>
                  <div class="map-route route-b"></div>
                  <div class="vehicle-marker"></div>
                  <div class="map-badge">导航主页</div>
                </div>
                <div class="home-side-widgets">
                  <div class="widget-card" :class="{ focused: isTileFocused('media') }">
                    <span class="section-label">媒体</span>
                    <strong>{{ panelState.media_playing ? "蓝牙音乐播放中" : "媒体已暂停" }}</strong>
                    <span>音量 {{ Math.round(panelState.volume) }}% · 单指画圈可直接调节</span>
                    <div class="mini-track"><div class="mini-fill" :style="{ width: `${panelState.volume}%` }"></div></div>
                  </div>
                  <div class="widget-card" :class="{ focused: isTileFocused('vehicle') }">
                    <span class="section-label">车辆</span>
                    <strong>{{ panelState.vehicle_status }}</strong>
                    <span>{{ modeSummary }}</span>
                  </div>
                </div>
              </div>
            </div>

            <div class="cockpit-media" v-else-if="activeCockpitPage === 'media'">
              <div class="media-layout">
                <div class="media-main" :class="{ focused: isTileFocused('media') }">
                  <span class="section-label">媒体中心</span>
                  <strong>{{ panelState.media_playing ? "行车伴音" : "媒体暂停" }}</strong>
                  <span>{{ mediaStatusText }}</span>
                  <div class="volume-knob">
                    <div class="volume-ring">
                      <div class="volume-ring-fill" :style="{ width: `${panelState.volume}%` }"></div>
                    </div>
                    <div class="volume-value">{{ Math.round(panelState.volume) }}</div>
                  </div>
                </div>
                <div class="media-side">
                  <div class="metric-card">
                    <span class="section-label">来源</span>
                    <strong>蓝牙音频</strong>
                    <span>握拳可确认播放/暂停</span>
                  </div>
                  <div class="metric-card">
                    <span class="section-label">推荐</span>
                    <strong>城市通勤</strong>
                    <span>单指画圈时自动前置媒体页</span>
                  </div>
                </div>
              </div>
            </div>

            <div class="cockpit-comfort" v-else-if="activeCockpitPage === 'comfort'">
              <div class="comfort-layout">
                <div class="comfort-main" :class="{ focused: isTileFocused('comfort') }">
                  <span class="section-label">舒适控制</span>
                  <strong>{{ panelState.climate_temperature }}°C</strong>
                  <span>{{ comfortStatusText }}</span>
                  <div class="climate-ring">
                    <div class="climate-ring-fill" :style="{ width: temperatureFillWidth }"></div>
                  </div>
                </div>
                <div class="comfort-side">
                  <div class="metric-card">
                    <span class="section-label">场景</span>
                    <strong>{{ panelState.comfort_scene }}</strong>
                    <span>握拳后执行舒适联动</span>
                  </div>
                  <div class="metric-card">
                    <span class="section-label">座椅</span>
                    <strong>主驾通风</strong>
                    <span>已与温控策略联动</span>
                  </div>
                  <div class="metric-card">
                    <span class="section-label">空气</span>
                    <strong>PM2.5 优</strong>
                    <span>净化系统持续工作</span>
                  </div>
                </div>
              </div>
            </div>

            <div class="cockpit-vehicle" v-else-if="activeCockpitPage === 'vehicle'">
              <div class="vehicle-layout">
                <div class="vehicle-main" :class="{ focused: isTileFocused('vehicle') }">
                  <span class="section-label">车辆状态</span>
                  <strong>{{ panelState.vehicle_status }}</strong>
                  <span>{{ vehicleStatusText }}</span>
                </div>
                <div class="vehicle-side">
                  <div class="metric-card">
                    <span class="section-label">续航</span>
                    <strong>438 km</strong>
                    <span>电池温控正常</span>
                  </div>
                  <div class="metric-card">
                    <span class="section-label">车门</span>
                    <strong>全部关闭</strong>
                    <span>锁止状态可握拳确认</span>
                  </div>
                  <div class="metric-card">
                    <span class="section-label">360 环视</span>
                    <strong>待命</strong>
                    <span>切换到本页后可查看车况</span>
                  </div>
                </div>
              </div>
            </div>

            <div class="cockpit-call" v-else>
              <div class="call-screen">
                <div class="caller-avatar">LI</div>
                <span class="section-label">蓝牙电话</span>
                <strong>{{ panelState.phone_call_active ? "李先生" : "来电提醒" }}</strong>
                <span>{{ callStatusText }}</span>
                <div class="call-action-row">
                  <div class="call-action decline">
                    <span>挂断</span>
                  </div>
                  <div class="call-action accept">
                    <span>{{ panelState.phone_call_active ? "通话中" : "接听" }}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div class="cockpit-dock">
            <div
              v-for="page in cockpitPages"
              :key="page.key"
              class="dock-item"
              :class="{ active: activeCockpitPage === page.key }"
            >
              {{ page.label }}
            </div>
          </div>
        </div>
      </article>
    </section>
  </section>
</template>

<script setup lang="ts">
import axios from "axios";
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from "vue";

import {
  fetchOwnerGestureApi,
  type OwnerControlPanelState,
  type OwnerGestureFrameResult,
} from "@/api/owner_gesture";

const loading = ref(false);
const error = ref("");
const result = ref<OwnerGestureFrameResult | null>(null);
const inputMode = ref<"camera" | "image">("camera");
const fileInputRef = ref<HTMLInputElement | null>(null);
const videoRef = ref<HTMLVideoElement | null>(null);
const imagePreviewRef = ref<HTMLImageElement | null>(null);
const overlayCanvasRef = ref<HTMLCanvasElement | null>(null);
const imageOverlayCanvasRef = ref<HTMLCanvasElement | null>(null);
const captureCanvasRef = ref<HTMLCanvasElement | null>(null);
const panelState = ref<OwnerControlPanelState>(createDefaultPanelState());
const cameraActive = ref(false);
const sessionId = ref("");
const cameraDeviceLabel = ref("");
const videoDevices = ref<Array<{ deviceId: string; label: string }>>([]);
const selectedDeviceId = ref("");
const sourcePreviewUrl = ref("");
const annotatedPreviewUrl = ref("");
const uiNow = ref(Date.now());

const baseFrameIntervalMs = 125;
const previewIdealWidth = 960;
const previewIdealHeight = 720;
const captureMaxWidth = 512;
const captureMaxHeight = 384;
const captureJpegQuality = 0.72;
let mediaStream: MediaStream | null = null;
let activeSourceVideo: HTMLVideoElement | null = null;
let captureTimer: number | null = null;
let requestInFlight = false;
let uiClockTimer: number | null = null;
let overlayAnimationFrame: number | null = null;
let overlayFromKeypoints: Array<{ x: number; y: number }> = [];
let overlayTargetKeypoints: Array<{ x: number; y: number }> = [];
let overlayDisplayKeypoints: Array<{ x: number; y: number }> = [];
let overlayTransitionStartedAt = 0;
let overlayTransitionDurationMs = 110;
let lastInferenceDurationMs = 120;

const HAND_CONNECTIONS: Array<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

const LABEL_MAP: Record<string, string> = {
  open_palm: "手掌张开",
  fist: "握拳",
  point: "单指伸出",
  index_circle: "单指画圈",
  swipe_left: "向左滑动",
  swipe_right: "向右滑动",
  thumbs_up: "拇指向上",
  thumbs_down: "拇指向下",
  wave: "挥手",
  unknown: "未识别",
  "未检测到手部": "未检测到手部",
};

const GESTURE_ACTION_MAP: Record<string, string> = {
  open_palm: "唤醒",
  fist: "确认",
  point: "待命",
  index_circle: "音量",
  swipe_left: "切换",
  swipe_right: "切换",
  thumbs_up: "接听",
  thumbs_down: "挂断",
  wave: "主页",
  unknown: "等待",
  "未检测到手部": "无动作",
};

const COMMAND_DISPLAY_MAP: Record<string, string> = {
  WakeSystem: "系统唤醒",
  ConfirmAction: "确认执行",
  AdjustVolume: "音量调节",
  SwitchPrevFeature: "切换功能",
  SwitchNextFeature: "切换功能",
  AnswerCall: "接听电话",
  HangUpCall: "挂断电话",
  ReturnHome: "返回主页",
};

const MODE_LABEL_MAP: Record<string, string> = {
  home: "主页",
  media: "媒体",
  comfort: "舒适",
  vehicle: "车辆",
  call: "通话",
  off: "关闭",
};

const cockpitPages = [
  { key: "home", label: "主页" },
  { key: "media", label: "媒体" },
  { key: "comfort", label: "舒适" },
  { key: "vehicle", label: "车辆" },
  { key: "call", label: "电话" },
] as const;

function createDefaultPanelState(): OwnerControlPanelState {
  return {
    system_awake: false,
    volume: 32,
    climate_temperature: 24,
    phone_call_active: false,
    current_mode: "home",
    media_playing: true,
    comfort_scene: "标准",
    vehicle_status: "就绪",
    focus_tile: "home",
    last_gesture: null,
    last_command: null,
    last_command_at: null,
    last_feedback: null,
    updated_at: null,
  };
}

const gestureLabel = computed(() => {
  const gesture = result.value?.gesture || panelState.value.last_gesture || "";
  if (!gesture) return "—";
  return LABEL_MAP[gesture] || gesture;
});

const gestureActionLabel = computed(() => {
  const key = result.value?.gesture || panelState.value.last_gesture || "unknown";
  return GESTURE_ACTION_MAP[key] || GESTURE_ACTION_MAP.unknown;
});

const recognitionConfidence = computed(() => `${((result.value?.confidence ?? 0) * 100).toFixed(1)}%`);

const gestureCommandLabel = computed(() => {
  const command = result.value?.control_command || panelState.value.last_command;
  return command ? COMMAND_DISPLAY_MAP[command] || command : "等待动作";
});

const systemAwake = computed(() => panelState.value.system_awake);

const activeCockpitPage = computed<"off" | "home" | "media" | "comfort" | "vehicle" | "call">(() => {
  if (!systemAwake.value) {
    return "off";
  }

  const currentMode = panelState.value.current_mode;
  if (currentMode === "call" || panelState.value.phone_call_active) {
    return "call";
  }

  if (currentMode === "media" || currentMode === "comfort" || currentMode === "vehicle") {
    return currentMode;
  }
  return "home";
});

const activePageLabel = computed(() => MODE_LABEL_MAP[activeCockpitPage.value] || "主页");

const focusTileLabel = computed(() => MODE_LABEL_MAP[panelState.value.focus_tile] || "主页");

const cockpitClock = computed(() =>
  new Date(uiNow.value).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit", hour12: false })
);

const modeSummary = computed(() => {
  if (!systemAwake.value) {
    return "系统默认关闭，识别到唤醒手势后进入主页。";
  }
  if (activeCockpitPage.value === "comfort") {
    return "舒适功能前置展示，便于确认空调与座舱联动。";
  }
  if (activeCockpitPage.value === "vehicle") {
    return "车辆页用于查看车况与整车状态。";
  }
  return "主页汇总导航、媒体、舒适与车辆信息。";
});

const mediaStatusText = computed(() => {
  return panelState.value.media_playing
    ? "单指画圈可直接提高音量，握拳可确认当前播放状态。"
    : "媒体已暂停，握拳后会恢复播放。";
});

const comfortStatusText = computed(() => {
  return panelState.value.comfort_scene === "舒享"
    ? "握拳后已执行舒享模式，温控与座椅同步强化。"
    : "左右滑动切换到此页后，可通过握拳执行舒适场景。";
});

const vehicleStatusText = computed(() => {
  return panelState.value.vehicle_status === "已完成整车检查"
    ? "整车检查已执行完成，关键车况正常。"
    : "可通过左右滑动切换到车辆页，再用握拳执行车况确认。";
});

const callStatusText = computed(() => {
  if (panelState.value.phone_call_active) {
    return "通话已接通，拇指向下可直接挂断。";
  }
  return "检测到来电场景时，拇指向上可进入通话界面。";
});

const showConfirmSpotlight = computed(() => {
  return (
    panelState.value.last_command === "ConfirmAction" &&
    isFreshCommand(panelState.value.last_command_at, 3200)
  );
});

const confirmSpotlightText = computed(() => {
  return panelState.value.last_feedback || `${focusTileLabel.value}功能已执行。`;
});

const temperatureFillWidth = computed(() => {
  const raw = ((panelState.value.climate_temperature - 18) / 12) * 100;
  return `${Math.max(0, Math.min(100, raw))}%`;
});

const cameraStatusText = computed(() => {
  if (!cameraActive.value) {
    return "摄像头未开启";
  }
  if (cameraDeviceLabel.value) {
    return loading.value
      ? `实时识别中 ... 当前设备：${cameraDeviceLabel.value}`
      : `摄像头已连接，等待下一帧 · ${cameraDeviceLabel.value}`;
  }
  return loading.value ? "实时识别中 ..." : "摄像头已连接，等待下一帧";
});

const inputStatusText = computed(() => {
  if (inputMode.value === "camera") {
    return cameraActive.value ? "实时模式：保留摄像头原画面，自适应节奏更新识别节点" : "摄像头未开启";
  }
  return sourcePreviewUrl.value ? "图片模式：显示上传原图，识别结果可直接驱动右侧 CMC" : "图片模式：等待上传";
});

const previewStatusText = computed(() => {
  if (inputMode.value === "camera") {
    if (!cameraActive.value) {
      return cameraStatusText.value;
    }
    return `${cameraStatusText.value} · 节点覆盖层已启用`;
  }
  if (loading.value) {
    return "图片识别中 ...";
  }
  if (sourcePreviewUrl.value) {
    return result.value?.keypoints?.length ? "已显示上传原图与节点标注" : "已加载上传原图";
  }
  return "上传图片后可直接查看原图与节点标注";
});

function isTileFocused(tile: string) {
  return panelState.value.focus_tile === tile;
}

function isFreshCommand(timestamp: string | null, windowMs: number) {
  if (!timestamp) return false;
  return uiNow.value - new Date(timestamp).getTime() <= windowMs;
}

function createSessionId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID().replace(/-/g, "").slice(0, 16);
  }
  return `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 10)}`.slice(0, 16);
}

function normalizeCameraError(error: unknown) {
  const err = error as DOMException | Error | undefined;
  const name = (err as DOMException | undefined)?.name || "";

  if (name === "NotAllowedError" || name === "PermissionDeniedError") {
    if (!window.isSecureContext) {
      return "当前页面不是安全上下文，浏览器会阻止摄像头。请使用 localhost 访问前端，不要用局域网 IP。";
    }
    return "浏览器阻止了摄像头访问。请检查地址栏右侧的摄像头权限，并确认没有被系统隐私设置禁用。";
  }
  if (name === "NotFoundError" || name === "DevicesNotFoundError") {
    return "未检测到可用摄像头，请确认设备已连接。";
  }
  if (name === "NotReadableError" || name === "TrackStartError") {
    return "摄像头正被其他程序占用，请关闭微信、QQ、浏览器会议页等再重试。";
  }
  if (name === "OverconstrainedError" || name === "ConstraintNotSatisfiedError") {
    return "当前设备不支持前置摄像头约束，已尝试切换通用摄像头。";
  }
  return err?.message || "无法启动摄像头";
}

function resetRecognitionState() {
  result.value = null;
  sessionId.value = "";
  panelState.value = createDefaultPanelState();
  annotatedPreviewUrl.value = "";
  resetOverlayState();
  clearImageOverlayCanvas();
}

function revokeSourcePreview() {
  if (!sourcePreviewUrl.value) return;
  URL.revokeObjectURL(sourcePreviewUrl.value);
  sourcePreviewUrl.value = "";
}

function switchInputMode(mode: "camera" | "image") {
  if (inputMode.value === mode) return;

  if (inputMode.value === "camera") {
    stopCamera();
  }
  if (inputMode.value === "image") {
    revokeSourcePreview();
  }

  error.value = "";
  resetRecognitionState();
  inputMode.value = mode;
}

function openImagePicker() {
  fileInputRef.value?.click();
}

async function onImageSelected(event: Event) {
  const target = event.target as HTMLInputElement;
  const file = target.files?.[0];
  if (!file) return;

  if (cameraActive.value) {
    stopCamera();
  }

  error.value = "";
  revokeSourcePreview();
  clearImageOverlayCanvas();
  inputMode.value = "image";
  sourcePreviewUrl.value = URL.createObjectURL(file);
  annotatedPreviewUrl.value = "";
  result.value = null;
  if (!sessionId.value) {
    sessionId.value = createSessionId();
    panelState.value = createDefaultPanelState();
  }

  const formData = new FormData();
  formData.append("file", file);
  formData.append("session_id", sessionId.value);
  formData.append("input_mode", "image");

  loading.value = true;
  try {
    const { data } = await fetchOwnerGestureApi(formData);
    result.value = data;
    applyResultToPanelState(data);
    annotatedPreviewUrl.value = "";
    await nextTick();
    drawImageOverlay();
  } catch (err: unknown) {
    if (axios.isAxiosError(err)) {
      error.value = String(err.response?.data?.detail || err.message || "识别失败");
    } else {
      error.value = (err as Error | undefined)?.message || "识别失败";
    }
  } finally {
    loading.value = false;
    target.value = "";
  }
}

function clearImageSelection() {
  revokeSourcePreview();
  error.value = "";
  resetRecognitionState();
}

function resetOverlayState() {
  overlayFromKeypoints = [];
  overlayTargetKeypoints = [];
  overlayDisplayKeypoints = [];
  overlayTransitionStartedAt = 0;
  clearOverlayCanvas();
}

function clearOverlayCanvas() {
  const canvas = overlayCanvasRef.value;
  const context = canvas?.getContext("2d");
  if (!canvas || !context) return;
  context.clearRect(0, 0, canvas.width, canvas.height);
}

function clearImageOverlayCanvas() {
  const canvas = imageOverlayCanvasRef.value;
  const context = canvas?.getContext("2d");
  if (!canvas || !context) return;
  context.clearRect(0, 0, canvas.width, canvas.height);
}

function syncImageOverlayCanvasSize() {
  const canvas = imageOverlayCanvasRef.value;
  const image = imagePreviewRef.value;
  if (!canvas || !image) return;

  const width = Math.max(1, Math.round(image.clientWidth));
  const height = Math.max(1, Math.round(image.clientHeight));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function drawImageOverlay() {
  const canvas = imageOverlayCanvasRef.value;
  const image = imagePreviewRef.value;
  const keypoints = result.value?.keypoints || [];
  const context = canvas?.getContext("2d");
  if (!canvas || !image || !context) return;

  syncImageOverlayCanvasSize();
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (!keypoints.length || !image.naturalWidth || !image.naturalHeight) {
    return;
  }

  const scale = Math.min(canvas.width / image.naturalWidth, canvas.height / image.naturalHeight);
  const drawWidth = image.naturalWidth * scale;
  const drawHeight = image.naturalHeight * scale;
  const offsetX = (canvas.width - drawWidth) / 2;
  const offsetY = (canvas.height - drawHeight) / 2;

  const perHand = 21;
  const handCount = Math.floor(keypoints.length / perHand);
  for (let handIndex = 0; handIndex < handCount; handIndex += 1) {
    const hand = keypoints.slice(handIndex * perHand, (handIndex + 1) * perHand);

    context.strokeStyle = "rgba(74, 192, 183, 0.92)";
    context.lineWidth = 2.2;
    context.lineCap = "round";
    context.lineJoin = "round";
    for (const [start, end] of HAND_CONNECTIONS) {
      const startPoint = hand[start];
      const endPoint = hand[end];
      if (!startPoint || !endPoint) continue;
      context.beginPath();
      context.moveTo(offsetX + startPoint.x * drawWidth, offsetY + startPoint.y * drawHeight);
      context.lineTo(offsetX + endPoint.x * drawWidth, offsetY + endPoint.y * drawHeight);
      context.stroke();
    }

    for (const point of hand) {
      const x = offsetX + point.x * drawWidth;
      const y = offsetY + point.y * drawHeight;
      context.beginPath();
      context.fillStyle = "rgba(247, 235, 222, 0.96)";
      context.arc(x, y, 3.2, 0, Math.PI * 2);
      context.fill();
      context.beginPath();
      context.strokeStyle = "rgba(74, 192, 183, 0.42)";
      context.lineWidth = 5.2;
      context.arc(x, y, 5.2, 0, Math.PI * 2);
      context.stroke();
    }
  }
}

function handleImagePreviewLoad() {
  drawImageOverlay();
}

function syncOverlayCanvasSize() {
  const canvas = overlayCanvasRef.value;
  const video = videoRef.value;
  if (!canvas || !video) return;

  const width = Math.max(1, Math.round(video.clientWidth || video.videoWidth || 0));
  const height = Math.max(1, Math.round(video.clientHeight || video.videoHeight || 0));
  if (width === 0 || height === 0) return;

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function updateOverlayKeypoints(keypoints: Array<{ x: number; y: number }>) {
  const now = performance.now();
  overlayFromKeypoints = overlayDisplayKeypoints.length ? overlayDisplayKeypoints.map((point) => ({ ...point })) : [];
  overlayTargetKeypoints = keypoints.map((point) => ({ x: point.x, y: point.y }));
  overlayTransitionStartedAt = now;
  overlayTransitionDurationMs = Math.max(80, Math.min(140, Math.round(lastInferenceDurationMs * 0.16)));

  if (overlayFromKeypoints.length !== overlayTargetKeypoints.length) {
    overlayFromKeypoints = overlayTargetKeypoints.map((point) => ({ ...point }));
    overlayDisplayKeypoints = overlayTargetKeypoints.map((point) => ({ ...point }));
  }
}

function interpolateOverlayKeypoints(now: number) {
  if (!overlayTargetKeypoints.length) {
    overlayDisplayKeypoints = [];
    return;
  }

  if (!overlayFromKeypoints.length || overlayFromKeypoints.length !== overlayTargetKeypoints.length) {
    overlayDisplayKeypoints = overlayTargetKeypoints.map((point) => ({ ...point }));
    return;
  }

  const progress = Math.min(1, (now - overlayTransitionStartedAt) / Math.max(overlayTransitionDurationMs, 1));
  const eased = 1 - (1 - progress) * (1 - progress);
  overlayDisplayKeypoints = overlayTargetKeypoints.map((point, index) => ({
    x: overlayFromKeypoints[index].x + (point.x - overlayFromKeypoints[index].x) * eased,
    y: overlayFromKeypoints[index].y + (point.y - overlayFromKeypoints[index].y) * eased,
  }));
}

function drawOverlayFrame(now: number) {
  if (!cameraActive.value || inputMode.value !== "camera") {
    clearOverlayCanvas();
    return;
  }

  syncOverlayCanvasSize();
  interpolateOverlayKeypoints(now);

  const canvas = overlayCanvasRef.value;
  const context = canvas?.getContext("2d");
  if (!canvas || !context) return;

  context.clearRect(0, 0, canvas.width, canvas.height);
  if (!overlayDisplayKeypoints.length) {
    return;
  }

  const perHand = 21;
  const handCount = Math.floor(overlayDisplayKeypoints.length / perHand);
  const width = canvas.width;
  const height = canvas.height;

  for (let handIndex = 0; handIndex < handCount; handIndex += 1) {
    const hand = overlayDisplayKeypoints.slice(handIndex * perHand, (handIndex + 1) * perHand);

    context.strokeStyle = "rgba(74, 192, 183, 0.92)";
    context.lineWidth = 2.4;
    context.lineCap = "round";
    context.lineJoin = "round";
    for (const [start, end] of HAND_CONNECTIONS) {
      const startPoint = hand[start];
      const endPoint = hand[end];
      if (!startPoint || !endPoint) continue;
      context.beginPath();
      context.moveTo(startPoint.x * width, startPoint.y * height);
      context.lineTo(endPoint.x * width, endPoint.y * height);
      context.stroke();
    }

    for (const point of hand) {
      const x = point.x * width;
      const y = point.y * height;
      context.beginPath();
      context.fillStyle = "rgba(247, 235, 222, 0.95)";
      context.arc(x, y, 3.2, 0, Math.PI * 2);
      context.fill();
      context.beginPath();
      context.strokeStyle = "rgba(74, 192, 183, 0.42)";
      context.lineWidth = 5.5;
      context.arc(x, y, 5.4, 0, Math.PI * 2);
      context.stroke();
    }
  }
}

function startOverlayLoop() {
  stopOverlayLoop();

  const animate = (now: number) => {
    drawOverlayFrame(now);
    overlayAnimationFrame = window.requestAnimationFrame(animate);
  };

  overlayAnimationFrame = window.requestAnimationFrame(animate);
}

function stopOverlayLoop() {
  if (overlayAnimationFrame !== null) {
    window.cancelAnimationFrame(overlayAnimationFrame);
    overlayAnimationFrame = null;
  }
}

async function startCamera() {
  if (cameraActive.value) return;

  inputMode.value = "camera";
  error.value = "";
  sessionId.value = createSessionId();
  result.value = null;
  panelState.value = createDefaultPanelState();
  annotatedPreviewUrl.value = "";

  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("当前浏览器不支持摄像头调用");
    }

    await refreshVideoDevices();
    const streamBinding = await requestPreferredStream();
    mediaStream = streamBinding.stream;
    cameraDeviceLabel.value = streamBinding.deviceLabel || "默认摄像头";

    if (!mediaStream || !videoRef.value) {
      throw new Error("摄像头初始化失败");
    }
    const track = mediaStream.getVideoTracks()[0];
    await refreshVideoDevices();
    if (!selectedDeviceId.value) {
      selectedDeviceId.value = track?.getSettings().deviceId || pickBestVideoDeviceId(videoDevices.value);
    }
    bindTrackEvents(track);
    videoRef.value.srcObject = mediaStream;
    activeSourceVideo = videoRef.value;
    cameraActive.value = true;
    await nextTick();
    await videoRef.value.play();
    await ensureVideoFrame(videoRef.value);
    syncOverlayCanvasSize();
    startOverlayLoop();
    startCaptureLoop();
  } catch (err: any) {
    stopCamera();
    error.value = normalizeCameraError(err);
  }
}

function stopCamera() {
  stopCaptureLoop();
  stopOverlayLoop();
  loading.value = false;
  requestInFlight = false;
  cameraActive.value = false;
  annotatedPreviewUrl.value = "";
  resetOverlayState();

  if (videoRef.value) {
    videoRef.value.pause();
    videoRef.value.srcObject = null;
  }
  if (activeSourceVideo) {
    activeSourceVideo.pause();
    activeSourceVideo.srcObject = null;
    activeSourceVideo = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
  cameraDeviceLabel.value = "";
}

function bindTrackEvents(track?: MediaStreamTrack) {
  if (!track) return;

  track.onmute = () => {
    if (!cameraActive.value) return;
    error.value = "摄像头已连接但没有输出画面，常见原因是微信、会议软件等正在占用摄像头。";
  };
  track.onended = () => {
    if (!cameraActive.value) return;
    stopCamera();
    error.value = "摄像头连接已中断，请关闭占用程序后重新开启。";
  };
}

async function ensureVideoFrame(video: HTMLVideoElement) {
  const deadline = Date.now() + 2500;

  while (Date.now() < deadline) {
    if (video.videoWidth > 0 && video.videoHeight > 0 && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      return;
    }
    await new Promise((resolve) => window.setTimeout(resolve, 120));
  }

  throw new Error("摄像头已连接但没有返回有效画面，请关闭微信等占用摄像头的软件后重试");
}

async function refreshVideoDevices() {
  if (!navigator.mediaDevices?.enumerateDevices) {
    return;
  }

  const devices = await navigator.mediaDevices.enumerateDevices();
  const videoInputs = devices
    .filter((device) => device.kind === "videoinput")
    .map((device, index) => ({
      deviceId: device.deviceId,
      label: device.label || `摄像头 ${index + 1}`,
    }));

  videoDevices.value = videoInputs;

  if (!selectedDeviceId.value && videoInputs.length > 0) {
    selectedDeviceId.value = pickBestVideoDeviceId(videoInputs);
  }
}

function pickBestVideoDeviceId(devices: Array<{ deviceId: string; label: string }>) {
  const scoreLabel = (label: string) => {
    const normalized = label.toLowerCase();
    let score = 0;

    if (normalized.includes("integrated")) score += 6;
    if (normalized.includes("hd")) score += 4;
    if (normalized.includes("webcam")) score += 4;
    if (normalized.includes("camera")) score += 3;
    if (normalized.includes("usb")) score += 2;

    if (normalized.includes("virtual")) score -= 10;
    if (normalized.includes("obs")) score -= 10;
    if (normalized.includes("snap")) score -= 10;
    if (normalized.includes("droidcam")) score -= 10;
    if (normalized.includes("epoccam")) score -= 10;
    if (normalized.includes("ndi")) score -= 8;
    if (normalized.includes("ir")) score -= 8;
    if (normalized.includes("infrared")) score -= 8;
    if (normalized.includes("hello")) score -= 8;

    return score;
  };

  const sorted = [...devices].sort((left, right) => scoreLabel(right.label) - scoreLabel(left.label));
  return sorted[0]?.deviceId || "";
}

async function requestPreferredStream(): Promise<{
  stream: MediaStream;
  deviceLabel: string;
}> {
  const probeVideo = createProbeVideo();

  const candidateIds = buildCandidateDeviceIds();
  if (candidateIds.length > 0) {
    let lastError: unknown = null;
    let selectedBinding: { stream: MediaStream; deviceLabel: string } | null = null;

    for (const deviceId of candidateIds) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: buildVideoConstraints({ deviceId: { exact: deviceId } }),
          audio: false,
        });

        const track = stream.getVideoTracks()[0];
        const deviceLabel = lookupDeviceLabel(deviceId) || track?.label || "默认摄像头";
        if (isLikelyUnsupportedCamera(deviceLabel)) {
          stream.getTracks().forEach((item) => item.stop());
          continue;
        }

        probeVideo.srcObject = stream;
        await probeVideo.play();
        await ensureVideoFrame(probeVideo);

        const isValid = await probeLiveVideoFrame(probeVideo);
        if (!isValid) {
          stream.getTracks().forEach((item) => item.stop());
          continue;
        }

        selectedDeviceId.value = deviceId;
        selectedBinding = {
          stream,
          deviceLabel,
        };
        break;
      } catch (requestError) {
        lastError = requestError;
      } finally {
        if (!selectedBinding) {
          probeVideo.pause();
          probeVideo.srcObject = null;
        }
      }
    }

    if (selectedBinding) {
      return selectedBinding;
    }

    if (lastError) {
      throw lastError;
    }
    throw new Error("未找到可输出有效画面的摄像头，请切换为真实内置摄像头。");
  }

  return await requestFallbackStream(probeVideo);
}

async function requestFallbackStream(probeVideo: HTMLVideoElement) {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: buildVideoConstraints({ facingMode: "user" }),
      audio: false,
    });
    const track = stream.getVideoTracks()[0];
    cameraDeviceLabel.value = track?.label || "默认摄像头";
    probeVideo.srcObject = stream;
    await probeVideo.play();
    await ensureVideoFrame(probeVideo);
    const isValid = await probeLiveVideoFrame(probeVideo);
    if (!isValid) {
      stream.getTracks().forEach((item) => item.stop());
      probeVideo.pause();
      probeVideo.srcObject = null;
      throw new Error("当前默认摄像头没有输出有效画面。");
    }
    return {
      stream,
      deviceLabel: cameraDeviceLabel.value || track?.label || "默认摄像头",
    };
  } catch (err: unknown) {
    const name = (err as DOMException | undefined)?.name || "";
    if (name !== "OverconstrainedError" && name !== "ConstraintNotSatisfiedError" && name !== "NotFoundError") {
      throw err;
    }

    const stream = await navigator.mediaDevices.getUserMedia({
      video: buildVideoConstraints(),
      audio: false,
    });
    const track = stream.getVideoTracks()[0];
    cameraDeviceLabel.value = track?.label || "默认摄像头";
    probeVideo.srcObject = stream;
    await probeVideo.play();
    await ensureVideoFrame(probeVideo);
    const isValid = await probeLiveVideoFrame(probeVideo);
    if (!isValid) {
      stream.getTracks().forEach((item) => item.stop());
      probeVideo.pause();
      probeVideo.srcObject = null;
      throw new Error("当前默认摄像头没有输出有效画面。");
    }
    return {
      stream,
      deviceLabel: cameraDeviceLabel.value || track?.label || "默认摄像头",
    };
  }
}

function buildCandidateDeviceIds() {
  if (videoDevices.value.length === 0) {
    return selectedDeviceId.value ? [selectedDeviceId.value] : [];
  }

  const preferred = selectedDeviceId.value || pickBestVideoDeviceId(videoDevices.value);
  const others = videoDevices.value
    .map((device) => device.deviceId)
    .filter((deviceId) => deviceId !== preferred);

  return preferred ? [preferred, ...others] : others;
}

function lookupDeviceLabel(deviceId: string) {
  return videoDevices.value.find((device) => device.deviceId === deviceId)?.label || "";
}

function isLikelyUnsupportedCamera(label: string) {
  const normalized = label.toLowerCase();
  return [
    "ir",
    "infrared",
    "hello",
    "virtual",
    "obs",
    "snap",
    "droidcam",
    "epoccam",
    "ndi",
  ].some((keyword) => normalized.includes(keyword));
}

function buildVideoConstraints(extra: MediaTrackConstraints = {}): MediaTrackConstraints {
  return {
    width: { ideal: previewIdealWidth, max: 1280 },
    height: { ideal: previewIdealHeight, max: 720 },
    frameRate: { ideal: 24, max: 30 },
    ...extra,
  };
}

function startCaptureLoop() {
  stopCaptureLoop();

  const tick = async () => {
    if (!cameraActive.value) return;
    await captureFrame();
    if (!cameraActive.value) return;
    const nextDelay = Math.max(70, Math.min(150, Math.round(lastInferenceDurationMs * 0.22 + baseFrameIntervalMs)));
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

  const video = activeSourceVideo;
  const captureCanvas = captureCanvasRef.value;
  if (!captureCanvas || !video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    return;
  }

  const captureSize = computeCaptureSize(video);
  captureCanvas.width = captureSize.width;
  captureCanvas.height = captureSize.height;

  const ctx = captureCanvas.getContext("2d");
  if (!ctx) return;
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

  const blob = await new Promise<Blob | null>((resolve) => {
    captureCanvas.toBlob(resolve, "image/jpeg", captureJpegQuality);
  });
  if (!blob) {
    error.value = "摄像头帧抓取失败";
    return;
  }

  requestInFlight = true;
  error.value = "";

  const formData = new FormData();
  formData.append("file", blob, "owner-gesture-frame.jpg");
  formData.append("session_id", sessionId.value);
  formData.append("input_mode", "camera");
  const inferStartedAt = performance.now();

  try {
    const { data } = await fetchOwnerGestureApi(formData);
    lastInferenceDurationMs = performance.now() - inferStartedAt;
    result.value = data;
    applyResultToPanelState(data);
    updateOverlayKeypoints(data.keypoints);
  } catch (err: any) {
    if (axios.isAxiosError(err)) {
      if (err.response?.status === 401 || err.response?.status === 403) {
        stopCamera();
        error.value = "登录状态已失效，请重新登录后再开启摄像头。";
        return;
      }
      error.value = String(err.response?.data?.detail || err.message || "识别失败");
    } else {
      error.value = err?.message || "识别失败";
    }
  } finally {
    requestInFlight = false;
  }
}

function applyResultToPanelState(data: OwnerGestureFrameResult) {
  if (data.panel_state) {
    panelState.value = data.panel_state;
    return;
  }

  const nextState: OwnerControlPanelState = {
    ...panelState.value,
    last_gesture: data.gesture,
    last_command: data.triggered ? data.control_command : panelState.value.last_command,
    last_command_at: data.triggered ? data.updated_at : panelState.value.last_command_at,
    updated_at: data.updated_at,
  };

  if (data.triggered && data.control_command) {
    if (data.control_command === "WakeSystem") {
      nextState.system_awake = true;
      nextState.phone_call_active = false;
      nextState.current_mode = "home";
      nextState.focus_tile = "home";
      nextState.last_feedback = "CMC 已唤醒，主页信息恢复显示。";
    } else if (data.control_command === "SwitchPrevFeature" && nextState.system_awake && !nextState.phone_call_active) {
      nextState.current_mode = shiftMode(nextState.current_mode, -1);
      nextState.focus_tile = nextState.current_mode;
      nextState.last_feedback = `已切换至${MODE_LABEL_MAP[nextState.current_mode] || "主页"}界面。`;
    } else if (data.control_command === "SwitchNextFeature" && nextState.system_awake && !nextState.phone_call_active) {
      nextState.current_mode = shiftMode(nextState.current_mode, 1);
      nextState.focus_tile = nextState.current_mode;
      nextState.last_feedback = `已切换至${MODE_LABEL_MAP[nextState.current_mode] || "主页"}界面。`;
    } else if (data.control_command === "AdjustVolume" && nextState.system_awake) {
      nextState.current_mode = "media";
      nextState.focus_tile = "media";
      nextState.media_playing = true;
      nextState.volume = Math.min(100, nextState.volume + 6);
      nextState.last_feedback = `媒体音量已调至 ${nextState.volume}%。`;
    } else if (data.control_command === "ConfirmAction" && nextState.system_awake) {
      applyConfirmActionLocally(nextState);
    } else if (data.control_command === "AnswerCall" && nextState.system_awake) {
      nextState.phone_call_active = true;
      nextState.current_mode = "call";
      nextState.focus_tile = "call";
      nextState.last_feedback = "蓝牙电话已接通，通话界面接管前台。";
    } else if (data.control_command === "HangUpCall" && nextState.system_awake) {
      nextState.phone_call_active = false;
      nextState.current_mode = "home";
      nextState.focus_tile = "home";
      nextState.last_feedback = "通话已挂断，系统已回到主页。";
    } else if (data.control_command === "ReturnHome" && nextState.system_awake && !nextState.phone_call_active) {
      nextState.current_mode = "home";
      nextState.focus_tile = "home";
      nextState.last_feedback = "已挥手返回主页。";
    }
  }

  panelState.value = nextState;
}

function shiftMode(currentMode: string, direction: number) {
  const modes = ["home", "media", "comfort", "vehicle"];
  const currentIndex = Math.max(0, modes.indexOf(currentMode));
  return modes[(currentIndex + direction + modes.length) % modes.length];
}

function applyConfirmActionLocally(state: OwnerControlPanelState) {
  if (state.current_mode === "media") {
    state.media_playing = !state.media_playing;
    state.focus_tile = "media";
    state.last_feedback = state.media_playing ? "媒体播放已确认继续。" : "媒体播放已确认暂停。";
    return;
  }

  if (state.current_mode === "comfort") {
    state.comfort_scene = "舒享";
    state.climate_temperature = 22;
    state.focus_tile = "comfort";
    state.last_feedback = "舒适模式已执行，空调与座椅联动完成。";
    return;
  }

  if (state.current_mode === "vehicle") {
    state.vehicle_status = "已完成整车检查";
    state.focus_tile = "vehicle";
    state.last_feedback = "车辆检查已执行，车况状态正常。";
    return;
  }

  if (state.current_mode === "call" && state.phone_call_active) {
    state.focus_tile = "call";
    state.last_feedback = "当前正在通话，确认动作已转为通话内操作。";
    return;
  }

  state.current_mode = "home";
  state.focus_tile = "home";
  state.last_feedback = "主页快捷操作已确认执行。";
}

function computeCaptureSize(video: HTMLVideoElement) {
  const sourceWidth = Math.max(1, video.videoWidth || captureMaxWidth);
  const sourceHeight = Math.max(1, video.videoHeight || captureMaxHeight);
  const scale = Math.min(1, captureMaxWidth / sourceWidth, captureMaxHeight / sourceHeight);

  return {
    width: Math.max(1, Math.round(sourceWidth * scale)),
    height: Math.max(1, Math.round(sourceHeight * scale)),
  };
}

async function probeLiveVideoFrame(video: HTMLVideoElement) {
  const canvas = document.createElement("canvas");
  const width = Math.max(64, video.videoWidth || 320);
  const height = Math.max(48, video.videoHeight || 240);
  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext("2d");
  if (!ctx) return false;

  const signatures: string[] = [];

  for (let round = 0; round < 3; round += 1) {
    ctx.drawImage(video, 0, 0, width, height);
    const signature = sampleFrameSignature(ctx, width, height);
    signatures.push(signature.signature);
    await new Promise((resolve) => window.setTimeout(resolve, 120));
  }

  const distinctCount = new Set(signatures).size;
  const lastSignature = signatures[signatures.length - 1];
  const [minLuma, maxLuma] = lastSignature.split("|").slice(0, 2).map((value) => Number(value));
  const contrast = maxLuma - minLuma;

  return Number.isFinite(contrast) && contrast >= 8 && distinctCount >= 2;
}

function createProbeVideo() {
  const video = document.createElement("video");
  video.autoplay = true;
  video.muted = true;
  video.playsInline = true;
  return video;
}

function sampleFrameSignature(ctx: CanvasRenderingContext2D, width: number, height: number) {
  const sampleWidth = Math.min(64, width);
  const sampleHeight = Math.min(48, height);
  const sample = ctx.getImageData(0, 0, sampleWidth, sampleHeight).data;

  let minLuma = 255;
  let maxLuma = 0;
  let checksum = 0;

  for (let index = 0; index < sample.length; index += 16) {
    const r = sample[index];
    const g = sample[index + 1];
    const b = sample[index + 2];
    const luma = Math.round((r * 299 + g * 587 + b * 114) / 1000);

    if (luma < minLuma) minLuma = luma;
    if (luma > maxLuma) maxLuma = luma;
    checksum = (checksum + r * 3 + g * 5 + b * 7 + index) % 10000019;
  }

  return {
    signature: `${minLuma}|${maxLuma}|${checksum}`,
    contrast: maxLuma - minLuma,
  };
}

function handleViewportResize() {
  syncOverlayCanvasSize();
  drawImageOverlay();
}

onMounted(() => {
  void refreshVideoDevices();
  window.addEventListener("resize", handleViewportResize);
  uiClockTimer = window.setInterval(() => {
    uiNow.value = Date.now();
  }, 1000);
});

onBeforeUnmount(() => {
  stopCamera();
  revokeSourcePreview();
  window.removeEventListener("resize", handleViewportResize);
  if (uiClockTimer !== null) {
    window.clearInterval(uiClockTimer);
    uiClockTimer = null;
  }
});
</script>

<style scoped lang="scss">
.two-col {
  grid-template-columns: minmax(0, 1.02fr) minmax(0, 0.98fr);
  align-items: stretch;
}

.panel {
  display: flex;
  flex-direction: column;
}

.upload-row {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 14px;
}

.mode-switch {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 4px;
  border-radius: 999px;
  background: rgba(201, 176, 153, 0.14);
}

.mode-chip {
  height: 36px;
  padding: 0 16px;
  border: none;
  border-radius: 999px;
  background: transparent;
  color: #7d6c5e;
  font-size: 13px;
  font-weight: 700;
  cursor: pointer;
  transition: background 0.2s ease, color 0.2s ease, box-shadow 0.2s ease;
}

.mode-chip.active {
  background: linear-gradient(180deg, rgba(201, 176, 153, 0.9), rgba(184, 154, 126, 0.82));
  color: #fffaf4;
  box-shadow: 0 8px 18px rgba(184, 154, 126, 0.22);
}

.control-button {
  border: none;
  cursor: pointer;
}

.control-button:disabled,
.device-refresh:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}

.secondary-button {
  background: #d8c8b5;
  color: #49372d;
}

.file-name,
.stream-meta {
  font-size: 13px;
  color: var(--muted-soft);
}

.preview-canvas-wrap {
  position: relative;
  min-height: 292px;
}

.preview-stage {
  position: relative;
  overflow: hidden;
  border-radius: 10px;
}

.image-preview-stage {
  background: #f0ece5;
}

.gesture-frame {
  height: 272px;
  margin-top: 0;
}

.preview-video,
.preview-image {
  display: block;
  width: 100%;
  min-height: 272px;
  aspect-ratio: 4 / 3;
  max-height: 408px;
  object-fit: cover;
  border-radius: 8px;
  background: #f0ece5;
}

.image-preview-fit {
  object-fit: contain;
  background: #f0ece5;
}

.overlay-canvas {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

.capture-canvas {
  display: none;
}

.hidden-file-input {
  display: none;
}

.recognition-summary {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  margin-top: 10px;
}

.recognition-chip,
.gesture-tag,
.gesture-action-chip,
.cockpit-page-tag,
.map-badge {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
}

.gesture-tag {
  padding: 6px 12px;
  background: linear-gradient(180deg, rgba(255, 247, 238, 0.98), rgba(243, 231, 219, 0.96));
  color: #5c483c;
  font-size: 13px;
  font-weight: 700;
  border: 1px solid rgba(184, 162, 141, 0.18);
}

.recognition-chip {
  padding: 6px 10px;
  background: rgba(201, 176, 153, 0.1);
  color: #8a7b6d;
  font-size: 12px;
}

.gesture-action-chip {
  padding: 6px 12px;
  background: rgba(201, 176, 153, 0.16);
  color: #6f5746;
  font-size: 13px;
  font-weight: 700;
}

.cockpit-header {
  display: flex;
  align-items: flex-start;
  justify-content: flex-start;
  gap: 16px;
  margin-bottom: 10px;
}

.cockpit-shell {
  flex: 1;
  padding: 16px;
  border-radius: 24px;
  background:
    radial-gradient(circle at top left, rgba(201, 176, 153, 0.22), transparent 30%),
    radial-gradient(circle at top right, rgba(184, 196, 208, 0.18), transparent 28%),
    linear-gradient(180deg, #fbf8f3 0%, #f2ebe3 100%);
  border: 1px solid rgba(184, 162, 141, 0.24);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.8);
}

.cockpit-topbar,
.cockpit-brand-group,
.cockpit-meta,
.call-action-row {
  display: flex;
  align-items: center;
}

.cockpit-topbar {
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
}

.cockpit-brand-group {
  gap: 10px;
}

.cockpit-brand {
  font-size: 14px;
  font-weight: 700;
  letter-spacing: 0.04em;
  color: #514238;
}

.cockpit-page-tag {
  height: 28px;
  padding: 0 10px;
  background: rgba(201, 176, 153, 0.18);
  color: #7f6a5a;
  font-size: 12px;
}

.cockpit-meta {
  gap: 12px;
  font-size: 12px;
  color: #a59a8c;
}

.section-label {
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: rgba(111, 95, 82, 0.68);
}

.cockpit-display {
  position: relative;
  min-height: 300px;
  padding: 12px;
  border-radius: 20px;
  background: linear-gradient(180deg, rgba(255, 251, 246, 0.96), rgba(245, 238, 229, 0.96));
  border: 1px solid rgba(184, 162, 141, 0.18);
  overflow: hidden;
}

.confirm-spotlight {
  position: absolute;
  top: 14px;
  right: 14px;
  z-index: 2;
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 180px;
  padding: 14px 16px;
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(201, 176, 153, 0.94), rgba(184, 154, 126, 0.92));
  color: #fff9f3;
  box-shadow: 0 14px 30px rgba(184, 154, 126, 0.26);
}

.confirm-spotlight strong {
  font-size: 26px;
  color: inherit;
}

.confirm-spotlight span:last-child {
  font-size: 12px;
  line-height: 1.5;
  color: rgba(255, 249, 243, 0.9);
}

.cockpit-off,
.cockpit-home,
.cockpit-media,
.cockpit-comfort,
.cockpit-vehicle,
.cockpit-call {
  min-height: 276px;
}

.off-screen,
.call-screen,
.media-main,
.comfort-main,
.vehicle-main,
.widget-card,
.metric-card {
  border-radius: 20px;
  border: 1px solid rgba(184, 162, 141, 0.18);
}

.off-screen {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 276px;
  overflow: hidden;
  background: linear-gradient(180deg, #f1e9df 0%, #e7ddd1 100%);
}

.off-glow {
  position: absolute;
  width: 180px;
  height: 180px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(201, 176, 153, 0.42) 0%, rgba(201, 176, 153, 0.08) 60%, transparent 72%);
  filter: blur(10px);
}

.off-copy {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  max-width: 320px;
  text-align: center;
  color: #5d4b3f;
}

.off-copy strong,
.widget-card strong,
.media-main strong,
.comfort-main strong,
.metric-card strong,
.vehicle-main strong,
.call-screen strong {
  font-size: 28px;
  color: #43352d;
}

.off-copy span:last-child,
.widget-card span:last-child,
.media-main span:last-child,
.comfort-main span:last-child,
.metric-card span:last-child,
.vehicle-main span:last-child,
.call-screen span:last-child {
  font-size: 12px;
  line-height: 1.5;
  color: #9d8f82;
}

.home-hero {
  display: grid;
  grid-template-columns: 1.6fr 1fr;
  gap: 14px;
  min-height: 276px;
  align-items: stretch;
}

.map-surface {
  position: relative;
  min-height: 218px;
  overflow: hidden;
  border-radius: 20px;
  background:
    linear-gradient(135deg, rgba(242, 236, 228, 0.96), rgba(232, 223, 212, 0.98)),
    linear-gradient(0deg, rgba(255, 255, 255, 0.02), rgba(255, 255, 255, 0.02));
  border: 1px solid rgba(184, 162, 141, 0.18);
}

.map-grid {
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(181, 164, 148, 0.16) 1px, transparent 1px),
    linear-gradient(90deg, rgba(181, 164, 148, 0.16) 1px, transparent 1px);
  background-size: 34px 34px;
}

.map-route {
  position: absolute;
  border-radius: 999px;
  background: linear-gradient(90deg, rgba(184, 196, 208, 0.28), rgba(184, 196, 208, 0.86));
}

.route-a {
  top: 44%;
  left: 12%;
  width: 58%;
  height: 10px;
  transform: rotate(-18deg);
}

.route-b {
  top: 34%;
  right: 10%;
  width: 28%;
  height: 10px;
  transform: rotate(62deg);
}

.vehicle-marker {
  position: absolute;
  top: 45%;
  left: 48%;
  width: 56px;
  height: 24px;
  border-radius: 14px;
  background: linear-gradient(90deg, #fffdf9, #d9c5b2);
  box-shadow: 0 0 18px rgba(184, 162, 141, 0.22);
}

.map-badge {
  position: absolute;
  left: 16px;
  bottom: 16px;
  padding: 7px 12px;
  background: rgba(255, 250, 244, 0.86);
  color: #6a5a4d;
  font-size: 12px;
}

.home-side-widgets,
.media-side,
.comfort-side,
.vehicle-side {
  display: grid;
  gap: 14px;
}

.home-side-widgets {
  grid-template-rows: repeat(2, minmax(0, 1fr));
}

.widget-card,
.metric-card,
.media-main,
.comfort-main,
.vehicle-main {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 16px;
  background: linear-gradient(180deg, rgba(255, 252, 247, 0.98), rgba(244, 236, 227, 0.96));
}

.widget-card strong,
.metric-card strong {
  font-size: 20px;
}

.focused {
  box-shadow: inset 0 0 0 1px rgba(184, 154, 126, 0.28), 0 12px 26px rgba(184, 154, 126, 0.16);
  background: linear-gradient(180deg, rgba(255, 250, 244, 1), rgba(242, 232, 219, 0.98));
}

.mini-track,
.climate-ring,
.volume-ring {
  position: relative;
  width: 100%;
  height: 10px;
  margin-top: 4px;
  overflow: hidden;
  border-radius: 999px;
  background: rgba(201, 176, 153, 0.16);
}

.mini-fill,
.climate-ring-fill,
.volume-ring-fill {
  position: absolute;
  inset: 0 auto 0 0;
  border-radius: 999px;
  background: linear-gradient(90deg, #b89a7e, #d9c5b2);
}

.media-layout,
.comfort-layout,
.vehicle-layout {
  display: grid;
  grid-template-columns: 1.2fr 1fr;
  gap: 14px;
  min-height: 276px;
}

.media-main,
.comfort-main,
.vehicle-main {
  justify-content: center;
}

.volume-knob {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-top: 8px;
}

.volume-value {
  font-size: 42px;
  font-weight: 700;
  color: #4b3a2e;
}

.call-screen {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  min-height: 276px;
  background:
    radial-gradient(circle at top, rgba(201, 176, 153, 0.26), transparent 34%),
    linear-gradient(180deg, rgba(254, 250, 244, 0.98), rgba(241, 232, 221, 0.98));
}

.caller-avatar {
  display: grid;
  place-items: center;
  width: 88px;
  height: 88px;
  border-radius: 50%;
  background: linear-gradient(180deg, #d9c5b2, #b89a7e);
  color: #fffaf4;
  font-size: 28px;
  font-weight: 700;
  box-shadow: 0 10px 24px rgba(184, 162, 141, 0.22);
}

.call-action-row {
  gap: 18px;
  margin-top: 10px;
}

.call-action {
  display: grid;
  place-items: center;
  width: 72px;
  height: 72px;
  border-radius: 50%;
  color: #fffaf4;
  font-size: 13px;
  font-weight: 700;
}

.call-action.accept {
  background: linear-gradient(180deg, #97c7b8, #7cb9a8);
}

.call-action.decline {
  background: linear-gradient(180deg, #e3c2b7, #dbb5a8);
}

.cockpit-dock {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 10px;
  margin-top: 14px;
}

.dock-item {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 46px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.62);
  color: #9b8c7d;
  font-size: 13px;
  border: 1px solid rgba(184, 162, 141, 0.12);
}

.dock-item.active {
  background: linear-gradient(180deg, rgba(201, 176, 153, 0.3), rgba(184, 154, 126, 0.22));
  color: #5d4c40;
}

.device-select {
  max-width: 240px;
  min-width: 180px;
  height: 42px;
  padding: 0 12px;
  border: 1px solid rgba(184, 162, 141, 0.35);
  border-radius: 12px;
  background: #f7f2eb;
  color: #6f5f52;
  font-size: 14px;
}

.device-refresh {
  height: 42px;
  padding: 0 16px;
  border: none;
  border-radius: 12px;
  background: #eadfce;
  color: #6f5f52;
  cursor: pointer;
}

@media (max-width: 960px) {
  .cockpit-header,
  .cockpit-topbar {
    flex-direction: column;
    align-items: flex-start;
  }

  .home-hero,
  .media-layout,
  .comfort-layout,
  .vehicle-layout,
  .cockpit-dock {
    grid-template-columns: 1fr;
  }
}
</style>
