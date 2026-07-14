<template>
  <section class="page-shell">
    <header class="page-header">
      <h1>手势控车</h1>
      <p>支持单张图片与实时视频输入；实时模式连续显示后端关键点与中文标注画面</p>
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
            <div class="camera-controls">
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
            </div>
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
            <iframe
              v-if="cameraActive && cameraDisplayUrl"
              class="preview-image stream-player"
              :src="cameraDisplayUrl"
              title="手势实时识别标注画面"
              allow="autoplay; fullscreen"
              allowfullscreen
            />
            <div v-else-if="cameraActive" class="image-placeholder gesture-frame">
              后端正在生成标注画面
              <div class="small">识别结果返回后将显示关键点、手势与功能</div>
            </div>
          </div>
          <div v-else-if="sourcePreviewUrl" class="preview-stage image-preview-stage">
            <img
              class="preview-image image-preview-fit"
              :src="imageDisplayUrl"
              alt="手势图片识别标注画面"
            />
          </div>
          <canvas ref="captureCanvasRef" class="capture-canvas" />
          <div
            v-if="(inputMode === 'camera' && !cameraActive) || (inputMode === 'image' && !sourcePreviewUrl)"
            class="image-placeholder gesture-frame"
          >
            {{ inputMode === "camera" ? "手势实时识别" : "手势图片识别" }}
            <div class="small">
              {{ inputMode === "camera" ? "点击“开启摄像头”开始检测驾驶员手势" : "上传单张手势图片，直接查看后端返回的标注结果" }}
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
              <span class="section-label">执行结果</span>
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
                    <span>音量 {{ Math.round(panelState.volume) }}% · 顺逆时针画圈可直接调节</span>
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
                    <span>画圈调音量时会自动前置媒体页</span>
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
  fetchOwnerGestureStreamStateApi,
  fetchOwnerGestureStreamResultApi,
  startOwnerGestureStreamApi,
  stopOwnerGestureStreamApi,
  type OwnerControlPanelState,
  type OwnerGestureFrameResult,
} from "@/api/owner_gesture";

const loading = ref(false);
const error = ref("");
const result = ref<OwnerGestureFrameResult | null>(null);
const inputMode = ref<"camera" | "image">("camera");
const fileInputRef = ref<HTMLInputElement | null>(null);
const videoRef = ref<HTMLVideoElement | null>(null);
const captureCanvasRef = ref<HTMLCanvasElement | null>(null);
const panelState = ref<OwnerControlPanelState>(createDefaultPanelState());
const cameraActive = ref(false);
const cameraSourceChoice = ref("0");
const customCameraSource = ref(4);
const streamVideoUrl = ref("");
const sessionId = ref("");
const cameraDeviceLabel = ref("");
const videoDevices = ref<Array<{ deviceId: string; label: string }>>([]);
const selectedDeviceId = ref("");
const sourcePreviewUrl = ref("");
const uiNow = ref(Date.now());

const baseFrameIntervalMs = 8;
const previewIdealWidth = 960;
const previewIdealHeight = 720;
const captureMaxWidth = 960;
const captureMaxHeight = 720;
const captureJpegQuality = 0.88;
let mediaStream: MediaStream | null = null;
let activeSourceVideo: HTMLVideoElement | null = null;
let captureTimer: number | null = null;
let requestInFlight = false;
let uiClockTimer: number | null = null;
let streamResultTimer: number | null = null;
let streamStarting = false;
let lastInferenceDurationMs = 120;

const LABEL_MAP: Record<string, string> = {
  open_palm: "手掌张开",
  palm: "手掌张开",
  fist: "握拳",
  point: "待机",
  pointing: "待机",
  index_circle: "待机",
  circle_cw: "顺时针画圈",
  circle_ccw: "逆时针画圈",
  swipe_left: "张开拳头",
  swipe_right: "收回拳头",
  thumbs_up: "拇指向上",
  thumb_up: "拇指向上",
  thumbs_down: "拇指向下",
  thumb_down: "拇指向下",
  thunb_index: "捏指",
  thumb_index: "捏指",
  wave: "挥手",
  idle: "待机",
  unknown: "未识别",
  "未检测到手部": "未检测到手部",
};

const GESTURE_ACTION_MAP: Record<string, string> = {
  open_palm: "唤醒",
  palm: "唤醒",
  fist: "确认",
  point: "待机",
  pointing: "待机",
  index_circle: "待机",
  circle_cw: "音量+",
  circle_ccw: "音量-",
  swipe_left: "上一个功能",
  swipe_right: "下一个功能",
  thumbs_up: "接听",
  thumb_up: "接听",
  thumbs_down: "挂断",
  thumb_down: "挂断",
  thunb_index: "主页",
  thumb_index: "主页",
  wave: "待机",
  idle: "待机",
  unknown: "等待",
  "未检测到手部": "无动作",
};

const ACTION_KEY_LABEL_MAP: Record<string, string> = {
  wake: "唤醒",
  confirm: "确认",
  volume_adjust: "音量",
  volume_up: "音量+",
  volume_down: "音量-",
  prev_func: "切换",
  next_func: "切换",
  call_answer: "接听",
  call_hangup: "挂断",
  home: "主页",
  idle: "待机",
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

const currentGestureKey = computed(() => {
  if (result.value?.gesture) {
    return result.value.gesture;
  }
  return panelState.value.last_gesture || "";
});

const gestureLabel = computed(() => {
  const gesture = currentGestureKey.value;
  if (!gesture) return "—";
  return LABEL_MAP[gesture] || gesture;
});

const gestureActionLabel = computed(() => {
  const action = result.value?.action;
  if (action) {
    return ACTION_KEY_LABEL_MAP[action] || action;
  }
  const key = currentGestureKey.value || "unknown";
  return GESTURE_ACTION_MAP[key] || GESTURE_ACTION_MAP.unknown;
});

const recognitionConfidence = computed(() => `${((result.value?.confidence ?? 0) * 100).toFixed(1)}%`);

const cameraDisplayUrl = computed(() => {
  if (inputMode.value !== "camera") return "";
  return streamVideoUrl.value;
});

const imageDisplayUrl = computed(() => result.value?.annotated_image || sourcePreviewUrl.value || "");

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
    ? "顺时针画圈升高音量，逆时针画圈降低音量，握拳可确认当前播放状态。"
    : "媒体已暂停，握拳后会恢复播放。";
});

const comfortStatusText = computed(() => {
  return panelState.value.comfort_scene === "舒享"
    ? "握拳后已执行舒享模式，温控与座椅同步强化。"
    : "张开拳头或收回拳头切换到此页后，可通过握拳执行舒适场景。";
});

const vehicleStatusText = computed(() => {
  return panelState.value.vehicle_status === "已完成整车检查"
    ? "整车检查已执行完成，关键车况正常。"
    : "可通过张开拳头或收回拳头切换到车辆页，再用握拳执行车况确认。";
});

const callStatusText = computed(() => {
  if (panelState.value.phone_call_active) {
    return "通话已接通，拇指向下可直接挂断。";
  }
  return "检测到来电场景时，拇指向上可进入通话界面。";
});

const showConfirmSpotlight = computed(() => {
  return (
    (panelState.value.last_command === "ConfirmAction" ||
      panelState.value.last_command === "WakeSystem") &&
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
    return cameraActive.value ? "实时模式：显示后端关键点与中文标注画面" : "摄像头未开启";
  }
  return sourcePreviewUrl.value ? "图片模式：仅识别静态手势，显示后端返回的标注结果" : "图片模式：等待上传";
});

const previewStatusText = computed(() => {
  if (inputMode.value === "camera") {
    if (!cameraActive.value) {
      return cameraStatusText.value;
    }
    return `${cameraStatusText.value} · 当前显示后端实时返回的标注画面`;
  }
  if (loading.value) {
    return "图片识别中 ... 仅支持静态手势";
  }
  if (sourcePreviewUrl.value) {
    return result.value?.annotated_image ? "已显示后端返回的标注结果 · 动态手势请使用实时模式" : "已加载上传图片";
  }
  return "上传图片后可查看后端标注结果，图片模式仅识别静态手势";
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
  inputMode.value = "image";
  sourcePreviewUrl.value = URL.createObjectURL(file);
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

async function startCamera() {
  if (cameraActive.value) return;

  inputMode.value = "camera";
  error.value = "";
  sessionId.value = createSessionId();
  result.value = null;
  panelState.value = createDefaultPanelState();

  try {
    await stopOwnerGestureStreamApi().catch(() => undefined);
    const cameraSource = cameraSourceChoice.value === "custom" ? customCameraSource.value : cameraSourceChoice.value;
    if (!Number.isInteger(Number(cameraSource)) || Number(cameraSource) < 0) {
      throw new Error("请选择有效的摄像头编号");
    }
    const { data } = await startOwnerGestureStreamApi(String(cameraSource), 15);
    if (!data.playback_url) throw new Error("后端未返回 MediaMTX 播放地址");
    cameraDeviceLabel.value = `正在打开摄像头 ${cameraSource}`;
    cameraActive.value = true;
    streamStarting = true;
    startStreamResultPolling();
    await waitForOwnerGestureStreamPublished(data.playback_url);
  } catch (err: any) {
    stopCamera();
    error.value = axios.isAxiosError(err)
      ? String(err.response?.data?.detail || err.message || "后端摄像头启动失败")
      : err?.message || "后端摄像头启动失败";
  }
}

function stopCamera() {
  void stopOwnerGestureStreamApi().catch(() => undefined);
  stopStreamResultPolling();
  streamStarting = false;
  streamVideoUrl.value = "";
  stopCaptureLoop();
  loading.value = false;
  requestInFlight = false;
  cameraActive.value = false;

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

function startStreamResultPolling() {
  stopStreamResultPolling();
  const poll = async () => {
    if (!cameraActive.value) return;
    try {
      const { data } = await fetchOwnerGestureStreamResultApi();
      result.value = data;
      if (data.panel_state) panelState.value = data.panel_state;
      error.value = "";
    } catch {
      if (cameraActive.value && !streamStarting) error.value = "无法获取后端手势识别结果";
    } finally {
      if (cameraActive.value) streamResultTimer = window.setTimeout(poll, 500);
    }
  };
  void poll();
}

async function waitForOwnerGestureStreamPublished(playbackUrl: string) {
  const deadline = Date.now() + 30_000;
  while (cameraActive.value && Date.now() < deadline) {
    const { data } = await fetchOwnerGestureStreamStateApi();
    if (data.last_error) throw new Error(data.last_error);
    if (!data.running) throw new Error("后端手势推流已停止");
    if (data.published) {
      cameraDeviceLabel.value = data.source === "0" ? "内置摄像头" : data.source === "1" ? "4K USB Camera" : `视频源 ${data.source}`;
      await new Promise((resolve) => window.setTimeout(resolve, 900));
      if (!cameraActive.value) return;
      streamVideoUrl.value = `${playbackUrl}?t=${Date.now()}`;
      streamStarting = false;
      error.value = "";
      return;
    }
    await new Promise((resolve) => window.setTimeout(resolve, 400));
  }
  throw new Error("等待 MediaMTX 手势流就绪超时");
}

function stopStreamResultPolling() {
  if (streamResultTimer !== null) {
    window.clearTimeout(streamResultTimer);
    streamResultTimer = null;
  }
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
    const nextDelay = Math.max(8, Math.min(24, Math.round(lastInferenceDurationMs * 0.04 + baseFrameIntervalMs)));
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
      const wasAwake = nextState.system_awake;
      nextState.system_awake = true;
      if (wasAwake) {
        nextState.last_feedback = "系统已唤醒。";
      } else {
        nextState.phone_call_active = false;
        nextState.current_mode = "home";
        nextState.focus_tile = "home";
        nextState.last_feedback = "CMC 已唤醒，主页信息恢复显示。";
      }
    } else if (data.control_command === "SwitchPrevFeature" && nextState.system_awake && !nextState.phone_call_active) {
      nextState.current_mode = shiftMode(nextState.current_mode, -1);
      nextState.focus_tile = nextState.current_mode;
      nextState.last_feedback = `已切换至${MODE_LABEL_MAP[nextState.current_mode] || "主页"}界面。`;
    } else if (data.control_command === "SwitchNextFeature" && nextState.system_awake && !nextState.phone_call_active) {
      nextState.current_mode = shiftMode(nextState.current_mode, 1);
      nextState.focus_tile = nextState.current_mode;
      nextState.last_feedback = `已切换至${MODE_LABEL_MAP[nextState.current_mode] || "主页"}界面。`;
    } else if (
      (data.control_command === "AdjustVolume" ||
        data.control_command === "AdjustVolumeUp" ||
        data.control_command === "AdjustVolumeDown") &&
      nextState.system_awake
    ) {
      nextState.current_mode = "media";
      nextState.focus_tile = "media";
      nextState.media_playing = true;
      if (data.control_command === "AdjustVolumeDown") {
        nextState.volume = Math.max(0, nextState.volume - 6);
        nextState.last_feedback = `媒体音量已下调至 ${nextState.volume}%。`;
      } else {
        nextState.volume = Math.min(100, nextState.volume + 6);
        nextState.last_feedback = `媒体音量已调至 ${nextState.volume}%。`;
      }
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
      nextState.last_feedback = "已捏指返回主页。";
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
  return;
}

onMounted(() => {
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

.camera-source-control {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  min-height: 36px;
  color: var(--muted-soft);
  font-size: 13px;
}

.camera-source-control select,
.camera-source-control input {
  height: 36px;
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

.camera-controls {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  flex: 0 0 auto;
  flex-wrap: nowrap;
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
  min-height: 0;
}

.preview-stage {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  border-radius: 10px;
  background: #f0ece5;
}

.image-preview-stage {
  background: #f0ece5;
}

.gesture-frame {
  width: 100%;
  height: auto;
  aspect-ratio: 4 / 3;
  margin-top: 0;
}

.stream-player {
  border: 0;
}

.preview-video,
.preview-image {
  display: block;
  width: 100%;
  min-height: 272px;
  aspect-ratio: 4 / 3;
  max-height: 408px;
  object-fit: contain;
  border-radius: 8px;
  background: #f0ece5;
}

.capture-video {
  position: absolute;
  width: 1px;
  height: 1px;
  opacity: 0;
  pointer-events: none;
  left: -9999px;
  top: -9999px;
}

.image-preview-fit {
  object-fit: contain;
  background: #f0ece5;
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
