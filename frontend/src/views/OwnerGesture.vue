<template>
  <section class="page-shell">
    <header class="page-header">
      <h1>手势控车</h1>
      <p>调用本机前置摄像头进行实时识别，完成非接触式控车交互</p>
    </header>

    <section class="two-col">
      <article class="panel">
        <div class="upload-row">
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
          <span class="file-name">{{ cameraActive ? "实时模式：约每 0.45 秒分析一帧" : "摄像头未开启" }}</span>
        </div>

        <div class="video-status" v-if="error">
          <span class="off">{{ error }}</span>
        </div>

        <div class="preview-canvas-wrap">
          <video
            ref="videoRef"
            class="preview-video"
            autoplay
            muted
            playsinline
            v-show="cameraActive"
            @loadedmetadata="onVideoReady"
          />
          <canvas
            ref="canvasRef"
            class="overlay-canvas"
            :width="canvasW"
            :height="canvasH"
          />
          <canvas ref="captureCanvasRef" class="capture-canvas" />
          <div v-if="!cameraActive" class="image-placeholder gesture-frame">
            手势实时识别
            <div class="small">点击“开启摄像头”开始检测驾驶员手势</div>
          </div>
        </div>

        <div class="stream-meta">{{ cameraStatusText }}</div>

        <template v-if="result">
          <div class="recognition-summary">
            <span class="gesture-tag">{{ gestureLabel }}</span>
            <span class="recognition-chip gesture-confidence">
              置信度 {{ (result.confidence * 100).toFixed(1) }}%
            </span>
            <span class="gesture-action-chip">{{ gestureActionLabel }}</span>
            <span class="recognition-chip">
              {{ result.keypoints?.length || 0 }} 点 · {{ (result.keypoints?.length || 0) / 21 }} 手
            </span>
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
              <span>23°C</span>
              <span>{{ cockpitClock }}</span>
            </div>
          </div>

          <div class="cockpit-display">
            <div class="cockpit-off" v-if="activeCockpitPage === 'off'">
              <div class="off-screen">
                <div class="off-glow"></div>
                <div class="off-copy">
                  <span class="section-label">待机状态</span>
                  <strong>CMC 已关闭</strong>
                  <span>检测到唤醒手势后将直接进入主页，不再停留在通话界面。</span>
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
                  <div class="widget-card media-widget">
                    <span class="section-label">媒体</span>
                    <strong>行车伴音</strong>
                    <span>音量 {{ Math.round(panelState.volume) }}% · 蓝牙已连接</span>
                    <div class="mini-track"><div class="mini-fill" :style="{ width: `${panelState.volume}%` }"></div></div>
                  </div>
                  <div class="widget-card vehicle-widget">
                    <span class="section-label">车辆</span>
                    <strong>{{ panelState.climate_temperature }}°C 舒适座舱</strong>
                    <span>{{ modeSummary }}</span>
                  </div>
                </div>
              </div>
            </div>

            <div class="cockpit-comfort" v-else-if="activeCockpitPage === 'comfort'">
              <div class="comfort-layout">
                <div class="comfort-main">
                  <span class="section-label">空调控制</span>
                  <strong>{{ panelState.climate_temperature }}°C</strong>
                  <span>{{ comfortStatusText }}</span>
                  <div class="climate-ring">
                    <div class="climate-ring-fill" :style="{ width: temperatureFillWidth }"></div>
                  </div>
                </div>
                <div class="comfort-side">
                  <div class="metric-card">
                    <span class="section-label">风量</span>
                    <strong>3 档</strong>
                    <span>维持前排柔风</span>
                  </div>
                  <div class="metric-card">
                    <span class="section-label">座椅</span>
                    <strong>主驾通风</strong>
                    <span>已同步舒适模式</span>
                  </div>
                  <div class="metric-card">
                    <span class="section-label">空气</span>
                    <strong>PM2.5 优</strong>
                    <span>座舱净化持续运行</span>
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
                    <span>{{ panelState.phone_call_active ? "静音" : "接听" }}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div class="cockpit-dock">
            <div class="dock-item" :class="{ active: activeCockpitPage === 'home' }">主页</div>
            <div class="dock-item" :class="{ active: activeCockpitPage === 'comfort' }">舒适</div>
            <div class="dock-item" :class="{ active: activeCockpitPage === 'call' }">电话</div>
            <div class="dock-item">车辆</div>
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
  type OwnerGestureResult,
} from "@/api/owner_gesture";

const loading = ref(false);
const error = ref("");
const result = ref<OwnerGestureResult | null>(null);
const videoRef = ref<HTMLVideoElement | null>(null);
const canvasRef = ref<HTMLCanvasElement | null>(null);
const captureCanvasRef = ref<HTMLCanvasElement | null>(null);
const canvasW = ref(300);
const canvasH = ref(220);
const panelState = ref<OwnerControlPanelState>(createDefaultPanelState());
const cameraActive = ref(false);
const sessionId = ref("");
const cameraDeviceLabel = ref("");
const videoDevices = ref<Array<{ deviceId: string; label: string }>>([]);
const selectedDeviceId = ref("");
const debugVideoWidth = ref(0);
const debugVideoHeight = ref(0);
const debugReadyState = ref(0);
const debugTrackState = ref("none");
const debugTrackMuted = ref("false");
const debugPreviewFrames = ref(0);
const debugCaptureMode = ref("video");
const debugRawFrameState = ref("已停用");
const debugFrameSignature = ref("");
const debugFrameContrast = ref(0);

const frameIntervalMs = 550;
const captureMaxWidth = 640;
const captureMaxHeight = 480;
const captureJpegQuality = 0.72;
let mediaStream: MediaStream | null = null;
let activeSourceVideo: HTMLVideoElement | null = null;
let captureTimer: number | null = null;
let requestInFlight = false;
let previewFrameHandle: number | null = null;

const LABEL_MAP: Record<string, string> = {
  open_palm: "手掌张开",
  fist: "握拳",
  thumbs_up: "拇指向上",
  thumbs_down: "拇指向下",
  unknown: "未识别",
  "未检测到手部": "未检测到手部",
};

const GESTURE_ACTION_MAP: Record<string, string> = {
  open_palm: "唤醒",
  fist: "确认",
  thumbs_up: "接听",
  thumbs_down: "挂断",
  unknown: "等待",
  "未检测到手部": "无动作",
};

function createDefaultPanelState(): OwnerControlPanelState {
  return {
    system_awake: false,
    volume: 32,
    climate_temperature: 24,
    phone_call_active: false,
    current_mode: "home",
    last_gesture: null,
    last_command: null,
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
const systemAwake = computed(() => panelState.value.system_awake);
const modeSummary = computed(() => {
  if (!systemAwake.value) {
    return "系统默认关闭，识别到唤醒手势后会立即开启并回到主页。";
  }
  if (panelState.value.phone_call_active) {
    return "当前通话已接管主界面，可直接观察通话状态。";
  }
  if (panelState.value.current_mode === "control") {
    return "舒适功能已被置于前台，温控与座舱参数突出显示。";
  }
  return "主页已经恢复显示，导航、媒体与车辆卡片进入待命状态。";
});
const cockpitPages = [
  { key: "home", label: "主页" },
  { key: "comfort", label: "舒适" },
  { key: "call", label: "通话" },
] as const;
type CockpitPageKey = "off" | (typeof cockpitPages)[number]["key"];
const activeCockpitPage = computed<CockpitPageKey>(() => {
  if (!systemAwake.value) {
    return "off";
  }
  if (panelState.value.phone_call_active) {
    return "call";
  }
  if (panelState.value.current_mode === "control") {
    return "comfort";
  }
  return "home";
});
const activePageLabel = computed(() => {
  if (activeCockpitPage.value === "off") {
    return "关闭";
  }
  return cockpitPages.find((page) => page.key === activeCockpitPage.value)?.label || "主页";
});
const comfortStatusText = computed(() => {
  if (panelState.value.current_mode === "control") {
    return "握拳确认后已切入舒适控制，温度与空气质量卡片高亮。";
  }
  return "当前维持舒适巡航设定，空调、座椅与净化联动待命。";
});
const callStatusText = computed(() => {
  if (panelState.value.phone_call_active) {
    return "通话已接通，方向盘与手势挂断逻辑均可接管结束操作。";
  }
  return "检测到来电场景时，拇指向上可进入通话界面。";
});
const cockpitClock = computed(() => {
  const base = panelState.value.updated_at ? new Date(panelState.value.updated_at) : new Date();
  return base.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit", hour12: false });
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

async function startCamera() {
  if (cameraActive.value) return;

  error.value = "";
  sessionId.value = createSessionId();
  result.value = null;
  panelState.value = createDefaultPanelState();
  resetDebugState();

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
    syncCanvasToPreview();
    startPreviewLoop();
    startCaptureLoop();
  } catch (err: any) {
    stopCamera();
    error.value = normalizeCameraError(err);
  }
}

function stopCamera() {
  stopCaptureLoop();
  stopPreviewLoop();
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

  clearCanvas();
  clearPreviewCanvas();
}

function bindTrackEvents(track?: MediaStreamTrack) {
  if (!track) return;

  debugTrackState.value = track.readyState;
  debugTrackMuted.value = String(track.muted);

  track.onmute = () => {
    debugTrackMuted.value = "true";
    if (!cameraActive.value) return;
    error.value = "摄像头已连接但没有输出画面，常见原因是微信、会议软件等正在占用摄像头。";
  };
  track.onunmute = () => {
    debugTrackMuted.value = "false";
  };
  track.onended = () => {
    debugTrackState.value = "ended";
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
        debugTrackState.value = track?.readyState || "unknown";
        debugTrackMuted.value = String(track?.muted ?? false);
        cameraDeviceLabel.value = deviceLabel;
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
      } catch (error) {
        lastError = error;
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

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: buildVideoConstraints({ facingMode: "user" }),
      audio: false,
    });
    const track = stream.getVideoTracks()[0];
    debugTrackState.value = track?.readyState || "unknown";
    debugTrackMuted.value = String(track?.muted ?? false);
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
    debugTrackState.value = track?.readyState || "unknown";
    debugTrackMuted.value = String(track?.muted ?? false);
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
    width: { ideal: captureMaxWidth, max: 1280 },
    height: { ideal: captureMaxHeight, max: 720 },
    frameRate: { ideal: 15, max: 24 },
    ...extra,
  };
}

function startCaptureLoop() {
  stopCaptureLoop();

  const tick = async () => {
    if (!cameraActive.value) return;
    await captureFrame();
    if (!cameraActive.value) return;
    captureTimer = window.setTimeout(() => {
      void tick();
    }, frameIntervalMs);
  };

  void tick();
}

function stopCaptureLoop() {
  if (captureTimer !== null) {
    window.clearTimeout(captureTimer);
    captureTimer = null;
  }
}

function startPreviewLoop() {
  stopPreviewLoop();
  debugPreviewFrames.value = 0;

  const draw = () => {
    if (!cameraActive.value) return;

    const video = activeSourceVideo;
    if (video) {
      debugVideoWidth.value = video.videoWidth || 0;
      debugVideoHeight.value = video.videoHeight || 0;
      debugReadyState.value = video.readyState;
      if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        debugPreviewFrames.value += 1;
      }
    }

    previewFrameHandle = window.requestAnimationFrame(draw);
  };

  previewFrameHandle = window.requestAnimationFrame(draw);
}

function stopPreviewLoop() {
  if (previewFrameHandle !== null) {
    window.cancelAnimationFrame(previewFrameHandle);
    previewFrameHandle = null;
  }
}

async function captureFrame() {
  if (!cameraActive.value || requestInFlight) return;

  const video = activeSourceVideo;
  const captureCanvas = captureCanvasRef.value;
  if (!captureCanvas) {
    return;
  }

  if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    return;
  }
  const captureSize = computeCaptureSize(video);
  captureCanvas.width = captureSize.width;
  captureCanvas.height = captureSize.height;

  const ctx = captureCanvas.getContext("2d");
  if (!ctx) return;
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
  debugCaptureMode.value = "video";

  updateDebugFrameStats(ctx, captureCanvas.width, captureCanvas.height);

  const blob = await new Promise<Blob | null>((resolve) => {
    captureCanvas.toBlob(resolve, "image/jpeg", captureJpegQuality);
  });
  if (!blob) {
    error.value = "摄像头帧抓取失败";
    return;
  }

  requestInFlight = true;
  loading.value = true;
  error.value = "";

  const formData = new FormData();
  formData.append("file", blob, "owner-gesture-frame.jpg");
  formData.append("session_id", sessionId.value);

  try {
    const { data } = await fetchOwnerGestureApi(formData);
    result.value = data;
    applyResultToPanelState(data);
    await nextTick();
    drawKeypoints(data.keypoints);
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
    loading.value = false;
    requestInFlight = false;
  }
}

function onVideoReady() {
  syncCanvasToPreview();
  startPreviewLoop();
  if (result.value?.keypoints?.length) {
    nextTick(() => drawKeypoints(result.value?.keypoints || []));
  }
}

function applyResultToPanelState(data: OwnerGestureResult) {
  if (data.panel_state) {
    panelState.value = data.panel_state;
    return;
  }

  const nextState: OwnerControlPanelState = {
    ...panelState.value,
    last_gesture: data.gesture,
    last_command: data.control_command,
    updated_at: data.updated_at,
  };

  if (data.triggered && data.control_command) {
    if (data.control_command === "WakeSystem") {
      nextState.system_awake = true;
      nextState.phone_call_active = false;
      nextState.current_mode = "home";
    } else if (data.control_command === "ConfirmAction" && nextState.system_awake) {
      nextState.current_mode = "control";
    } else if (data.control_command === "AnswerCall" && nextState.system_awake) {
      nextState.phone_call_active = true;
    } else if (data.control_command === "HangUpCall" && nextState.system_awake) {
      nextState.phone_call_active = false;
      nextState.current_mode = "home";
    }
  }

  panelState.value = nextState;
}

function computeCaptureSize(video: HTMLVideoElement) {
  const sourceWidth = Math.max(1, debugVideoWidth.value || video.videoWidth || captureMaxWidth);
  const sourceHeight = Math.max(1, debugVideoHeight.value || video.videoHeight || captureMaxHeight);
  const scale = Math.min(1, captureMaxWidth / sourceWidth, captureMaxHeight / sourceHeight);

  return {
    width: Math.max(1, Math.round(sourceWidth * scale)),
    height: Math.max(1, Math.round(sourceHeight * scale)),
  };
}

function syncCanvasToPreview() {
  const wrapper = videoRef.value?.parentElement;
  if (!wrapper) return;

  const width = wrapper.clientWidth || 300;
  const videoWidth = activeSourceVideo?.videoWidth || 4;
  const videoHeight = activeSourceVideo?.videoHeight || 3;
  const ratio = videoWidth / videoHeight;
  const computedHeight = width / ratio;

  canvasW.value = width;
  canvasH.value = Math.min(360, Math.max(220, computedHeight));
}

const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

function drawKeypoints(keypoints: Array<{ x: number; y: number }>) {
  const canvas = canvasRef.value;
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!keypoints?.length) return;

  const perHand = 21;
  const numHands = Math.floor(keypoints.length / perHand);

  for (let handIndex = 0; handIndex < numHands; handIndex += 1) {
    const hand = keypoints.slice(handIndex * perHand, (handIndex + 1) * perHand);
    const width = canvas.width;
    const height = canvas.height;

    ctx.strokeStyle = "#2dd4bf";
    ctx.lineWidth = 2;
    for (const [start, end] of HAND_CONNECTIONS) {
      const startPoint = hand[start];
      const endPoint = hand[end];
      if (startPoint && endPoint) {
        ctx.beginPath();
        ctx.moveTo(startPoint.x * width, startPoint.y * height);
        ctx.lineTo(endPoint.x * width, endPoint.y * height);
        ctx.stroke();
      }
    }

    ctx.fillStyle = "#c9b099";
    for (const point of hand) {
      ctx.beginPath();
      ctx.arc(point.x * width, point.y * height, 3, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}

function clearCanvas() {
  const canvas = canvasRef.value;
  const ctx = canvas?.getContext("2d");
  if (!canvas || !ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function clearPreviewCanvas() {
  const canvas = canvasRef.value;
  const ctx = canvas?.getContext("2d");
  if (!canvas || !ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
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
    debugFrameSignature.value = signature.signature;
    debugFrameContrast.value = signature.contrast;
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

function updateDebugFrameStats(ctx: CanvasRenderingContext2D, width: number, height: number) {
  const frame = sampleFrameSignature(ctx, width, height);
  debugFrameSignature.value = frame.signature;
  debugFrameContrast.value = frame.contrast;
}

function resetDebugState() {
  debugVideoWidth.value = 0;
  debugVideoHeight.value = 0;
  debugReadyState.value = 0;
  debugTrackState.value = "none";
  debugTrackMuted.value = "false";
  debugPreviewFrames.value = 0;
  debugCaptureMode.value = "video";
  debugRawFrameState.value = "已停用";
  debugFrameSignature.value = "";
  debugFrameContrast.value = 0;
}

onMounted(() => {
  void refreshVideoDevices();
});

onBeforeUnmount(() => {
  stopCamera();
});
</script>

<style scoped lang="scss">
.two-col {
  grid-template-columns: minmax(0, 1.04fr) minmax(0, 0.96fr);
  align-items: stretch;
}

.panel {
  display: flex;
  flex-direction: column;
}

.gesture-frame {
  height: 272px;
  margin-top: 0;
}

.stream-meta {
  margin-top: 6px;
  font-size: 13px;
  color: var(--muted-soft);
}

.recognition-summary {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  margin-top: 10px;
}

.recognition-chip {
  display: inline-flex;
  align-items: center;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(201, 176, 153, 0.1);
  color: #8a7b6d;
  font-size: 12px;
}

.gesture-action-chip {
  display: inline-flex;
  align-items: center;
  padding: 6px 12px;
  border-radius: 999px;
  background: rgba(201, 176, 153, 0.14);
  color: #7b6453;
  font-size: 13px;
  font-weight: 600;
}

.cockpit-header {
  display: flex;
  align-items: flex-start;
  justify-content: flex-start;
  gap: 16px;
  margin-bottom: 10px;
}

.section-label {
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: rgba(111, 95, 82, 0.68);
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

.cockpit-topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 14px;
}

.cockpit-brand-group {
  display: flex;
  align-items: center;
  gap: 10px;
}

.cockpit-brand {
  font-size: 14px;
  font-weight: 700;
  letter-spacing: 0.04em;
  color: #514238;
}

.cockpit-page-tag {
  display: inline-flex;
  align-items: center;
  height: 28px;
  padding: 0 10px;
  border-radius: 999px;
  background: rgba(201, 176, 153, 0.18);
  color: #7f6a5a;
  font-size: 12px;
}

.cockpit-meta {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 12px;
  color: #a59a8c;
}

.cockpit-display {
  min-height: 300px;
  padding: 12px;
  border-radius: 20px;
  background: linear-gradient(180deg, rgba(255, 251, 246, 0.96), rgba(245, 238, 229, 0.96));
  border: 1px solid rgba(184, 162, 141, 0.18);
}

.cockpit-off,
.cockpit-home,
.cockpit-comfort,
.cockpit-call {
  min-height: 276px;
}

.off-screen {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 276px;
  overflow: hidden;
  border-radius: 18px;
  background: linear-gradient(180deg, #f1e9df 0%, #e7ddd1 100%);
  border: 1px solid rgba(184, 162, 141, 0.18);
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
.shortcut-title,
.comfort-main strong,
.metric-card strong,
.call-screen strong {
  font-size: 28px;
  color: #43352d;
}

.off-copy span:last-child,
.widget-card span:last-child,
.shortcut-subtitle,
.comfort-main span:last-child,
.metric-card span:last-child,
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
  border-radius: 999px;
  background: rgba(255, 250, 244, 0.86);
  color: #6a5a4d;
  font-size: 12px;
  border: 1px solid rgba(184, 162, 141, 0.18);
}

.home-side-widgets {
  display: grid;
  grid-template-rows: repeat(2, minmax(0, 1fr));
  gap: 14px;
}

.widget-card,
.metric-card {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 16px;
  border-radius: 18px;
  background: linear-gradient(180deg, rgba(255, 252, 247, 0.98), rgba(244, 236, 227, 0.96));
  border: 1px solid rgba(184, 162, 141, 0.18);
}

.widget-card strong,
.metric-card strong {
  font-size: 20px;
}

.mini-track,
.climate-ring {
  position: relative;
  width: 100%;
  height: 10px;
  margin-top: 4px;
  overflow: hidden;
  border-radius: 999px;
  background: rgba(201, 176, 153, 0.16);
}

.mini-fill,
.climate-ring-fill {
  position: absolute;
  inset: 0 auto 0 0;
  border-radius: 999px;
  background: linear-gradient(90deg, #b89a7e, #d9c5b2);
}

.comfort-layout {
  display: grid;
  grid-template-columns: 1.2fr 1fr;
  gap: 14px;
  min-height: 276px;
}

.comfort-main {
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 12px;
  padding: 24px;
  border-radius: 20px;
  background:
    radial-gradient(circle at top left, rgba(201, 176, 153, 0.22), transparent 34%),
    linear-gradient(180deg, rgba(254, 251, 245, 0.98), rgba(243, 235, 225, 0.98));
  border: 1px solid rgba(184, 162, 141, 0.2);
}

.comfort-side {
  display: grid;
  gap: 14px;
}

.call-screen {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  min-height: 276px;
  border-radius: 20px;
  background:
    radial-gradient(circle at top, rgba(201, 176, 153, 0.26), transparent 34%),
    linear-gradient(180deg, rgba(254, 250, 244, 0.98), rgba(241, 232, 221, 0.98));
  border: 1px solid rgba(184, 162, 141, 0.2);
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
  display: flex;
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
  grid-template-columns: repeat(4, minmax(0, 1fr));
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

@media (max-width: 960px) {
  .cockpit-header {
    flex-direction: column;
  }

  .home-hero,
  .comfort-layout,
  .cockpit-dock {
    grid-template-columns: 1fr;
  }
}

.upload-row {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 14px;
}

.control-button {
  border: none;
  cursor: pointer;
}

.control-button:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}

.secondary-button {
  background: #d8c8b5;
  color: #49372d;
}

.file-name {
  font-size: 13px;
  color: var(--muted);
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

.device-refresh:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}

.preview-canvas-wrap {
  position: relative;
  margin-top: 6px;
  min-height: 292px;
}

.preview-video {
  display: block;
  width: 100%;
  min-height: 272px;
  aspect-ratio: 4 / 3;
  max-height: 408px;
  object-fit: cover;
  border-radius: 8px;
  background: #f0ece5;
}

.overlay-canvas {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

.capture-canvas {
  display: none;
}
</style>
