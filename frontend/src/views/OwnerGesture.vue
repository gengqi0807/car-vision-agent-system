<template>
  <section class="page-shell">
    <header class="page-header">
      <h1>手势控车</h1>
      <p>通过手势对车辆功能进行非接触式控制</p>
    </header>

    <section class="two-col">
      <!-- 左侧：上传 + 识别结果 -->
      <article class="panel">
        <div class="upload-row">
          <label class="btn-video upload-label">
            选择图片
            <input
              type="file"
              accept="image/*"
              class="file-input"
              @change="onFileChange"
            />
          </label>
          <span class="file-name" v-if="fileName">{{ fileName }}</span>
        </div>

        <div class="video-status" v-if="error">
          <span class="off">{{ error }}</span>
        </div>

        <!-- 图片预览 + 骨架画布 -->
        <div class="preview-canvas-wrap">
          <img
            v-if="previewUrl"
            :src="previewUrl"
            class="preview-img"
            @load="onImgLoad"
          />
          <canvas
            ref="canvasRef"
            class="overlay-canvas"
            :width="canvasW"
            :height="canvasH"
          />
          <div v-if="!previewUrl" class="image-placeholder gesture-frame">
            手势识别图片
            <div class="small">上传一张手部照片进行识别</div>
          </div>
        </div>

        <!-- 加载 -->
        <div class="stream-meta" v-if="loading">识别中 ...</div>

        <!-- 结果 -->
        <template v-if="result">
          <div style="margin-top: 10px">
            <span class="gesture-tag">{{ gestureLabel }}</span>
            <span class="gesture-confidence">
              置信度 {{ (result.confidence * 100).toFixed(1) }}%
            </span>
          </div>
          <div class="stream-meta">
            检测到 {{ result.keypoints?.length || 0 }} 个关键点 · {{ (result.keypoints?.length || 0) / 21 }} 只手
          </div>
        </template>
      </article>

      <!-- 右侧：控制面板 -->
      <article class="panel">
        <h4>模拟控制面板</h4>
        <div class="slider-group">
          <label>音量</label>
          <div class="track"><div class="fill" style="width: 65%"></div></div>
        </div>
        <div class="slider-group">
          <label>空调温度</label>
          <div class="track"><div class="fill" style="width: 35%"></div></div>
        </div>
        <div class="btn-group">
          <button class="primary" type="button">接听电话</button>
          <button type="button">挂断电话</button>
          <button type="button">返回主页</button>
        </div>
        <div class="control-status">
          当前手势：
          <span class="highlight">{{ gestureLabel || '—' }}</span>
        </div>
        <div class="support-info">
          支持 6 种手势：手掌张开 · 握拳 · 单指画圈 · 左右滑动 · 拇指向上 · 拇指向下
        </div>
      </article>
    </section>
  </section>
</template>

<script setup lang="ts">
import { ref, computed, nextTick } from "vue";
import { fetchOwnerGestureApi } from "@/api/owner_gesture";

// ----- state -----
const loading = ref(false);
const error = ref("");
const fileName = ref("");
const previewUrl = ref("");
const result = ref<any>(null);
const canvasRef = ref<HTMLCanvasElement | null>(null);
const canvasW = ref(300);
const canvasH = ref(220);

// ----- gesture label mapping -----
const LABEL_MAP: Record<string, string> = {
  open_palm: "手掌张开",
  fist: "握拳",
  thumbs_up: "拇指向上",
  thumbs_down: "拇指向下",
  unknown: "未识别",
};

const gestureLabel = computed(() => {
  if (!result.value) return "—";
  const g = result.value.gesture || "";
  return LABEL_MAP[g] || g;
});


// ----- file select → upload -----
function onFileChange(e: Event) {
  const input = e.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;

  error.value = "";
  fileName.value = file.name;
  loading.value = true;

  // preview
  const reader = new FileReader();
  reader.onload = (ev) => {
    previewUrl.value = (ev.target?.result as string) || "";
  };
  reader.readAsDataURL(file);

  // upload
  const fd = new FormData();
  fd.append("file", file);
  fetchOwnerGestureApi(fd)
    .then((res) => {
      result.value = res.data;
      nextTick(() => drawKeypoints(res.data.keypoints));
    })
    .catch((err) => {
      error.value = err?.response?.data?.detail || err.message || "识别失败";
    })
    .finally(() => {
      loading.value = false;
    });
}

// ----- canvas sizing -----
function onImgLoad() {
  const c = canvasRef.value;
  if (!c || !c.parentElement) return;
  const rect = c.parentElement.getBoundingClientRect();
  canvasW.value = rect.width;
  canvasH.value = c.parentElement.querySelector("img")?.clientHeight || 220;
}

// ----- draw keypoints -----
const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // thumb
  [0, 5], [5, 6], [6, 7], [7, 8],       // index
  [0, 9], [9, 10], [10, 11], [11, 12],   // middle
  [0, 13], [13, 14], [14, 15], [15, 16], // ring
  [0, 17], [17, 18], [18, 19], [19, 20], // pinky
  [5, 9], [9, 13], [13, 17],             // across knuckles
];

function drawKeypoints(keypoints: Array<{ x: number; y: number }>) {
  const c = canvasRef.value;
  if (!c || !keypoints?.length) return;
  const ctx = c.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, c.width, c.height);

  const perHand = 21;
  const numHands = Math.floor(keypoints.length / perHand);

  for (let h = 0; h < numHands; h++) {
    const hand = keypoints.slice(h * perHand, (h + 1) * perHand);
    const w = c.width;
    const ht = c.height;

    // connections
    ctx.strokeStyle = "#2dd4bf";
    ctx.lineWidth = 2;
    for (const [a, b] of HAND_CONNECTIONS) {
      const p0 = hand[a];
      const p1 = hand[b];
      if (p0 && p1) {
        ctx.beginPath();
        ctx.moveTo(p0.x * w, p0.y * ht);
        ctx.lineTo(p1.x * w, p1.y * ht);
        ctx.stroke();
      }
    }

    // points
    ctx.fillStyle = "#c9b099";
    for (const pt of hand) {
      ctx.beginPath();
      ctx.arc(pt.x * w, pt.y * ht, 3, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}
</script>

<style scoped lang="scss">
.gesture-frame {
  height: 220px;
  margin-top: 0;
}

.stream-meta {
  margin-top: 6px;
  font-size: 13px;
  color: var(--muted-soft);
}

.upload-row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 14px;
}

.upload-label {
  display: inline-block;
  cursor: pointer;
  position: relative;
}

.file-input {
  position: absolute;
  inset: 0;
  opacity: 0;
  cursor: pointer;
}

.file-name {
  font-size: 13px;
  color: var(--muted);
}

.preview-canvas-wrap {
  position: relative;
  margin-top: 6px;
}

.preview-img {
  display: block;
  width: 100%;
  max-height: 360px;
  object-fit: contain;
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
</style>
