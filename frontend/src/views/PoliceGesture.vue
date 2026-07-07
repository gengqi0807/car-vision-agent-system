<template>
  <section class="page-shell">
    <header class="page-header">
      <h1>交警手势识别</h1>
      <p>实时视频流分析，识别 8 种中国标准交警指挥手势</p>
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
          <div v-if="!previewUrl" class="image-placeholder police-frame">
            交警手势图片
            <div class="small">上传一张交警照片进行识别</div>
          </div>
        </div>

        <div class="stream-meta" v-if="loading">识别中 ...</div>

        <template v-if="result">
          <div style="margin-top: 10px">
            <span class="gesture-tag">{{ gestureLabel }}</span>
            <span class="gesture-confidence">
              置信度 {{ (result.confidence * 100).toFixed(1) }}%
            </span>
          </div>
          <div class="stream-meta">
            检测到 {{ result.keypoints?.length || 0 }} 个关键点 · {{ (result.keypoints?.length || 0) / 33 }} 人
          </div>
        </template>
      </article>

      <!-- 右侧：识别结果列表 -->
      <article class="panel">
        <h4>识别结果</h4>
        <div class="result-item" v-for="item in candidateList" :key="item.label" :class="{ inactive: item.label !== gestureLabel }">
          <span>{{ item.label }}</span>
          <span class="val">{{ item.label === gestureLabel ? (result?.confidence ?? 0) * 100 : item.fallback }}%</span>
        </div>
        <div class="support-label">支持 8 种标准手势</div>
        <div class="support-tags">停止 · 直行 · 左转弯 · 左待转 · 右转弯 · 变道 · 减速 · 靠边</div>
      </article>
    </section>
  </section>
</template>

<script setup lang="ts">
import { ref, computed, nextTick } from "vue";
import { fetchPoliceGestureApi } from "@/api/police_gesture";

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
  stop: "停止信号",
  left_turn: "左转弯信号",
  right_turn: "右转弯信号",
  go_straight: "直行信号",
  unknown: "未识别",
};

const gestureLabel = computed(() => {
  if (!result.value) return "—";
  const g = result.value.gesture || "";
  return LABEL_MAP[g] || g;
});

const candidateList = [
  { label: "停止信号", fallback: 28.7 },
  { label: "直行信号", fallback: 45.1 },
  { label: "左转弯信号", fallback: 12.4 },
  { label: "右转弯信号", fallback: 22.3 },
];

// ----- file select → upload -----
function onFileChange(e: Event) {
  const input = e.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;

  error.value = "";
  fileName.value = file.name;
  loading.value = true;

  const reader = new FileReader();
  reader.onload = (ev) => {
    previewUrl.value = (ev.target?.result as string) || "";
  };
  reader.readAsDataURL(file);

  const fd = new FormData();
  fd.append("file", file);
  fetchPoliceGestureApi(fd)
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

function onImgLoad() {
  const c = canvasRef.value;
  if (!c || !c.parentElement) return;
  canvasW.value = c.parentElement.getBoundingClientRect().width;
  canvasH.value = c.parentElement.querySelector("img")?.clientHeight || 220;
}

// ----- draw pose keypoints -----
const POSE_CONNECTIONS: [number, number][] = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], // arms
  [11, 23], [12, 24], [23, 24], // torso
  [23, 25], [25, 27], [24, 26], [26, 28], // legs
  [0, 1], [0, 4], [1, 2], [2, 3], [4, 5], [5, 6], // face
];

function drawKeypoints(keypoints: Array<{ x: number; y: number }>) {
  const c = canvasRef.value;
  if (!c || !keypoints?.length) return;
  const ctx = c.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, c.width, c.height);

  const perPerson = 33;
  const numPeople = Math.floor(keypoints.length / perPerson);

  for (let p = 0; p < numPeople; p++) {
    const pts = keypoints.slice(p * perPerson, (p + 1) * perPerson);
    const w = c.width;
    const ht = c.height;

    ctx.strokeStyle = "#2dd4bf";
    ctx.lineWidth = 2;
    for (const [a, b] of POSE_CONNECTIONS) {
      const p0 = pts[a];
      const p1 = pts[b];
      if (p0 && p1) {
        ctx.beginPath();
        ctx.moveTo(p0.x * w, p0.y * ht);
        ctx.lineTo(p1.x * w, p1.y * ht);
        ctx.stroke();
      }
    }

    ctx.fillStyle = "#c9b099";
    for (const pt of pts) {
      ctx.beginPath();
      ctx.arc(pt.x * w, pt.y * ht, 3, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}
</script>

<style scoped lang="scss">
.police-frame {
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
