<template>
  <section class="page-shell">
    <header class="page-header">
      <h1>车牌识别</h1>
      <p>上传道路场景图片或输入视频流，识别车牌号码、颜色与车型</p>
    </header>

    <section class="two-col">
      <div>
        <label class="upload-zone">
          <input class="hidden-input" accept="image/*" type="file" @change="handleFileChange" />
          <div class="main">点击上传图片 或 输入视频流地址</div>
          <div class="sub">支持 JPG · PNG · MP4 · HLS</div>
        </label>

        <article class="panel">
          <div class="preview-shell">
            <img v-if="previewUrl" :src="previewUrl" alt="道路场景图像预览" class="preview-image" />
            <div v-else class="image-placeholder">
              道路场景图像
              <div class="small">检测框与识别结果将在这里叠加显示</div>
            </div>
          </div>

          <div
            v-for="(item, index) in detectionRows"
            :key="`${item.plate}-${index}`"
            :style="{ marginTop: index === 0 ? '0' : '4px' }"
          >
            <div class="detection-box" :class="{ blue: index === 1 }">
              <span class="label">{{ item.plate }}</span>
              <span class="conf">{{ item.confidence }}</span>
            </div>
            <span class="detection-meta">{{ item.meta }}</span>
          </div>

          <div class="result-meta">{{ resultMeta }}</div>
          <div v-if="requestError" class="request-error">{{ requestError }}</div>
        </article>
      </div>

      <article class="panel history-panel">
        <h4>历史记录</h4>
        <input v-model="keyword" class="search-box" placeholder="搜索车牌号或车型..." />
        <div class="history-row header">
          <span>车牌号码</span>
          <span>颜色</span>
          <span>车型</span>
          <span class="time">时间</span>
        </div>
        <div v-for="record in filteredRecords" :key="record.id" class="history-row record">
          <span>{{ record.plate }}</span>
          <span>{{ record.color }}</span>
          <span>{{ record.vehicleType }}</span>
          <span class="time">{{ record.time }}</span>
        </div>
      </article>
    </section>
  </section>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";

import {
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

const keyword = ref("");
const previewUrl = ref("");
const requestError = ref("");
const detections = ref<PlateDetection[]>([]);
const historyRecords = ref<PlateRecordSummary[]>([]);

const fallbackRecords: HistoryRecordView[] = [
  { id: 1, plate: "京A·88888", color: "蓝色", vehicleType: "轿车", time: "14:23" },
  { id: 2, plate: "沪B·66666", color: "绿色", vehicleType: "SUV", time: "14:15" },
  { id: 3, plate: "粤C·12345", color: "蓝色", vehicleType: "货车", time: "13:50" },
  { id: 4, plate: "京A·99999", color: "黄色", vehicleType: "客车", time: "13:20" }
];

const fallbackDetections = [
  { plate: "京A · 88888", confidence: "99.2%", meta: "蓝色 · 轿车" },
  { plate: "沪B · 66666", confidence: "97.8%", meta: "绿色 · SUV" }
];

const displayHistory = computed<HistoryRecordView[]>(() => {
  if (historyRecords.value.length === 0) {
    return fallbackRecords;
  }

  return historyRecords.value.map((record) => ({
    id: record.id,
    plate: record.plate_number,
    color: record.plate_color,
    vehicleType: "轿车",
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

  return displayHistory.value.filter((record) => `${record.plate}${record.vehicleType}`.includes(term));
});

const detectionRows = computed(() => {
  if (detections.value.length === 0) {
    return fallbackDetections;
  }

  return detections.value.map((item) => ({
    plate: item.plate_number,
    confidence: `${(item.confidence * 100).toFixed(1)}%`,
    meta: `${item.plate_color} · 轿车`
  }));
});

const resultMeta = computed(() => `检测到 ${detectionRows.value.length} 辆车牌 · 处理耗时 1.23 秒`);

function resetPreviewUrl() {
  if (previewUrl.value) {
    URL.revokeObjectURL(previewUrl.value);
    previewUrl.value = "";
  }
}

async function loadHistory() {
  try {
    const { data } = await fetchPlateHistoryApi();
    historyRecords.value = data;
  } catch {
    historyRecords.value = [];
  }
}

async function handleFileChange(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];

  resetPreviewUrl();
  requestError.value = "";

  if (!file) {
    return;
  }

  previewUrl.value = URL.createObjectURL(file);

  try {
    const { data } = await recognizePlateImageApi(file);
    detections.value = data.detections;
    await loadHistory();
  } catch {
    detections.value = [];
    requestError.value = "后端未连接，当前展示原型数据。";
  }
}

onMounted(() => {
  void loadHistory();
});

onBeforeUnmount(() => {
  resetPreviewUrl();
});
</script>

<style scoped lang="scss">
.hidden-input {
  display: none;
}

.preview-shell {
  margin-bottom: 12px;
}

.preview-image {
  display: block;
  width: 100%;
  height: 200px;
  object-fit: cover;
  border-radius: 8px;
}

.detection-meta {
  margin-left: 8px;
  font-size: 13px;
  color: var(--text-soft);
}

.history-panel {
  height: 100%;
}

.search-box {
  margin-bottom: 14px;
}

.request-error {
  margin-top: 12px;
  font-size: 13px;
  color: #b27f75;
  text-align: center;
}
</style>
