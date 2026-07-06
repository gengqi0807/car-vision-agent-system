<template>
  <section class="page-shell">
    <div class="content-grid">
      <VideoPlayer description="支持车载图片上传与视频流接入，后续接 YOLO + OCR 流水线。" />
      <OverlayCanvas :items="overlayItems" />
    </div>
    <section class="card history-card">
      <h3 class="section-title">识别历史</h3>
      <div class="history-row" v-for="record in records" :key="record.plate">
        <strong>{{ record.plate }}</strong>
        <span>{{ record.color }}</span>
        <span class="muted">{{ record.time }}</span>
      </div>
    </section>
  </section>
</template>

<script setup lang="ts">
import OverlayCanvas from "../components/OverlayCanvas.vue";
import VideoPlayer from "../components/VideoPlayer.vue";

const overlayItems = [
  { label: "检测目标", value: "3 辆车" },
  { label: "最高置信度", value: "0.94" },
  { label: "当前车牌", value: "浙A12345" },
  { label: "车牌颜色", value: "蓝牌" }
];

const records = [
  { plate: "浙A12345", color: "蓝牌", time: "2026-07-06 16:30:00" },
  { plate: "沪B8C888", color: "绿牌", time: "2026-07-06 16:28:31" }
];
</script>

<style scoped lang="scss">
.content-grid {
  display: grid;
  grid-template-columns: 1.2fr 0.8fr;
  gap: 20px;
}

.history-card {
  padding: 20px;
}

.history-row {
  display: grid;
  grid-template-columns: 120px 80px 1fr;
  gap: 12px;
  padding: 14px 0;
  border-bottom: 1px solid var(--line);
}

.history-row:last-child {
  border-bottom: 0;
}

@media (max-width: 960px) {
  .content-grid {
    grid-template-columns: 1fr;
  }
}
</style>
