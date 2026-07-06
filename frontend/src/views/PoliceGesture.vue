<template>
  <section class="page-shell">
    <div class="content-grid">
      <VideoPlayer description="支持实时视频流，后续接入人体关键点与时序动作分类。" />
      <OverlayCanvas :items="overlayItems" />
    </div>
    <StatisticsCharts :title="'标准手势识别覆盖计划'" :metrics="metrics" />
  </section>
</template>

<script setup lang="ts">
import OverlayCanvas from "../components/OverlayCanvas.vue";
import StatisticsCharts from "../components/StatisticsCharts.vue";
import VideoPlayer from "../components/VideoPlayer.vue";

const overlayItems = [
  { label: "当前手势", value: "停止信号" },
  { label: "置信度", value: "88%" },
  { label: "关键点数量", value: "33" },
  { label: "复杂背景鲁棒性", value: "待优化" }
];

const metrics = [
  { label: "已规划手势", value: 8 },
  { label: "当前接入", value: 1 },
  { label: "目标输入", value: "实时视频流" },
  { label: "识别链路", value: "Pose + Classifier" }
];
</script>

<style scoped lang="scss">
.content-grid {
  display: grid;
  grid-template-columns: 1.2fr 0.8fr;
  gap: 20px;
}

@media (max-width: 960px) {
  .content-grid {
    grid-template-columns: 1fr;
  }
}
</style>
