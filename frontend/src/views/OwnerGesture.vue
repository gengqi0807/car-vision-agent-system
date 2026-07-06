<template>
  <section class="page-shell">
    <div class="content-grid">
      <VideoPlayer description="支持车内摄像头输入，后续接入手部关键点追踪与误触抑制机制。" />
      <OverlayCanvas :items="overlayItems" />
    </div>
    <ControlPanel :volume="32" :temperature="24" :phone-active="false" mode="媒体模式" />
  </section>
</template>

<script setup lang="ts">
import ControlPanel from "../components/ControlPanel.vue";
import OverlayCanvas from "../components/OverlayCanvas.vue";
import VideoPlayer from "../components/VideoPlayer.vue";

const overlayItems = [
  { label: "当前手势", value: "手掌张开" },
  { label: "追踪状态", value: "稳定" },
  { label: "关键点数量", value: "21" },
  { label: "误触抑制", value: "持续时间阈值" }
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
