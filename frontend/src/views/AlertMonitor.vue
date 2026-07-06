<template>
  <section class="page-shell">
    <StatisticsCharts :title="'告警统计'" :metrics="metrics" />
    <div class="content-grid">
      <AlertList :items="alerts" />
      <section class="card timeline-card">
        <h3 class="section-title">告警时间线</h3>
        <div v-for="item in alerts" :key="item.id" class="timeline-item">
          <strong>{{ item.title }}</strong>
          <p>{{ item.summary }}</p>
          <span class="muted">{{ item.time }}</span>
        </div>
      </section>
    </div>
  </section>
</template>

<script setup lang="ts">
import AlertList from "../components/AlertList.vue";
import StatisticsCharts from "../components/StatisticsCharts.vue";

const metrics = [
  { label: "总告警数", value: 4 },
  { label: "严重级别", value: 1 },
  { label: "警告级别", value: 2 },
  { label: "通知渠道", value: "WebSocket" }
];

const alerts = [
  {
    id: 1,
    level: "critical",
    title: "LLM API 调用超时",
    summary: "连续请求超时，建议检查网络与令牌额度。",
    time: "2026-07-06 16:25:10"
  },
  {
    id: 2,
    level: "warning",
    title: "车牌识别连续失败",
    summary: "建议检查视频流清晰度和推理服务状态。",
    time: "2026-07-06 16:27:41"
  }
];
</script>

<style scoped lang="scss">
.content-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}

.timeline-card {
  padding: 20px;
}

.timeline-item {
  padding: 14px 0;
  border-bottom: 1px solid var(--line);
}

.timeline-item:last-child {
  border-bottom: 0;
}

.timeline-item p {
  margin: 6px 0;
  color: var(--muted);
}

@media (max-width: 960px) {
  .content-grid {
    grid-template-columns: 1fr;
  }
}
</style>
