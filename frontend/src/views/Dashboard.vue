<template>
  <section class="page-shell">
    <StatisticsCharts :title="'系统概览'" :metrics="metrics" />
    <div class="dashboard-grid">
      <AlertList :items="alerts" />
      <section class="card quick-panel">
        <h3 class="section-title">开发节奏</h3>
        <div class="milestone">
          <strong>Day 1</strong>
          <span class="muted">前后端基础框架、路由、接口占位、页面骨架</span>
        </div>
        <div class="milestone">
          <strong>Day 2</strong>
          <span class="muted">接入数据库、日志链路、WebSocket 告警推送</span>
        </div>
        <div class="milestone">
          <strong>Day 3+</strong>
          <span class="muted">逐步落地车牌识别、交警手势、车主手势控制能力</span>
        </div>
      </section>
    </div>
  </section>
</template>

<script setup lang="ts">
import AlertList from "../components/AlertList.vue";
import StatisticsCharts from "../components/StatisticsCharts.vue";

const metrics = [
  { label: "车牌识别接口", value: "Ready" },
  { label: "交警手势接口", value: "Ready" },
  { label: "车主手势接口", value: "Ready" },
  { label: "告警智能体", value: "Scaffold" }
];

const alerts = [
  {
    id: 1,
    level: "warning",
    title: "识别稳定性待接入真实模型",
    summary: "当前前后端以 mock 数据联调，下一阶段接入真实推理能力。"
  },
  {
    id: 2,
    level: "info",
    title: "页面框架已完成",
    summary: "后续可以按模块分工推进页面细节、状态管理和接口联调。"
  }
];
</script>

<style scoped lang="scss">
.dashboard-grid {
  display: grid;
  grid-template-columns: 1.2fr 0.8fr;
  gap: 20px;
}

.quick-panel {
  padding: 20px;
}

.milestone {
  display: grid;
  gap: 6px;
  padding: 14px 0;
  border-bottom: 1px solid var(--line);
}

.milestone:last-child {
  border-bottom: 0;
}

@media (max-width: 960px) {
  .dashboard-grid {
    grid-template-columns: 1fr;
  }
}
</style>
