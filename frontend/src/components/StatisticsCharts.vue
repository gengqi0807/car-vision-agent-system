<template>
  <section class="card chart-card">
    <h3 class="section-title">{{ title }}</h3>
    <div class="chart-placeholder">
      <div v-for="metric in metrics" :key="metric.label" class="metric-card">
        <span class="muted">{{ metric.label }}</span>
        <strong>{{ metric.value }}</strong>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
defineProps<{
  title: string;
  metrics: Array<{ label: string; value: string | number }>;
}>();
</script>

<style scoped lang="scss">
.chart-card {
  padding: 20px;
}

.chart-placeholder {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px;
}

.metric-card {
  display: grid;
  gap: 8px;
  padding: 16px;
  border-radius: 16px;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02));
}

.metric-card strong {
  font-size: 26px;
}
</style>
