<template>
  <div class="overlay card">
    <h3 class="section-title">识别叠加层</h3>
    <div class="overlay-grid">
      <div v-for="item in items" :key="item.label" class="overlay-item">
        <strong>{{ item.label }}</strong>
        <span class="muted">{{ item.value }}</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  items: Array<{ label: string; value: string }>;
}>();
</script>

<style scoped lang="scss">
.overlay {
  padding: 20px;
}

.overlay-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px;
}

.overlay-item {
  display: grid;
  gap: 6px;
  padding: 14px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.04);
}
</style>
