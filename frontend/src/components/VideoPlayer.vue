<template>
  <div class="player card">
    <div class="screen">
      <div class="scanline"></div>
      <p>视频流区域</p>
      <span class="muted">{{ description }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  description: string;
}>();
</script>

<style scoped lang="scss">
.player {
  padding: 16px;
}

.screen {
  position: relative;
  display: grid;
  place-items: center;
  min-height: 280px;
  border-radius: 18px;
  border: 1px dashed rgba(45, 212, 191, 0.35);
  background: linear-gradient(180deg, rgba(9, 19, 31, 0.96), rgba(12, 25, 39, 0.82));
  overflow: hidden;
  text-align: center;
}

.scanline {
  position: absolute;
  inset: 0;
  background: linear-gradient(180deg, transparent 0%, rgba(45, 212, 191, 0.08) 48%, transparent 100%);
  animation: sweep 3.2s linear infinite;
}

@keyframes sweep {
  from {
    transform: translateY(-100%);
  }
  to {
    transform: translateY(100%);
  }
}
</style>
