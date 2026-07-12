<template>
  <section class="card control-panel">
    <h3 class="section-title">模拟控车面板</h3>
    <div class="panel-grid">
      <label>
        <span>音量</span>
        <input type="range" min="0" max="100" :value="volume" />
      </label>
      <label>
        <span>空调温度</span>
        <input type="range" min="18" max="30" :value="temperature" />
      </label>
      <button type="button" class="panel-btn">{{ phoneActive ? "挂断电话" : "接听电话" }}</button>
      <button type="button" class="panel-btn ghost">{{ mode }}</button>
    </div>
  </section>
</template>

<script setup lang="ts">
defineProps<{
  volume: number;
  temperature: number;
  phoneActive: boolean;
  mode: string;
}>();
</script>

<style scoped lang="scss">
.control-panel {
  padding: 20px;
}

.panel-grid {
  display: grid;
  gap: 14px;
}

label {
  display: grid;
  gap: 8px;
}

input[type="range"] {
  width: 100%;
}

.panel-btn {
  height: 44px;
  border: 0;
  border-radius: 14px;
  background: var(--accent);
  color: #05211f;
  font-weight: 700;
}

.panel-btn.ghost {
  background: rgba(255, 255, 255, 0.06);
  color: var(--text);
}
</style>
