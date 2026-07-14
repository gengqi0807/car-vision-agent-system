<template>
  <section class="card alert-list">
    <div class="header">
      <h3 class="section-title">实时告警</h3>
      <span class="muted">{{ items.length }} 条</span>
    </div>
    <div class="list">
      <article v-for="item in items" :key="item.id" class="item">
        <div class="badge" :data-level="item.level">{{ item.level }}</div>
        <div>
          <strong>{{ item.title }}</strong>
          <p>{{ item.summary }}</p>
        </div>
      </article>
    </div>
  </section>
</template>

<script setup lang="ts">
defineProps<{
  items: Array<{ id: number; level: string; title: string; summary: string }>;
}>();
</script>

<style scoped lang="scss">
.alert-list {
  padding: 20px;
}

.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.list {
  display: grid;
  gap: 12px;
}

.item {
  display: grid;
  grid-template-columns: 88px 1fr;
  gap: 14px;
  padding: 14px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.04);
}

.item p {
  margin: 6px 0 0;
  color: var(--muted);
}

.badge {
  display: grid;
  place-items: center;
  height: 30px;
  border-radius: 999px;
  font-size: 12px;
  text-transform: uppercase;
}

.badge[data-level="critical"] {
  background: rgba(239, 68, 68, 0.18);
  color: #ffb3b3;
}

.badge[data-level="warning"] {
  background: rgba(245, 158, 11, 0.18);
  color: #ffd48a;
}

.badge[data-level="info"] {
  background: rgba(45, 212, 191, 0.18);
  color: #8cf7e7;
}
</style>
