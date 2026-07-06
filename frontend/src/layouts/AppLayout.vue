<template>
  <div class="layout">
    <aside class="sidebar card">
      <div>
        <p class="brand-kicker">CMS</p>
        <h1>智能车载视觉感知与交互系统</h1>
      </div>
      <nav class="nav">
        <RouterLink v-for="item in navItems" :key="item.to" class="nav-link" :to="item.to">
          <span class="status-dot"></span>
          <span>{{ item.label }}</span>
        </RouterLink>
      </nav>
      <div class="sidebar-footer">
        <p class="muted">Day 1 基础框架</p>
        <p>当前阶段以页面骨架与接口联调为主</p>
      </div>
    </aside>

    <main class="main">
      <header class="topbar card">
        <div>
          <p class="muted">Vehicle Vision Workspace</p>
          <h2>{{ pageTitle }}</h2>
        </div>
        <div class="topbar-actions">
          <div class="status-chip">
            <span class="status-dot"></span>
            <span>系统在线</span>
          </div>
          <button class="ghost-btn" type="button" @click="logout">退出</button>
        </div>
      </header>
      <RouterView />
    </main>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";
import { useRoute, useRouter } from "vue-router";

const route = useRoute();
const router = useRouter();

const navItems = [
  { label: "系统总览", to: "/" },
  { label: "车牌识别", to: "/plate-recognition" },
  { label: "交警手势", to: "/police-gesture" },
  { label: "车主手势控车", to: "/owner-gesture" },
  { label: "告警监控", to: "/alert-monitor" }
];

const titleMap: Record<string, string> = {
  dashboard: "系统总览",
  "plate-recognition": "道路车辆车牌识别",
  "police-gesture": "交警手势识别",
  "owner-gesture": "车主手势控车",
  "alert-monitor": "日志监控与告警智能体"
};

const pageTitle = computed(() => titleMap[String(route.name)] ?? "智能车载视觉平台");

function logout() {
  localStorage.removeItem("cvms_token");
  router.push({ name: "login" });
}
</script>

<style scoped lang="scss">
.layout {
  display: grid;
  grid-template-columns: 300px 1fr;
  min-height: 100vh;
  gap: 20px;
  padding: 20px;
}

.sidebar,
.topbar {
  padding: 24px;
}

.sidebar {
  display: grid;
  gap: 28px;
  align-content: start;
}

.brand-kicker {
  margin: 0 0 8px;
  color: var(--accent);
  letter-spacing: 0.28em;
  font-size: 12px;
  text-transform: uppercase;
}

.sidebar h1,
.topbar h2 {
  margin: 0;
}

.nav {
  display: grid;
  gap: 10px;
}

.nav-link {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 14px 16px;
  border: 1px solid transparent;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.02);
  transition: 0.2s ease;
}

.nav-link.router-link-active {
  border-color: rgba(45, 212, 191, 0.3);
  background: var(--accent-soft);
}

.sidebar-footer {
  padding: 18px;
  border: 1px solid var(--line);
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.03);
}

.sidebar-footer p {
  margin: 0;
}

.sidebar-footer p + p {
  margin-top: 8px;
}

.main {
  display: grid;
  gap: 20px;
  align-content: start;
}

.topbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.topbar-actions {
  display: flex;
  align-items: center;
  gap: 14px;
}

.status-chip,
.ghost-btn {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  height: 42px;
  padding: 0 16px;
  border-radius: 999px;
}

.status-chip {
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid var(--line);
}

.ghost-btn {
  border: 1px solid rgba(255, 255, 255, 0.12);
  background: transparent;
  color: var(--text);
  cursor: pointer;
}

@media (max-width: 1080px) {
  .layout {
    grid-template-columns: 1fr;
  }
}
</style>
