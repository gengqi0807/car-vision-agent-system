<template>
  <div class="app-shell">
    <header class="top-nav">
      <span class="brand">Car Vision System</span>
      <RouterLink
        v-for="item in navItems"
        :key="item.to"
        :to="item.to"
        class="nav-link"
        :class="{ active: route.name === item.name }"
      >
        {{ item.label }}
      </RouterLink>
      <span class="spacer"></span>
      <button type="button" class="avatar" title="退出登录" @click="logout">
        {{ avatarText }}
      </button>
    </header>

    <main class="main-content">
      <RouterView />
    </main>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";
import { useRoute, useRouter } from "vue-router";

import { useUserStore } from "../stores/user";

const route = useRoute();
const router = useRouter();
const userStore = useUserStore();

const navItems = [
  { label: "仪表盘", to: "/", name: "dashboard" },
  { label: "车牌识别", to: "/plate-recognition", name: "plate-recognition" },
  { label: "交警手势", to: "/police-gesture", name: "police-gesture" },
  { label: "手势控车", to: "/owner-gesture", name: "owner-gesture" },
  { label: "告警监控", to: "/alert-monitor", name: "alert-monitor" }
];

const avatarText = computed(() => {
  const base = (userStore.username || "A").trim();
  return base.slice(0, 1).toUpperCase();
});

function logout() {
  userStore.logout();
  router.push({ name: "login" });
}
</script>

<style scoped lang="scss">
.app-shell {
  max-width: 1360px;
  min-height: 100vh;
  margin: 0 auto;
  padding: 24px 32px;
}

.top-nav {
  display: flex;
  align-items: center;
  gap: 28px;
  margin-bottom: 28px;
  padding: 14px 28px;
  background: var(--surface);
  border: 1px solid var(--line-strong);
  border-radius: 14px;
  flex-wrap: wrap;
}

.brand {
  margin-right: 16px;
  font-size: 19px;
  font-weight: 700;
  cursor: default;
}

.nav-link {
  padding: 4px 0;
  font-size: 14px;
  color: var(--muted);
  border-bottom: 2px solid transparent;
  transition: 0.2s ease;
}

.nav-link:hover {
  color: var(--text-soft);
}

.nav-link.active {
  color: var(--accent);
  font-weight: 600;
  border-bottom-color: var(--accent);
}

.spacer {
  flex: 1;
}

.avatar {
  width: 36px;
  height: 36px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: #8a7a6a;
  font-size: 14px;
  font-weight: 600;
  background: #f0ece5;
  border-radius: 50%;
  cursor: pointer;
}

.avatar:hover {
  background: #e5ddd5;
}

@media (max-width: 720px) {
  .app-shell {
    padding: 20px 16px;
  }

  .top-nav {
    gap: 16px;
    padding: 14px 16px;
  }

  .brand {
    width: 100%;
    margin-right: 0;
  }
}
</style>
