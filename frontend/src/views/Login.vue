<template>
  <div class="auth-page">
    <section class="auth-card card">
      <p class="brand-kicker">CAR VISION CMS</p>
      <h1>登录平台</h1>
      <p class="muted">今天先完成开发骨架，后续再按原型补齐交互细节。</p>
      <form class="auth-form" @submit.prevent="submit">
        <input v-model="form.username" placeholder="账号" />
        <input v-model="form.password" type="password" placeholder="密码" />
        <button type="submit">进入系统</button>
      </form>
      <RouterLink class="muted" to="/register">没有账号？前往注册</RouterLink>
    </section>
  </div>
</template>

<script setup lang="ts">
import { reactive } from "vue";
import { useRouter } from "vue-router";

import { useUserStore } from "../stores/user";

const router = useRouter();
const userStore = useUserStore();
const form = reactive({
  username: "demo_admin",
  password: "123456"
});

function submit() {
  userStore.setToken("demo-token");
  router.push({ name: "dashboard" });
}
</script>

<style scoped lang="scss">
.auth-page {
  display: grid;
  place-items: center;
  min-height: 100vh;
  padding: 20px;
}

.auth-card {
  width: min(460px, 100%);
  padding: 36px;
}

.auth-card h1 {
  margin: 0 0 8px;
}

.auth-form {
  display: grid;
  gap: 14px;
  margin: 24px 0 14px;
}

input,
button {
  height: 48px;
  border-radius: 14px;
}

input {
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.04);
  color: var(--text);
  padding: 0 14px;
}

button {
  border: 0;
  background: var(--accent);
  color: #05211f;
  font-weight: 700;
  cursor: pointer;
}
</style>
