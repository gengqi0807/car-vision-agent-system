<template>
  <div class="auth-page">
    <div class="login-container">
      <section class="login-brand">
        <h1>智能车载视觉</h1>
        <h1>感知系统</h1>
        <span class="divider"></span>
        <p>车牌识别 · 交警手势 · 手势控车</p>
        <p class="subtitle">智能告警一体化平台</p>
      </section>

      <section class="login-card">
        <h2>登录</h2>
        <p class="sub">输入凭证以继续</p>
        <form @submit.prevent="submit">
          <label for="username">用户名</label>
          <input id="username" v-model="form.username" class="auth-input" type="text" placeholder="输入用户名" />

          <label for="password">密码</label>
          <input id="password" v-model="form.password" class="auth-input" type="password" placeholder="输入密码" />

          <button type="submit" class="login-btn">登录</button>
        </form>

        <div class="login-divider">其他登录方式</div>
        <div class="social-icons">
          <div class="icon">
            微
            <span>微信</span>
          </div>
          <div class="icon">
            邮
            <span>邮箱</span>
          </div>
          <div class="icon">
            手
            <span>手机</span>
          </div>
          <div class="icon">
            支
            <span>支付</span>
          </div>
        </div>

        <div class="register-link">
          没有账号？
          <RouterLink to="/register">立即注册</RouterLink>
        </div>
      </section>
    </div>
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
  if (!form.username || !form.password) {
    return;
  }

  userStore.setSession(form.username, "demo-token");
  router.push({ name: "dashboard" });
}
</script>

<style scoped lang="scss">
.auth-page {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px 32px;
}

.login-container {
  width: 100%;
  max-width: 1100px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 60px;
}

.login-brand h1 {
  margin: 0;
  font-size: 44px;
  font-weight: 700;
  line-height: 1.2;
}

.divider {
  display: block;
  width: 90px;
  height: 4px;
  margin: 18px 0 22px;
  background: var(--accent);
  border-radius: 2px;
}

.login-brand p {
  margin: 0;
  font-size: 17px;
  line-height: 1.6;
  color: #9a8b7a;
}

.login-brand .subtitle {
  margin-top: 2px;
}

.login-card {
  width: 400px;
  padding: 36px 38px 32px;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 16px;
  flex-shrink: 0;
}

.login-card h2 {
  margin: 0;
  font-size: 26px;
  font-weight: 600;
  text-align: center;
}

.sub {
  margin: 2px 0 28px;
  font-size: 14px;
  color: var(--muted);
  text-align: center;
}

form label {
  display: block;
  margin: 16px 0 4px;
  font-size: 14px;
  font-weight: 500;
  color: var(--text-soft);
}

.login-btn {
  width: 100%;
  margin-top: 24px;
  padding: 14px;
  color: #ffffff;
  font-size: 17px;
  font-weight: 600;
  background: var(--accent);
  border-radius: 10px;
  cursor: pointer;
  transition: 0.2s ease;
}

.login-btn:hover {
  background: var(--accent-strong);
}

.login-divider {
  margin: 20px 0 14px;
  font-size: 12px;
  color: var(--muted-soft);
  text-align: center;
}

.social-icons {
  display: flex;
  justify-content: center;
  gap: 20px;
}

.icon {
  width: 52px;
  height: 52px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 1px;
  font-size: 16px;
  font-weight: 500;
  color: var(--text-soft);
  background: var(--surface-muted);
  border: 1px solid var(--line);
  border-radius: 50%;
}

.icon span {
  font-size: 10px;
  font-weight: 400;
  color: var(--muted-soft);
}

.register-link {
  margin-top: 20px;
  font-size: 14px;
  color: var(--muted);
  text-align: center;
}

.register-link a {
  color: var(--accent);
  font-weight: 500;
}

@media (max-width: 900px) {
  .login-container {
    flex-direction: column;
    align-items: stretch;
    gap: 28px;
  }

  .login-card {
    width: 100%;
  }
}

@media (max-width: 720px) {
  .auth-page {
    padding: 20px 16px;
  }

  .login-brand h1 {
    font-size: 34px;
  }

  .login-card {
    padding: 28px 22px 24px;
  }
}
</style>
