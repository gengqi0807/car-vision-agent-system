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
          <div class="icon clickable" role="button" tabindex="0" @click="openEmailLogin" @keydown.enter="openEmailLogin">
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

    <div v-if="showEmailDialog" class="dialog-mask" @click.self="closeEmailLogin">
      <section class="dialog-card">
        <h3>邮箱验证码登录</h3>
        <p class="dialog-sub">输入已绑定邮箱并完成验证码登录</p>
        <form @submit.prevent="submitEmailLogin">
          <label for="email">邮箱</label>
          <input id="email" v-model="emailForm.email" class="auth-input" type="email" placeholder="输入绑定邮箱" />

          <label for="code">验证码</label>
          <div class="code-row">
            <input id="code" v-model="emailForm.code" class="auth-input" type="text" maxlength="6" placeholder="6 位验证码" />
            <button type="button" class="secondary-action" @click="sendEmailCode">
              {{ sendingCode ? "发送中..." : countdown > 0 ? `${countdown}s` : "发送验证码" }}
            </button>
          </div>

          <div class="dialog-actions">
            <button type="button" class="cancel-btn" @click="closeEmailLogin">取消</button>
            <button type="submit" class="login-btn">立即登录</button>
          </div>
        </form>
      </section>
    </div>
  </div>
</template>

<script setup lang="ts">
import axios from "axios";
import { onBeforeUnmount, reactive, ref } from "vue";
import { useRouter } from "vue-router";

import { emailLoginApi, loginApi, sendEmailCodeApi } from "../api/auth";
import { useUserStore } from "../stores/user";

const router = useRouter();
const userStore = useUserStore();

const form = reactive({
  username: "demo_admin",
  password: "123456"
});
const showEmailDialog = ref(false);
const sendingCode = ref(false);
const countdown = ref(0);
const emailForm = reactive({
  email: "",
  code: ""
});
let countdownTimer: number | null = null;

function submit() {
  if (!form.username || !form.password) {
    return;
  }

  void handleLogin();
}

async function handleLogin() {
  try {
    const { data } = await loginApi(form);
    userStore.setSession(data.user.username, data.access_token);
    userStore.setProfile(data.user);
    router.push({ name: "dashboard" });
  } catch (error) {
    if (axios.isAxiosError(error)) {
      window.alert(String(error.response?.data?.detail ?? "登录失败，请检查后端服务和账号密码。"));
    } else {
      window.alert("登录失败，请稍后重试。");
    }
  }
}

function openEmailLogin() {
  showEmailDialog.value = true;
}

function closeEmailLogin() {
  showEmailDialog.value = false;
  emailForm.code = "";
}

async function sendEmailCode() {
  if (!emailForm.email || sendingCode.value || countdown.value > 0) {
    return;
  }

  sendingCode.value = true;
  try {
    await sendEmailCodeApi({ email: emailForm.email.trim() });
    window.alert("验证码已发送，请查收邮箱。");
    startCountdown();
  } catch (error) {
    if (axios.isAxiosError(error)) {
      window.alert(String(error.response?.data?.detail ?? "验证码发送失败，请稍后重试。"));
    } else {
      window.alert("验证码发送失败，请稍后重试。");
    }
  } finally {
    sendingCode.value = false;
  }
}

async function submitEmailLogin() {
  if (!emailForm.email || !emailForm.code) {
    return;
  }

  try {
    const { data } = await emailLoginApi({
      email: emailForm.email.trim(),
      code: emailForm.code.trim()
    });
    userStore.setSession(data.user.username, data.access_token);
    userStore.setProfile(data.user);
    closeEmailLogin();
    router.push({ name: "dashboard" });
  } catch (error) {
    if (axios.isAxiosError(error)) {
      window.alert(String(error.response?.data?.detail ?? "邮箱登录失败，请检查验证码。"));
    } else {
      window.alert("邮箱登录失败，请稍后重试。");
    }
  }
}

function startCountdown() {
  countdown.value = 60;
  if (countdownTimer !== null) {
    window.clearInterval(countdownTimer);
  }
  countdownTimer = window.setInterval(() => {
    countdown.value -= 1;
    if (countdown.value <= 0 && countdownTimer !== null) {
      window.clearInterval(countdownTimer);
      countdownTimer = null;
    }
  }, 1000);
}

onBeforeUnmount(() => {
  if (countdownTimer !== null) {
    window.clearInterval(countdownTimer);
  }
});
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

.icon.clickable {
  cursor: pointer;
}

.icon.clickable:hover {
  background: #ede7df;
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

.dialog-mask {
  position: fixed;
  inset: 0;
  display: grid;
  place-items: center;
  padding: 20px;
  background: rgba(61, 53, 41, 0.18);
}

.dialog-card {
  width: min(420px, 100%);
  padding: 28px 30px 24px;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 16px;
  box-shadow: 0 18px 50px rgba(61, 53, 41, 0.12);
}

.dialog-card h3 {
  margin: 0;
  font-size: 22px;
  font-weight: 600;
}

.dialog-sub {
  margin: 6px 0 18px;
  font-size: 14px;
  color: var(--muted);
}

.code-row {
  display: grid;
  grid-template-columns: 1fr 124px;
  gap: 10px;
}

.secondary-action,
.cancel-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 0 14px;
  color: var(--text-soft);
  background: var(--surface-muted);
  border: 1px solid var(--line);
  border-radius: 8px;
  cursor: pointer;
}

.dialog-actions {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  margin-top: 22px;
}

.dialog-actions .login-btn {
  width: auto;
  margin-top: 0;
  padding: 10px 22px;
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

  .code-row {
    grid-template-columns: 1fr;
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
