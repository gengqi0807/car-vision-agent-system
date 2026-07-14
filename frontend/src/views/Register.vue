<template>
  <div class="auth-page">
    <section class="register-card">
      <p class="kicker">Create Account</p>
      <h1>注册入口</h1>
      <p class="desc">
        当前先保留与原型风格一致的注册占位页，后续可以继续补短信验证码、邮箱验证和扫码登录。
      </p>
      <form class="register-form" @submit.prevent="submit">
        <label for="username">用户名</label>
        <input id="username" v-model="form.username" class="auth-input" type="text" placeholder="输入用户名" />

        <label for="email">邮箱</label>
        <input id="email" v-model="form.email" class="auth-input" type="email" placeholder="输入邮箱（可选）" />

        <label for="phone">手机号</label>
        <input id="phone" v-model="form.phone" class="auth-input" type="text" placeholder="输入手机号（可选）" />

        <label for="password">密码</label>
        <input id="password" v-model="form.password" class="auth-input" type="password" placeholder="至少 6 位密码" />

        <label for="confirmPassword">确认密码</label>
        <input
          id="confirmPassword"
          v-model="confirmPassword"
          class="auth-input"
          type="password"
          placeholder="再次输入密码"
        />

        <button type="submit" class="register-btn">立即注册</button>
      </form>
      <RouterLink class="back-link" to="/login">返回登录</RouterLink>
    </section>
  </div>
</template>

<script setup lang="ts">
import axios from "axios";
import { reactive, ref } from "vue";
import { useRouter } from "vue-router";

import { registerApi } from "../api/auth";

const router = useRouter();
const form = reactive({
  username: "",
  email: "",
  phone: "",
  password: ""
});
const confirmPassword = ref("");

function validateContactFields() {
  const email = form.email.trim();
  const phone = form.phone.trim();
  const emailPattern = /^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+$/;
  const allowedEmailSuffixes = [".com", ".cn", ".com.cn", ".net", ".org", ".edu", ".edu.cn", ".gov.cn"];
  const phonePattern = /^1[3-9]\d{9}$/;

  if (email && (!emailPattern.test(email) || !allowedEmailSuffixes.some((suffix) => email.toLowerCase().endsWith(suffix)))) {
    window.alert("邮箱格式不正确，仅支持 .com、.cn、.com.cn、.net、.org、.edu、.edu.cn、.gov.cn 后缀。");
    return false;
  }
  if (phone && !phonePattern.test(phone)) {
    window.alert("手机号格式不正确，请输入 11 位中国大陆手机号。");
    return false;
  }
  return true;
}

async function submit() {
  if (!form.username || !form.password) {
    window.alert("请至少填写用户名和密码。");
    return;
  }

  if (form.password !== confirmPassword.value) {
    window.alert("两次输入的密码不一致。");
    return;
  }

  if (!validateContactFields()) {
    return;
  }

  try {
    await registerApi({
      username: form.username.trim(),
      password: form.password,
      email: form.email.trim() || undefined,
      phone: form.phone.trim() || undefined
    });
    window.alert("注册成功，请使用新账号登录。");
    router.push({ name: "login" });
  } catch (error) {
    if (axios.isAxiosError(error)) {
      window.alert(String(error.response?.data?.detail ?? "注册失败，请检查输入信息。"));
    } else {
      window.alert("注册失败，请稍后重试。");
    }
  }
}
</script>

<style scoped lang="scss">
.auth-page {
  min-height: 100vh;
  display: grid;
  place-items: center;
  padding: 24px 16px;
}

.register-card {
  width: min(460px, 100%);
  padding: 36px;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 16px;
}

.kicker {
  margin: 0 0 12px;
  font-size: 12px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--accent);
}

h1 {
  margin: 0 0 10px;
}

.desc {
  margin: 0 0 20px;
  line-height: 1.7;
  color: var(--muted);
}

.register-form label {
  display: block;
  margin: 16px 0 4px;
  font-size: 14px;
  font-weight: 500;
  color: var(--text-soft);
}

.register-btn {
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

.register-btn:hover {
  background: var(--accent-strong);
}

.back-link {
  display: inline-block;
  margin-top: 20px;
  color: var(--accent);
  font-weight: 500;
}
</style>
