<template>
  <div class="auth-page">
    <section class="register-card">
      <p class="kicker">Create Account</p>
      <h1>注册入口</h1>
      <p class="desc">
        当前先保留与原型风格一致的注册占位页，后续可以继续补短信验证码、邮箱验证和扫码登录。
      </p>
      <RouterLink class="back-link" to="/login">返回登录</RouterLink>
    </section>
  </div>
</template>

<script setup lang="ts">
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

.back-link {
  color: var(--accent);
  font-weight: 500;
}
</style>
