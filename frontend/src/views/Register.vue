<template>
  <div class="auth-page">
    <section class="auth-card card">
      <p class="brand-kicker">SIGN UP</p>
      <h1>注册占位页</h1>
      <p class="muted">后续可扩展短信验证码、邮箱验证码、微信扫码等登录方式。</p>
      <RouterLink to="/login">返回登录</RouterLink>
    </section>
  </div>
</template>

<script setup lang="ts">
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
</style>
