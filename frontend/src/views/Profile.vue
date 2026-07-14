<template>
  <section class="page-shell">
    <header class="page-header">
      <h1>个人资料</h1>
      <p>维护用户名、邮箱和手机号，便于后续使用邮箱验证码登录</p>
    </header>

    <section class="profile-grid">
      <article class="panel">
        <h4>资料编辑</h4>
        <form class="profile-form" @submit.prevent="submit">
          <label for="profile-username">用户名</label>
          <input id="profile-username" v-model="form.username" class="auth-input" type="text" placeholder="输入用户名" />

          <label for="profile-email">邮箱</label>
          <input id="profile-email" v-model="form.email" class="auth-input" type="email" placeholder="输入常用邮箱" />

          <label for="profile-phone">手机号</label>
          <input id="profile-phone" v-model="form.phone" class="auth-input" type="text" placeholder="输入手机号" />

          <div class="profile-actions">
            <button type="submit" class="primary-btn">保存资料</button>
            <button type="button" class="ghost-btn" @click="logout">退出登录</button>
          </div>
        </form>
      </article>

      <article class="panel">
        <h4>账号信息</h4>
        <div class="info-row">
          <span>UID</span>
          <strong>{{ profile?.uid || "加载中" }}</strong>
        </div>
        <div class="info-row">
          <span>当前用户</span>
          <strong>{{ profile?.username || userStore.username }}</strong>
        </div>
        <div class="info-row">
          <span>绑定邮箱</span>
          <strong>{{ profile?.email || "未绑定" }}</strong>
        </div>
        <div class="info-row">
          <span>绑定手机号</span>
          <strong>{{ profile?.phone || "未绑定" }}</strong>
        </div>
        <div class="info-row">
          <span>角色</span>
          <strong>{{ profile?.role || "user" }}</strong>
        </div>
      </article>
    </section>
  </section>
</template>

<script setup lang="ts">
import axios from "axios";
import { onMounted, reactive, ref } from "vue";
import { useRouter } from "vue-router";

import { fetchProfileApi, type UserProfile, updateProfileApi } from "../api/auth";
import { useUserStore } from "../stores/user";

const router = useRouter();
const userStore = useUserStore();
const profile = ref<UserProfile | null>(null);
const form = reactive({
  username: "",
  email: "",
  phone: ""
});

type ProfileFieldKey = "username" | "email" | "phone";

const profileFieldLabels: Record<ProfileFieldKey, string> = {
  username: "用户名",
  email: "邮箱",
  phone: "手机号"
};

function normalizeProfileValue(value: string | null | undefined) {
  return (value ?? "").trim();
}

function displayProfileValue(value: string | null | undefined) {
  const normalizedValue = normalizeProfileValue(value);
  return normalizedValue || "未绑定";
}

function buildProfileUpdateMessage(before: UserProfile | null, after: UserProfile) {
  const changedFields = (["username", "email", "phone"] as ProfileFieldKey[])
    .map((field) => {
      const beforeValue = normalizeProfileValue(before?.[field]);
      const afterValue = normalizeProfileValue(after[field]);
      if (beforeValue === afterValue) {
        return null;
      }
      return `${profileFieldLabels[field]}：${displayProfileValue(beforeValue)} -> ${displayProfileValue(afterValue)}`;
    })
    .filter((item): item is string => item !== null);
  const changeContent = changedFields.length > 0 ? changedFields.join("；") : "无字段变化";

  return `资料保存成功\n修改账号UID：${after.uid}\n修改内容：${changeContent}`;
}

async function loadProfile() {
  try {
    const { data } = await fetchProfileApi();
    profile.value = data;
    userStore.setProfile(data);
    form.username = data.username;
    form.email = data.email ?? "";
    form.phone = data.phone ?? "";
  } catch (error) {
    if (axios.isAxiosError(error)) {
      window.alert(String(error.response?.data?.detail ?? "读取个人资料失败。"));
    } else {
      window.alert("读取个人资料失败。");
    }
  }
}

async function submit() {
  try {
    const beforeProfile = profile.value ? { ...profile.value } : null;
    const { data } = await updateProfileApi({
      username: form.username.trim(),
      email: form.email.trim() || undefined,
      phone: form.phone.trim() || undefined
    });
    profile.value = data;
    userStore.setProfile(data);
    window.alert(buildProfileUpdateMessage(beforeProfile, data));
  } catch (error) {
    if (axios.isAxiosError(error)) {
      window.alert(String(error.response?.data?.detail ?? "更新资料失败，请检查输入内容。"));
    } else {
      window.alert("更新资料失败，请稍后重试。");
    }
  }
}

function logout() {
  userStore.logout();
  router.push({ name: "login" });
}

onMounted(() => {
  void loadProfile();
});
</script>

<style scoped lang="scss">
.profile-grid {
  display: grid;
  grid-template-columns: 1.1fr 0.9fr;
  gap: 24px;
}

.profile-form label {
  display: block;
  margin: 16px 0 4px;
  font-size: 14px;
  font-weight: 500;
  color: var(--text-soft);
}

.profile-actions {
  display: flex;
  gap: 12px;
  margin-top: 24px;
}

.info-row {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  padding: 14px 0;
  border-bottom: 1px solid #f0ece5;
}

.info-row:last-child {
  border-bottom: none;
}

.info-row span {
  color: var(--muted);
}

@media (max-width: 960px) {
  .profile-grid {
    grid-template-columns: 1fr;
  }
}
</style>
