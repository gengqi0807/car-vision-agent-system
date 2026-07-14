<template>
  <section class="page-shell">
    <header class="page-header">
      <h1>仪表盘</h1>
      <p>系统运行概览与关键指标</p>
    </header>

    <div v-if="errorMessage" class="dashboard-error">{{ errorMessage }}</div>

    <section class="stats-grid" :aria-busy="loading">
      <article class="stat-card card">
        <div class="label">车牌识别</div>
        <div class="number">{{ formatNumber(dashboard.counts.plates) }}</div>
      </article>
      <article class="stat-card card">
        <div class="label">交警手势</div>
        <div class="number">{{ formatNumber(dashboard.counts.police_gestures) }}</div>
      </article>
      <article class="stat-card card">
        <div class="label">手势控车</div>
        <div class="number">{{ formatNumber(dashboard.counts.owner_gestures) }}</div>
      </article>
      <article class="stat-card card">
        <div class="label">告警</div>
        <div class="number critical">{{ formatNumber(dashboard.counts.alerts) }}</div>
      </article>
    </section>

    <section class="charts-grid">
      <article class="panel">
        <h4>最近 7 天识别趋势</h4>
        <div class="bar-chart">
          <div v-for="item in dashboard.trend" :key="item.date" class="bar-item">
            <span class="bar-value">{{ item.total }}</span>
            <div
              class="bar"
              :style="{ height: `${barHeight(item.total)}px` }"
              :title="trendTitle(item)"
            ></div>
            <span class="bar-label">{{ item.label }}</span>
          </div>
        </div>
        <div v-if="!loading && dashboard.trend.every((item) => item.total === 0)" class="empty-state">
          最近 7 天暂无识别记录
        </div>
      </article>

      <article class="panel">
        <h4>最新告警</h4>
        <div v-if="loading" class="empty-state">正在加载仪表盘数据...</div>
        <div v-else-if="dashboard.latest_alerts.length === 0" class="empty-state">暂无告警记录</div>
        <div
          v-for="alert in dashboard.latest_alerts"
          v-else
          :key="alert.id"
          class="alert-item"
          :class="normalizeLevel(alert.level)"
        >
          <span class="dot" :class="normalizeLevel(alert.level)"></span>
          <div class="content">
            <span class="tag" :class="normalizeLevel(alert.level)">{{ levelText(alert.level) }}</span>
            <div class="msg">{{ alert.title }}</div>
            <div class="desc">{{ alert.summary }}</div>
          </div>
          <span class="time">{{ formatTime(alert.created_at) }}</span>
        </div>
      </article>
    </section>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref } from "vue";
import axios from "axios";

import { fetchDashboardApi, type DashboardOverview, type DashboardTrendPoint } from "../api/dashboard";

const dashboard = reactive<DashboardOverview>({
  counts: { plates: 0, police_gestures: 0, owner_gestures: 0, alerts: 0 },
  trend: [],
  latest_alerts: []
});
const loading = ref(true);
const errorMessage = ref("");
const maxTrendValue = computed(() => Math.max(1, ...dashboard.trend.map((item) => item.total)));

function formatNumber(value: number) {
  return new Intl.NumberFormat("zh-CN").format(value || 0);
}

function barHeight(value: number) {
  if (value <= 0) return 4;
  return Math.max(12, Math.round((value / maxTrendValue.value) * 112));
}

function trendTitle(item: DashboardTrendPoint) {
  return `车牌 ${item.plates} · 交警手势 ${item.police_gestures} · 手势控车 ${item.owner_gestures}`;
}

function normalizeLevel(level: string) {
  return ["critical", "warning", "info"].includes(level) ? level : "info";
}

function levelText(level: string) {
  return { critical: "严重", warning: "警告", info: "提示" }[normalizeLevel(level)] ?? "提示";
}

function formatTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "--";
  return date.toLocaleString("zh-CN", { hour12: false, month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}

async function loadDashboard() {
  loading.value = true;
  errorMessage.value = "";
  try {
    const { data } = await fetchDashboardApi();
    Object.assign(dashboard.counts, data.counts);
    dashboard.trend = data.trend;
    dashboard.latest_alerts = data.latest_alerts;
  } catch (error) {
    errorMessage.value = axios.isAxiosError(error)
      ? String(error.response?.data?.detail || "仪表盘数据加载失败，请检查后端与数据库连接。")
      : "仪表盘数据加载失败，请稍后重试。";
  } finally {
    loading.value = false;
  }
}

onMounted(loadDashboard);
</script>

<style scoped>
.dashboard-error {
  padding: 12px 16px;
  border: 1px solid #e8c9c1;
  border-radius: 6px;
  color: #9d493d;
  background: #fff7f5;
}

.bar-value {
  min-height: 16px;
  font-size: 12px;
  color: var(--text-soft);
}

.empty-state {
  padding: 18px 0;
  text-align: center;
  color: var(--muted);
}
</style>
