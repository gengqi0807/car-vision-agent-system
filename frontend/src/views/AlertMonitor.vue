<template>
  <section class="page-shell">
    <header class="page-header">
      <h1>告警监控</h1>
      <p>实时告警推送、历史时间线与多模块行为日志</p>
    </header>

    <section v-if="showHealthyBanner" class="status-banner healthy">
      <div class="status-title">目前无异常</div>
      <div class="status-desc">车牌识别、交警手势与手势控车模块运行正常，暂未发现告警。</div>
    </section>

    <section class="alert-stats-grid">
      <article class="alert-stat-card">
        <div class="label">总告警</div>
        <div class="number">{{ overview.total }}</div>
      </article>
      <article class="alert-stat-card">
        <div class="label">严重</div>
        <div class="number critical">{{ overview.critical }}</div>
      </article>
      <article class="alert-stat-card">
        <div class="label">警告</div>
        <div class="number warning">{{ overview.warning }}</div>
      </article>
      <article class="alert-stat-card">
        <div class="label">提示</div>
        <div class="number info">{{ overview.info }}</div>
      </article>
    </section>

    <section class="two-col">
      <article class="panel">
        <h4>告警时间线</h4>
        <div v-if="timeline.length === 0" class="empty-state">当前没有异常告警记录。</div>
        <div v-for="item in timeline" :key="item.id" class="timeline-item">
          <span class="dot" :class="item.level"></span>
          <div class="body">
            <span class="tag" :class="item.level">{{ getAlertLevelLabel(item.level) }}</span>
            <div class="title">{{ item.title }}</div>
            <div class="desc">{{ item.summary }}</div>
          </div>
          <span class="time">{{ formatClock(item.created_at) }}</span>
        </div>
      </article>

      <article class="panel">
        <h4>告警级别分布</h4>
        <div class="vertical-bars">
          <div class="vbar-group">
            <div class="bar critical" :style="{ height: buildBarHeight(overview.critical) }"></div>
            <div class="bar-value">{{ overview.critical }}</div>
            <div class="bar-label">严重</div>
          </div>
          <div class="vbar-group">
            <div class="bar warning" :style="{ height: buildBarHeight(overview.warning) }"></div>
            <div class="bar-value">{{ overview.warning }}</div>
            <div class="bar-label">警告</div>
          </div>
          <div class="vbar-group">
            <div class="bar info" :style="{ height: buildBarHeight(overview.info) }"></div>
            <div class="bar-value">{{ overview.info }}</div>
            <div class="bar-label">提示</div>
          </div>
        </div>
        <div class="chart-footer">共 {{ overview.total }} 条告警</div>
      </article>
    </section>

    <article class="panel">
      <div class="panel-heading">
        <h4>行为日志</h4>
        <span class="panel-hint">实时追加车牌识别、交警手势、手势控车交互</span>
      </div>
      <div class="behavior-log-board">
        <div v-if="behaviorLogs.length === 0" class="empty-state">暂无真实行为日志。</div>
        <div v-for="log in paginatedBehaviorLogs" :key="log.id" class="behavior-log-item">
          <div class="behavior-log-main">
            <span class="behavior-source">{{ getBehaviorSourceLabel(log.source) }}</span>
            <div class="title">{{ log.title }}</div>
            <div class="desc">{{ log.summary }}</div>
          </div>
          <span class="time">{{ formatDateTime(log.created_at) }}</span>
        </div>
      </div>
      <div v-if="totalBehaviorPages > 1" class="pagination-row">
        <button type="button" class="page-arrow" :disabled="behaviorPage === 1" @click="goToPreviousPage">
          ‹
        </button>
        <div class="page-current">{{ behaviorPage }}</div>
        <button
          type="button"
          class="page-arrow"
          :disabled="behaviorPage === totalBehaviorPages"
          @click="goToNextPage"
        >
          ›
        </button>
        <div class="page-jump">
          <input
            v-model="pageInput"
            class="page-input"
            type="number"
            min="1"
            :max="totalBehaviorPages"
            @keydown.enter.prevent="jumpToBehaviorPage"
            @blur="normalizeBehaviorPageInput"
          />
          <span class="page-total">/ {{ totalBehaviorPages }}</span>
          <button type="button" class="page-jump-btn" @click="jumpToBehaviorPage">跳转</button>
        </div>
      </div>
    </article>
  </section>
</template>

<script setup lang="ts">
import axios from "axios";
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from "vue";

import {
  fetchAlertOverviewApi,
  fetchAlertTimelineApi,
  fetchBehaviorLogsApi,
  type AlertEvent,
  type AlertOverview,
  type BehaviorLogRecord
} from "../api/alert";
import { formatDateTime } from "../utils/format";

const overview = reactive<AlertOverview>({
  total: 0,
  critical: 0,
  warning: 0,
  info: 0,
  latest: []
});
const timeline = ref<AlertEvent[]>([]);
const behaviorLogs = ref<BehaviorLogRecord[]>([]);
const behaviorPage = ref(1);
const pageInput = ref("1");

const behaviorPageSize = 5;

let refreshTimer: number | undefined;
let hasShownLoadError = false;

const showHealthyBanner = computed(() => overview.critical === 0 && overview.warning === 0);
const totalBehaviorPages = computed(() => Math.max(1, Math.ceil(behaviorLogs.value.length / behaviorPageSize)));
const paginatedBehaviorLogs = computed(() => {
  const start = (behaviorPage.value - 1) * behaviorPageSize;
  return behaviorLogs.value.slice(start, start + behaviorPageSize);
});

watch(behaviorPage, (page) => {
  pageInput.value = String(page);
});

function getAlertLevelLabel(level: AlertEvent["level"]) {
  if (level === "critical") {
    return "严重";
  }
  if (level === "warning") {
    return "警告";
  }
  return "提示";
}

function getBehaviorSourceLabel(source: string) {
  if (source === "plate-recognition") {
    return "车牌识别";
  }
  if (source === "police-gesture") {
    return "交警手势";
  }
  if (source === "owner-gesture") {
    return "手势控车";
  }
  return source;
}

function buildBarHeight(value: number) {
  return `${Math.max(28, value * 28)}px`;
}

function setBehaviorPage(page: number) {
  behaviorPage.value = Math.min(Math.max(1, page), totalBehaviorPages.value);
}

function goToPreviousPage() {
  setBehaviorPage(behaviorPage.value - 1);
}

function goToNextPage() {
  setBehaviorPage(behaviorPage.value + 1);
}

function jumpToBehaviorPage() {
  const page = Number(pageInput.value);
  if (Number.isNaN(page)) {
    pageInput.value = String(behaviorPage.value);
    return;
  }

  setBehaviorPage(page);
}

function normalizeBehaviorPageInput() {
  const page = Number(pageInput.value);
  if (Number.isNaN(page)) {
    pageInput.value = String(behaviorPage.value);
    return;
  }

  setBehaviorPage(page);
}

function formatClock(value: string) {
  return new Date(value).toLocaleTimeString("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false
  });
}

async function loadAlertMonitor() {
  try {
    const [overviewResponse, timelineResponse, behaviorResponse] = await Promise.all([
      fetchAlertOverviewApi(),
      fetchAlertTimelineApi(),
      fetchBehaviorLogsApi(24)
    ]);

    Object.assign(overview, overviewResponse.data);
    timeline.value = timelineResponse.data;
    behaviorLogs.value = behaviorResponse.data;
    if (behaviorPage.value > totalBehaviorPages.value) {
      behaviorPage.value = totalBehaviorPages.value;
    } else {
      pageInput.value = String(behaviorPage.value);
    }
    hasShownLoadError = false;
  } catch (error) {
    if (hasShownLoadError) {
      return;
    }
    hasShownLoadError = true;

    if (axios.isAxiosError(error)) {
      window.alert(String(error.response?.data?.detail ?? "告警监控数据加载失败，请检查后端服务。"));
      return;
    }
    window.alert("告警监控数据加载失败，请稍后重试。");
  }
}

onMounted(() => {
  void loadAlertMonitor();
  refreshTimer = window.setInterval(() => {
    void loadAlertMonitor();
  }, 5000);
});

onBeforeUnmount(() => {
  if (refreshTimer) {
    window.clearInterval(refreshTimer);
  }
});
</script>
