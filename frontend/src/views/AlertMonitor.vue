<template>
  <section class="page-shell">
    <header class="page-header">
      <h1>告警监控</h1>
      <p>集中查看智能体告警、监控日志、原因分析与事件回放。</p>
    </header>

    <section v-if="showHealthyBanner" class="status-banner healthy">
      <div class="status-title">当前未发现严重告警</div>
      <div class="status-desc">系统监控链路在线，最近一轮采样中未出现严重或警告级别事件。</div>
    </section>

    <section class="alert-stats-grid">
      <article class="alert-stat-card">
        <div class="label">告警总数</div>
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
        <div class="label">监控日志</div>
        <div class="number info">{{ dashboard.total_logs }}</div>
      </article>
    </section>

    <section class="two-col">
      <article class="panel">
        <div class="panel-heading">
          <h4>告警时间线</h4>
          <span class="panel-hint">点击任一条可查看原因分析与事件回放</span>
        </div>
        <div v-if="timeline.length === 0" class="empty-state">暂无告警事件。</div>
        <div
          v-for="item in timeline"
          :key="item.id"
          class="timeline-item"
          :style="{ cursor: 'pointer', opacity: selectedAlertId === item.id ? '1' : '0.92' }"
          @click="selectReplay(item.id)"
        >
          <span class="dot" :class="item.level"></span>
          <div class="body">
            <span class="tag" :class="item.level">{{ getAlertLevelLabel(item.level) }}</span>
            <div class="title">{{ item.title }}</div>
            <div class="desc">{{ item.summary }}</div>
            <div class="desc">
              {{ getSourceLabel(item.source) }}
              <template v-if="item.event_type"> · {{ item.event_type }}</template>
            </div>
          </div>
          <span class="time">{{ formatClock(item.created_at) }}</span>
        </div>
      </article>

      <article class="panel">
        <div class="panel-heading">
          <h4>统计仪表盘</h4>
          <span class="panel-hint">按来源、根因与通知渠道汇总</span>
        </div>
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
        <div class="chart-footer">最近告警来源：{{ joinMetricLabels(dashboard.top_sources) }}</div>
      </article>
    </section>

    <section class="two-col">
      <article class="panel">
        <div class="panel-heading">
          <h4>原因分析</h4>
          <span class="panel-hint">智能体自动生成的根因、影响范围与建议动作</span>
        </div>
        <div v-if="!selectedReplay" class="empty-state">选择左侧告警后，这里会显示详细分析。</div>
        <template v-else>
          <div class="result-item">
            <span>异常类型</span>
            <span class="val">{{ selectedReplay.alert.event_type || "未知" }}</span>
          </div>
          <div class="result-item">
            <span>影响范围</span>
            <span class="val">{{ selectedReplay.alert.impact_scope || "未提供" }}</span>
          </div>
          <div class="result-item">
            <span>根因</span>
            <span class="val">{{ selectedReplay.alert.root_cause || "未提供" }}</span>
          </div>
          <div class="result-item">
            <span>建议处置</span>
            <span class="val">{{ selectedReplay.alert.suggested_action || "未提供" }}</span>
          </div>
          <div class="support-label">回放摘要</div>
          <div class="support-info">{{ selectedReplay.reason_summary }}</div>
          <div class="support-label">通知渠道分布</div>
          <div class="support-tags">{{ joinMetricPairs(overview.notification_breakdown) }}</div>
        </template>
      </article>

      <article class="panel">
        <div class="panel-heading">
          <h4>事件回放</h4>
          <span class="panel-hint">围绕当前告警聚合相关监控日志与推送记录</span>
        </div>
        <div v-if="!selectedReplay" class="empty-state">暂无回放数据。</div>
        <template v-else>
          <div class="history-row header">
            <span>时间</span>
            <span>来源</span>
            <span>状态</span>
            <span>摘要</span>
          </div>
          <div v-for="log in selectedReplay.related_logs.slice(0, 6)" :key="log.id" class="history-row record">
            <span>{{ formatDateTime(log.created_at) }}</span>
            <span>{{ getSourceLabel(log.source) }}</span>
            <span>{{ log.status || "-" }}</span>
            <span>{{ log.title }}</span>
          </div>
          <div class="support-label">推送记录</div>
          <div class="support-info">
            {{ selectedReplay.push_logs.length === 0 ? "暂无推送记录" : replayPushText }}
          </div>
        </template>
      </article>
    </section>

    <section class="two-col">
      <article class="panel">
        <div class="panel-heading">
          <h4>监控日志</h4>
          <span class="panel-hint">覆盖车牌识别、手势识别、交警手势与认证访问</span>
        </div>
        <div v-if="monitorLogs.length === 0" class="empty-state">暂无监控日志。</div>
        <div v-for="log in monitorLogs" :key="log.id" class="behavior-log-item">
          <div class="behavior-log-main">
            <span class="behavior-source">{{ getSourceLabel(log.source) }}</span>
            <div class="title">{{ log.title }}</div>
            <div class="desc">{{ log.summary }}</div>
            <div class="desc">
              {{ log.event_type }} · {{ log.status || "无状态" }}
              <template v-if="typeof log.confidence === 'number'"> · 置信度 {{ log.confidence.toFixed(2) }}</template>
            </div>
          </div>
          <span class="time">{{ formatDateTime(log.created_at) }}</span>
        </div>
      </article>

      <article class="panel">
        <div class="panel-heading">
          <h4>用户操作日志</h4>
          <span class="panel-hint">用于审计登录、注册、资料更新等行为</span>
        </div>
        <div v-if="operationLogs.length === 0" class="empty-state">暂无用户操作日志。</div>
        <div v-for="record in operationLogs" :key="record.id" class="behavior-log-item">
          <div class="behavior-log-main">
            <span class="behavior-source">用户 {{ record.user_id }}</span>
            <div class="title">{{ record.operation_type }}</div>
            <div class="desc">状态：{{ record.response_status || "未知" }}</div>
          </div>
          <span class="time">{{ formatDateTime(record.created_at) }}</span>
        </div>
      </article>
    </section>

    <article class="panel">
      <div class="panel-heading">
        <h4>行为日志</h4>
        <span class="panel-hint">识别任务的普通运行记录，可用于辅助排查。</span>
      </div>
      <div class="behavior-log-board">
        <div v-if="behaviorLogs.length === 0" class="empty-state">暂无行为日志。</div>
        <div v-for="log in paginatedBehaviorLogs" :key="log.id" class="behavior-log-item">
          <div class="behavior-log-main">
            <span class="behavior-source">{{ getSourceLabel(log.source) }}</span>
            <div class="title">{{ log.title }}</div>
            <div class="desc">{{ log.summary }}</div>
          </div>
          <span class="time">{{ formatDateTime(log.created_at) }}</span>
        </div>
      </div>
      <div v-if="totalBehaviorPages > 1" class="pagination-row">
        <button type="button" class="page-arrow" :disabled="behaviorPage === 1" @click="goToPreviousPage">
          上一页
        </button>
        <div class="page-current">{{ behaviorPage }}</div>
        <button type="button" class="page-arrow" :disabled="behaviorPage === totalBehaviorPages" @click="goToNextPage">
          下一页
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
  fetchAlertDashboardApi,
  fetchAlertReplayApi,
  fetchAlertTimelineApi,
  fetchBehaviorLogsApi,
  fetchMonitorLogsApi,
  fetchOperationLogsApi,
  type AlertDashboard,
  type AlertEvent,
  type AlertOverview,
  type AlertReplay,
  type BehaviorLogRecord,
  type MetricPoint,
  type MonitorLogRecord,
  type OperationLogRecord
} from "../api/alert";
import { formatDateTime } from "../utils/format";

const overview = reactive<AlertOverview>({
  total: 0,
  critical: 0,
  warning: 0,
  info: 0,
  latest: [],
  source_breakdown: [],
  root_cause_breakdown: [],
  notification_breakdown: []
});

const dashboard = reactive<AlertDashboard>({
  total_logs: 0,
  alert_overview: {
    total: 0,
    critical: 0,
    warning: 0,
    info: 0,
    latest: [],
    source_breakdown: [],
    root_cause_breakdown: [],
    notification_breakdown: []
  },
  latest_alerts: [],
  latest_logs: [],
  latest_operations: [],
  top_sources: [],
  top_event_types: []
});

const timeline = ref<AlertEvent[]>([]);
const behaviorLogs = ref<BehaviorLogRecord[]>([]);
const monitorLogs = ref<MonitorLogRecord[]>([]);
const operationLogs = ref<OperationLogRecord[]>([]);
const selectedReplay = ref<AlertReplay | null>(null);
const selectedAlertId = ref<number | null>(null);
const behaviorPage = ref(1);
const pageInput = ref("1");

const behaviorPageSize = 5;

let refreshTimer: number | undefined;
let socket: WebSocket | null = null;
let hasShownLoadError = false;

const showHealthyBanner = computed(() => overview.critical === 0 && overview.warning === 0);
const totalBehaviorPages = computed(() => Math.max(1, Math.ceil(behaviorLogs.value.length / behaviorPageSize)));
const paginatedBehaviorLogs = computed(() => {
  const start = (behaviorPage.value - 1) * behaviorPageSize;
  return behaviorLogs.value.slice(start, start + behaviorPageSize);
});
const replayPushText = computed(() =>
  (selectedReplay.value?.push_logs ?? [])
    .map((item) => `${item.channel} -> ${item.target}（${item.success ? "成功" : "失败"}）`)
    .join("；")
);

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

function getSourceLabel(source: string) {
  if (source === "plate-recognition") {
    return "车牌识别";
  }
  if (source === "police-gesture") {
    return "交警手势";
  }
  if (source === "owner-gesture") {
    return "车主手势";
  }
  if (source === "auth") {
    return "用户认证";
  }
  return source;
}

function buildBarHeight(value: number) {
  return `${Math.max(28, value * 28)}px`;
}

function joinMetricLabels(items: MetricPoint[] | undefined) {
  if (!items || items.length === 0) {
    return "暂无";
  }
  return items.map((item) => `${item.label}（${item.value}）`).join("、");
}

function joinMetricPairs(items: MetricPoint[] | undefined) {
  if (!items || items.length === 0) {
    return "暂无";
  }
  return items.map((item) => `${item.label}: ${item.value}`).join("；");
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

async function selectReplay(alertId: number) {
  selectedAlertId.value = alertId;
  try {
    const response = await fetchAlertReplayApi(alertId);
    selectedReplay.value = response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      window.alert(String(error.response?.data?.detail ?? "告警回放加载失败。"));
      return;
    }
    window.alert("告警回放加载失败。");
  }
}

async function loadAlertMonitor() {
  try {
    const [dashboardResponse, timelineResponse, behaviorResponse, monitorResponse, operationResponse] =
      await Promise.all([
        fetchAlertDashboardApi({ latest_limit: 6, log_limit: 12 }),
        fetchAlertTimelineApi(),
        fetchBehaviorLogsApi(24),
        fetchMonitorLogsApi({ limit: 12 }),
        fetchOperationLogsApi({ limit: 12 })
      ]);

    Object.assign(dashboard, dashboardResponse.data);
    Object.assign(overview, dashboardResponse.data.alert_overview);
    timeline.value = timelineResponse.data;
    behaviorLogs.value = behaviorResponse.data;
    monitorLogs.value = monitorResponse.data;
    operationLogs.value = operationResponse.data;

    if (timeline.value.length > 0) {
      const candidateId = selectedAlertId.value ?? timeline.value[0].id;
      if (candidateId !== selectedAlertId.value || selectedReplay.value === null) {
        await selectReplay(candidateId);
      }
    } else {
      selectedAlertId.value = null;
      selectedReplay.value = null;
    }

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

function connectSocket() {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const host =
    window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
      ? "127.0.0.1:8000"
      : window.location.host;
  socket = new WebSocket(`${protocol}://${host}/api/v1/alerts/ws`);

  socket.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      if (payload?.type === "alert.created") {
        void loadAlertMonitor();
      }
    } catch {
      return;
    }
  };

  socket.onclose = () => {
    socket = null;
  };
}

onMounted(() => {
  void loadAlertMonitor();
  connectSocket();
  refreshTimer = window.setInterval(() => {
    void loadAlertMonitor();
  }, 8000);
});

onBeforeUnmount(() => {
  if (refreshTimer) {
    window.clearInterval(refreshTimer);
  }
  socket?.close();
});
</script>
