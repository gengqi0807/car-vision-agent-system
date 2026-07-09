<template>
  <section class="page-shell">
    <header class="page-header">
      <h1>告警监控</h1>
      <p>集中查看智能体告警、监控日志、原因分析与行为记录。</p>
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
        <div class="panel-fixed-body panel-fixed-body--timeline">
          <div v-if="timelineState.items.length === 0" class="empty-state">暂无告警事件。</div>
          <div
            v-for="item in timelineState.items"
            :key="item.id"
            class="timeline-item"
            :style="{ cursor: 'pointer', opacity: selectedAlertId === item.id ? '1' : '0.92' }"
            @click="selectReplay(item.id)"
          >
            <span class="dot" :class="item.level"></span>
            <div class="body">
              <span class="tag" :class="item.level">{{ getAlertLevelLabel(item.level) }}</span>
              <div class="title">{{ item.title }}</div>
              <div class="desc clamp-2">{{ item.summary }}</div>
              <div class="desc">
                {{ getSourceLabel(item.source) }}
                <template v-if="item.event_type"> · {{ item.event_type }}</template>
              </div>
            </div>
            <span class="time">{{ formatClock(item.created_at) }}</span>
          </div>
        </div>
        <div class="pagination-row">
          <button
            type="button"
            class="page-arrow"
            :disabled="timelineState.page <= 1"
            @click="goToPreviousPage(timelineState, loadTimelinePage)"
          >
            &lt;
          </button>
          <div class="page-status">第 {{ timelineState.page }} / {{ timelineState.totalPages }} 页</div>
          <button
            type="button"
            class="page-arrow"
            :disabled="timelineState.page >= timelineState.totalPages"
            @click="goToNextPage(timelineState, loadTimelinePage)"
          >
            &gt;
          </button>
          <div class="page-jump">
            <input
              v-model="timelineState.pageInput"
              class="page-input"
              type="number"
              min="1"
              :max="timelineState.totalPages"
              @keydown.enter.prevent="jumpToPage(timelineState, loadTimelinePage)"
              @blur="normalizePageInput(timelineState)"
            />
            <span class="page-range">范围 1 - {{ timelineState.totalPages }}</span>
          </div>
        </div>
      </article>

      <article class="panel">
        <div class="panel-heading">
          <h4>统计仪表盘</h4>
          <span class="panel-hint">按来源、根因与通知渠道汇总</span>
        </div>
        <div class="pie-chart-card">
          <div class="pie-chart-wrap">
            <svg viewBox="0 0 220 220" class="pie-chart" aria-label="Alert level pie chart">
              <circle cx="110" cy="110" r="78" fill="#f3ede4" />
              <template v-if="pieChartSegments.length > 0">
                <path
                  v-for="segment in pieChartSegments"
                  :key="segment.key"
                  :d="segment.path"
                  :fill="segment.color"
                />
                <text
                  v-for="segment in pieChartSegments"
                  :key="`${segment.key}-label`"
                  :x="segment.labelX"
                  :y="segment.labelY"
                  class="pie-slice-label"
                >
                  {{ segment.value }}
                </text>
              </template>
              <template v-else>
                <circle cx="110" cy="110" r="78" fill="#eadfce" />
              </template>
              <circle cx="110" cy="110" r="46" fill="#fffaf3" />
              <text x="110" y="102" text-anchor="middle" class="pie-center-title">总数</text>
              <text x="110" y="126" text-anchor="middle" class="pie-center-value">{{ pieChartTotal }}</text>
            </svg>
          </div>
          <div class="pie-legend">
            <div v-for="segment in pieLegendItems" :key="segment.key" class="pie-legend-item">
              <span class="pie-legend-dot" :style="{ backgroundColor: segment.color }"></span>
              <span class="pie-legend-label">{{ segment.label }}</span>
              <span class="pie-legend-value">{{ segment.value }}</span>
            </div>
          </div>
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
        <div class="panel-fixed-body">
          <div v-if="monitorState.items.length === 0" class="empty-state">暂无监控日志。</div>
          <div v-for="log in monitorState.items" :key="log.id" class="behavior-log-item">
            <div class="behavior-log-main">
              <span class="behavior-source">{{ getSourceLabel(log.source) }}</span>
              <div class="title">{{ log.title }}</div>
              <div class="desc clamp-2">{{ log.summary }}</div>
              <div class="desc">
                {{ log.event_type }} · {{ log.status || "无状态" }}
                <template v-if="typeof log.confidence === 'number'"> · 置信度 {{ log.confidence.toFixed(2) }}</template>
              </div>
            </div>
            <span class="time">{{ formatDateTime(log.created_at) }}</span>
          </div>
        </div>
        <div class="pagination-row">
          <button
            type="button"
            class="page-arrow"
            :disabled="monitorState.page <= 1"
            @click="goToPreviousPage(monitorState, loadMonitorLogsPage)"
          >
            &lt;
          </button>
          <div class="page-status">第 {{ monitorState.page }} / {{ monitorState.totalPages }} 页</div>
          <button
            type="button"
            class="page-arrow"
            :disabled="monitorState.page >= monitorState.totalPages"
            @click="goToNextPage(monitorState, loadMonitorLogsPage)"
          >
            &gt;
          </button>
          <div class="page-jump">
            <input
              v-model="monitorState.pageInput"
              class="page-input"
              type="number"
              min="1"
              :max="monitorState.totalPages"
              @keydown.enter.prevent="jumpToPage(monitorState, loadMonitorLogsPage)"
              @blur="normalizePageInput(monitorState)"
            />
            <span class="page-range">范围 1 - {{ monitorState.totalPages }}</span>
          </div>
        </div>
      </article>

      <article class="panel">
        <div class="panel-heading">
          <h4>用户操作日志</h4>
          <span class="panel-hint">用于审计登录、注册、资料更新等行为</span>
        </div>
        <div class="panel-fixed-body">
          <div v-if="operationState.items.length === 0" class="empty-state">暂无用户操作日志。</div>
          <div v-for="record in operationState.items" :key="record.id" class="behavior-log-item">
            <div class="behavior-log-main">
              <span class="behavior-source">用户 {{ record.user_id }}</span>
              <div class="title">{{ record.operation_type }}</div>
              <div class="desc">状态：{{ record.response_status || "未知" }}</div>
            </div>
            <span class="time">{{ formatDateTime(record.created_at) }}</span>
          </div>
        </div>
        <div class="pagination-row">
          <button
            type="button"
            class="page-arrow"
            :disabled="operationState.page <= 1"
            @click="goToPreviousPage(operationState, loadOperationLogsPage)"
          >
            &lt;
          </button>
          <div class="page-status">第 {{ operationState.page }} / {{ operationState.totalPages }} 页</div>
          <button
            type="button"
            class="page-arrow"
            :disabled="operationState.page >= operationState.totalPages"
            @click="goToNextPage(operationState, loadOperationLogsPage)"
          >
            &gt;
          </button>
          <div class="page-jump">
            <input
              v-model="operationState.pageInput"
              class="page-input"
              type="number"
              min="1"
              :max="operationState.totalPages"
              @keydown.enter.prevent="jumpToPage(operationState, loadOperationLogsPage)"
              @blur="normalizePageInput(operationState)"
            />
            <span class="page-range">范围 1 - {{ operationState.totalPages }}</span>
          </div>
        </div>
      </article>
    </section>

    <article class="panel">
      <div class="panel-heading">
        <h4>行为日志</h4>
        <span class="panel-hint">识别任务的普通运行记录，可用于辅助排查。</span>
      </div>
      <div class="panel-fixed-body panel-fixed-body--wide">
        <div v-if="behaviorState.items.length === 0" class="empty-state">暂无行为日志。</div>
        <div v-for="log in behaviorState.items" :key="log.id" class="behavior-log-item">
          <div class="behavior-log-main">
            <span class="behavior-source">{{ getSourceLabel(log.source) }}</span>
            <div class="title">{{ log.title }}</div>
            <div class="desc clamp-2">{{ log.summary }}</div>
          </div>
          <span class="time">{{ formatDateTime(log.created_at) }}</span>
        </div>
      </div>
      <div class="pagination-row">
        <button
          type="button"
          class="page-arrow"
          :disabled="behaviorState.page <= 1"
          @click="goToPreviousPage(behaviorState, loadBehaviorLogsPage)"
        >
          &lt;
        </button>
        <div class="page-status">第 {{ behaviorState.page }} / {{ behaviorState.totalPages }} 页</div>
        <button
          type="button"
          class="page-arrow"
          :disabled="behaviorState.page >= behaviorState.totalPages"
          @click="goToNextPage(behaviorState, loadBehaviorLogsPage)"
        >
          &gt;
        </button>
        <div class="page-jump">
          <input
            v-model="behaviorState.pageInput"
            class="page-input"
            type="number"
            min="1"
            :max="behaviorState.totalPages"
            @keydown.enter.prevent="jumpToPage(behaviorState, loadBehaviorLogsPage)"
            @blur="normalizePageInput(behaviorState)"
          />
          <span class="page-range">范围 1 - {{ behaviorState.totalPages }}</span>
        </div>
      </div>
    </article>
  </section>
</template>

<script setup lang="ts">
import axios from "axios";
import { computed, onBeforeUnmount, onMounted, reactive, ref } from "vue";

import {
  fetchAlertDashboardApi,
  fetchAlertReplayApi,
  fetchAlertTimelinePageApi,
  fetchBehaviorLogsPageApi,
  fetchMonitorLogsPageApi,
  fetchOperationLogsPageApi,
  type AlertDashboard,
  type AlertEvent,
  type AlertOverview,
  type AlertReplay,
  type BehaviorLogRecord,
  type MetricPoint,
  type MonitorLogRecord,
  type OperationLogRecord,
  type PagedResult
} from "../api/alert";
import { formatDateTime } from "../utils/format";

interface PagedState<T> {
  items: T[];
  page: number;
  pageInput: string;
  total: number;
  totalPages: number;
  pageSize: number;
}

type PageLoader<T> = (page?: number) => Promise<void>;

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

const selectedReplay = ref<AlertReplay | null>(null);
const selectedAlertId = ref<number | null>(null);

const timelineState = createPagedState<AlertEvent>(5);
const monitorState = createPagedState<MonitorLogRecord>(5);
const operationState = createPagedState<OperationLogRecord>(5);
const behaviorState = createPagedState<BehaviorLogRecord>(5);

let refreshTimer: number | undefined;
let socket: WebSocket | null = null;
let hasShownLoadError = false;

const showHealthyBanner = computed(() => overview.critical === 0 && overview.warning === 0);
const replayPushText = computed(() =>
  (selectedReplay.value?.push_logs ?? [])
    .map((item) => `${item.channel} -> ${item.target} (${item.success ? "成功" : "失败"})`)
    .join("；")
);
const pieLegendItems = computed(() => [
  { key: "critical", label: "严重", value: overview.critical, color: "#c95d4c" },
  { key: "warning", label: "警告", value: overview.warning, color: "#d8a25c" },
  { key: "info", label: "提示", value: overview.info, color: "#7ea0c6" }
]);
const pieChartTotal = computed(() => pieLegendItems.value.reduce((sum, item) => sum + item.value, 0));
const pieChartSegments = computed(() => {
  const total = pieChartTotal.value;
  if (total <= 0) {
    return [] as Array<{
      key: string;
      value: number;
      color: string;
      path: string;
      labelX: number;
      labelY: number;
    }>;
  }

  let startAngle = -90;
  return pieLegendItems.value
    .filter((item) => item.value > 0)
    .map((item) => {
      const angle = (item.value / total) * 360;
      const endAngle = startAngle + angle;
      const midAngle = startAngle + angle / 2;
      const labelPoint = polarToCartesian(110, 110, 62, midAngle);
      const segment = {
        key: item.key,
        value: item.value,
        color: item.color,
        path: describePieSlice(110, 110, 78, startAngle, endAngle),
        labelX: labelPoint.x,
        labelY: labelPoint.y
      };
      startAngle = endAngle;
      return segment;
    });
});

function createPagedState<T>(pageSize: number): PagedState<T> {
  return reactive({
    items: [] as T[],
    page: 1,
    pageInput: "1",
    total: 0,
    totalPages: 1,
    pageSize
  }) as PagedState<T>;
}

function applyPageData<T>(state: PagedState<T>, data: PagedResult<T>) {
  state.items = data.items;
  state.page = data.page;
  state.pageInput = String(data.page);
  state.total = data.total;
  state.totalPages = data.total_pages;
}

function normalizeRequestedPage<T>(state: PagedState<T>, rawPage: number) {
  if (!Number.isFinite(rawPage)) {
    return state.page;
  }
  return Math.min(Math.max(1, Math.trunc(rawPage)), Math.max(1, state.totalPages));
}

function normalizePageInput<T>(state: PagedState<T>) {
  state.pageInput = String(normalizeRequestedPage(state, Number(state.pageInput)));
}

async function jumpToPage<T>(state: PagedState<T>, loader: PageLoader<T>) {
  await loader(normalizeRequestedPage(state, Number(state.pageInput)));
}

async function goToPreviousPage<T>(state: PagedState<T>, loader: PageLoader<T>) {
  await loader(state.page - 1);
}

async function goToNextPage<T>(state: PagedState<T>, loader: PageLoader<T>) {
  await loader(state.page + 1);
}

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

function formatClock(value: string) {
  return new Date(value).toLocaleTimeString("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false
  });
}

function polarToCartesian(centerX: number, centerY: number, radius: number, angleInDegrees: number) {
  const angleInRadians = (angleInDegrees * Math.PI) / 180;
  return {
    x: centerX + radius * Math.cos(angleInRadians),
    y: centerY + radius * Math.sin(angleInRadians)
  };
}

function describePieSlice(centerX: number, centerY: number, radius: number, startAngle: number, endAngle: number) {
  if (endAngle - startAngle >= 360) {
    return [
      `M ${centerX} ${centerY}`,
      `m -${radius}, 0`,
      `a ${radius},${radius} 0 1,0 ${radius * 2},0`,
      `a ${radius},${radius} 0 1,0 -${radius * 2},0`
    ].join(" ");
  }

  const start = polarToCartesian(centerX, centerY, radius, startAngle);
  const end = polarToCartesian(centerX, centerY, radius, endAngle);
  const largeArcFlag = endAngle - startAngle > 180 ? 1 : 0;

  return [
    `M ${centerX} ${centerY}`,
    `L ${start.x} ${start.y}`,
    `A ${radius} ${radius} 0 ${largeArcFlag} 1 ${end.x} ${end.y}`,
    "Z"
  ].join(" ");
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

async function loadTimelinePage(page = timelineState.page) {
  const response = await fetchAlertTimelinePageApi({
    page: normalizeRequestedPage(timelineState, page),
    page_size: timelineState.pageSize
  });
  applyPageData(timelineState, response.data);

  if (timelineState.items.length === 0) {
    selectedAlertId.value = null;
    selectedReplay.value = null;
    return;
  }

  if (selectedAlertId.value === null) {
    await selectReplay(timelineState.items[0].id);
    return;
  }

  try {
    await selectReplay(selectedAlertId.value);
  } catch {
    return;
  }
}

async function loadMonitorLogsPage(page = monitorState.page) {
  const response = await fetchMonitorLogsPageApi({
    page: normalizeRequestedPage(monitorState, page),
    page_size: monitorState.pageSize
  });
  applyPageData(monitorState, response.data);
}

async function loadOperationLogsPage(page = operationState.page) {
  const response = await fetchOperationLogsPageApi({
    page: normalizeRequestedPage(operationState, page),
    page_size: operationState.pageSize
  });
  applyPageData(operationState, response.data);
}

async function loadBehaviorLogsPage(page = behaviorState.page) {
  const response = await fetchBehaviorLogsPageApi({
    page: normalizeRequestedPage(behaviorState, page),
    page_size: behaviorState.pageSize
  });
  applyPageData(behaviorState, response.data);
}

async function loadAlertMonitor() {
  try {
    const [dashboardResponse] = await Promise.all([
      fetchAlertDashboardApi({ latest_limit: 6, log_limit: 12 }),
      loadTimelinePage(),
      loadBehaviorLogsPage(),
      loadMonitorLogsPage(),
      loadOperationLogsPage()
    ]);

    Object.assign(dashboard, dashboardResponse.data);
    Object.assign(overview, dashboardResponse.data.alert_overview);
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

<style scoped lang="scss">
.panel-fixed-body {
  display: flex;
  flex-direction: column;
  gap: 12px;
  height: 420px;
  overflow: hidden;
}

.panel-fixed-body--timeline {
  height: 440px;
}

.panel-fixed-body--wide {
  height: 360px;
}

.pagination-row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-top: 14px;
  flex-wrap: wrap;
}

.page-arrow {
  width: 38px;
  height: 38px;
  border: 1px solid rgba(184, 162, 141, 0.35);
  border-radius: 10px;
  background: #f8f3ec;
  color: #5e4d41;
  font-size: 16px;
  cursor: pointer;
}

.page-arrow:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.page-status {
  min-width: 92px;
  color: #6e5e52;
  font-size: 13px;
  font-weight: 600;
}

.page-jump {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-left: auto;
}

.page-input {
  width: 84px;
  height: 38px;
  padding: 0 10px;
  border: 1px solid rgba(184, 162, 141, 0.35);
  border-radius: 10px;
  background: #fffdf9;
  color: #5e4d41;
}

.page-range {
  color: #938477;
  font-size: 12px;
}

.clamp-2 {
  display: -webkit-box;
  overflow: hidden;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
}

.vertical-bars {
  display: none;
}

.pie-chart-card {
  display: grid;
  grid-template-columns: minmax(180px, 240px) minmax(0, 1fr);
  gap: 18px;
  align-items: center;
  min-height: 260px;
}

.pie-chart-wrap {
  display: flex;
  justify-content: center;
  align-items: center;
}

.pie-chart {
  width: 220px;
  height: 220px;
  overflow: visible;
}

.pie-center-title {
  fill: #9a8978;
  font-size: 12px;
  font-weight: 600;
}

.pie-center-value {
  fill: #5d4c40;
  font-size: 24px;
  font-weight: 700;
}

.pie-slice-label {
  fill: #fffaf3;
  font-size: 14px;
  font-weight: 700;
  text-anchor: middle;
  dominant-baseline: middle;
}

.pie-legend {
  display: grid;
  gap: 12px;
}

.pie-legend-item {
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: 10px;
  align-items: center;
  padding: 10px 12px;
  border-radius: 14px;
  background: rgba(248, 243, 236, 0.9);
  border: 1px solid rgba(184, 162, 141, 0.18);
}

.pie-legend-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
}

.pie-legend-label {
  color: #6e5e52;
  font-size: 13px;
  font-weight: 600;
}

.pie-legend-value {
  color: #5d4c40;
  font-size: 15px;
  font-weight: 700;
}

.timeline-item,
.behavior-log-item {
  min-height: 0;
}

@media (max-width: 960px) {
  .panel-fixed-body,
  .panel-fixed-body--timeline,
  .panel-fixed-body--wide {
    height: auto;
    min-height: 320px;
  }

  .page-jump {
    margin-left: 0;
  }

  .pie-chart-card {
    grid-template-columns: 1fr;
    justify-items: center;
  }

  .pie-legend {
    width: 100%;
  }
}
</style>
