import request from "./request";

export interface AlertEvent {
  id: number;
  level: "critical" | "warning" | "info";
  source: string;
  event_type?: string | null;
  title: string;
  summary: string;
  impact_scope?: string | null;
  root_cause?: string | null;
  suggested_action?: string | null;
  analysis?: Record<string, unknown> | null;
  created_at: string;
}

export interface AlertOverview {
  total: number;
  critical: number;
  warning: number;
  info: number;
  latest: AlertEvent[];
  source_breakdown?: MetricPoint[];
  root_cause_breakdown?: MetricPoint[];
  notification_breakdown?: MetricPoint[];
}

export interface BehaviorLogRecord {
  id: number;
  source: string;
  title: string;
  summary: string;
  created_at: string;
}

export interface OperationLogRecord {
  id: number;
  user_id: number;
  operation_type: string;
  response_status?: string | null;
  created_at: string;
}

export interface MonitorLogRecord {
  id: number;
  category: string;
  source: string;
  event_type: string;
  level: string;
  title: string;
  summary: string;
  status?: string | null;
  trace_id?: string | null;
  user_id?: number | null;
  alert_id?: number | null;
  confidence?: number | null;
  details?: Record<string, unknown> | null;
  created_at: string;
}

export interface AlertPushRecord {
  id: number;
  channel: string;
  target: string;
  success: boolean;
  created_at: string;
}

export interface MetricPoint {
  label: string;
  value: number;
}

export interface AlertReplay {
  alert: AlertEvent;
  related_logs: MonitorLogRecord[];
  push_logs: AlertPushRecord[];
  reason_summary: string;
}

export interface AlertDashboard {
  total_logs: number;
  alert_overview: AlertOverview;
  latest_alerts: AlertEvent[];
  latest_logs: MonitorLogRecord[];
  latest_operations: OperationLogRecord[];
  top_sources: MetricPoint[];
  top_event_types: MetricPoint[];
}

export interface PagedResult<T> {
  items: T[];
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
}

export const fetchAlertOverviewApi = () => request.get<AlertOverview>("/alerts/overview");
export const fetchAlertTimelineApi = () => request.get<AlertEvent[]>("/alerts/timeline");
export const fetchAlertTimelinePageApi = (params?: {
  page?: number;
  page_size?: number;
  level?: string;
  source?: string;
}) => request.get<PagedResult<AlertEvent>>("/alerts/timeline-page", { params });
export const fetchBehaviorLogsApi = (limit = 24) =>
  request.get<BehaviorLogRecord[]>("/alerts/behavior-logs", {
    params: { limit }
  });
export const fetchBehaviorLogsPageApi = (params?: {
  page?: number;
  page_size?: number;
}) => request.get<PagedResult<BehaviorLogRecord>>("/alerts/behavior-logs-page", { params });

export const fetchMonitorLogsApi = (params?: {
  limit?: number;
  category?: string;
  source?: string;
  level?: string;
}) => request.get<MonitorLogRecord[]>("/alerts/monitor-logs", { params });
export const fetchMonitorLogsPageApi = (params?: {
  page?: number;
  page_size?: number;
  category?: string;
  source?: string;
  level?: string;
}) => request.get<PagedResult<MonitorLogRecord>>("/alerts/monitor-logs-page", { params });

export const fetchOperationLogsApi = (params?: {
  limit?: number;
  user_id?: number;
  operation_type?: string;
}) => request.get<OperationLogRecord[]>("/alerts/operation-logs", { params });
export const fetchOperationLogsPageApi = (params?: {
  page?: number;
  page_size?: number;
  user_id?: number;
  operation_type?: string;
}) => request.get<PagedResult<OperationLogRecord>>("/alerts/operation-logs-page", { params });

export const fetchAlertReplayApi = (alertId: number) =>
  request.get<AlertReplay>(`/alerts/replay/${alertId}`);

export const fetchAlertDashboardApi = (params?: {
  latest_limit?: number;
  log_limit?: number;
}) => request.get<AlertDashboard>("/alerts/dashboard", { params });
