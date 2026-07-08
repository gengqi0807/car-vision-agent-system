import request from "./request";

export interface AlertEvent {
  id: number;
  level: "critical" | "warning" | "info";
  source: string;
  title: string;
  summary: string;
  created_at: string;
}

export interface AlertOverview {
  total: number;
  critical: number;
  warning: number;
  info: number;
  latest: AlertEvent[];
}

export interface BehaviorLogRecord {
  id: number;
  source: string;
  title: string;
  summary: string;
  created_at: string;
}

export const fetchAlertOverviewApi = () => request.get<AlertOverview>("/alerts/overview");
export const fetchAlertTimelineApi = () => request.get<AlertEvent[]>("/alerts/timeline");
export const fetchBehaviorLogsApi = (limit = 24) =>
  request.get<BehaviorLogRecord[]>("/alerts/behavior-logs", {
    params: { limit }
  });
