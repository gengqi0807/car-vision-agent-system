import request from "./request";

export interface DashboardCounts {
  plates: number;
  police_gestures: number;
  owner_gestures: number;
  alerts: number;
}

export interface DashboardTrendPoint {
  date: string;
  label: string;
  plates: number;
  police_gestures: number;
  owner_gestures: number;
  total: number;
}

export interface DashboardAlert {
  id: number;
  level: string;
  title: string;
  summary: string;
  created_at: string;
}

export interface DashboardOverview {
  counts: DashboardCounts;
  trend: DashboardTrendPoint[];
  latest_alerts: DashboardAlert[];
}

export const fetchDashboardApi = () =>
  request.get<DashboardOverview>("/dashboard", { params: { days: 7, latest_limit: 5 } });
