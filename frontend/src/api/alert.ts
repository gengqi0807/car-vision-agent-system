import request from "./request";

export const fetchAlertOverviewApi = () => request.get("/alerts/overview");
export const fetchAlertTimelineApi = () => request.get("/alerts/timeline");
