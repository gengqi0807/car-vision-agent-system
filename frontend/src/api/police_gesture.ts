import request from "./request";

export const fetchPoliceGestureApi = () => request.get("/police-gesture/current");
export const fetchPoliceGestureHistoryApi = () => request.get("/police-gesture/history");
