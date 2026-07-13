import request from "./request";

export const fetchOwnerGestureApi = () => request.get("/owner-gesture/current");
export const fetchOwnerPanelApi = () => request.get("/owner-gesture/panel");
