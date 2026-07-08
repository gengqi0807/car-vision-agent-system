import request from "./request";

export const fetchPoliceGestureApi = (formData: FormData) =>
  request.post("/police-gesture/current", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

export const fetchPoliceGestureHistoryApi = () =>
  request.get("/police-gesture/history");
