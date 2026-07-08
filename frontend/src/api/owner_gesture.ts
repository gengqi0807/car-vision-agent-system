import request from "./request";

export const fetchOwnerGestureApi = (formData: FormData) =>
  request.post("/owner-gesture/current", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

export const fetchOwnerPanelApi = () => request.get("/owner-gesture/panel");
