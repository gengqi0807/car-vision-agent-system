import request from "./request";

export interface OwnerGestureKeypoint {
  x: number;
  y: number;
  score: number;
}

export interface OwnerControlPanelState {
  system_awake: boolean;
  volume: number;
  climate_temperature: number;
  phone_call_active: boolean;
  current_mode: string;
  last_gesture: string | null;
  last_command: string | null;
  updated_at: string | null;
}

export interface OwnerGestureResult {
  gesture: string;
  confidence: number;
  keypoints: OwnerGestureKeypoint[];
  control_command: string | null;
  triggered: boolean;
  panel_state: OwnerControlPanelState | null;
  updated_at: string;
}

export const fetchOwnerGestureApi = (formData: FormData) =>
  request.post<OwnerGestureResult>("/owner-gesture/current", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

export const fetchOwnerPanelApi = () => request.get<OwnerControlPanelState>("/owner-gesture/panel");
