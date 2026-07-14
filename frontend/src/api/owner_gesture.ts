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
  media_playing: boolean;
  comfort_scene: string;
  vehicle_status: string;
  focus_tile: string;
  last_gesture: string | null;
  last_command: string | null;
  last_command_at: string | null;
  last_feedback: string | null;
  updated_at: string | null;
}

export interface OwnerGestureFrameResult {
  gesture: string;
  action: string | null;
  confidence: number;
  keypoints: OwnerGestureKeypoint[];
  annotated_image: string | null;
  control_command: string | null;
  triggered: boolean;
  panel_state: OwnerControlPanelState | null;
  updated_at: string;
}

export interface OwnerGestureStreamResult {
  gesture: string;
  action: string;
  confidence: number;
  keypoints: OwnerGestureKeypoint[];
  annotated_image: string | null;
  hand_count: number;
  control_command: string | null;
  triggered: boolean;
  panel_state: OwnerControlPanelState | null;
  updated_at: string;
}

export interface OwnerGestureStreamState {
  running: boolean;
  source: string;
  fps: number;
  published: boolean;
  publish_rtsp_url: string | null;
  playback_url: string | null;
  last_error: string | null;
  started_at: string | null;
}

export const fetchOwnerGestureApi = (formData: FormData) =>
  request.post<OwnerGestureFrameResult>("/owner-gesture/current", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 60 * 1000,
  });

export const fetchOwnerPanelApi = () => request.get<OwnerControlPanelState>("/owner-gesture/panel");

export const startOwnerGestureStreamApi = (source = "0", fps = 15) =>
  request.post<OwnerGestureStreamState>("/owner-gesture/stream/start", { command: "start", source, fps });

export const stopOwnerGestureStreamApi = () => request.post("/owner-gesture/stream/stop");

export const fetchOwnerGestureStreamStateApi = () =>
  request.get<OwnerGestureStreamState>("/owner-gesture/stream");

export const fetchOwnerGestureStreamResultApi = () =>
  request.get<OwnerGestureStreamResult>("/owner-gesture/current");

export const ownerGestureVideoFeedUrl = () => {
  const base = String(request.defaults.baseURL || "").replace(/\/$/, "");
  return `${base}/owner-gesture/video-feed`;
};

// ── 自定义手势 API ────────────────────────────────────────────────

export interface CustomGestureItem {
  id: number;
  name: string;
  display_name: string;
  description: string;
  sample_count: number;
  is_trained: boolean;
  created_at: string;
  updated_at: string;
}

export interface CustomGestureListOut {
  gestures: CustomGestureItem[];
  total: number;
}

export interface CustomGestureSampleItem {
  id: number;
  gesture_id: number;
  keypoints: { x: number; y: number; z: number }[];
  source_type: string;
  filename: string;
  created_at: string;
}

export interface CustomGestureSampleListOut {
  samples: CustomGestureSampleItem[];
  total: number;
}

export interface CustomGestureTrainOut {
  status: string;
  message: string;
  n_samples: number;
  n_classes: number;
  class_names: string[];
  model_path: string;
  evaluation: Record<string, any>;
}

export const fetchCustomGesturesApi = () =>
  request.get<CustomGestureListOut>("/owner-gesture/custom-gesture/");

export const createCustomGestureApi = (payload: {
  name: string;
  display_name?: string;
  description?: string;
}) => request.post<CustomGestureItem>("/owner-gesture/custom-gesture/", payload);

export const deleteCustomGestureApi = (name: string) =>
  request.delete(`/owner-gesture/custom-gesture/${encodeURIComponent(name)}`);

export const fetchCustomGestureSamplesApi = (name: string, limit = 50, offset = 0) =>
  request.get<CustomGestureSampleListOut>(
    `/owner-gesture/custom-gesture/${encodeURIComponent(name)}/samples`,
    { params: { limit, offset } }
  );

export const addCustomGestureSampleApi = (name: string, formData: FormData) =>
  request.post<CustomGestureSampleItem>(
    `/owner-gesture/custom-gesture/${encodeURIComponent(name)}/samples`,
    formData,
  );

export const deleteCustomGestureSampleApi = (sampleId: number) =>
  request.delete(`/owner-gesture/custom-gesture/samples/${sampleId}`);

export const triggerCustomGestureTrainApi = (gestureNames?: string[]) =>
  request.post<CustomGestureTrainOut>("/owner-gesture/custom-gesture/train", {
    gesture_names: gestureNames || [],
  });
