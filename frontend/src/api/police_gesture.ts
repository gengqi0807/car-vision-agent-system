import request from "./request";

export interface PoliceGestureKeypoint {
  x: number;
  y: number;
  score: number;
}

export interface PoliceGestureFrameResult {
  gesture: string;
  confidence: number;
  keypoints: PoliceGestureKeypoint[];
  annotated_image?: string | null;
  updated_at: string;
}

export interface PoliceGestureVideoResult {
  source_filename: string;
  gesture: string;
  confidence: number;
  keypoints: PoliceGestureKeypoint[];
  task_id?: string | null;
  processed_video_url: string;
  processed_frame_count: number;
  duration_seconds?: number | null;
  updated_at: string;
}

export interface PoliceGestureVideoProgress {
  task_id: string;
  source_filename: string;
  status: string;
  progress: number;
  message: string;
  processed_frame_count: number;
  total_frames?: number | null;
  gesture?: string | null;
  confidence?: number | null;
  annotated_frame?: string | null;
  playback_url?: string | null;
  processed_video_url?: string | null;
  duration_seconds?: number | null;
  events: Array<{
    gesture: string;
    confidence: number;
    frame_index: number;
    timestamp_seconds?: number | null;
    message: string;
    updated_at: string;
  }>;
  updated_at: string;
}

export interface PoliceGestureVideoJobCreateResponse {
  task_id: string;
  status: string;
}

export interface PoliceGestureHistoryItem {
  gesture: string;
  confidence: number;
  source_path?: string | null;
  updated_at: string;
}

export interface PoliceGestureStreamState {
  running: boolean;
  source: string;
  fps: number;
  published: boolean;
  publish_rtsp_url?: string | null;
  playback_url?: string | null;
  last_error?: string | null;
  started_at?: string | null;
}

export const fetchPoliceGestureApi = (formData: FormData) =>
  request.post<PoliceGestureFrameResult>("/police-gesture/current", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

export const recognizePoliceGestureVideoApi = (formData: FormData, taskId?: string) => {
  if (taskId) {
    formData.set("task_id", taskId);
  }

  return request.post<PoliceGestureVideoResult>("/police-gesture/video", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 10 * 60 * 1000,
  });
};

export const createPoliceGestureVideoJobApi = (formData: FormData, taskId: string) => {
  formData.set("task_id", taskId);
  return request.post<PoliceGestureVideoJobCreateResponse>("/police-gesture/video/jobs", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 60 * 1000,
  });
};

export const cancelPoliceGestureVideoJobApi = (taskId: string) =>
  request.post<PoliceGestureVideoProgress>(`/police-gesture/video/jobs/${taskId}/cancel`);

export const fetchPoliceGestureVideoProgressApi = (taskId: string) =>
  request.get<PoliceGestureVideoProgress>(`/police-gesture/video/progress/${taskId}`);

export const fetchPoliceGestureHistoryApi = () =>
  request.get<PoliceGestureHistoryItem[]>("/police-gesture/history");

export const startPoliceGestureStreamApi = (source = "auto", fps = 15) =>
  request.post<PoliceGestureStreamState>("/police-gesture/stream/start", { command: "start", source, fps });

export const stopPoliceGestureStreamApi = () =>
  request.post<PoliceGestureStreamState>("/police-gesture/stream/stop");

export const fetchPoliceGestureStreamStateApi = () =>
  request.get<PoliceGestureStreamState>("/police-gesture/stream");

export const fetchPoliceGestureStreamResultApi = () =>
  request.get<PoliceGestureFrameResult>("/police-gesture/stream/result");

export const policeGestureVideoFeedUrl = () => {
  const base = String(request.defaults.baseURL || "").replace(/\/$/, "");
  return `${base}/police-gesture/stream/video-feed`;
};
