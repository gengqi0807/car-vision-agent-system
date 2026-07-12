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

export interface PoliceGestureHistoryItem {
  gesture: string;
  confidence: number;
  source_path?: string | null;
  updated_at: string;
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

export const fetchPoliceGestureVideoProgressApi = (taskId: string) =>
  request.get<PoliceGestureVideoProgress>(`/police-gesture/video/progress/${taskId}`);

export const fetchPoliceGestureHistoryApi = () =>
  request.get<PoliceGestureHistoryItem[]>("/police-gesture/history");
