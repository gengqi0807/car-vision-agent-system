import request from "./request";

export interface PlateDetection {
  plate_number: string;
  plate_color: string;
  vehicle_type: string;
  confidence: number;
  bbox: number[];
}

export interface PlateRecognitionResponse {
  frame_id: string;
  detections: PlateDetection[];
}

export interface PlateVideoRecognitionResponse {
  source_filename: string;
  processed_video_url: string;
  detections: PlateDetection[];
  processed_frame_count: number;
  duration_seconds?: number | null;
}

export interface PlateVideoJobCreateResponse {
  job_id: string;
  status: string;
}

export interface PlateVideoJobStatusResponse {
  job_id: string;
  source_filename: string;
  status: string;
  progress: number;
  processed_frame_count: number;
  total_frames: number;
  detections: PlateDetection[];
  preview_image_url?: string | null;
  processed_video_url?: string | null;
  unread_samples: string[];
  duration_seconds?: number | null;
  error_message?: string | null;
}

export interface PlateRecordSummary {
  id: number;
  plate_number: string;
  plate_color: string;
  vehicle_type: string;
  created_at: string;
}

export interface PlateStreamControlResponse {
  running: boolean;
  published?: boolean;
  publisher_started?: boolean;
  phase?: string;
  status_message?: string | null;
  process_frames?: boolean;
  source_type?: "rtsp" | "camera";
  rtsp_url?: string | null;
  camera_index?: number | null;
  stream_name?: string | null;
  publish_rtsp_url?: string | null;
  playback_url?: string | null;
  last_error?: string | null;
  started_at?: string | null;
}

export const fetchPlateHistoryApi = () => request.get<PlateRecordSummary[]>("/plate/history");

export function recognizePlateImageApi(file: File) {
  const formData = new FormData();
  formData.append("file", file);
  return request.post<PlateRecognitionResponse>("/plate/image", formData, {
    timeout: 60 * 1000
  });
}

export function recognizePlateVideoApi(file: File) {
  const formData = new FormData();
  formData.append("file", file);
  return request.post<PlateVideoRecognitionResponse>("/plate/video", formData, {
    timeout: 10 * 60 * 1000
  });
}

export function createPlateVideoJobApi(file: File) {
  const formData = new FormData();
  formData.append("file", file);
  return request.post<PlateVideoJobCreateResponse>("/plate/video/jobs", formData, {
    timeout: 2 * 60 * 1000
  });
}

export const fetchPlateVideoJobStatusApi = (jobId: string) =>
  request.get<PlateVideoJobStatusResponse>(`/plate/video/jobs/${jobId}`);

export function startPlatePushStreamApi(payload: {
  sourceType?: "rtsp" | "camera";
  rtspUrl?: string;
  cameraIndex?: number;
  streamName?: string;
  processFrames?: boolean;
}) {
  return request.post<PlateStreamControlResponse>("/plate/stream/start", {
    source_type: payload.sourceType ?? "rtsp",
    rtsp_url: payload.rtspUrl,
    camera_index: payload.cameraIndex,
    stream_name: payload.streamName,
    process_frames: payload.processFrames ?? true
  });
}

export const stopPlatePushStreamApi = () => request.post<PlateStreamControlResponse>("/plate/stream/stop");

export const fetchPlatePushStreamStatusApi = () => request.get<PlateStreamControlResponse>("/plate/stream/status");
