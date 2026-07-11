import request from "./request";

export interface PlateDetection {
  plate_number: string;
  plate_color: string;
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
  unread_samples: string[];
  processed_frame_count: number;
  duration_seconds?: number | null;
}

export interface PlateRecordSummary {
  id: number;
  plate_number: string;
  plate_color: string;
  created_at: string;
}

export interface PlateStreamControlResponse {
  running: boolean;
  published?: boolean;
  rtsp_url?: string | null;
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
    timeout: 2 * 60 * 1000
  });
}

export function recognizePlateVideoApi(file: File) {
  const formData = new FormData();
  formData.append("file", file);
  return request.post<PlateVideoRecognitionResponse>("/plate/video", formData, {
    timeout: 10 * 60 * 1000
  });
}

export function startPlatePushStreamApi(rtspUrl: string, streamName?: string) {
  return request.post<PlateStreamControlResponse>("/plate/stream/start", {
    rtsp_url: rtspUrl,
    stream_name: streamName
  });
}

export const stopPlatePushStreamApi = () => request.post<PlateStreamControlResponse>("/plate/stream/stop");

export const fetchPlatePushStreamStatusApi = () => request.get<PlateStreamControlResponse>("/plate/stream/status");
