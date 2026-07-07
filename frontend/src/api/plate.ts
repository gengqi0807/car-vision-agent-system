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

export interface PlateRecordSummary {
  id: number;
  plate_number: string;
  plate_color: string;
  created_at: string;
}

export const fetchPlateHistoryApi = () => request.get<PlateRecordSummary[]>("/plate/history");

export function recognizePlateImageApi(file: File) {
  const formData = new FormData();
  formData.append("file", file);
  return request.post<PlateRecognitionResponse>("/plate/image", formData);
}

function normalizeBasePath(path: string) {
  if (!path) {
    return "";
  }
  return path.startsWith("/") ? path.replace(/\/$/, "") : `/${path.replace(/\/$/, "")}`;
}

export function buildPlateStreamWebSocketUrl(rtspUrl: string) {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const apiBase = normalizeBasePath((import.meta.env.VITE_API_BASE as string) || "/api/v1");
  const explicitWsBase = (import.meta.env.VITE_WS_BASE as string | undefined)?.replace(/\/$/, "");
  const defaultOrigin = import.meta.env.DEV ? `${protocol}//127.0.0.1:8000` : `${protocol}//${window.location.host}`;
  const origin = explicitWsBase || defaultOrigin;
  return `${origin}${apiBase}/plate/ws/stream?rtsp_url=${encodeURIComponent(rtspUrl)}`;
}
