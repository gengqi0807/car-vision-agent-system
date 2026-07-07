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
