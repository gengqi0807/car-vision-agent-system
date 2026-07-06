import request from "./request";

export const fetchPlateHistoryApi = () => request.get("/plate/history");
