import request from "./request";

export interface LoginPayload {
  username: string;
  password: string;
}

export const loginApi = (payload: LoginPayload) => request.post("/auth/login", payload);
export const fetchProfileApi = () => request.get("/auth/me");
