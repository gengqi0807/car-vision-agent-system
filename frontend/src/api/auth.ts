import request from "./request";

export interface LoginPayload {
  username: string;
  password: string;
}

export interface RegisterPayload {
  username: string;
  password: string;
  email?: string;
  phone?: string;
}

export interface UserProfile {
  id: number;
  username: string;
  email: string | null;
  phone: string | null;
  role: string;
  created_at: string;
  last_login: string | null;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  user: UserProfile;
}

export const loginApi = (payload: LoginPayload) => request.post<LoginResponse>("/auth/login", payload);
export const registerApi = (payload: RegisterPayload) => request.post<UserProfile>("/auth/register", payload);
export const fetchProfileApi = () => request.get<UserProfile>("/auth/me");
