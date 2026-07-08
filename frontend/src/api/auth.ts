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

export interface EmailCodePayload {
  email: string;
}

export interface EmailLoginPayload {
  email: string;
  code: string;
}

export interface UpdateProfilePayload {
  username: string;
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
export const sendEmailCodeApi = (payload: EmailCodePayload) => request.post("/auth/email-code", payload);
export const emailLoginApi = (payload: EmailLoginPayload) => request.post<LoginResponse>("/auth/email-login", payload);
export const updateProfileApi = (payload: UpdateProfilePayload) => request.put<UserProfile>("/auth/profile", payload);
