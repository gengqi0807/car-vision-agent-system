import axios from "axios";

const apiBaseURL = import.meta.env.VITE_API_BASE || "/api/v1";

const request = axios.create({
  baseURL: apiBaseURL,
  timeout: 15000
});

request.interceptors.request.use((config) => {
  const token = localStorage.getItem("cvms_token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export default request;
