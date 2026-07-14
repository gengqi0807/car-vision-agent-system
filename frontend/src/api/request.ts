import axios from "axios";

const request = axios.create({
  baseURL: import.meta.env.VITE_API_BASE,
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
