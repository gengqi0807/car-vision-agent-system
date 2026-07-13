import { defineStore } from "pinia";

export const useUserStore = defineStore("user", {
  state: () => ({
    username: localStorage.getItem("cvms_username") ?? "A",
    token: localStorage.getItem("cvms_token") ?? ""
  }),
  actions: {
    setSession(username: string, token: string) {
      this.username = username;
      this.token = token;
      localStorage.setItem("cvms_username", username);
      localStorage.setItem("cvms_token", token);
    },
    setToken(token: string) {
      this.token = token;
      localStorage.setItem("cvms_token", token);
    },
    logout() {
      this.token = "";
      this.username = "A";
      localStorage.removeItem("cvms_token");
      localStorage.removeItem("cvms_username");
    }
  }
});
