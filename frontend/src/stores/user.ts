import { defineStore } from "pinia";

export const useUserStore = defineStore("user", {
  state: () => ({
    username: "demo_admin",
    token: localStorage.getItem("cvms_token") ?? ""
  }),
  actions: {
    setToken(token: string) {
      this.token = token;
      localStorage.setItem("cvms_token", token);
    }
  }
});
