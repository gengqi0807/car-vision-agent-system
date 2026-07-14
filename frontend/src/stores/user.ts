import { defineStore } from "pinia";

import type { UserProfile } from "../api/auth";

export const useUserStore = defineStore("user", {
  state: () => ({
    username: localStorage.getItem("cvms_username") ?? "A",
    token: localStorage.getItem("cvms_token") ?? "",
    profile: null as UserProfile | null
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
    setProfile(profile: UserProfile) {
      this.profile = profile;
      this.username = profile.username;
      localStorage.setItem("cvms_username", profile.username);
    },
    logout() {
      this.token = "";
      this.username = "A";
      this.profile = null;
      localStorage.removeItem("cvms_token");
      localStorage.removeItem("cvms_username");
    }
  }
});
