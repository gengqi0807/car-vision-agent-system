import { defineStore } from "pinia";

export const useWebsocketStore = defineStore("websocket", {
  state: () => ({
    connected: false
  }),
  actions: {
    setConnected(status: boolean) {
      this.connected = status;
    }
  }
});
