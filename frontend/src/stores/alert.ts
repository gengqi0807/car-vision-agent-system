import { defineStore } from "pinia";

export interface AlertItem {
  id: number;
  level: string;
  title: string;
  summary: string;
  created_at: string;
}

export const useAlertStore = defineStore("alert", {
  state: () => ({
    items: [] as AlertItem[]
  }),
  actions: {
    setItems(items: AlertItem[]) {
      this.items = items;
    }
  }
});
