import { createRouter, createWebHistory } from "vue-router";

import AppLayout from "../layouts/AppLayout.vue";
import AlertMonitor from "../views/AlertMonitor.vue";
import Dashboard from "../views/Dashboard.vue";
import Login from "../views/Login.vue";
import OwnerGesture from "../views/OwnerGesture.vue";
import PlateRecognition from "../views/PlateRecognition.vue";
import PoliceGesture from "../views/PoliceGesture.vue";
import Register from "../views/Register.vue";
import { authGuard } from "./guards";

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: "/login",
      name: "login",
      component: Login
    },
    {
      path: "/register",
      name: "register",
      component: Register
    },
    {
      path: "/",
      component: AppLayout,
      meta: { requiresAuth: true },
      children: [
        {
          path: "",
          name: "dashboard",
          component: Dashboard
        },
        {
          path: "plate-recognition",
          name: "plate-recognition",
          component: PlateRecognition
        },
        {
          path: "police-gesture",
          name: "police-gesture",
          component: PoliceGesture
        },
        {
          path: "owner-gesture",
          name: "owner-gesture",
          component: OwnerGesture
        },
        {
          path: "alert-monitor",
          name: "alert-monitor",
          component: AlertMonitor
        }
      ]
    }
  ]
});

router.beforeEach(authGuard);

export default router;
