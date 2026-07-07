import type { NavigationGuardNext, RouteLocationNormalized } from "vue-router";

export function authGuard(to: RouteLocationNormalized, _from: RouteLocationNormalized, next: NavigationGuardNext) {
  const hasToken = Boolean(localStorage.getItem("cvms_token"));
  if (to.meta.requiresAuth && !hasToken) {
    next({ name: "login" });
    return;
  }
  next();
}
