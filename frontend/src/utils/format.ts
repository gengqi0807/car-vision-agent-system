const BEIJING_TIME_ZONE = "Asia/Shanghai";

function parseBackendDate(value: string) {
  const normalized = value.trim();
  const hasExplicitTimezone = /(?:Z|[+-]\d{2}:\d{2})$/i.test(normalized);
  if (hasExplicitTimezone) {
    return new Date(normalized);
  }

  const isoLikeValue = normalized.includes(" ") ? normalized.replace(" ", "T") : normalized;
  return new Date(`${isoLikeValue}Z`);
}

function formatInBeijingTime(value: string, options: Intl.DateTimeFormatOptions) {
  const date = parseBackendDate(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return new Intl.DateTimeFormat("zh-CN", {
    timeZone: BEIJING_TIME_ZONE,
    hour12: false,
    ...options
  }).format(date);
}

export function formatDateTime(value: string) {
  return formatInBeijingTime(value, {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  });
}

export function formatClock(value: string) {
  return formatInBeijingTime(value, {
    hour: "2-digit",
    minute: "2-digit"
  });
}
