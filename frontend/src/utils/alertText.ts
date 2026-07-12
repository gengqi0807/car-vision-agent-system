const EVENT_TYPE_LABELS: Record<string, string> = {
  behavior_event: "行为记录",
  unauthorized_access: "未授权访问",
  register: "用户注册",
  login: "用户登录",
  email_login: "邮箱登录",
  update_profile: "更新资料",
  plate_recognition_success: "车牌识别成功",
  plate_recognition_no_detection: "车牌未识别到结果",
  plate_recognition_failure: "车牌识别失败",
  plate_recognition_timeout: "车牌识别超时",
  owner_gesture_success: "车主手势识别成功",
  owner_gesture_low_confidence: "车主手势低置信度",
  owner_gesture_decode_error: "车主手势图像解析失败",
  police_gesture_success: "交警手势识别成功",
  police_gesture_low_confidence: "交警手势低置信度",
  police_gesture_decode_error: "交警手势图像解析失败"
};

const STATUS_LABELS: Record<string, string> = {
  Success: "成功",
  success: "成功",
  Rejected: "拒绝",
  rejected: "拒绝",
  Failed: "失败",
  failed: "失败",
  Timeout: "超时",
  timeout: "超时",
  NoDetection: "未识别到结果",
  no_detection: "未识别到结果",
  NoHandDetected: "未检测到手部",
  processed: "已处理",
  empty: "空结果",
  recorded: "已记录",
  pending: "待处理"
};

const OPERATION_TYPE_LABELS: Record<string, string> = {
  register: "注册",
  login: "登录",
  email_login: "邮箱登录",
  update_profile: "更新资料",
  plate_recognition: "车牌识别",
  owner_gesture_recognition: "车主手势识别",
  police_gesture_recognition: "交警手势识别"
};

const TITLE_LABELS: Record<string, string> = {
  "Police gesture not recognized": "交警手势未识别",
  "Police gesture frame processed": "交警手势帧处理完成",
  "Police gesture frame processing failed": "交警手势帧处理失败"
};

const TOKEN_LABELS: Record<string, string> = {
  open_palm: "张开手掌",
  fist: "握拳",
  point: "指向",
  index_circle: "画圈",
  swipe_left: "向左挥动",
  swipe_right: "向右挥动",
  thumbs_up: "竖起大拇指",
  thumbs_down: "倒拇指",
  wave: "挥手",
  idle: "待机",
  unknown: "未知",
  stop: "停止",
  WakeSystem: "唤醒系统",
  ConfirmAction: "确认操作",
  AdjustVolume: "调节音量",
  SwitchPrevFeature: "切换上一个功能",
  SwitchNextFeature: "切换下一个功能",
  AnswerCall: "接听电话",
  HangUpCall: "挂断电话",
  ReturnHome: "返回主页",
  None: "无"
};

function escapeRegExp(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function replaceTokenLabel(text: string, token: string, label: string) {
  const pattern = new RegExp(`\\b${escapeRegExp(token)}\\b`, "g");
  return text.replace(pattern, label);
}

function replaceKnownTokens(text: string) {
  let localized = text;
  for (const [token, label] of Object.entries(TOKEN_LABELS)) {
    localized = replaceTokenLabel(localized, token, label);
  }

  return localized
    .split("Cannot decode image bytes.").join("无法解析图像字节数据。")
    .split("gesture=").join("手势=")
    .split("confidence=").join("置信度=")
    .split("poses=").join("姿态数=")
    .split("processed:").join("已处理：");
}

export function localizeEventType(value?: string | null) {
  if (!value) {
    return "";
  }
  return EVENT_TYPE_LABELS[value] ?? replaceKnownTokens(value);
}

export function localizeStatus(value?: string | null) {
  if (!value) {
    return "";
  }
  return STATUS_LABELS[value] ?? replaceKnownTokens(value);
}

export function localizeOperationType(value?: string | null) {
  if (!value) {
    return "";
  }
  return OPERATION_TYPE_LABELS[value] ?? replaceKnownTokens(value);
}

export function localizeLogTitle(value: string) {
  return TITLE_LABELS[value] ?? replaceKnownTokens(value);
}

export function localizeLogSummary(value: string) {
  return replaceKnownTokens(value)
    .replace(
      /^(.+?) did not produce a recognized police gesture\. 手势=(.+), 姿态数=(\d+)\.$/,
      "$1 未识别出有效的交警手势。手势=$2，姿态数=$3。"
    )
    .replace(
      /^(.+?) 已处理： 手势=(.+), 置信度=([\d.]+), 姿态数=(\d+)\.$/,
      "$1 已处理完成：手势=$2，置信度=$3，姿态数=$4。"
    );
}
