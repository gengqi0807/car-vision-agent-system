from __future__ import annotations

import json
from datetime import datetime
from urllib import request

from app.core.config import settings


class LLMClient:
    def build_summary(self, source: str, payload: dict) -> dict:
        fallback = self._build_fallback_summary(source=source, payload=payload)
        if not settings.llm_api_base or not settings.llm_api_key:
            return fallback

        try:
            return self._build_remote_summary(source=source, payload=payload, fallback=fallback)
        except Exception:
            return fallback

    def _build_remote_summary(self, *, source: str, payload: dict, fallback: dict) -> dict:
        prompt = {
            "source": source,
            "event_type": payload.get("event_type"),
            "level": payload.get("level"),
            "title": payload.get("title"),
            "summary": payload.get("summary"),
            "impact_scope": payload.get("impact_scope"),
            "root_cause_hint": payload.get("root_cause"),
            "suggested_action_hint": payload.get("suggested_action"),
            "analysis": payload.get("analysis", {}),
        }
        body = {
            "model": "gpt-4.1-mini",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是告警分析助手。请返回 JSON，对象中必须包含以下键："
                        "summary、root_cause、impact_scope、suggested_action。"
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }
        base = settings.llm_api_base.rstrip("/")
        if not base.endswith("/chat/completions"):
            base = f"{base}/chat/completions"

        req = request.Request(
            base,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {settings.llm_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=settings.llm_request_timeout_seconds) as response:
            response_payload = json.loads(response.read().decode("utf-8"))

        content = response_payload["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        return {
            **fallback,
            "summary": parsed.get("summary") or fallback["summary"],
            "root_cause": parsed.get("root_cause") or fallback["root_cause"],
            "impact_scope": parsed.get("impact_scope") or fallback["impact_scope"],
            "suggested_action": parsed.get("suggested_action") or fallback["suggested_action"],
            "llm_mode": "remote",
        }

    def _build_fallback_summary(self, *, source: str, payload: dict) -> dict:
        title = payload.get("title", "系统告警")
        level = payload.get("level", "info")
        event_type = payload.get("event_type", "unknown")
        timestamp = payload.get("created_at") or datetime.utcnow().isoformat()
        root_cause = payload.get("root_cause") or self._default_root_cause(event_type, source)
        impact_scope = payload.get("impact_scope") or self._default_impact_scope(source, level)
        suggested_action = payload.get("suggested_action") or self._default_action(event_type, source, level)
        evidence = payload.get("summary", "")

        summary = (
            f"【{self._level_label(level)}】{title}。"
            f"发生时间：{timestamp}。"
            f"异常来源：{self._source_label(source)}，事件类型：{event_type}。"
            f"影响范围：{impact_scope}。"
            f"监控依据：{evidence}。"
            f"建议处置：{suggested_action}。"
        )

        return {
            "source": source,
            "event_type": event_type,
            "level": level,
            "title": title,
            "summary": summary,
            "root_cause": root_cause,
            "impact_scope": impact_scope,
            "suggested_action": suggested_action,
            "created_at": timestamp,
            "llm_mode": "fallback",
        }

    def _default_root_cause(self, event_type: str, source: str) -> str:
        if "timeout" in event_type:
            return "上游依赖或模型推理执行时间超过了系统设定阈值。"
        if "low_confidence" in event_type:
            return "输入质量较低，或模型置信度持续低于设定阈值。"
        if "unauthorized" in event_type or source == "auth":
            return "身份认证或授权校验未通过。"
        if "token" in event_type:
            return "LLM Token 使用量超过了系统配置的预算限制。"
        if "failure" in event_type:
            return "识别链路连续返回失败结果。"
        return "监控智能体检测到了异常运行信号。"

    def _default_impact_scope(self, source: str, level: str) -> str:
        if source == "plate-recognition":
            return "影响车牌识别请求以及相关历史记录。"
        if source == "owner-gesture":
            return "影响车内手势控车交互。"
        if source == "police-gesture":
            return "影响交警手势识别结果。"
        if source == "auth":
            return "影响需要认证保护的接口与用户访问流程。"
        if level == "critical":
            return "可能影响多个用户或系统核心功能。"
        return "影响范围主要集中在当前功能模块。"

    def _default_action(self, event_type: str, source: str, level: str) -> str:
        if "timeout" in event_type:
            return "检查超时阈值、模型加载耗时以及服务器资源占用情况。"
        if "low_confidence" in event_type:
            return "优化摄像头取景与光照条件，并确认模型资源已正确加载。"
        if "unauthorized" in event_type or source == "auth":
            return "核查访问方身份，必要时吊销可疑令牌，并审查访问日志。"
        if "token" in event_type:
            return "降低 Token 消耗，必要时轮换凭据，并检查服务商额度。"
        if level == "critical":
            return "立即通知运维人员，并优先排查相关监控日志。"
        return "检查最近的监控日志，确认该事件是否符合预期。"

    def _level_label(self, level: str) -> str:
        return {
            "critical": "严重",
            "warning": "警告",
            "info": "提示",
        }.get(level, level)

    def _source_label(self, source: str) -> str:
        return {
            "plate-recognition": "车牌识别",
            "owner-gesture": "车主手势",
            "police-gesture": "交警手势",
            "auth": "用户认证",
        }.get(source, source)
