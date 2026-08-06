"""Minimal OpenAI-compatible client for the DeepSeek Chat Completions API."""

from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol
from urllib import error, request


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"


class DeepSeekConfigurationError(ValueError):
    """Raised when required local configuration is missing or invalid."""


class DeepSeekAPIError(RuntimeError):
    """Raised for transport errors and invalid API responses."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class ModelCompletion:
    message: dict[str, Any]
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    model: str | None = None


class ChatModelClient(Protocol):
    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelCompletion:
        ...


Transport = Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]]


def _extract_error_message(body: str, fallback: str) -> str:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return fallback
    api_error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(api_error, dict) and isinstance(api_error.get("message"), str):
        return api_error["message"]
    return fallback


def _http_transport(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    http_request = request.Request(url, data=encoded, headers=headers, method="POST")
    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        message = _extract_error_message(body, f"DeepSeek API 返回 HTTP {exc.code}。")
        raise DeepSeekAPIError(message, status_code=exc.code) from exc
    except (error.URLError, socket.timeout, TimeoutError) as exc:
        raise DeepSeekAPIError(f"无法连接 DeepSeek API：{exc.reason if hasattr(exc, 'reason') else exc}") from exc

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise DeepSeekAPIError("DeepSeek API 返回了无效 JSON。") from exc
    if not isinstance(parsed, dict):
        raise DeepSeekAPIError("DeepSeek API 返回结构不是 JSON 对象。")
    return parsed


class DeepSeekClient:
    """Small testable client; secrets only live in process memory."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        max_tokens: int = 1200,
        thinking_enabled: bool = False,
        transport: Transport | None = None,
    ) -> None:
        if not api_key.strip():
            raise DeepSeekConfigurationError("DEEPSEEK_API_KEY 不能为空。")
        if timeout <= 0:
            raise DeepSeekConfigurationError("timeout 必须大于 0。")
        if max_tokens <= 0:
            raise DeepSeekConfigurationError("max_tokens 必须大于 0。")
        self._api_key = api_key.strip()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.thinking_enabled = thinking_enabled
        self._transport = transport or _http_transport

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        max_tokens: int = 1200,
    ) -> "DeepSeekClient":
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise DeepSeekConfigurationError(
                "未设置 DEEPSEEK_API_KEY。请先在当前终端导出该环境变量。"
            )
        return cls(
            api_key=api_key,
            model=model or os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL),
            base_url=base_url or os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
            timeout=timeout,
            max_tokens=max_tokens,
        )

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelCompletion:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "stream": False,
            "thinking": {"type": "enabled" if self.thinking_enabled else "disabled"},
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        response = self._transport(
            f"{self.base_url}/chat/completions", payload, headers, self.timeout
        )

        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise DeepSeekAPIError("DeepSeek API 响应缺少 choices。")
        choice = choices[0]
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            raise DeepSeekAPIError("DeepSeek API 响应缺少 assistant message。")
        message = dict(choice["message"])
        message.setdefault("role", "assistant")
        return ModelCompletion(
            message=message,
            finish_reason=choice.get("finish_reason"),
            usage=response.get("usage") if isinstance(response.get("usage"), dict) else {},
            model=response.get("model") if isinstance(response.get("model"), str) else self.model,
        )
