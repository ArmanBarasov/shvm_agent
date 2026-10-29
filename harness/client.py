"""Клиент Chat Completions.

Один и тот же код ходит и в шлюз стенда, и в Ollama.
Отличается только профиль. Обрезанный JSON не считается
вызовом инструмента: цикл повторит запрос более коротким.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

from harness.config import Profile

log = logging.getLogger(__name__)


class ModelError(Exception):
    def __init__(self, message: str, *, fold: bool = False) -> None:
        super().__init__(message)
        self.fold = fold


@dataclass
class Completion:
    message: dict
    tool_name: str | None
    tool_args: dict | None
    tool_call_id: str | None
    truncated: bool


class ModelClient:
    def __init__(self, profile: Profile) -> None:
        self.profile = profile
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ModelError("не установлен пакет openai") from exc
        self._sdk = OpenAI(
            base_url=profile.base_url,
            api_key=profile.api_key,
            max_retries=0,
        )
        # Конкурсный шлюз принимает required. Часть прокси — только auto.
        self._force_tool = True
        self._rate_waits = 0

    def complete(self, messages: list[dict], tools: list[dict], timeout: float) -> Completion:
        raw = json.dumps(messages, ensure_ascii=False, default=str).encode("utf-8")
        if len(raw) > self.profile.max_request_bytes:
            raise ModelError("запрос больше лимита шлюза", fold=True)
        try:
            response = self._create(messages, tools, timeout, rich=True)
        except Exception as exc:
            response = self._recover(exc, messages, tools, timeout)
        self._rate_waits = 0
        choice = response.choices[0]
        message = message_dict(choice.message)
        completion = parse_completion(message, choice.finish_reason)
        usage = getattr(response, "usage", None)
        if usage is not None:
            log.info(
                "токены: вход %s, выход %s",
                getattr(usage, "prompt_tokens", "?"),
                getattr(usage, "completion_tokens", "?"),
            )
        return completion

    def _recover(self, exc: Exception, messages: list[dict], tools: list[dict], timeout: float):
        status = getattr(exc, "status_code", None)
        text = str(exc).lower()
        if status == 429:
            self._rate_waits += 1
            if self._rate_waits > 6:
                raise ModelError("шлюз слишком часто отвечает 429") from exc
            delay = min(30, 2 ** self._rate_waits)
            log.warning("лимит запросов, жду %s с", delay)
            time.sleep(delay)
            try:
                return self._create(messages, tools, timeout, rich=True)
            except Exception as second:
                return self._recover(second, messages, tools, timeout)
        if status == 413 or "payload too large" in text:
            raise ModelError("шлюз вернул 413", fold=True) from exc
        if status == 400 and ("reasoning_content" in text or "context length" in text):
            raise ModelError("шлюз отклонил историю", fold=True) from exc
        if status == 400 and "tool_choice" in text and self._force_tool:
            self._force_tool = False
            log.warning("шлюз не принимает обязательный tool call, повторяю с auto")
            try:
                return self._create(messages, tools, timeout, rich=True)
            except Exception as second:
                return self._recover(second, messages, tools, timeout)
        if status == 400 and any(
            word in text for word in ("temperature", "reasoning", "thinking", "unknown", "unrecognized")
        ):
            log.warning("шлюз не принял параметры thinking, повторяю простой запрос")
            try:
                return self._create(messages, tools, timeout, rich=False)
            except Exception as second:
                raise ModelError(str(second)) from second
        raise ModelError(str(exc)) from exc

    def _create(self, messages: list[dict], tools: list[dict], timeout: float, *, rich: bool):
        kwargs: dict = {
            "model": self.profile.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "required" if self._force_tool else "auto",
            "max_tokens": self.profile.max_output_tokens,
        }
        if rich:
            kwargs["temperature"] = self.profile.temperature
            kwargs["reasoning_effort"] = self.profile.reasoning_effort
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        return self._sdk.with_options(timeout=timeout).chat.completions.create(**kwargs)


def message_dict(message) -> dict:
    data = {"role": "assistant", "content": message.content or ""}
    reasoning = getattr(message, "reasoning_content", None)
    extra = getattr(message, "model_extra", None) or {}
    if not reasoning:
        reasoning = extra.get("reasoning_content")
    if reasoning:
        data["reasoning_content"] = reasoning
    calls = []
    for call in message.tool_calls or []:
        calls.append(
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments or "{}",
                },
            }
        )
    if calls:
        data["tool_calls"] = calls
    return data


def parse_completion(message: dict, finish_reason: str | None) -> Completion:
    hit_limit = finish_reason in {"length", "incomplete"}
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        return Completion(message, None, None, None, hit_limit)
    if len(tool_calls) > 1:
        message = dict(message)
        message["tool_calls"] = tool_calls[:1]
    first = message["tool_calls"][0]
    function = first.get("function") or {}
    name = function.get("name")
    call_id = first.get("id")
    if not name:
        return Completion(message, None, None, call_id, True)
    try:
        args = json.loads(function.get("arguments") or "{}")
    except json.JSONDecodeError:
        # Обрезанный текст не исполняем. Целый JSON при лимите токенов
        # обычно значит, что оборвался reasoning, а вызов уже собран.
        return Completion(message, None, None, call_id, True)
    if not isinstance(args, dict):
        return Completion(message, None, None, call_id, True)
    return Completion(message, name, args, call_id, False)
