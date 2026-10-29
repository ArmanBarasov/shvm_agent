"""Два профиля одной и той же схемы запроса.

Конкурс читает только OPENAI_*. Разработка ходит в Ollama.
Это переключатель запуска, а не запасной провайдер: при ошибке
конкурсного шлюза запрос никуда не переезжает.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse

CONTEST_MODEL = "deepseek-v4.1-flash"


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class Profile:
    name: str
    base_url: str
    api_key: str
    model: str
    reasoning_effort: str
    temperature: float
    max_output_tokens: int
    max_request_bytes: int
    task_timeout_s: int

    @property
    def host(self) -> str:
        return urlparse(self.base_url).netloc or self.base_url


def load_profile() -> Profile:
    mode = _selected_mode()
    if mode == "contest":
        return _contest_profile()
    return _dev_profile()


def _selected_mode() -> str:
    chosen = os.environ.get("AGENT_PROFILE", "").strip().lower()
    if chosen in {"contest", "dev"}:
        return chosen
    # На стенде профиль не передают: достаточно переменных шлюза.
    if os.environ.get("OPENAI_BASE_URL"):
        return "contest"
    return "dev"


def _contest_profile() -> Profile:
    base_url = _required("OPENAI_BASE_URL")
    api_key = _required("OPENAI_API_KEY")
    model = _required("OPENAI_MODEL")
    if model != CONTEST_MODEL:
        raise ConfigError(
            f"конкурсный профиль принимает только {CONTEST_MODEL}, получено {model}"
        )
    if "ollama" in urlparse(base_url).netloc:
        raise ConfigError("конкурсный профиль не отправляет запросы в Ollama")
    return Profile(
        name="contest",
        base_url=base_url,
        api_key=api_key,
        model=model,
        reasoning_effort="high",
        temperature=0.2,
        max_output_tokens=4096,
        max_request_bytes=4 * 1024 * 1024,
        task_timeout_s=600,
    )


def _dev_profile() -> Profile:
    # MATCH_LIMITS=0 ослабляет потолок токенов, чтобы отлаживать промпт.
    # Такой прогон не похож на стенд и балл не предсказывает.
    match_limits = os.environ.get("MATCH_LIMITS", "1") != "0"
    if match_limits:
        max_tokens, max_bytes = 4096, 4 * 1024 * 1024
    else:
        max_tokens, max_bytes = 8192, 8 * 1024 * 1024
    return Profile(
        name="dev",
        base_url=os.environ.get("OLLAMA_BASE_URL", "https://ollama.com/v1").strip(),
        api_key=_required("OLLAMA_API_KEY"),
        model=os.environ.get("OLLAMA_MODEL", CONTEST_MODEL).strip() or CONTEST_MODEL,
        reasoning_effort=os.environ.get("DEV_REASONING", "high").strip() or "high",
        temperature=0.2,
        max_output_tokens=max_tokens,
        max_request_bytes=max_bytes,
        task_timeout_s=600,
    )


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(f"не задана переменная {name}")
    return value
