"""Память одной задачи.

В запрос каждый раз уходят задача, досье прочитанных фрагментов
и последний результат инструмента. Старые ответы модели не
пересылаются: их reasoning раздувает тело, а текст файла от этого
не становится ближе. Досье и есть состояние.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def clip(text: str, limit: int = 8000) -> str:
    text = redact(text)
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = max(0, limit - head - 20)
    return text[:head] + "\n...\n" + text[-tail:]


def redact(text: str) -> str:
    for name in ("OPENAI_API_KEY", "OLLAMA_API_KEY"):
        secret = os.environ.get(name, "")
        if len(secret) >= 8:
            text = text.replace(secret, "***")
    return text


def clip_issue(issue: str) -> str:
    if len(issue) <= 14_000:
        return issue
    return issue[:12_000] + "\n\n...[issue truncated]...\n\n" + issue[-2_000:]


@dataclass
class Window:
    path: str
    start: int
    end: int
    raw: str


@dataclass
class Dossier:
    windows: list[Window] = field(default_factory=list)

    def store(self, path: str, start: int, end: int, raw: str) -> None:
        self.windows = [
            window
            for window in self.windows
            if not (window.path == path and window.start == start and window.end == end)
        ]
        self.windows.append(Window(path, start, end, raw))
        del self.windows[:-3]

    def has(self, path: str, start: int, end: int) -> bool:
        return any(
            window.path == path and window.start == start and window.end == end
            for window in self.windows
        )

    def drop(self, path: str) -> None:
        self.windows = [window for window in self.windows if window.path != path]

    def shrink(self) -> bool:
        if not self.windows:
            return False
        self.windows.pop(0)
        return True

    def render(self) -> str:
        if not self.windows:
            return "empty"
        blocks: list[str] = []
        for window in self.windows:
            numbered = _number(window)
            blocks.append(
                f"File {window.path} lines {window.start}-{window.end}\n"
                f"{numbered}\n\n"
                f"Exact text for edit:\n{window.raw}"
            )
        return clip("\n\n".join(blocks), 24_000)


def _number(window: Window) -> str:
    rows = []
    for offset, line in enumerate(window.raw.splitlines()):
        rows.append(f"{window.start + offset:5}|{line}")
    return "\n".join(rows)


@dataclass
class Scratchpad:
    facts: list[str] = field(default_factory=list)
    reads: int = 0
    regression: bool = False
    repro_base: str = ""
    repro_after: str = ""
    passed_before: set[str] = field(default_factory=set)

    def add(self, fact: str) -> None:
        self.facts.append(fact.replace("\n", " ")[:240])
        del self.facts[:-12]

    def render(self) -> str:
        lines = list(self.facts)
        if self.repro_base:
            lines.append(f"repro before edit: {self.repro_base}")
        if self.repro_after:
            lines.append(f"repro after edit: {self.repro_after}")
        if self.regression:
            lines.append("a test that passed before the edit now fails")
        return ("\n".join(lines) or "empty")[:4000]


class Transcript:
    def __init__(self, issue: str) -> None:
        self.issue = issue
        self.scratch = Scratchpad()
        self.dossier = Dossier()
        self.last_result = ""

    def build(self, system: str, hint: str) -> list[dict]:
        user = (
            f"{hint}\n\n"
            f"Dossier:\n{self.dossier.render()}\n\n"
            f"Scratchpad:\n{self.scratch.render()}\n\n"
            f"Last result:\n{clip(self.last_result, 4000) or 'none'}\n\n"
            f"Issue:\n{clip_issue(self.issue)}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def note(self, result: str) -> None:
        self.last_result = result

    def shrink_last_observation(self) -> bool:
        if len(self.last_result) < 80:
            return False
        self.last_result = self.last_result[: len(self.last_result) // 2] + "\n[shortened]"
        return True

    def fold(self) -> bool:
        shrunk = self.dossier.shrink()
        if len(self.last_result) > 500:
            self.last_result = clip(self.last_result, 500)
            shrunk = True
        if shrunk:
            self.scratch.add("earlier file text was folded")
        return shrunk
