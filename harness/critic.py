"""Проверка дерева перед выходом.

Критик не вызывает модель. Он смотрит на правки: синтаксис,
закрытые файлы и тест, который до правки проходил, а после упал.
Упавший repro и красный тест бага diff не отменяют: пустой diff
на стенде даёт ноль, частичный ещё может зачесть FAIL_TO_PASS.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from harness.files import SKIP_DIRS
from harness.guard import forbidden_reason
from harness.tools import ToolContext

log = logging.getLogger(__name__)

JUNK_NAMES = {
    "repro.py",
    "patch.txt",
    "notes.md",
    "agent_notes.md",
    "AGENT_NOTES.md",
}


@dataclass(frozen=True)
class Report:
    ok: bool
    text: str


def review(ctx: ToolContext) -> Report:
    problems: list[str] = []
    if not ctx.edits.touched:
        problems.append("diff is empty")
    for path in ctx.edits.paths():
        if not path.exists():
            continue
        reason = forbidden_reason(path, ctx.workspace, ctx.issue)
        if reason:
            problems.append(reason)
        if path.suffix == ".py":
            problem = _syntax(path)
            if problem:
                problems.append(problem)
    if ctx.scratch.regression:
        problems.append("a test that passed before the edit now fails")
    if problems:
        return Report(False, "; ".join(problems))
    return Report(True, "ok")


def remove_junk(workspace: Path) -> list[str]:
    removed: list[str] = []
    for relative in _untracked(workspace):
        path = workspace / relative
        if path.name not in JUNK_NAMES or not path.is_file():
            continue
        path.unlink()
        removed.append(relative)
    return removed


def shutdown(ctx: ToolContext, *, revert: bool) -> int:
    changed = ", ".join(ctx.edits.relative_paths(ctx.workspace)) or "нет"
    if revert:
        log.info("откатываю правки, тесты рядом с фиксом упали: %s", changed)
        ctx.edits.restore_all()
        changed = "нет"
    removed = remove_junk(ctx.workspace)
    if removed:
        log.info("убрал служебные файлы: %s", ", ".join(removed))
    log.info("изменённые файлы: %s", changed)
    return 0


def _syntax(path: Path) -> str | None:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return f"syntax: {path.name}: {exc}"
    try:
        compile(source, str(path), "exec")
    except SyntaxError as exc:
        return f"syntax: {path.name}: line {exc.lineno}"
    return None


def _untracked(workspace: Path) -> list[str]:
    if not (workspace / ".git").exists():
        return []
    try:
        output = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if output.returncode != 0:
        return []
    names: list[str] = []
    for line in output.stdout.splitlines():
        relative = line.strip()
        if not relative or Path(relative).parts[0] in SKIP_DIRS:
            continue
        names.append(relative)
    return names
