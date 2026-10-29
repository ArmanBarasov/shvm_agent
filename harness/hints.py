"""Поиск файлов-кандидатов до первого вызова модели.

Сначала берём пути из traceback и текста задачи. Если их нет,
ищем буквальные цитаты и редкие идентификаторы. Эмбеддингов нет:
на стенде они запрещены.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from harness.files import walk_python
from harness.guard import CONFIG_NAMES

FILE_LINE = re.compile(r'File "([^"]+\.py)", line \d+')
PLAIN_PATH = re.compile(r"(?:[\w.-]+/)+[\w.-]+\.py")
QUOTED = re.compile(r"[\"'`]([^\"'`]{4,80})[\"'`]")
IDENT = re.compile(r"\b(?:[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+|[a-z]+(?:_[a-z0-9]+)+)\b")


CALL = re.compile(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b")


def focus_window(issue: str, workspace: Path, files: list[str]) -> tuple[str, int, int, str] | None:
    """Функция, которую можно заменить целиком, без точного old."""
    root = workspace.resolve()
    for module, func in CALL.findall(issue)[:6]:
        for path in _named(root, module):
            span = _function_text(path, func)
            if span is None:
                continue
            return (path.relative_to(root).as_posix(), *span)
    for match in FILE_LINE.finditer(issue):
        path = locate(workspace, match.group(1))
        if path is None:
            continue
        line = int(re.search(r"line (\d+)", match.group(0)).group(1))
        span = _enclosing(path, line)
        if span is None:
            continue
        return (path.relative_to(root).as_posix(), *span)
    names = re.findall(r"\b([A-Za-z_]\w*)\s*\(", issue)
    for rel in files:
        if _noisy(rel):
            continue
        for name in names[:8]:
            span = _function_text(root / rel, name)
            if span is None:
                continue
            return (rel, *span)
    return None


def _named(workspace: Path, stem: str) -> list[Path]:
    found: list[Path] = []
    for path in workspace.rglob(stem + ".py"):
        if any(part in {".git", ".venv", "venv", "__pycache__"} for part in path.parts):
            continue
        found.append(path)
        if len(found) >= 5:
            break
    return found


def _function_text(path: Path, name: str) -> tuple[int, int, str] | None:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, UnicodeError, SyntaxError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return _slice(source, node.lineno, node.end_lineno or node.lineno)
    return None


def _enclosing(path: Path, line: int) -> tuple[int, int, str] | None:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, UnicodeError, SyntaxError):
        return None
    best: tuple[int, int] | None = None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = node.end_lineno or node.lineno
        if node.lineno <= line <= end and (best is None or node.lineno > best[0]):
            best = (node.lineno, end)
    if best is None:
        return None
    return _slice(source, best[0], best[1])


def _slice(source: str, start: int, end: int) -> tuple[int, int, str]:
    end = min(end, start + 79)
    rows = source.splitlines()
    return start, end, "\n".join(rows[start - 1 : end])


def candidate_files(issue: str, workspace: Path) -> list[str]:
    root = workspace.resolve()
    mentioned = [path for path in _paths_mentioned(issue, root) if not _noisy(path)]
    if mentioned:
        return mentioned[:5]
    quoted = [path for path in _files_containing(root, _quotes(issue)) if not _noisy(path)]
    if quoted:
        return quoted[:3]
    return [path for path in _files_by_identifiers(issue, root) if not _noisy(path)][:3]


def locate(workspace: Path, raw: str) -> Path | None:
    raw = raw.replace("\\", "/").strip()
    if not raw or raw == "py":
        return None
    parts = tuple(part for part in raw.split("/") if part not in {"", "."})
    options = [Path(raw), workspace / raw]
    for index in range(len(parts)):
        options.append(workspace.joinpath(*parts[index:]))
    root = workspace.resolve()
    for option in options:
        try:
            resolved = option.resolve()
            resolved.relative_to(root)
        except (OSError, ValueError):
            continue
        if resolved.is_file():
            return resolved
    return None


def _paths_mentioned(issue: str, workspace: Path) -> list[str]:
    found: list[str] = []
    for pattern in (FILE_LINE, PLAIN_PATH):
        for match in pattern.finditer(issue):
            path = locate(workspace, match.group(1) if pattern is FILE_LINE else match.group(0))
            _add(found, workspace, path)
            if len(found) >= 5:
                return found
    return found


def _quotes(issue: str) -> list[str]:
    values: list[str] = []
    for match in QUOTED.finditer(issue):
        text = match.group(1).strip()
        if text and text not in values and not text.endswith(".py"):
            values.append(text)
        if len(values) >= 4:
            break
    return values


def _files_containing(workspace: Path, needles: list[str]) -> list[str]:
    if not needles:
        return []
    found: list[str] = []
    for path in walk_python(workspace):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if any(needle in text for needle in needles):
            _add(found, workspace, path)
        if len(found) >= 3:
            break
    return found


def _files_by_identifiers(issue: str, workspace: Path) -> list[str]:
    tokens = []
    for match in IDENT.finditer(issue):
        token = match.group(0)
        if token not in tokens:
            tokens.append(token)
        if len(tokens) >= 8:
            break
    if not tokens:
        return []
    scored: list[tuple[int, str]] = []
    for path in walk_python(workspace):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        score = sum(1 for token in tokens if token in text)
        if score:
            scored.append((score, path.relative_to(workspace.resolve()).as_posix()))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [name for _, name in scored]


def _noisy(relative: str) -> bool:
    """Тесты, сборка и бенчмарки не должны занимать первые чтения."""
    parts = Path(relative).parts
    name = parts[-1]
    if name in CONFIG_NAMES or name == "conftest.py":
        return True
    if name.startswith("test_") or name.endswith("_test.py"):
        return True
    return any(part in {"tests", "benchmarks", "benchmark"} for part in parts)


def _add(found: list[str], workspace: Path, path: Path | None) -> None:
    if path is None:
        return
    relative = path.resolve().relative_to(workspace.resolve()).as_posix()
    if relative not in found:
        found.append(relative)
