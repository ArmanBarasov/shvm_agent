"""Что модели можно менять в дереве.

Читать можно почти всё, кроме `.git`. Писать в тесты и файлы
сборки нельзя, пока текст задачи не называет их путь целиком.

Каталог `django/test` — это код библиотеки, не набор тестов,
поэтому запрещается часть пути `tests`, а не любое имя `test`.
"""

from __future__ import annotations

from pathlib import Path

CONFIG_NAMES = {
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
    "tox.ini",
    "pytest.ini",
}


class GuardError(Exception):
    pass


def resolve_inside(workspace: Path, raw: str) -> Path:
    if not raw or raw.strip() in {".", ""}:
        return workspace.resolve()
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = workspace / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(workspace.resolve())
    except ValueError as exc:
        raise GuardError("path escapes the workspace") from exc
    if ".git" in resolved.relative_to(workspace.resolve()).parts:
        raise GuardError(".git is closed")
    return resolved


def is_test_path(path: Path) -> bool:
    name = path.name
    if name == "conftest.py" or name.startswith("test_") or name.endswith("_test.py"):
        return True
    return "tests" in path.parts


def is_config_path(path: Path) -> bool:
    return path.name in CONFIG_NAMES


def named_in_issue(workspace: Path, path: Path, issue: str) -> bool:
    relative = path.relative_to(workspace.resolve()).as_posix()
    return relative in issue.replace("\\", "/")


def forbidden_reason(path: Path, workspace: Path, issue: str) -> str | None:
    """Почему файл нельзя править. None значит, что правка разрешена."""
    try:
        relative = path.relative_to(workspace.resolve())
    except ValueError:
        return "path escapes the workspace"
    if ".git" in relative.parts:
        return ".git is closed"
    if is_test_path(path) or is_config_path(path):
        if named_in_issue(workspace, path, issue):
            return None
        shown = relative.as_posix()
        return f"{shown} is closed unless the issue names that path"
    return None
