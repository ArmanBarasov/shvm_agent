"""Обход репозитория и чтение файлов без смены переводов строк.

Обычный `Path.read_text` на Windows превращает CRLF в LF. Для diff
это уже правка, поэтому файл открывается с `newline=""`.
"""

from __future__ import annotations

import os
from pathlib import Path

SKIP_DIRS = {
    ".git",
    ".hg",
    ".tox",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".pytest_cache",
    ".mypy_cache",
}

MAX_FILE_BYTES = 512_000


def read_text(path: Path) -> str:
    # newline="" есть у open() с 3.11. У Path.read_text его ещё нет.
    with path.open(encoding="utf-8", newline="") as handle:
        return handle.read()


def write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def walk_python(root: Path, limit: int = 3000) -> list[Path]:
    """Python-файлы под каталогом, без служебных папок и больших файлов."""
    if root.is_file():
        return [root] if root.suffix == ".py" else []

    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name
            for name in dirnames
            if name not in SKIP_DIRS and not name.startswith(".")
        ]
        for name in filenames:
            if not name.endswith(".py"):
                continue
            path = Path(dirpath) / name
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            found.append(path)
            if len(found) >= limit:
                return found
    return found
