"""Память о наших правках.

Хранится первый текст каждого файла. Откат в конце задачи
возвращает дерево к состоянию до агента и не трогает git.
"""

from __future__ import annotations

from pathlib import Path

from harness.files import write_text


class EditSession:
    def __init__(self) -> None:
        # None означает, что файла не было и мы его создали.
        self._original: dict[Path, str | None] = {}

    def remember(self, path: Path, original: str | None) -> None:
        self._original.setdefault(path, original)

    @property
    def touched(self) -> bool:
        return bool(self._original)

    def paths(self) -> list[Path]:
        return list(self._original)

    def relative_paths(self, workspace: Path) -> list[str]:
        names: list[str] = []
        root = workspace.resolve()
        for path in self._original:
            try:
                names.append(path.resolve().relative_to(root).as_posix())
            except ValueError:
                names.append(str(path))
        return names

    def restore_all(self) -> None:
        for path, original in self._original.items():
            if original is None:
                if path.exists():
                    path.unlink()
                continue
            write_text(path, original)
        self._original.clear()
