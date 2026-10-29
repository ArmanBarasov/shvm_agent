"""Тексты для модели.

Промпты на английском: задачи SWE-bench и вызовы инструментов
так модель выполняет устойчивее. Комментарии в коде — на русском,
чтобы агента можно было читать.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You fix one bug in a repository. The workspace is the current directory.
You receive the issue, a dossier of file text you already read, and the last tool result.

Use exactly one tool per reply. Keep arguments small. Do not write a long explanation.

Rules:
- Read the production file before editing. A read returns at most 180 lines, with line numbers.
- The dossier keeps that text. Edit from the dossier. Do not re-read a range that is already there.
- Prefer replace. It takes path, start, end and only the new lines. Do not send the old text.
- Use edit only when replace is not enough. Copy the snippet without line numbers.
- The snippet must occur once and stay within 40 lines.
- Do not edit tests, pyproject.toml, setup.cfg, setup.py, tox.ini, or pytest.ini
  unless the issue names that exact file.
- Do not reformat unrelated code.
- A reproduction script is optional and lives outside the repo.
  An import error means this environment cannot load the project. It is not a verdict on the fix.
- If the issue names a test file, you may run that file. Never run the whole suite.
- Call finish once a source edit has landed. Do not wait for a perfect repro.
- If a result says the reply was cut off, send a shorter tool call.
- Do not repeat the same tool call.
"""


def status_text(*, has_read: bool, has_edit: bool) -> str:
    if has_edit:
        return (
            "A source edit is already in the tree. "
            "Call finish. If a named test file exists and pytest is available, you may run it first. "
            "An import error is not a reason to undo the edit."
        )
    if not has_read:
        return (
            "Read the production file from the candidates or the issue. "
            "Skip tests, setup.py, and benchmarks unless the bug is inside them."
        )
    return (
        "Call replace now. Use the path and line numbers shown in the dossier. "
        "Put only the new lines in new. Do not copy the old text."
    )
