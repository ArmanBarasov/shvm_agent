"""Цикл одной задачи.

Один бюджет на все инструменты. Фаза не закрывает ход:
модель может читать и править, пока не кончились вызовы или время.
За 20 секунд до конца новая правка не начинается, уже лежащий
в дереве diff сохраняется.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from harness.client import ModelClient, ModelError
from harness.config import Profile
from harness.context import Transcript
from harness.critic import remove_junk, review, shutdown
from harness.edits import EditSession
from harness.hints import candidate_files, focus_window
from harness.prompts import SYSTEM_PROMPT, status_text
from harness.tools import ToolContext, dispatch, tool_specs

log = logging.getLogger(__name__)

CALL_BUDGET = 18


def run_agent(
    task_file: Path,
    workspace: Path,
    profile: Profile,
    client: ModelClient | None = None,
) -> int:
    issue = task_file.read_text(encoding="utf-8")
    workspace = workspace.resolve()
    transcript = Transcript(issue)
    ctx = ToolContext(
        workspace=workspace,
        issue=issue,
        edits=EditSession(),
        scratch=transcript.scratch,
        dossier=transcript.dossier,
    )
    model = client or ModelClient(profile)
    specs = tool_specs()
    failures = 0
    calls = 0
    last_signature: str | None = None
    started = time.monotonic()
    _note_candidates(issue, workspace, transcript)

    def remaining() -> float:
        return profile.task_timeout_s - (time.monotonic() - started)

    while remaining() > 20 and calls < CALL_BUDGET:
        messages = transcript.build(SYSTEM_PROMPT, _hint(ctx))
        try:
            completion = _ask(model, transcript, messages, specs, remaining())
        except ModelError as exc:
            if exc.fold and transcript.fold():
                log.info("сжал историю: %s", exc)
                continue
            failures += 1
            log.warning("модель недоступна: %s", exc)
            if failures >= 3:
                break
            continue
        except Exception:
            failures += 1
            log.exception("сбой шага")
            if failures >= 3:
                break
            continue

        if completion is None or completion.tool_name is None or completion.tool_call_id is None:
            transcript.note("The reply was cut off or had no tool call. Send one shorter tool call.")
            calls += 1
            continue

        signature = f"{completion.tool_name}:{_stable(completion.tool_args)}"
        if signature == last_signature:
            log.info("повтор %s", completion.tool_name)
            transcript.note("That call was repeated. Choose a different action.")
            last_signature = None
            calls += 1
            continue
        last_signature = signature

        calls += 1
        log.info("ход %s -> %s", calls, completion.tool_name)
        if completion.tool_name == "finish":
            remove_junk(workspace)
            report = review(ctx)
            if report.ok:
                log.info("критик принял diff")
                return shutdown(ctx, revert=False)
            transcript.note("critic: " + report.text)
            log.info("критик отклонил finish: %s", report.text)
            continue

        if completion.tool_name == "read" and ctx.scratch.reads >= 4:
            result = "error: stop reading. Call replace with the line numbers from the dossier."
        else:
            result = dispatch(completion.tool_name, completion.tool_args, ctx)
            if completion.tool_name == "read" and not result.startswith("error"):
                ctx.scratch.reads += 1
        transcript.note(result)
        transcript.scratch.add(completion.tool_name)
        log.info("результат: %s", result.splitlines()[0][:200])
        if result.startswith("replaced ") and "syntax error" not in result:
            remove_junk(workspace)
            report = review(ctx)
            if report.ok:
                log.info("критик принял diff")
                return shutdown(ctx, revert=False)

    return _close(ctx)


def _close(ctx: ToolContext) -> int:
    remove_junk(ctx.workspace)
    report = review(ctx)
    log.info("сдаём diff: %s", report.text)
    # Регрессия уже зелёного теста стирает правку. Всё остальное оставляем.
    return shutdown(ctx, revert=ctx.scratch.regression)


def _hint(ctx: ToolContext) -> str:
    return status_text(has_read=bool(ctx.dossier and ctx.dossier.windows), has_edit=ctx.edits.touched)


def _note_candidates(issue: str, workspace: Path, transcript: Transcript) -> None:
    found = candidate_files(issue, workspace)
    window = focus_window(issue, workspace, found)
    if window is not None:
        path, start, end, raw = window
        transcript.dossier.store(path, start, end, raw)
        transcript.scratch.add(f"replace {path} lines {start}-{end}")
        log.info("функция для замены: %s строки %s-%s", path, start, end)
        return
    if found:
        transcript.scratch.add("candidates: " + ", ".join(found))
        log.info("файлы из текста задачи: %s", ", ".join(found))
        return
    log.info("в тексте нет боевого файла, начнём с поиска")


def _ask(model, transcript: Transcript, messages, specs, remaining_s: float):
    timeout = min(90.0, max(5.0, remaining_s - 5))
    completion = model.complete(messages, specs, timeout=timeout)
    if not completion.truncated:
        return completion
    if transcript.shrink_last_observation():
        log.info("ответ обрезан, сжимаю последний результат")
        messages = transcript.build(SYSTEM_PROMPT, messages[1]["content"].split("\n\n", 1)[0])
        completion = model.complete(messages, specs, timeout=timeout)
    if completion.truncated:
        return None
    return completion


def _stable(args: dict | None) -> str:
    return json.dumps(args or {}, sort_keys=True, ensure_ascii=False)
