"""Инструменты агента.

Один вызов за ход. Схемы собраны из pydantic-моделей, чтобы
описание для модели и проверка аргументов не разъезжались.
"""

from __future__ import annotations

import ast
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from harness.context import Dossier, Scratchpad, clip
from harness.edits import EditSession
from harness.files import read_text, walk_python, write_text
from harness.guard import GuardError, forbidden_reason, is_test_path, resolve_inside

log = logging.getLogger(__name__)

MAX_HUNK_LINES = 40
RUN_TIMEOUT_S = 45


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pattern: str = Field(min_length=2, max_length=200, description="Literal text, not a regex")
    path: str = Field(default=".", max_length=300, description="File or directory inside the repo")


class OutlineArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1, max_length=300)


class ReadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1, max_length=300)
    start: int = Field(default=1, ge=1)
    end: int = Field(default=180, ge=1)


class EditArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1, max_length=300)
    old: str = Field(max_length=4000, description="Exact existing snippet, once in the file")
    new: str = Field(max_length=4000, description="Replacement, at most 40 lines")


class ReplaceArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1, max_length=300)
    start: int = Field(ge=1, description="First line to replace, from the dossier")
    end: int = Field(ge=1, description="Last line to replace, inclusive")
    new: str = Field(max_length=4000, description="New lines only. Do not repeat the old text")


class ReproArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str = Field(min_length=1, max_length=6000, description="Python script, stored outside the repo")


class RunArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: Literal["repro", "pytest"]
    paths: list[str] = Field(default_factory=list, description="One or two test files or node ids")


class FinishArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass
class ToolContext:
    workspace: Path
    issue: str
    edits: EditSession
    scratch: Scratchpad
    dossier: Dossier | None = None


_PYTHON: list[str] | None = None  # conda testbed на стенде, иначе текущий Python


def tool_specs() -> list[dict]:
    rows = [
        (SearchArgs, "search", "Find a literal string. Returns at most 40 hits."),
        (OutlineArgs, "outline", "List top-level classes and functions with line numbers."),
        (ReadArgs, "read", "Read a line range. Lines are prefixed with numbers; do not copy the numbers into edit."),
        (EditArgs, "edit", "Replace one exact snippet, or create a file when old is empty."),
        (
            ReplaceArgs,
            "replace",
            "Replace a line range. Send only the new lines. The old text is not required.",
        ),
        (ReproArgs, "repro", "Write the reproduction script outside the repository."),
        (RunArgs, "run", "Run the repro, or pytest on one or two existing test files."),
        (FinishArgs, "finish", "Submit the current diff once a source edit has landed."),
    ]
    return [_schema(model, name, description) for model, name, description in rows]


def dispatch(name: str, raw: dict | None, ctx: ToolContext) -> str:
    handler = HANDLERS.get(name)
    model = MODELS.get(name)
    if handler is None or model is None:
        return f"error: unknown tool {name}"
    try:
        args = model.model_validate(raw or {})
    except ValidationError as exc:
        err = exc.errors()[0]
        loc = ".".join(str(part) for part in err["loc"]) or "args"
        return f"error: {loc}: {err['msg']}"
    try:
        return handler(ctx, args)
    except GuardError as exc:
        return f"error: {exc}"


def search_tool(ctx: ToolContext, args: SearchArgs) -> str:
    root = resolve_inside(ctx.workspace, args.path)
    if not root.exists():
        return "error: path does not exist"
    hits: list[str] = []
    files = [root] if root.is_file() else walk_python(root)
    for file in files:
        if len(hits) >= 40:
            break
        try:
            text = read_text(file)
        except (OSError, UnicodeError):
            continue
        if args.pattern not in text:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if args.pattern not in line:
                continue
            relative = file.relative_to(ctx.workspace.resolve()).as_posix()
            hits.append(f"{relative}:{number}: {line.strip()[:160]}")
            if len(hits) >= 40:
                break
    if not hits:
        return "no matches"
    return "\n".join(hits)


def outline_tool(ctx: ToolContext, args: OutlineArgs) -> str:
    path = _python_file(ctx, args.path)
    try:
        tree = ast.parse(read_text(path))
    except SyntaxError as exc:
        return f"error: syntax at line {exc.lineno}"
    rows: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            rows.append(f"class {node.name}  line {node.lineno}")
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    rows.append(f"  def {child.name}({_arg_names(child)})  line {child.lineno}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            rows.append(f"def {node.name}({_arg_names(node)})  line {node.lineno}")
    if not rows:
        return "no top-level functions or classes"
    return "\n".join(rows[:200])


def read_tool(ctx: ToolContext, args: ReadArgs) -> str:
    path = resolve_inside(ctx.workspace, args.path)
    if not path.is_file():
        return "error: not a file"
    try:
        lines = read_text(path).splitlines()
    except UnicodeError:
        return "error: file is not utf-8"
    start = args.start
    end = min(len(lines), args.end)
    if start > len(lines):
        return f"error: file has {len(lines)} lines"
    if end < start:
        return "error: empty range"
    clamped = False
    if end - start + 1 > 180:
        end = start + 179
        clamped = True
    selected = lines[start - 1 : end]
    if ctx.dossier is not None:
        relative = path.relative_to(ctx.workspace.resolve()).as_posix()
        if ctx.dossier.has(relative, start, end):
            return "error: this range is already in the dossier. Call edit and copy the exact-text block."
        ctx.dossier.store(relative, start, end, "\n".join(selected))
    chunk = [f"{number:5}|{selected[number - start]}" for number in range(start, end + 1)]
    if clamped:
        chunk.append("[range clamped to 180 lines]")
    return "\n".join(chunk)


def edit_tool(ctx: ToolContext, args: EditArgs) -> str:
    path = resolve_inside(ctx.workspace, args.path)
    reason = forbidden_reason(path, ctx.workspace, ctx.issue)
    if reason:
        return "error: " + reason
    if path.suffix not in {".py", ".pyi"}:
        return "error: only .py and .pyi files can be edited"
    if path.exists() and args.old == "":
        return "error: old snippet is empty"
    if _hunk_too_wide(args.old, args.new):
        return f"error: hunk is larger than {MAX_HUNK_LINES} lines; split it"
    if path.exists():
        try:
            text = read_text(path)
        except UnicodeError:
            return "error: file is not utf-8"
        if "\x00" in text:
            return "error: binary file"
        count = text.count(args.old)
        if count == 0:
            stripped = _strip_line_numbers(args.old)
            if stripped != args.old and text.count(stripped) == 1:
                args = EditArgs(path=args.path, old=stripped, new=_strip_line_numbers(args.new))
                count = 1
        if count == 0:
            return "error: snippet not found; copy it exactly, without line numbers"
        if count > 1:
            return f"error: snippet found {count} times; include more surrounding lines"
        ctx.edits.remember(path, text)
        write_text(path, text.replace(args.old, args.new, 1))
    else:
        if args.old != "":
            return "error: file does not exist; to create it, pass an empty old snippet"
        if not path.parent.exists():
            return "error: directory does not exist"
        ctx.edits.remember(path, None)
        write_text(path, args.new)
    relative = path.relative_to(ctx.workspace.resolve()).as_posix()
    if ctx.dossier is not None:
        ctx.dossier.drop(relative)
    return f"edited {relative}"


def repro_tool(ctx: ToolContext, args: ReproArgs) -> str:
    path = repro_file()
    write_text(path, args.content if args.content.endswith("\n") else args.content + "\n")
    ctx.scratch.add("repro script written")
    return f"wrote repro at {path}"


def run_tool(ctx: ToolContext, args: RunArgs) -> str:
    if args.target == "repro":
        script = repro_file()
        if not script.is_file():
            return "error: write a repro first"
        code, output = execute(python_argv() + [str(script)], ctx.workspace)
        if _unusable(output):
            ctx.scratch.add("repro could not import the project")
            return f"repro exit {code} (environment)\n{output}"
        status = "fail" if code else "pass"
        if ctx.edits.touched:
            ctx.scratch.repro_after = status
        else:
            ctx.scratch.repro_base = status
        return f"repro exit {code} ({status})\n{output}"

    if not args.paths or len(args.paths) > 2:
        return "error: pass one or two test files"
    targets: list[str] = []
    for raw in args.paths:
        targets.append(_pytest_target(ctx, raw))
    argv = python_argv() + [
        "-m",
        "pytest",
        "-q",
        "--tb=line",
        "-p",
        "no:cacheprovider",
        *targets,
    ]
    code, output = execute(argv, ctx.workspace)
    if "No module named pytest" in output or _unusable(output):
        return f"error: the environment could not run this test\n{output}"
    note_pytest(ctx, ", ".join(targets), code)
    state = "pass" if code == 0 else "fail"
    ctx.scratch.add(f"pytest {state}: {', '.join(targets)}")
    return f"pytest exit {code} ({state})\n{output}"


def replace_tool(ctx: ToolContext, args: ReplaceArgs) -> str:
    """Меняет диапазон строк. Старый текст модели копировать не нужно."""
    path = resolve_inside(ctx.workspace, args.path)
    reason = forbidden_reason(path, ctx.workspace, ctx.issue)
    if reason:
        return "error: " + reason
    if path.suffix not in {".py", ".pyi"}:
        return "error: only .py and .pyi files can be edited"
    if args.end < args.start:
        return "error: end is before start"
    span = args.end - args.start + 1
    new_lines = args.new.splitlines()
    if span > 80 or len(new_lines) > 80:
        return "error: range is larger than 80 lines"
    try:
        original = read_text(path)
    except UnicodeError:
        return "error: file is not utf-8"
    lines = original.splitlines()
    if args.start > len(lines):
        return f"error: file has {len(lines)} lines"
    end = min(args.end, len(lines))
    ctx.edits.remember(path, original)
    updated = lines[: args.start - 1] + new_lines + lines[end:]
    text = "\n".join(updated)
    if original.endswith("\n"):
        text += "\n"
    write_text(path, text)
    relative = path.relative_to(ctx.workspace.resolve()).as_posix()
    if ctx.dossier is not None:
        ctx.dossier.drop(relative)
    try:
        compile(text, str(path), "exec")
    except SyntaxError as exc:
        return f"replaced {relative} but syntax error at line {exc.lineno}"
    return f"replaced {relative} lines {args.start}-{end}"


def finish_tool(ctx: ToolContext, args: FinishArgs) -> str:
    # Решение принимает цикл после критика. Сюда попадаем только если
    # фаза почему-то пропустила вызов напрямую.
    return "finish requested"


HANDLERS = {
    "search": search_tool,
    "outline": outline_tool,
    "read": read_tool,
    "edit": edit_tool,
    "replace": replace_tool,
    "repro": repro_tool,
    "run": run_tool,
    "finish": finish_tool,
}

MODELS = {
    "search": SearchArgs,
    "outline": OutlineArgs,
    "read": ReadArgs,
    "edit": EditArgs,
    "replace": ReplaceArgs,
    "repro": ReproArgs,
    "run": RunArgs,
    "finish": FinishArgs,
}


def repro_file() -> Path:
    if os.name == "nt":
        default = Path.home() / "AppData" / "Local" / "shvm-agent"
    else:
        default = Path("/tmp/agent")
    root = Path(os.environ.get("AGENT_TMP", default))
    root.mkdir(parents=True, exist_ok=True)
    return root / "repro.py"


def python_argv() -> list[str]:
    global _PYTHON
    if _PYTHON is not None:
        return list(_PYTHON)
    conda = shutil.which("conda")
    if conda:
        try:
            probe = subprocess.run(
                [conda, "run", "-n", "testbed", "python", "-c", "import sys"],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired):
            probe = None
        if probe is not None and probe.returncode == 0:
            log.info("интерпретатор: conda env testbed")
            _PYTHON = [conda, "run", "--no-capture-output", "-n", "testbed", "python"]
            return list(_PYTHON)
    log.info("интерпретатор: %s", sys.executable)
    _PYTHON = [sys.executable]
    return list(_PYTHON)


def execute(argv: list[str], cwd: Path, timeout: int = RUN_TIMEOUT_S) -> tuple[int, str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(cwd), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    for name in ("OPENAI_API_KEY", "OLLAMA_API_KEY"):
        env.pop(name, None)
    proc = subprocess.Popen(
        argv,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    try:
        output, _ = proc.communicate(timeout=timeout)
        code = proc.returncode if proc.returncode is not None else 1
    except subprocess.TimeoutExpired:
        proc.kill()
        output, _ = proc.communicate()
        code = 124
        output = (output or "") + f"\n[timed out after {timeout}s]"
    return code, clip(output or "")


def _python_file(ctx: ToolContext, raw: str) -> Path:
    path = resolve_inside(ctx.workspace, raw)
    if not path.is_file():
        raise GuardError("not a file")
    if path.suffix != ".py":
        raise GuardError("outline expects a .py file")
    return path


def _pytest_target(ctx: ToolContext, raw: str) -> str:
    file_part = raw.split("::", 1)[0]
    path = resolve_inside(ctx.workspace, file_part)
    if not path.is_file():
        raise GuardError("pytest target is not a file")
    if not is_test_path(path):
        raise GuardError("run a test file, not the source")
    if "::" in raw:
        node = raw.split("::", 1)[1]
        return path.relative_to(ctx.workspace.resolve()).as_posix() + "::" + node
    return path.relative_to(ctx.workspace.resolve()).as_posix()


def _arg_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    names = [arg.arg for arg in node.args.args]
    return ", ".join(names)


def _strip_line_numbers(snippet: str) -> str:
    rows = []
    for line in snippet.splitlines():
        match = re.match(r"^\s*\d+\|(.*)$", line)
        rows.append(match.group(1) if match else line)
    return "\n".join(rows)


def note_pytest(ctx: ToolContext, key: str, code: int) -> None:
    """Красный тест бага diff не отменяет.

    Откат нужен только когда тот же узел до правки был зелёным:
    это регрессия PASS_TO_PASS, она обнуляет вторичный балл.
    """
    if code == 0 and not ctx.edits.touched:
        ctx.scratch.passed_before.add(key)
        return
    if code != 0 and key in ctx.scratch.passed_before:
        ctx.scratch.regression = True


def _unusable(output: str) -> bool:
    return any(marker in output for marker in ("ModuleNotFoundError", "ImportError", "No module named"))


def _hunk_too_wide(old: str, new: str) -> bool:
    return max(len(old.splitlines()), len(new.splitlines())) > MAX_HUNK_LINES


def _schema(model: type[BaseModel], name: str, description: str) -> dict:
    parameters = model.model_json_schema()
    parameters.pop("title", None)
    parameters.pop("description", None)
    parameters["additionalProperties"] = False
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }
