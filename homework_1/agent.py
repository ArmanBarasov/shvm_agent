"""Домашнее задание 1: агент уровня 1 для AgentScore (planerverse.ru).

Платформа запускает агента так:
    python /agent/agent.py --task-file /task/TASK.md --workspace /testbed
Агент читает задачу, меняет файлы в /testbed и завершается. Оценивается git diff.
Модель и ключ приходят из окружения: OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL.

Уровень 1 — один запрос, без инструментов:
  1) выбрать по задаче несколько файлов, а не весь репозиторий;
  2) попросить короткую замену фрагмента, а не файл целиком;
  3) не разбирать ответ, если его оборвало по лимиту токенов.
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from openai import OpenAI

# Один файл в промпте и весь промпт. Иначе шлюз отвечает 413.
MAX_FILE_CHARS = 20_000
MAX_CONTEXT_CHARS = 120_000
MAX_FILES = 8

INSTRUCTIONS = """
Исправь код по задаче. Меняй только то, что нужно для исправления.
Не присылай файл целиком. Пришли одну или две короткие замены.
Фрагмент old должен совпасть с текстом файла символ в символ и встретиться один раз.
Без пояснений и без markdown.
<edit path="относительный/путь.py">
<old>
строки, которые уже есть в файле
</old>
<new>
те же строки после исправления
</new>
</edit>
""".strip()

EDIT_RE = re.compile(
    r'<edit path="([^"]+)">\s*<old>\n?(.*?)\n?</old>\s*<new>\n?(.*?)\n?</new>\s*</edit>',
    re.DOTALL,
)
PATH_RE = re.compile(
    r"(?:[\w.-]+/)*[\w.-]+\.(?:py|pyi|txt|rst|cfg|ini|toml|yml|yaml|json|md)"
)


def tracked_files(workspace: Path) -> list[str]:
    """Список файлов, которые видит git. Служебная папка .git сюда не входит."""
    listing = subprocess.run(
        ["git", "ls-files"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
    )
    return [
        line
        for line in listing.stdout.splitlines()
        if line and ".git" not in Path(line).parts
    ]


def read_text(workspace: Path, rel: str) -> str | None:
    path = workspace / rel
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def mentioned_paths(task: str, files: list[str]) -> list[str]:
    """Пути, которые прямо написаны в задаче, и файлы с подходящим именем."""
    found: list[str] = []
    seen: set[str] = set()

    def add(rel: str) -> None:
        if rel not in seen:
            seen.add(rel)
            found.append(rel)

    hints = [hint.lstrip("./") for hint in PATH_RE.findall(task.replace("\\", "/"))]
    for hint in hints:
        for rel in files:
            if rel == hint or rel.endswith("/" + hint):
                add(rel)

    # `stats.average([])` не похоже на путь, но имя до точки — это модуль stats.py.
    for token in re.findall(r"[A-Za-z_]\w*(?:\.\w+)+", task):
        stem = token.split(".", 1)[0]
        for rel in files:
            if Path(rel).stem == stem:
                add(rel)

    for token in re.findall(r"`([^`]+)`", task):
        name = token.strip()
        for rel in files:
            if Path(rel).name == name:
                add(rel)
    return found


def choose_paths(task: str, workspace: Path, files: list[str]) -> list[str]:
    chosen = mentioned_paths(task, files)
    if chosen:
        return chosen[:MAX_FILES]

    # В задаче нет пути: ищем слово из текста внутри файлов, но берём мало.
    words = [
        word
        for word in re.findall(r"[A-Za-z_]\w{3,}", task)
        if word not in {"Error", "Exception", "Traceback", "File", "line"}
    ]
    hits: list[str] = []
    for rel in files:
        text = read_text(workspace, rel)
        if text and any(word in text for word in words[:12]):
            hits.append(rel)
        if len(hits) >= 3:
            break
    return hits


def for_prompt(text: str, task: str) -> str:
    """Большой файл целиком в запрос не кладём: только окно вокруг строки из traceback."""
    if len(text) <= MAX_FILE_CHARS:
        return text
    match = re.search(r"line (\d+)", task)
    lines = text.splitlines()
    center = int(match.group(1)) - 1 if match else 0
    start = max(0, center - 40)
    end = min(len(lines), center + 40)
    return "\n".join(lines[start:end])


def build_context(workspace: Path, task: str) -> str:
    """Текст репозитория для модели: несколько нужных файлов, не весь проект."""
    files = tracked_files(workspace)
    blocks = []
    total = 0
    for rel in choose_paths(task, workspace, files):
        text = read_text(workspace, rel)
        if text is None:
            continue
        block = f'<file path="{rel}">\n{for_prompt(text, task)}\n</file>'
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)


def parse_files(answer: str, workspace: Path) -> dict[str, str]:
    """Короткие замены из ответа. На выход — полное новое содержимое файла."""
    updated: dict[str, str] = {}
    for rel, old, new in EDIT_RE.findall(answer):
        rel = rel.strip().replace("\\", "/")
        current = updated.get(rel)
        if current is None:
            current = read_text(workspace, rel)
        if current is None:
            print(f"нет файла для правки: {rel}", file=sys.stderr)
            continue
        count = current.count(old)
        if count != 1:
            print(f"фрагмент найден {count} раз, пропускаю: {rel}", file=sys.stderr)
            continue
        updated[rel] = current.replace(old, new, 1)
    return updated


def write_files(workspace: Path, files: dict[str, str]) -> list[str]:
    """Записываем файлы только внутри workspace и никогда не трогаем .git."""
    written = []
    for rel, content in files.items():
        target = (workspace / rel).resolve()
        if workspace not in target.parents or ".git" in target.parts:
            print(f"пропускаю путь вне репозитория: {rel}", file=sys.stderr)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(rel)
    return written


def ask(client: OpenAI, model: str, prompt: str):
    response = client.responses.create(model=model, instructions=INSTRUCTIONS, input=prompt)
    usage = response.usage
    print(f"статус {response.status}, токены {usage.input_tokens}/{usage.output_tokens}")
    return response


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    task = Path(args.task_file).read_text(encoding="utf-8")
    client = OpenAI(base_url=args.base_url)  # ключ SDK берёт из OPENAI_API_KEY

    prompt = f"# Задача\n{task}\n\n# Репозиторий\n{build_context(workspace, task)}"
    response = ask(client, args.model, prompt)
    if response.status == "incomplete":
        retry = prompt + (
            "\n\nПрошлый ответ оборвался. Верни одну правку на несколько строк, без целого файла."
        )
        response = ask(client, args.model, retry)
    if response.status == "incomplete":
        print("ответ оборван, файлы не меняю", file=sys.stderr)
        return 0

    files = parse_files(response.output_text or "", workspace)
    written = write_files(workspace, files)
    print("изменены файлы:", ", ".join(written) or "нет")
    return 0


if __name__ == "__main__":
    sys.exit(main())
