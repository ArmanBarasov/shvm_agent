"""Домашнее задание 1: агент уровня 1 для AgentScore (planerverse.ru).

Платформа запускает агента так:
    python /agent/agent.py --task-file /task/TASK.md --workspace /testbed
Агент читает задачу, меняет файлы в /testbed и завершается. Оценивается git diff.
Модель и ключ приходят из окружения: OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL.

Уровень 1 — без инструментов, один запрос к модели (всё из занятия 1.1):
  1) собрать контекст: задача + файлы проекта;
  2) поставить задачу промптом и задать формат ответа;
  3) разобрать ответ и записать изменённые файлы.
Места, которые нужно дописать, помечены TODO.
"""
import argparse
import os
import sys
from pathlib import Path

from openai import OpenAI

# TODO 1. Инструкции для модели: задача, правила, формат ответа.
# Формат должен быть таким, чтобы программа могла надёжно достать из ответа
# путь и новое содержимое каждого изменённого файла.
INSTRUCTIONS = """..."""


def build_context(workspace: Path) -> str:
    """TODO 2. Собрать из репозитория текст для модели.

    Подсказки:
    - список файлов проекта можно получить командой `git ls-files` (cwd=workspace);
    - берите текстовые файлы с кодом, пропускайте .git и слишком большие файлы;
    - отделяйте файлы друг от друга разделителями с путём, например <file path="...">...</file>;
    - помните про окно контекста и цену: весь репозиторий отправлять не нужно.
    """
    raise NotImplementedError


def parse_files(answer: str) -> dict[str, str]:
    """TODO 3. Достать из ответа модели {путь: новое содержимое файла}.

    Разбор должен соответствовать формату из INSTRUCTIONS.
    """
    raise NotImplementedError


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

    prompt = f"# Задача\n{task}\n\n# Репозиторий\n{build_context(workspace)}"
    response = client.responses.create(model=args.model, instructions=INSTRUCTIONS, input=prompt)
    print(f"статус {response.status}, токены {response.usage.input_tokens}/{response.usage.output_tokens}")

    files = parse_files(response.output_text or "")
    # TODO 4 (по желанию). Если ответ не удалось разобрать — повторить запрос
    # и напомнить модели формат.

    written = write_files(workspace, files)
    print("изменены файлы:", ", ".join(written) or "нет")
    return 0


if __name__ == "__main__":
    sys.exit(main())
