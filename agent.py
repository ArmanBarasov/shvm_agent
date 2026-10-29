"""Точка входа стенда.

Стенд запускает файл так:
python /agent/agent.py --task-file /task/TASK.md --workspace /testbed
Каталог скрипта попадает в sys.path, поэтому пакет harness
импортируется и когда рабочий каталог уже /testbed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from harness.config import ConfigError, load_profile
from harness.loop import run_agent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Починить одну задачу в репозитории")
    parser.add_argument("--task-file", required=True, help="Путь к TASK.md")
    parser.add_argument("--workspace", required=True, help="Корень репозитория задачи")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    log = logging.getLogger("agent")

    try:
        profile = load_profile()
    except ConfigError as exc:
        log.error("%s", exc)
        return 2

    task_file = Path(args.task_file)
    workspace = Path(args.workspace)
    if not task_file.is_file():
        log.error("файл задачи не найден: %s", task_file)
        return 2
    if not workspace.is_dir():
        log.error("репозиторий не найден: %s", workspace)
        return 2

    log.info("профиль %s, модель %s, хост %s", profile.name, profile.model, profile.host)
    try:
        return run_agent(task_file, workspace, profile)
    except Exception:
        log.exception("агент остановился с ошибкой")
        return 0


if __name__ == "__main__":
    sys.exit(main())
