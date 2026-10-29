"""Прогон агента по 36 задачам SWE-bench Verified.

Берёт открытый набор, по несколько задач из каждого репозитория.
Для задачи скачивает репозиторий на base_commit, запускает agent.py
и сохраняет diff. Золотые патчи и скрытые тесты агенту не передаются.

Повторный запуск продолжает с первой незавершённой задачи.
Официальный вердикт «решено / не решено» этот скрипт не ставит:
для него нужен Docker-harness SWE-bench.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / ".bench"
REPOS = BENCH / "repos"
RESULTS = BENCH / "results"
LOGS = BENCH / "logs"
SELECTION = BENCH / "selection.json"
TARGET = 36
PER_REPO = 4
AGENT_TIMEOUT_S = 720


def main() -> int:
    limit, out = _args()
    results = out / "results" if out else RESULTS
    logs = out / "logs" if out else LOGS
    BENCH.mkdir(exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    tasks = load_selection()
    pending = [task for task in tasks if not (results / f"{task['instance_id']}.json").exists()]
    if limit:
        pending = pending[:limit]
    print(f"выбрано {len(tasks)}, осталось {len(pending)}", flush=True)
    python = venv_python()
    env = agent_env()
    for index, task in enumerate(pending, start=1):
        print(f"[{index}/{len(pending)}] {task['instance_id']}", flush=True)
        run_one(task, python, env, results, logs)
    print("прогон выборки завершён", flush=True)
    return 0


def _args() -> tuple[int, Path | None]:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out
    if out is not None and not out.is_absolute():
        out = ROOT / out
    return args.limit, out


def load_selection() -> list[dict]:
    if SELECTION.exists():
        return json.loads(SELECTION.read_text(encoding="utf-8"))
    from datasets import load_dataset

    print("скачиваю список SWE-bench Verified", flush=True)
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    groups: dict[str, list] = defaultdict(list)
    for row in dataset:
        groups[row["repo"]].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: row["instance_id"])

    picked: list[dict] = []
    cursor = 0
    while len(picked) < TARGET:
        grew = False
        for repo in sorted(groups):
            taken = sum(1 for item in picked if item["repo"] == repo)
            if taken >= PER_REPO or cursor >= len(groups[repo]):
                continue
            row = groups[repo][cursor]
            picked.append(
                {
                    "instance_id": row["instance_id"],
                    "repo": row["repo"],
                    "base_commit": row["base_commit"],
                    "problem_statement": row["problem_statement"],
                }
            )
            grew = True
            if len(picked) >= TARGET:
                break
        if not grew:
            break
        cursor += 1
    SELECTION.write_text(json.dumps(picked, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"сохранил выборку: {SELECTION}", flush=True)
    return picked


def run_one(task: dict, python: Path, env: dict[str, str], results: Path, logs: Path) -> None:
    instance_id = task["instance_id"]
    started = time.monotonic()
    record = {
        "instance_id": instance_id,
        "repo": task["repo"],
        "base_commit": task["base_commit"],
    }
    try:
        workspace = prepare_repo(task["repo"], task["base_commit"])
        task_file = BENCH / "tasks" / instance_id / "TASK.md"
        task_file.parent.mkdir(parents=True, exist_ok=True)
        task_file.write_text(task["problem_statement"], encoding="utf-8")
        log_path = logs / f"{instance_id}.log"
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(
                [
                    str(python),
                    str(ROOT / "agent.py"),
                    "--task-file",
                    str(task_file),
                    "--workspace",
                    str(workspace),
                ],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=AGENT_TIMEOUT_S,
                check=False,
            )
        record["exit_code"] = proc.returncode
        record["patch"] = collect_patch(workspace)
    except subprocess.TimeoutExpired:
        record["exit_code"] = 124
        record["patch"] = ""
        record["error"] = "agent timeout"
    except Exception as exc:
        record["exit_code"] = 1
        record["patch"] = ""
        record["error"] = str(exc)
    record["seconds"] = round(time.monotonic() - started, 1)
    record["patch_empty"] = not record["patch"].strip()
    (results / f"{instance_id}.diff").write_text(record["patch"], encoding="utf-8")
    saved = {key: value for key, value in record.items() if key != "patch"}
    (results / f"{instance_id}.json").write_text(
        json.dumps(saved, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    state = "пустой diff" if record["patch_empty"] else f"{len(record['patch'].splitlines())} строк diff"
    print(f"  {state}, {record['seconds']} с, код {record['exit_code']}", flush=True)


def prepare_repo(repo: str, commit: str) -> Path:
    path = REPOS / repo.replace("/", "__")
    if not (path / ".git").exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        git(
            [
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                f"https://github.com/{repo}.git",
                str(path),
            ]
        )
    git(["fetch", "--filter=blob:none", "origin", commit], cwd=path)
    git(["checkout", "-f", commit], cwd=path)
    git(["clean", "-fdx"], cwd=path)
    return path


def collect_patch(workspace: Path) -> str:
    git(["add", "-A"], cwd=workspace)
    patch = git(["diff", "--cached"], cwd=workspace)
    return patch


def git(args: list[str], cwd: Path | None = None) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"git {' '.join(args[:3])} failed: {detail[:500]}")
    return proc.stdout


def agent_env() -> dict[str, str]:
    env = os.environ.copy()
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    env["PYTHONUNBUFFERED"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def venv_python() -> Path:
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    if not python.exists():
        return Path(sys.executable)
    return python


if __name__ == "__main__":
    sys.exit(main())
