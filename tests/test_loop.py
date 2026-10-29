"""Прогон на подставной модели.

Скрипт отвечает вместо DeepSeek, поэтому тест не ходит в сеть
и проверяет, что контроллер доходит до правки и сдачи без фаз.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from harness.client import Completion
from harness.config import Profile
from harness.files import write_text
from harness.loop import run_agent


def tool_call(name: str, args: dict) -> Completion:
    call_id = "call-" + name
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ],
    }
    return Completion(message, name, args, call_id, truncated=False)


class ScriptedModel:
    def complete(self, messages, tools, timeout):
        text = messages[1]["content"]
        if "edited pkg.py" in text:
            return tool_call("finish", {})
        if "return a - b" not in text:
            return tool_call("read", {"path": "pkg.py", "start": 1, "end": 20})
        return tool_call("edit", {"path": "pkg.py", "old": "return a - b", "new": "return a + b"})


class LoopTests(unittest.TestCase):
    def test_scripted_run_fixes_the_function(self):
        import harness.tools as tools

        tools._PYTHON = [sys.executable]
        root = Path(tempfile.mkdtemp())
        write_text(root / "pkg.py", "def add(a, b):\n    return a - b\n")
        task = root / "TASK.md"
        write_text(task, 'add() subtracts.\nFile "pkg.py", line 2, in add\n')
        tmp = Path(tempfile.mkdtemp())
        profile = Profile(
            name="dev",
            base_url="https://ollama.com/v1",
            api_key="test",
            model="deepseek-v4.1-flash",
            reasoning_effort="high",
            temperature=0.2,
            max_output_tokens=4096,
            max_request_bytes=4_000_000,
            task_timeout_s=120,
        )
        with unittest.mock.patch.dict(os.environ, {"AGENT_TMP": str(tmp)}):
            code = run_agent(task, root, profile, client=ScriptedModel())
        self.assertEqual(code, 0)
        self.assertIn("return a + b", (root / "pkg.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
