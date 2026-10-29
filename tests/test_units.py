"""Поведение без сети: границы, профиль, разбор ответа модели."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from harness.client import message_dict, parse_completion
from harness.config import CONTEST_MODEL, ConfigError, load_profile
from harness.context import Scratchpad, Transcript
from harness.critic import review
from harness.edits import EditSession
from harness.files import write_text
from harness.guard import forbidden_reason
from harness.hints import candidate_files
from harness.prompts import SYSTEM_PROMPT
from harness.tools import EditArgs, ToolContext, edit_tool, note_pytest


def repo(files: dict[str, str]) -> Path:
    root = Path(tempfile.mkdtemp())
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write_text(path, text)
    return root


class GuardTests(unittest.TestCase):
    def test_library_named_test_can_be_edited(self):
        root = repo({"django/test/client.py": "x = 1\n"})
        path = root / "django" / "test" / "client.py"
        self.assertIsNone(forbidden_reason(path, root, "bug"))

    def test_test_suite_stays_closed(self):
        root = repo({"tests/test_client.py": "def test_a():\n    pass\n"})
        path = root / "tests" / "test_client.py"
        reason = forbidden_reason(path, root, "client is wrong")
        self.assertIn("closed", reason or "")

    def test_issue_can_name_a_test_file(self):
        root = repo({"tests/test_client.py": "x = 1\n"})
        path = root / "tests" / "test_client.py"
        issue = "Fix tests/test_client.py itself"
        self.assertIsNone(forbidden_reason(path, root, issue))

    def test_edit_rejects_a_wide_hunk_and_a_repeated_snippet(self):
        root = repo({"pkg.py": "anchor = 1\n" + "\n".join(f"x{i} = {i}" for i in range(41)) + "\n"})
        ctx = ToolContext(root, "pkg.py", EditSession(), Scratchpad())
        wide = "\n".join(f"x{i} = {i}" for i in range(41))
        result = edit_tool(ctx, EditArgs(path="pkg.py", old=wide, new="y = 1"))
        self.assertIn("40 lines", result)
        self.assertFalse(ctx.edits.touched)

        write_text(root / "pkg.py", "value = 1\nvalue = 1\n")
        result = edit_tool(ctx, EditArgs(path="pkg.py", old="value = 1", new="value = 2"))
        self.assertIn("2 times", result)

    def test_restore_returns_the_file_we_had_at_the_start(self):
        root = repo({"pkg.py": "a = 1\n"})
        path = root / "pkg.py"
        session = EditSession()
        session.remember(path, "a = 1\n")
        write_text(path, "a = 2\n")
        session.restore_all()
        self.assertEqual(path.read_text(encoding="utf-8"), "a = 1\n")
        self.assertFalse(session.touched)


class HintTests(unittest.TestCase):
    def test_traceback_wins_over_a_search(self):
        root = repo(
            {
                "pkg/mod.py": "def annotate(self):\n    return 1\n",
                "pkg/other.py": "def unused():\n    return 0\n",
            }
        )
        issue = 'Traceback\nFile "pkg/mod.py", line 1, in annotate\n'
        self.assertEqual(candidate_files(issue, root), ["pkg/mod.py"])

    def test_quoted_error_finds_a_file_when_no_path_is_given(self):
        root = repo({"pkg/mod.py": "raise ValueError('needle-text-unique')\n"})
        issue = "Users see `needle-text-unique` and nothing else."
        self.assertEqual(candidate_files(issue, root), ["pkg/mod.py"])

    def test_setup_and_tests_do_not_crowd_out_the_library(self):
        root = repo(
            {
                "setup.py": "x = 1\n",
                "pkg/mod.py": "def annotate(self):\n    return 1\n",
                "tests/test_mod.py": "def test_a():\n    pass\n",
            }
        )
        issue = 'File "setup.py", line 1\nFile "tests/test_mod.py", line 1\nFile "pkg/mod.py", line 1\n'
        self.assertEqual(candidate_files(issue, root), ["pkg/mod.py"])


class CriticTests(unittest.TestCase):
    def test_a_failed_repro_does_not_reject_the_edit(self):
        root = repo({"pkg.py": "a = 1\n"})
        path = root / "pkg.py"
        session = EditSession()
        session.remember(path, "a = 1\n")
        write_text(path, "a = 2\n")
        scratch = Scratchpad()
        scratch.repro_after = "fail"
        ctx = ToolContext(root, "bug", session, scratch)
        self.assertTrue(review(ctx).ok)

    def test_a_red_test_reverts_only_when_it_was_green_before(self):
        root = repo({"pkg.py": "a = 1\n"})
        session = EditSession()
        scratch = Scratchpad()
        ctx = ToolContext(root, "bug", session, scratch)
        note_pytest(ctx, "tests/test_pkg.py", 1)
        self.assertFalse(scratch.regression)
        note_pytest(ctx, "tests/test_pkg.py", 0)
        self.assertIn("tests/test_pkg.py", scratch.passed_before)
        session.remember(root / "pkg.py", "a = 1\n")
        note_pytest(ctx, "tests/test_pkg.py", 1)
        self.assertTrue(scratch.regression)


class ProfileTests(unittest.TestCase):
    def test_gateway_variables_select_contest(self):
        env = {
            "OPENAI_BASE_URL": "http://agent-bench-model-gateway:3130/v1",
            "OPENAI_API_KEY": "temp",
            "OPENAI_MODEL": CONTEST_MODEL,
        }
        with mock.patch.dict(os.environ, env, clear=True):
            profile = load_profile()
        self.assertEqual(profile.name, "contest")
        self.assertEqual(profile.max_output_tokens, 4096)
        self.assertEqual(profile.reasoning_effort, "high")

    def test_contest_refuses_another_model_and_ollama(self):
        env = {
            "AGENT_PROFILE": "contest",
            "OPENAI_BASE_URL": "https://ollama.com/v1",
            "OPENAI_API_KEY": "temp",
            "OPENAI_MODEL": CONTEST_MODEL,
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ConfigError):
                load_profile()

        env["OPENAI_BASE_URL"] = "http://gateway:3130/v1"
        env["OPENAI_MODEL"] = "other-model"
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ConfigError):
                load_profile()

    def test_dev_profile_uses_ollama_and_can_relax_the_cap(self):
        env = {"AGENT_PROFILE": "dev", "OLLAMA_API_KEY": "temp", "MATCH_LIMITS": "0"}
        with mock.patch.dict(os.environ, env, clear=True):
            profile = load_profile()
        self.assertEqual(profile.host, "ollama.com")
        self.assertEqual(profile.model, CONTEST_MODEL)
        self.assertEqual(profile.max_output_tokens, 8192)


class ClientParseTests(unittest.TestCase):
    def test_broken_tool_json_is_not_executed(self):
        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path":'},
                }
            ],
        }
        parsed = parse_completion(message, "length")
        self.assertTrue(parsed.truncated)
        self.assertIsNone(parsed.tool_name)

    def test_complete_tool_json_is_kept_when_reasoning_hit_the_cap(self):
        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path": "a.py"}'},
                },
                {
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                },
            ],
        }
        parsed = parse_completion(message, "length")
        self.assertFalse(parsed.truncated)
        self.assertEqual(parsed.tool_name, "read")
        self.assertEqual(len(parsed.message["tool_calls"]), 1)

    def test_reasoning_is_copied_onto_the_assistant_message(self):
        call = SimpleNamespace(id="c1", function=SimpleNamespace(name="finish", arguments="{}"))
        message = SimpleNamespace(
            content="",
            reasoning_content="short thought",
            tool_calls=[call],
            model_extra={},
        )
        data = message_dict(message)
        self.assertEqual(data["reasoning_content"], "short thought")
        self.assertEqual(data["tool_calls"][0]["function"]["name"], "finish")


class TranscriptTests(unittest.TestCase):
    def test_request_keeps_the_dossier_not_old_turns(self):
        transcript = Transcript("issue")
        transcript.note("first result")
        transcript.note("second result")
        messages = transcript.build(SYSTEM_PROMPT, "Edit from the dossier.")
        self.assertEqual(len(messages), 2)
        self.assertIn("second result", messages[1]["content"])
        self.assertNotIn("first result", messages[1]["content"])

    def test_build_includes_the_issue_and_the_file_text(self):
        transcript = Transcript("the bug is here")
        transcript.dossier.store("pkg.py", 1, 1, "return a - b")
        messages = transcript.build(SYSTEM_PROMPT, "Edit from the dossier.")
        self.assertIn("the bug is here", messages[1]["content"])
        self.assertIn("return a - b", messages[1]["content"])
        self.assertNotIn("Phase:", messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
