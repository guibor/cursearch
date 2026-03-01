import os
import unittest
from pathlib import Path
from unittest import mock


def load_cursearch_namespace():
    """Load cursearch functions without executing main()."""
    source_path = Path(__file__).resolve().parents[1] / "cursearch.py"
    code = source_path.read_text(encoding="utf-8")
    code = code.rsplit("\nmain()", 1)[0] + "\n"
    ns = {}
    exec(code, ns)
    return ns


class TestSmartCaseOrderedMatching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_cursearch_namespace()

    def test_empty_query_matches_all(self):
        self.assertTrue(self.ns["ordered_tokens_match"]("anything here", ""))

    def test_ordered_tokens_gap_allowed(self):
        self.assertTrue(self.ns["ordered_tokens_match"]("alpha ... many words ... beta", "alpha beta"))
        self.assertFalse(self.ns["ordered_tokens_match"]("beta then alpha", "alpha beta"))

    def test_lowercase_token_is_case_insensitive(self):
        self.assertTrue(self.ns["ordered_tokens_match"]("Reversible Street Alert", "reversible street"))

    def test_uppercase_token_is_case_sensitive(self):
        self.assertTrue(self.ns["ordered_tokens_match"]("Alpha Beta", "Alpha Beta"))
        self.assertFalse(self.ns["ordered_tokens_match"]("alpha beta", "Alpha Beta"))

    def test_full_substring_per_token_not_char_fuzzy(self):
        self.assertFalse(self.ns["ordered_tokens_match"]("reversible street", "rvrsble"))


class TestSkillLaunchUniqueness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_cursearch_namespace()

    def test_launch_create_skills_deduplicates_paths(self):
        launch = self.ns["launch_create_skills_agent"]

        with mock.patch("os.execvp") as execvp_mock, mock.patch("sys.stderr"):
            launch(["./a.jsonl", "./a.jsonl", "./b.jsonl"])

        execvp_mock.assert_called_once()
        _, argv = execvp_mock.call_args[0]
        self.assertEqual(argv[0], "agent")
        self.assertEqual(argv[1], "--plan")
        prompt = argv[2]
        self.assertIn("- " + os.path.abspath("./a.jsonl"), prompt)
        self.assertIn("- " + os.path.abspath("./b.jsonl"), prompt)
        self.assertEqual(prompt.count(os.path.abspath("./a.jsonl")), 1)

