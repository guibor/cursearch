import os
import json
import sqlite3
import tempfile
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


class TestFullSearchIndexing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_cursearch_namespace()

    def test_build_session_search_record_indexes_full_transcript(self):
        huge_prefix = "x" * 350_000
        transcript = (
            '{"role":"user","message":{"content":[{"type":"text","text":"'
            + huge_prefix
            + ' tail_marker_query"}]}}\n'
        )
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tmp:
            tmp.write(transcript)
            tmp_path = tmp.name

        try:
            record = self.ns["build_session_search_record"](tmp_path)
        finally:
            os.unlink(tmp_path)

        self.assertIn("tail_marker_query", record["searchable_all"])
        self.assertTrue(
            self.ns["ordered_tokens_match"](record["searchable_all"], "tail_marker_query")
        )

    def test_make_search_excerpt_centers_near_match(self):
        excerpt = self.ns["make_search_excerpt"](
            "alpha beta gamma delta epsilon zeta eta theta",
            "epsilon",
            max_chars=20,
        )
        self.assertIn("epsilon", excerpt)

    def test_score_query_text_prefers_more_matching_tokens(self):
        scorer = self.ns["_score_query_text"]
        tokens = self.ns["parse_query_tokens"]("tsr tickets")
        both = scorer("tsr tickets discussed here", "tsr tickets", tokens)
        one = scorer("tsr only", "tsr tickets", tokens)
        none = scorer("nothing useful", "tsr tickets", tokens)
        self.assertGreater(both, one)
        self.assertGreater(one, none)
        self.assertEqual(none, 0)

    def test_preview_session_handles_empty_selection(self):
        with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
            self.ns["preview_session"]("", latest_match=True)

    def test_parse_cursor_store_reads_json_messages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "store.db"
            conn = sqlite3.connect(str(db_path))
            conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
            conn.execute("CREATE TABLE blobs (id TEXT PRIMARY KEY, data BLOB)")
            meta = {
                "agentId": "11111111-2222-3333-4444-555555555555",
                "latestRootBlobId": "root",
                "name": "Test Session",
                "createdAt": 1734465600000,
            }
            conn.execute("INSERT INTO meta (key, value) VALUES ('0', ?)", (json.dumps(meta).encode().hex(),))
            conn.execute(
                "INSERT INTO blobs (id, data) VALUES (?, ?)",
                (
                    "blob1",
                    json.dumps({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "<user_info>Workspace Path: /Users/md/code/demo</user_info>"},
                            {"type": "text", "text": "tsr tickets"},
                        ],
                    }).encode("utf-8"),
                ),
            )
            conn.execute(
                "INSERT INTO blobs (id, data) VALUES (?, ?)",
                (
                    "blob2",
                    json.dumps({
                        "role": "assistant",
                        "content": [
                            {"type": "reasoning", "text": "thinking..."},
                            {"type": "text", "text": "I found the tickets."},
                        ],
                    }).encode("utf-8"),
                ),
            )
            conn.commit()
            conn.close()

            messages = self.ns["parse_cursor_store"](db_path)
            self.assertEqual(messages[0][0], "user")
            self.assertIn("Workspace Path: /Users/md/code/demo", messages[0][1])
            self.assertIn("tsr tickets", messages[0][1])
            self.assertEqual(messages[1][0], "assistant")

            record = self.ns["build_session_search_record"](db_path)
            self.assertEqual(record["source"], "cursor_store")
            self.assertEqual(record["workspace_path"], "/Users/md/code/demo")
            self.assertIn("tsr tickets", record["searchable_all"])

    def test_parse_cursor_store_skips_non_message_lists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "store.db"
            conn = sqlite3.connect(str(db_path))
            conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
            conn.execute("CREATE TABLE blobs (id TEXT PRIMARY KEY, data BLOB)")
            meta = {
                "agentId": "11111111-2222-3333-4444-555555555555",
                "latestRootBlobId": "root",
                "name": "Test Session",
                "createdAt": 1734465600000,
            }
            conn.execute("INSERT INTO meta (key, value) VALUES ('0', ?)", (json.dumps(meta).encode().hex(),))
            conn.execute(
                "INSERT INTO blobs (id, data) VALUES (?, ?)",
                (
                    "blob1",
                    json.dumps([
                        {"key": "TICKET-1", "summary": "Stuttgart issue"},
                        {"key": "TICKET-2", "summary": "Other issue"},
                    ]).encode("utf-8"),
                ),
            )
            conn.execute(
                "INSERT INTO blobs (id, data) VALUES (?, ?)",
                (
                    "blob2",
                    json.dumps({
                        "role": "user",
                        "content": [{"type": "text", "text": "Stuttgart"}],
                    }).encode("utf-8"),
                ),
            )
            conn.commit()
            conn.close()

            messages = self.ns["parse_cursor_store"](db_path)
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0][0], "user")
            self.assertIn("Stuttgart", messages[0][1])

    def test_session_catalog_cache_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "cache.sqlite3"
            old_db = self.ns["SEARCH_CACHE_DB"]
            self.ns["SEARCH_CACHE_DB"] = db_path
            try:
                conn = self.ns["open_search_cache"]()
                record = {
                    "project": "demo",
                    "filepath": "/tmp/demo.jsonl",
                    "session_id": "abcd",
                    "created": self.ns["datetime"].fromtimestamp(1000),
                    "modified": self.ns["datetime"].fromtimestamp(2000),
                    "format": ".jsonl",
                    "source": "transcript",
                    "workspace_path": "/Users/md/code/demo",
                    "msg_count": 2,
                    "summary": "hello world",
                    "searchable_all": "hello world tsr tickets",
                    "searchable_user": "tsr tickets",
                    "searchable_assistant": "hello world",
                }
                fingerprint = "fingerprint123"
                self.ns["_save_session_catalog"](conn, fingerprint, [record])
                conn.commit()
                loaded = self.ns["_load_session_catalog"](conn, fingerprint)
                self.assertEqual(len(loaded), 1)
                self.assertEqual(loaded[0]["filepath"], record["filepath"])
                self.assertEqual(loaded[0]["summary"], record["summary"])
                self.assertEqual(loaded[0]["msg_count"], record["msg_count"])
                conn.close()
            finally:
                self.ns["SEARCH_CACHE_DB"] = old_db

    def test_build_search_lines_uses_cached_session_text(self):
        sessions = [
            {
                "project": "demo",
                "filepath": "/tmp/demo.jsonl",
                "session_id": "abcd",
                "created": self.ns["datetime"].fromtimestamp(1000),
                "modified": self.ns["datetime"].fromtimestamp(2000),
                "format": ".jsonl",
                "source": "transcript",
                "workspace_path": "/Users/md/code/demo",
                "msg_count": 2,
                "summary": "hello world",
                "searchable_all": "hello world tsr tickets",
                "searchable_user": "tsr tickets",
                "searchable_assistant": "hello world",
            }
        ]
        lines = self.ns["build_search_lines"](sessions, query="tsr tickets")
        self.assertEqual(len(lines), 1)
        self.assertIn("tsr", lines[0].lower())

