#!/usr/bin/env python3
"""cursearch — fuzzy search through Cursor CLI agent transcripts."""

import html as html_mod
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import webbrowser
from collections import Counter
from datetime import datetime
from pathlib import Path

# * Configuration

CURSOR_PROJECTS_DIR = Path.home() / ".cursor" / "projects"
HOME_PREFIX = str(Path.home()).lstrip("/").replace("/", "-") + "-"
MAX_SUMMARY_QUERIES = 5
SEARCH_CACHE_DB = Path(tempfile.gettempdir()) / "cursearch_search_cache.sqlite3"
SEARCH_EXCERPT_CHARS = 180

# * ANSI colors

BOLD = "\033[1m"
DIM = "\033[2m"
UNDERLINE = "\033[4m"
RESET = "\033[0m"
CYAN = "\033[36m"
GREEN = "\033[32m"
BLUE = "\033[34m"
BRIGHT_BLUE = "\033[94m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
WHITE = "\033[37m"
GRAY = "\033[90m"
MATCH_HL = "\033[1;32;4m"  # bold + green + underline for match highlights


# * Parsing

def parse_jsonl(filepath, max_bytes=None):
    """Parse a .jsonl transcript. Returns list of (role, text) tuples."""
    messages = []
    bytes_read = 0
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if max_bytes and bytes_read > max_bytes:
                break
            bytes_read += len(line)
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                role = obj.get("role", "unknown")
                content_parts = obj.get("message", {}).get("content", [])
                text = " ".join(
                    p.get("text", "") for p in content_parts if p.get("type") == "text"
                )
                if text.strip():
                    messages.append((role, text.strip()))
            except (json.JSONDecodeError, KeyError):
                continue
    return messages


def parse_txt(filepath, max_bytes=None):
    """Parse a .txt transcript. Returns list of (role, text) tuples."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read(max_bytes) if max_bytes else f.read()

    blocks = re.split(r"^(user|assistant):\n", content, flags=re.MULTILINE)
    messages = []
    for i in range(1, len(blocks) - 1, 2):
        role = blocks[i]
        text = blocks[i + 1].strip()
        if text:
            messages.append((role, text))
    return messages


def parse_transcript(filepath, max_bytes=None):
    """Auto-detect format and parse a transcript file."""
    filepath = Path(filepath)
    if filepath.suffix == ".jsonl":
        return parse_jsonl(filepath, max_bytes)
    elif filepath.suffix == ".txt":
        return parse_txt(filepath, max_bytes)
    return []


# * XML/tag stripping

NOISE_TAGS_RE = re.compile(
    r"</?(?:user_query|user_info|git_status|system_reminder|rules|"
    r"agent_requestable_workspace_rules|agent_requestable_workspace_rule|"
    r"user_rules|user_rule|agent_transcripts|manually_attached_skills|"
    r"tone_and_style|tool_calling|making_code_changes|citing_code|"
    r"inline_line_numbers|task_management|mermaid_syntax|"
    r"committing-changes-with-git|creating-pull-requests|"
    r"no_thinking_in_code_or_commands|other-common-operations)(?:\\s[^>]*)?>",
    re.IGNORECASE,
)
BLOCK_TAGS_RE = re.compile(
    r"<(?:system_reminder|git_status|user_info|rules|agent_transcripts|"
    r"manually_attached_skills)>.*?</(?:system_reminder|git_status|user_info|"
    r"rules|agent_transcripts|manually_attached_skills)>",
    re.DOTALL,
)


def strip_tags(text):
    """Remove XML wrapper tags, system blocks, and framework noise."""
    text = BLOCK_TAGS_RE.sub("", text)
    text = NOISE_TAGS_RE.sub("", text)
    return text.strip()


# * Scanning

def get_file_times(filepath):
    """Return (created, modified) as datetime objects. Uses birthtime on macOS."""
    st = os.stat(filepath)
    mtime = datetime.fromtimestamp(st.st_mtime)
    try:
        ctime = datetime.fromtimestamp(st.st_birthtime)
    except AttributeError:
        ctime = datetime.fromtimestamp(st.st_ctime)
    return ctime, mtime


def short_project_name(dirname):
    """Strip the home-dir prefix from a project directory name."""
    if dirname.startswith(HOME_PREFIX):
        short = dirname[len(HOME_PREFIX):]
        return short if short else "~"
    return dirname


def scan_all_transcripts():
    """Walk all Cursor project dirs and collect transcript metadata."""
    sessions = []
    seen_ids = {}

    if not CURSOR_PROJECTS_DIR.is_dir():
        return sessions

    for project_dir in sorted(CURSOR_PROJECTS_DIR.iterdir()):
        transcripts_dir = project_dir / "agent-transcripts"
        if not transcripts_dir.is_dir():
            continue

        project_name = short_project_name(project_dir.name)

        for fpath in sorted(transcripts_dir.iterdir()):
            if fpath.suffix not in (".jsonl", ".txt"):
                continue

            session_id = fpath.stem

            if session_id in seen_ids:
                if fpath.suffix == ".jsonl":
                    sessions[seen_ids[session_id]]["filepath"] = str(fpath)
                    sessions[seen_ids[session_id]]["format"] = fpath.suffix
                continue

            created, modified = get_file_times(fpath)
            seen_ids[session_id] = len(sessions)
            sessions.append({
                "project": project_name,
                "filepath": str(fpath),
                "session_id": session_id,
                "created": created,
                "modified": modified,
                "format": fpath.suffix,
            })

    sessions.sort(key=lambda s: s["modified"], reverse=True)
    return sessions


# * Search index building

def make_summary(messages):
    """Short topic summary from first few user messages."""
    user_msgs = [strip_tags(t) for r, t in messages if r == "user"]
    parts = []
    for msg in user_msgs[:MAX_SUMMARY_QUERIES]:
        line = msg.replace("\n", " ").strip()
        if len(line) > 80:
            line = line[:77] + "..."
        if line:
            parts.append(line)
    return " | ".join(parts) if parts else "(empty session)"


def parse_query_tokens(query):
    """Split query into non-empty whitespace-separated tokens."""
    return [t for t in query.split() if t]


def token_contains(haystack, token):
    """Smart-case token search: lowercase token -> case-insensitive, else sensitive."""
    if token.islower():
        return haystack.lower().find(token.lower())
    return haystack.find(token)


def ordered_tokens_match(haystack, query):
    """Ordered token match with smart-case and gap allowance between tokens."""
    tokens = parse_query_tokens(query)
    if not tokens:
        return True
    start = 0
    for token in tokens:
        idx = token_contains(haystack[start:], token)
        if idx < 0:
            return False
        start += idx + len(token)
    return True


def highlight_matches(text, query, restore=""):
    """Insert ANSI highlight around matching tokens in text.

    restore is the ANSI code to re-apply after each highlight reset
    (e.g. DIM so surrounding dim text stays dim).
    """
    tokens = parse_query_tokens(query)
    if not tokens:
        return text
    for token in tokens:
        if token.islower():
            lower_text = text.lower()
            lower_tok = token.lower()
            parts = []
            pos = 0
            while pos < len(text):
                idx = lower_text.find(lower_tok, pos)
                if idx < 0:
                    parts.append(text[pos:])
                    break
                parts.append(text[pos:idx])
                parts.append(f"{MATCH_HL}{text[idx:idx+len(token)]}{RESET}{restore}")
                pos = idx + len(token)
            text = "".join(parts)
        else:
            text = text.replace(token, f"{MATCH_HL}{token}{RESET}{restore}")
    return text


def make_search_excerpt(text, query="", max_chars=SEARCH_EXCERPT_CHARS):
    """Return a short contextual excerpt for display in the search UI."""
    text = text.replace("\n", " ").strip()
    if not text:
        return "(no transcript text)"

    start = 0
    tokens = parse_query_tokens(query)
    if tokens:
        idx = token_contains(text, tokens[0])
        if idx >= 0:
            start = max(0, idx - max_chars // 3)

    excerpt = text[start:start + max_chars].strip()
    if start > 0:
        excerpt = "..." + excerpt
    if start + max_chars < len(text):
        excerpt = excerpt + "..."
    return excerpt


def open_search_cache():
    """Open the persistent SQLite FTS5 search cache.

    Schema: a metadata table for cache invalidation plus an FTS5
    virtual table for fast full-text matching.
    """
    conn = sqlite3.connect(str(SEARCH_CACHE_DB))
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS transcript_meta (
            filepath TEXT PRIMARY KEY,
            mtime_ns INTEGER NOT NULL,
            size_bytes INTEGER NOT NULL,
            msg_count INTEGER NOT NULL,
            summary TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS transcript_fts USING fts5(
            filepath UNINDEXED,
            searchable_all,
            searchable_user,
            searchable_assistant,
            tokenize = 'unicode61 remove_diacritics 2 tokenchars _'
        )
        """
    )
    return conn


def build_session_search_record(filepath):
    """Build full-text search fields for a transcript file."""
    messages = parse_transcript(filepath)
    cleaned_messages = []
    for role, text in messages:
        text = strip_tags(text)
        if not text:
            continue
        cleaned_messages.append((role, text))

    searchable_all = " ".join(t.replace("\n", " ") for _, t in cleaned_messages)
    searchable_user = " ".join(t.replace("\n", " ") for r, t in cleaned_messages if r == "user")
    searchable_assistant = " ".join(
        t.replace("\n", " ") for r, t in cleaned_messages if r == "assistant"
    )
    return {
        "msg_count": len(cleaned_messages),
        "summary": make_summary(cleaned_messages),
        "searchable_all": searchable_all,
        "searchable_user": searchable_user,
        "searchable_assistant": searchable_assistant,
    }


def _fts_delete_row(conn, filepath):
    """Remove a row from the FTS5 table by filepath."""
    old = conn.execute(
        "SELECT rowid, searchable_all, searchable_user, searchable_assistant "
        "FROM transcript_fts WHERE filepath = ?",
        (filepath,),
    ).fetchone()
    if old:
        conn.execute(
            "INSERT INTO transcript_fts(transcript_fts, rowid, filepath, "
            "searchable_all, searchable_user, searchable_assistant) "
            "VALUES('delete', ?, ?, ?, ?, ?)",
            (old["rowid"], filepath, old["searchable_all"],
             old["searchable_user"], old["searchable_assistant"]),
        )


def ensure_search_cache(conn, sessions):
    """Ensure the SQLite FTS5 cache has current full-text data for all sessions."""
    valid_paths = set()
    for s in sessions:
        filepath = s["filepath"]
        valid_paths.add(filepath)
        stat = os.stat(filepath)
        row = conn.execute(
            "SELECT mtime_ns, size_bytes FROM transcript_meta WHERE filepath = ?",
            (filepath,),
        ).fetchone()
        if row and row["mtime_ns"] == stat.st_mtime_ns and row["size_bytes"] == stat.st_size:
            continue

        record = build_session_search_record(filepath)

        _fts_delete_row(conn, filepath)

        conn.execute(
            """
            INSERT INTO transcript_meta (filepath, mtime_ns, size_bytes, msg_count, summary)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(filepath) DO UPDATE SET
                mtime_ns = excluded.mtime_ns,
                size_bytes = excluded.size_bytes,
                msg_count = excluded.msg_count,
                summary = excluded.summary
            """,
            (filepath, stat.st_mtime_ns, stat.st_size,
             record["msg_count"], record["summary"]),
        )
        conn.execute(
            "INSERT INTO transcript_fts (filepath, searchable_all, "
            "searchable_user, searchable_assistant) VALUES (?, ?, ?, ?)",
            (filepath, record["searchable_all"],
             record["searchable_user"], record["searchable_assistant"]),
        )

    stale_paths = [
        row["filepath"]
        for row in conn.execute("SELECT filepath FROM transcript_meta").fetchall()
        if row["filepath"] not in valid_paths
    ]
    for p in stale_paths:
        _fts_delete_row(conn, p)
        conn.execute("DELETE FROM transcript_meta WHERE filepath = ?", (p,))

    conn.commit()


def _build_fts_query(query):
    """Translate a user query string into an FTS5 MATCH expression.

    Each whitespace-separated token becomes an FTS5 prefix token (tok*),
    and they are ANDed together so all tokens must appear.
    """
    tokens = parse_query_tokens(query)
    if not tokens:
        return None
    fts_tokens = []
    for tok in tokens:
        safe = tok.replace('"', '""')
        fts_tokens.append(f'"{safe}" *')
    return " AND ".join(fts_tokens)


def _fts_scope_column(scope):
    """Return the FTS5 column name for the given scope."""
    if scope == "user":
        return "searchable_user"
    if scope == "assistant":
        return "searchable_assistant"
    return "searchable_all"


def search_sessions(conn, query, scope="all"):
    """Search sessions using FTS5. Returns set of matching filepaths."""
    fts_expr = _build_fts_query(query)
    if fts_expr is None:
        return None

    col = _fts_scope_column(scope)
    try:
        rows = conn.execute(
            f"SELECT filepath FROM transcript_fts WHERE {col} MATCH ?",
            (fts_expr,),
        ).fetchall()
    except sqlite3.OperationalError:
        return set()
    return {row["filepath"] for row in rows}


def get_searchable_text(conn, filepath, scope="all"):
    """Retrieve the cached searchable text for a session."""
    col = _fts_scope_column(scope)
    row = conn.execute(
        f"SELECT {col} FROM transcript_fts WHERE filepath = ?",
        (filepath,),
    ).fetchone()
    return row[col] if row else ""


def build_search_lines(sessions, scope="all", sort_by="modified", query=""):
    """Build lines for fzf — one per session.

    Each line is a single fzf entry with tab-separated fields:
      filepath \\t visible_card

    Uses FTS5 for fast full-text filtering when a query is present.
    """
    if sort_by == "created":
        ordered_sessions = sorted(sessions, key=lambda s: s["created"], reverse=True)
    else:
        ordered_sessions = sorted(sessions, key=lambda s: s["modified"], reverse=True)

    conn = open_search_cache()
    ensure_search_cache(conn, ordered_sessions)

    matching_paths = search_sessions(conn, query, scope) if query.strip() else None

    lines = []
    for s in ordered_sessions:
        filepath = s["filepath"]

        if matching_paths is not None and filepath not in matching_paths:
            continue

        meta = conn.execute(
            "SELECT msg_count, summary FROM transcript_meta WHERE filepath = ?",
            (filepath,),
        ).fetchone()
        if not meta:
            continue

        modified_str = s["modified"].strftime("%Y-%m-%d %H:%M")
        created_str = s["created"].strftime("%m-%d %H:%M")
        summary = meta["summary"]
        msg_count = meta["msg_count"]

        searchable = get_searchable_text(conn, filepath, scope) if query.strip() else ""
        excerpt = make_search_excerpt(searchable, query) if query.strip() else ""

        hl_project = highlight_matches(s['project'], query, CYAN + BOLD) if query else s['project']
        hl_summary = highlight_matches(summary, query, DIM) if query else summary
        hl_excerpt = highlight_matches(excerpt, query, DIM) if query else excerpt

        excerpt_line = f"\n  {GRAY}{DIM}{hl_excerpt}{RESET}" if excerpt else ""
        card = (
            f"{CYAN}{BOLD}{hl_project}{RESET}"
            f"  {GRAY}({msg_count}){RESET}"
            f"  {DIM}{modified_str}{RESET}"
            f"  {GRAY}cre {created_str}{RESET}\n"
            f"  {DIM}{hl_summary}{RESET}"
            f"{excerpt_line}"
        )

        lines.append(f"{s['filepath']}\t{card}")

    conn.close()
    return lines


def join_lines(lines):
    """Join lines with null byte for fzf --read0."""
    return "\0".join(lines)


# * Preview (called by fzf subprocess)

def preview_session(filepath, reverse=False, query="", latest_match=False):
    """Print a colored preview of a session transcript."""
    messages = parse_transcript(filepath)
    project_dir = Path(filepath).parent.parent.name
    project = short_project_name(project_dir)
    created, modified = get_file_times(filepath)

    header = (
        f"{BRIGHT_BLUE}{BOLD}{project}{RESET}"
        f"  {GRAY}{len(messages)} messages{RESET}\n"
        f"{DIM}{created.strftime('%Y-%m-%d %H:%M')} → {modified.strftime('%Y-%m-%d %H:%M')}{RESET}\n"
        f"{GRAY}{'─' * 50}{RESET}"
    )

    display_msgs = list(reversed(messages)) if reverse else messages

    if latest_match and query.strip():
        last_idx = -1
        for i, (_, text) in enumerate(messages):
            if ordered_tokens_match(text, query):
                last_idx = i
        if last_idx >= 0:
            start = max(0, last_idx - 6)
            end = min(len(messages), last_idx + 8)
            display_msgs = messages[start:end]
        else:
            msg = f"{YELLOW}[no match]{RESET} No message matches query: {query}"
            print(header)
            print(msg)
            return

    msg_lines = []
    for role, text in display_msgs:
        text = strip_tags(text)
        if not text:
            continue
        wrapped = textwrap.fill(text, width=72)
        if role == "user":
            msg_lines.append(f"\n{GREEN}{BOLD}USER{RESET}\n{wrapped}")
        else:
            msg_lines.append(f"\n{BRIGHT_BLUE}{BOLD}ASST{RESET}\n{DIM}{wrapped}{RESET}")

    print(header)
    print("\n".join(msg_lines))


# * HTML export

# ** Markdown to HTML converter

_MD_FENCED_CODE_RE = re.compile(
    r"^```(\w*)\n(.*?)^```\s*$", re.MULTILINE | re.DOTALL
)
_MD_INLINE_CODE_RE = re.compile(r"`([^`\n]+?)`")
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
_MD_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)|(?<!_)_(?!_)(.+?)(?<!_)_(?!_)")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_MD_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
_MD_HR_RE = re.compile(r"^---+\s*$", re.MULTILINE)


def _md_to_html(text):
    """Convert markdown-formatted text to HTML for the export view.

    Handles fenced code blocks, inline code, bold, italic, links,
    headings, horizontal rules, and list items. Escapes HTML first
    to prevent injection, then applies formatting patterns.
    """
    code_blocks = {}
    counter = [0]

    def _stash_code(m):
        key = f"\x00CODE{counter[0]}\x00"
        counter[0] += 1
        lang = m.group(1) or ""
        body = html_mod.escape(m.group(2).rstrip("\n"))
        lang_attr = f' data-lang="{html_mod.escape(lang)}"' if lang else ""
        code_blocks[key] = f'<pre class="code-block"><code{lang_attr}>{body}</code></pre>'
        return key

    text = _MD_FENCED_CODE_RE.sub(_stash_code, text)

    inline_codes = {}

    def _stash_inline(m):
        key = f"\x00IC{counter[0]}\x00"
        counter[0] += 1
        inline_codes[key] = f'<code>{html_mod.escape(m.group(1))}</code>'
        return key

    text = _MD_INLINE_CODE_RE.sub(_stash_inline, text)

    text = html_mod.escape(text)

    for key, replacement in code_blocks.items():
        text = text.replace(html_mod.escape(key), replacement)
    for key, replacement in inline_codes.items():
        text = text.replace(html_mod.escape(key), replacement)

    text = _MD_BOLD_RE.sub(lambda m: f'<strong>{m.group(1) or m.group(2)}</strong>', text)
    text = _MD_ITALIC_RE.sub(lambda m: f'<em>{m.group(1) or m.group(2)}</em>', text)
    text = _MD_LINK_RE.sub(lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', text)

    def _heading_repl(m):
        level = min(len(m.group(1)) + 2, 6)
        return f'<h{level} class="msg-heading">{m.group(2)}</h{level}>'

    text = _MD_HEADING_RE.sub(_heading_repl, text)
    text = _MD_HR_RE.sub('<hr class="msg-hr">', text)

    out_lines = []
    in_list = False
    for line in text.split("\n"):
        stripped = line.strip()
        is_li = re.match(r"^[-*]\s+", stripped) or re.match(r"^\d+\.\s+", stripped)
        if is_li:
            if not in_list:
                out_lines.append("<ul>")
                in_list = True
            li_text = re.sub(r"^[-*]\s+|^\d+\.\s+", "", stripped)
            out_lines.append(f"<li>{li_text}</li>")
        else:
            if in_list:
                out_lines.append("</ul>")
                in_list = False
            out_lines.append(line)
    if in_list:
        out_lines.append("</ul>")
    text = "\n".join(out_lines)

    paragraphs = re.split(r"\n{2,}", text)
    processed = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if re.match(r"<(?:pre|h[1-6]|hr|ul|ol)", p):
            processed.append(p)
        else:
            processed.append(f"<p>{p}</p>")
    return "\n".join(processed)


# ** HTML template (Tufte-inspired)

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cursor Session — {project} — {modified}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

  :root {{
    --bg: #fffff8;
    --fg: #111;
    --fg-secondary: #555;
    --fg-muted: #888;
    --user-accent: #2d6a4f;
    --asst-accent: #3a5a8c;
    --border: #ddd;
    --code-bg: #f5f5f0;
    --user-bg: rgba(45, 106, 79, 0.04);
    --asst-bg: rgba(58, 90, 140, 0.04);
  }}

  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: 'Crimson Pro', 'Palatino Linotype', Palatino, 'Book Antiqua', Georgia, serif;
    font-size: 19px;
    line-height: 1.7;
    color: var(--fg);
    background: var(--bg);
    -webkit-font-smoothing: antialiased;
  }}

  /* -- Header -- */
  .header {{
    max-width: 740px;
    margin: 0 auto;
    padding: 60px 20px 30px;
    border-bottom: 1px solid var(--border);
  }}
  .header h1 {{
    font-size: 28px;
    font-weight: 600;
    letter-spacing: -0.5px;
    color: var(--fg);
    margin-bottom: 8px;
  }}
  .header .meta {{
    font-size: 14px;
    color: var(--fg-muted);
    font-family: 'JetBrains Mono', 'SF Mono', monospace;
    letter-spacing: 0.02em;
  }}

  /* -- Message container -- */
  .messages {{
    max-width: 740px;
    margin: 0 auto;
    padding: 20px 20px 80px;
  }}

  /* -- Individual message -- */
  .msg {{
    margin: 28px 0;
    padding: 0;
  }}
  .msg + .msg {{
    border-top: 1px solid #eee;
    padding-top: 28px;
  }}

  .role {{
    font-family: 'JetBrains Mono', 'SF Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 10px;
  }}
  .msg.user .role {{ color: var(--user-accent); }}
  .msg.asst .role {{ color: var(--asst-accent); }}

  .msg.user {{ background: var(--user-bg); padding: 20px 24px; border-radius: 4px; }}
  .msg.asst {{ }}

  .msg .body {{ color: var(--fg); }}
  .msg .body p {{ margin-bottom: 0.85em; }}
  .msg .body p:last-child {{ margin-bottom: 0; }}

  /* -- Typography inside messages -- */
  .msg .body strong {{ font-weight: 600; }}
  .msg .body em {{ font-style: italic; }}
  .msg .body a {{ color: var(--asst-accent); text-decoration: underline; text-decoration-thickness: 1px; text-underline-offset: 2px; }}
  .msg .body a:hover {{ color: var(--fg); }}

  .msg .body code {{
    font-family: 'JetBrains Mono', 'SF Mono', monospace;
    font-size: 0.82em;
    background: var(--code-bg);
    padding: 2px 5px;
    border-radius: 3px;
    color: #333;
  }}

  .msg .body pre.code-block {{
    font-family: 'JetBrains Mono', 'SF Mono', monospace;
    font-size: 0.78em;
    line-height: 1.55;
    background: var(--code-bg);
    border: 1px solid #e8e8e0;
    border-radius: 4px;
    padding: 16px 20px;
    margin: 16px 0;
    overflow-x: auto;
    white-space: pre;
  }}
  .msg .body pre.code-block code {{
    background: none;
    padding: 0;
    border-radius: 0;
    font-size: inherit;
  }}

  .msg .body .msg-heading {{
    font-family: 'Crimson Pro', Georgia, serif;
    color: var(--fg);
    margin: 20px 0 8px;
    line-height: 1.3;
  }}
  h3.msg-heading {{ font-size: 1.15em; font-weight: 600; }}
  h4.msg-heading {{ font-size: 1.05em; font-weight: 600; }}
  h5.msg-heading {{ font-size: 0.95em; font-weight: 600; color: var(--fg-secondary); }}
  h6.msg-heading {{ font-size: 0.88em; font-weight: 600; color: var(--fg-muted); }}

  .msg .body ul, .msg .body ol {{
    padding-left: 1.4em;
    margin: 10px 0;
  }}
  .msg .body li {{
    margin-bottom: 4px;
  }}
  .msg .body .msg-hr {{
    border: none;
    border-top: 1px solid var(--border);
    margin: 20px 0;
  }}

  /* -- Footer -- */
  .footer {{
    max-width: 740px;
    margin: 0 auto;
    padding: 20px;
    border-top: 1px solid var(--border);
    font-size: 13px;
    color: var(--fg-muted);
    font-family: 'JetBrains Mono', 'SF Mono', monospace;
    text-align: center;
  }}

  @media (max-width: 800px) {{
    body {{ font-size: 17px; }}
    .header, .messages, .footer {{ padding-left: 16px; padding-right: 16px; }}
    .header {{ padding-top: 30px; }}
  }}

  @media print {{
    body {{ font-size: 12pt; }}
    .header {{ padding-top: 0; }}
    .msg.user {{ background: #f8f8f4; }}
  }}
</style>
</head>
<body>
<div class="header">
  <h1>{project}</h1>
  <div class="meta">{created} &rarr; {modified} &middot; {count} messages</div>
</div>
<div class="messages">
{messages_html}
</div>
<div class="footer">
  cursearch export
</div>
</body>
</html>
"""


def export_html(filepath):
    """Export a session transcript to HTML and open in browser."""
    messages = parse_transcript(filepath)
    project_dir = Path(filepath).parent.parent.name
    project = short_project_name(project_dir)
    created, modified = get_file_times(filepath)

    msg_blocks = []
    for role, text in messages:
        text = strip_tags(text)
        if not text:
            continue
        css_class = "user" if role == "user" else "asst"
        label = "USER" if role == "user" else "ASSISTANT"
        body_html = _md_to_html(text)
        msg_blocks.append(
            f'<div class="msg {css_class}">'
            f'<div class="role">{label}</div>'
            f'<div class="body">{body_html}</div>'
            f'</div>'
        )

    page = HTML_TEMPLATE.format(
        project=html_mod.escape(project),
        created=created.strftime("%Y-%m-%d %H:%M"),
        modified=modified.strftime("%Y-%m-%d %H:%M"),
        count=len(messages),
        messages_html="\n".join(msg_blocks),
    )

    fd, tmp_path = tempfile.mkstemp(suffix=".html", prefix="cursearch_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(page)
    webbrowser.open(f"file://{tmp_path}")


# * Org-mode export

def make_heading_text(text, max_len=100):
    """First line of text, truncated, for use as an Org heading."""
    line = text.split("\n")[0].strip()
    if len(line) > max_len:
        line = line[:max_len - 3] + "..."
    return line


def normalize_export_messages(messages):
    """Return chronological, cleaned transcript messages for export."""
    normalized = []
    for idx, (role, text) in enumerate(messages, start=1):
        text = strip_tags(text)
        if not text:
            continue
        normalized.append((idx, role, text))
    return normalized


def export_org(filepath):
    """Export a session transcript to Org-mode and open in Emacs/default editor."""
    messages = parse_transcript(filepath)
    project_dir = Path(filepath).parent.parent.name
    project = short_project_name(project_dir)
    created, modified = get_file_times(filepath)
    export_messages = normalize_export_messages(messages)

    lines = [
        f"#+title: Cursor Session — {project}",
        f"#+date: [{created.strftime('%Y-%m-%d %a')}]",
        "#+options: toc:nil num:nil ^:{}",
        "",
        f"- *Project:* {project}",
        f"- *Created:* [{created.strftime('%Y-%m-%d %a %H:%M')}]",
        f"- *Modified:* [{modified.strftime('%Y-%m-%d %a %H:%M')}]",
        f"- *Messages:* {len(export_messages)}",
        "",
    ]

    for idx, role, text in export_messages:
        label = "User" if role == "user" else "Assistant"
        lines.append(f"* {idx:03d} {label}")
        lines.append("")
        lines.append(text)
        lines.append("")

    fd, tmp_path = tempfile.mkstemp(suffix=".org", prefix="cursearch_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    subprocess.Popen(["open", tmp_path])


# * Markdown export

def export_markdown(filepath):
    """Export a session transcript to Markdown and open in default editor."""
    messages = parse_transcript(filepath)
    project_dir = Path(filepath).parent.parent.name
    project = short_project_name(project_dir)
    created, modified = get_file_times(filepath)
    export_messages = normalize_export_messages(messages)

    lines = [
        f"# Cursor Session — {project}",
        "",
        f"- **Project:** {project}",
        f"- **Created:** {created.strftime('%Y-%m-%d %H:%M')}",
        f"- **Modified:** {modified.strftime('%Y-%m-%d %H:%M')}",
        f"- **Messages:** {len(export_messages)}",
        "",
        "---",
        "",
    ]

    for idx, role, text in export_messages:
        label = "User" if role == "user" else "Assistant"
        lines.append(f"## {idx:03d} {label}")
        lines.append("")
        lines.append(text)
        lines.append("")

    fd, tmp_path = tempfile.mkstemp(suffix=".md", prefix="cursearch_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    subprocess.Popen(["open", tmp_path])


# * Help overlay

HELP_TEXT = f"""\
{BRIGHT_BLUE}{BOLD}cursearch{RESET} {DIM}— Cursor session search{RESET}

{CYAN}Navigation{RESET}
  {WHITE}j / k{RESET}            up / down {DIM}(browse){RESET}
  {WHITE}ctrl-n / ctrl-p{RESET}  up / down {DIM}(always){RESET}
  {WHITE}alt-j / alt-k{RESET}    up / down {DIM}(always){RESET}

{CYAN}Mode{RESET}
  {WHITE}Enter{RESET}            resume session
  {WHITE}Tab{RESET}              enter browse mode
  {WHITE}Esc{RESET}              browse → search
  {WHITE}ctrl-c{RESET}           quit
  {WHITE}/{RESET}                back to search {DIM}(browse){RESET}

{CYAN}Selection{RESET} {DIM}(browse mode){RESET}
  {WHITE}x{RESET}                toggle mark
  {WHITE}u{RESET}                unmark + move down

{CYAN}Toggles{RESET}
  {WHITE}`{RESET}                scope: all / user-only
  {WHITE}ctrl-y{RESET}           sort: modified / created
  {WHITE}ctrl-o{RESET}           preview: latest-match / newest / chrono

{CYAN}Actions{RESET}
  {WHITE}ctrl-g{RESET}           create skills from selected
  {WHITE}alt-s{RESET}            summarize selected {DIM}(agent --print){RESET}
  {WHITE}alt-h{RESET}            export to HTML
  {WHITE}ctrl-i{RESET}           export to Org-mode
  {WHITE}alt-m{RESET}            export to Markdown

{DIM}focus any item to restore preview{RESET}
"""


def print_help_overlay():
    """Print the formatted help text for the fzf preview pane."""
    print(HELP_TEXT)


# * Resume session

def decode_project_path(dirname):
    """Decode a Cursor project directory name back to a real filesystem path.

    Cursor encodes paths by replacing /, ., _, and spaces with dashes.
    Uses DFS with filesystem checks to find the correct decoding.
    """
    segments = dirname.split("-")
    if len(segments) <= 1:
        return "/" + dirname
    best = ["/"]

    def try_decode(parent, leaf, idx):
        if idx >= len(segments):
            full = os.path.join(parent, leaf)
            if os.path.isdir(full):
                return full
            if len(parent) > len(best[0]):
                best[0] = parent
            return None
        seg = segments[idx]
        full_current = os.path.join(parent, leaf)
        if os.path.isdir(full_current):
            if len(full_current) > len(best[0]):
                best[0] = full_current
            result = try_decode(full_current, seg, idx + 1)
            if result:
                return result
        for sep in ["-", ".", "_", " "]:
            result = try_decode(parent, leaf + sep + seg, idx + 1)
            if result:
                return result
        return None

    result = try_decode("/", segments[0], 1)
    return result or best[0]


ABS_PATH_RE = re.compile(r"/Users/[^\n\"'`<>]+")
WORKDIR_RE = re.compile(r'working_directory:\s*(?:"([^"\n]+)"|([^\s"\n]+))')
REL_PATH_RE = re.compile(r"(?:`|\\b)([A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)+)")


def infer_resume_cwd(transcript_path, project_path):
    """Infer the best resume cwd from absolute paths inside transcript text."""
    try:
        with open(transcript_path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read(300_000)
    except OSError:
        return project_path

    project_real = os.path.realpath(project_path)
    candidates = Counter()

    for match in ABS_PATH_RE.findall(raw):
        p = match.strip().rstrip(".,;:)]}")
        if "/.cursor/" in p:
            continue
        real = os.path.realpath(p)
        if os.path.isdir(real):
            dir_path = real
        elif os.path.isfile(real):
            dir_path = os.path.dirname(real)
        else:
            continue
        try:
            if os.path.commonpath([dir_path, project_real]) != project_real:
                continue
        except ValueError:
            continue
        candidates[dir_path] += 1

    for m in WORKDIR_RE.finditer(raw):
        wd = (m.group(1) or m.group(2) or "").strip()
        if not wd:
            continue
        if wd.startswith("/"):
            real = os.path.realpath(wd)
        else:
            real = os.path.realpath(os.path.join(project_real, wd))
        if not os.path.isdir(real):
            continue
        try:
            if os.path.commonpath([real, project_real]) != project_real:
                continue
        except ValueError:
            continue
        candidates[real] += 2

    # Learn likely cwd from relative path mentions like "tsr-japan/file.org".
    for m in REL_PATH_RE.finditer(raw):
        rel_path = (m.group(1) or "").strip().strip("`")
        if not rel_path or rel_path.startswith(".cursor/"):
            continue
        parts = rel_path.split("/")
        if not parts:
            continue
        top = parts[0]
        top_dir = os.path.realpath(os.path.join(project_real, top))
        if os.path.isdir(top_dir):
            try:
                if os.path.commonpath([top_dir, project_real]) != project_real:
                    continue
            except ValueError:
                continue
            candidates[top_dir] += 3

    if not candidates:
        return project_path

    # Prefer frequently referenced and deeper paths (more specific cwd).
    best_dir = max(candidates.items(), key=lambda kv: (kv[1], kv[0].count("/")))[0]
    return best_dir


def resume_session(filepath):
    """Resume a Cursor CLI agent session. Replaces the current process."""
    filepath_str = str(filepath)
    session_id = Path(filepath_str).stem
    # Defensive fallback: extract canonical UUID from path if stem is malformed.
    if len(session_id) != 36:
        m = re.search(
            r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
            filepath_str,
            re.IGNORECASE,
        )
        if m:
            session_id = m.group(1)
    project_dirname = Path(filepath).parent.parent.name
    project_path = decode_project_path(project_dirname)
    # Bulletproof default: use workspace encoded by transcript container dir.
    # Heuristic subdir inference can point at the wrong workspace and cause
    # "resume" to start a new empty session context.
    resume_cwd = project_path
    cmd = ["agent", "--resume", session_id, "--workspace", resume_cwd]
    print(f"[cursearch] project {project_path}", file=sys.stderr)
    print(f"[cursearch] cd {resume_cwd}", file=sys.stderr)
    print(f"[cursearch] session_id {session_id} (len={len(session_id)})", file=sys.stderr)
    print(f"[cursearch] {' '.join(cmd)}", file=sys.stderr)
    os.chdir(resume_cwd)
    os.execvp("agent", cmd)


def summarize_sessions(filepaths):
    """Summarize selected sessions via headless agent --print."""
    excerpts = []
    for fp in filepaths:
        fp = fp.strip()
        if not fp:
            continue
        messages = parse_transcript(fp)
        if not messages:
            continue
        project_dir = Path(fp).parent.parent.name
        project = short_project_name(project_dir)
        _, modified = get_file_times(fp)
        lines = [f"=== {project} ({modified.strftime('%Y-%m-%d')}) ==="]
        budget = 6000
        for role, text in messages:
            text = strip_tags(text)
            if not text:
                continue
            label = "USER" if role == "user" else "ASST"
            if role != "user" and len(text) > 300:
                text = text[:297] + "..."
            entry = f"{label}: {text}"
            if len(entry) > budget:
                entry = entry[:budget]
            lines.append(entry)
            budget -= len(entry)
            if budget <= 0:
                break
        excerpts.append("\n".join(lines))

    if not excerpts:
        print("No transcript content to summarize.", file=sys.stderr)
        return

    prompt = (
        "Summarize these Cursor agent sessions concisely. "
        "Output a single bulleted list of main topics, decisions, and outcomes. "
        "Fit in one terminal screen (~30 lines max). No headers, no fluff.\n\n"
        + "\n\n".join(excerpts)
    )
    os.execvp("agent", ["agent", "--print", "--model", "auto", "--trust", prompt])


def launch_create_skills_agent(filepaths):
    """Launch agent with a generated prompt to create skills from sessions."""
    # Final safety dedupe before handoff to agent.
    unique_paths = []
    seen = set()
    for fp in filepaths:
        abs_fp = os.path.abspath(fp)
        if abs_fp in seen:
            continue
        seen.add(abs_fp)
        unique_paths.append(abs_fp)

    if not unique_paths:
        return

    path_lines = "\n".join(f"- {p}" for p in unique_paths)
    prompt = (
        "Create actionable skills from the selected Cursor agent sessions.\n\n"
        "Use the create-skill skill to produce a few high-quality, reusable skills.\n\n"
        "Selected transcript files:\n"
        f"{path_lines}\n\n"
        "Requirements:\n"
        "- Read those transcripts first.\n"
        "- Identify repeated workflows and decision patterns.\n"
        "- Create 3-7 practical skills with clear triggers, steps, and constraints.\n"
        "- Prefer concise, executable skills over generic advice.\n"
    )
    cmd = ["agent", "--plan", prompt]
    print(f"[cursearch] launching skill synthesis from {len(unique_paths)} sessions", file=sys.stderr)
    os.execvp("agent", cmd)


# * FZF integration

def run_fzf(search_lines):
    """Launch fzf with two modes: search and browse.

    Search mode: type to filter. Enter/alt-Enter resumes selected session.
                 Tab enters browse mode.
    Browse mode: j/k navigate. Enter resumes. Esc or / returns to search.

    Column layout: 1=filepath  2=card_with_search_text
    Mode is tracked via prompt: '> ' = search, '>> ' = browse.
    All transforms use POSIX shell (grep) to detect mode from prompt.
    """
    sp = os.path.abspath(__file__)

    # ** Enter: always accept (resume selected session)
    enter_bind = "enter:accept"

    # ** Tab: search->browse (freeze search, enable j/k/x/u/?)
    # No-op if already in browse mode (empty echo = no fzf action).
    tab_bind = (
        "tab:transform:"
        "if echo \"$FZF_PROMPT\" | grep -q '>>'; then "
        "true; "
        "else "
        "SCOPE=$(echo \"$FZF_PROMPT\" | grep -oE 'all|user'); "
        "SORT=$(echo \"$FZF_PROMPT\" | grep -oE 'mod|cre'); "
        "echo \"rebind(j,k,/,x,u,?)+disable-search"
        "+change-prompt(cursearch [$SCOPE|$SORT]>> )\"; "
        "fi"
    )

    # ** Esc: browse->search (no-op if already in search; ctrl-c to quit)
    esc_bind = (
        "esc:transform:"
        "if echo \"$FZF_PROMPT\" | grep -q '>>'; then "
        "SCOPE=$(echo \"$FZF_PROMPT\" | grep -oE 'all|user'); "
        "SORT=$(echo \"$FZF_PROMPT\" | grep -oE 'mod|cre'); "
        "echo \"unbind(j,k,/,x,u,?)+enable-search"
        "+change-prompt(cursearch [$SCOPE|$SORT]> )\"; "
        "fi"
    )

    # ** / (browse mode only, unbound in search): back to search
    slash_bind = (
        "/:transform:"
        "SCOPE=$(echo \"$FZF_PROMPT\" | grep -oE 'all|user'); "
        "SORT=$(echo \"$FZF_PROMPT\" | grep -oE 'mod|cre'); "
        "echo \"unbind(j,k,/,x,u,?)+enable-search"
        "+change-prompt(cursearch [$SCOPE|$SORT]> )\""
    )

    # ** Backtick: toggle scope, preserve mode marker
    scope_toggle = (
        "`:transform:"
        "if echo \"$FZF_PROMPT\" | grep -q '>>'; then SEP='>> '; "
        "else SEP='> '; fi; "
        "SORT=$(echo \"$FZF_PROMPT\" | grep -oE 'mod|cre'); "
        f"if echo \"$FZF_PROMPT\" | grep -q all; then "
        f"echo \"reload(python3 '{sp}' --lines user --sort $SORT --query {{q}})"
        "+change-prompt(cursearch [user|$SORT]$SEP)\"; "
        f"else "
        f"echo \"reload(python3 '{sp}' --lines all --sort $SORT --query {{q}})"
        "+change-prompt(cursearch [all|$SORT]$SEP)\"; "
        "fi"
    )

    # ** ctrl-y: toggle sort key, preserve scope and browse/search mode
    sort_toggle = (
        "ctrl-y:transform:"
        "if echo \"$FZF_PROMPT\" | grep -q '>>'; then SEP='>> '; "
        "else SEP='> '; fi; "
        "SCOPE=$(echo \"$FZF_PROMPT\" | grep -oE 'all|user'); "
        f"if echo \"$FZF_PROMPT\" | grep -q '|mod'; then "
        f"echo \"reload(python3 '{sp}' --lines $SCOPE --sort cre --query {{q}})"
        "+change-prompt(cursearch [$SCOPE|cre]$SEP)\"; "
        "else "
        f"echo \"reload(python3 '{sp}' --lines $SCOPE --sort mod --query {{q}})"
        "+change-prompt(cursearch [$SCOPE|mod]$SEP)\"; "
        "fi"
    )

    # ** query change: python-side filtering with ordered smart-case semantics
    query_change = (
        "change:transform:"
        "SCOPE=$(echo \"$FZF_PROMPT\" | grep -oE 'all|user'); "
        "SORT=$(echo \"$FZF_PROMPT\" | grep -oE 'mod|cre'); "
        f"echo \"reload(python3 '{sp}' --lines $SCOPE --sort $SORT --query {{q}})\""
    )

    # ** ctrl-o: cycle preview mode via label; preview() is one-shot so it
    # doesn't bake in {1}. focus:transform below re-renders on every move.
    order_toggle = (
        "ctrl-o:transform:"
        f"if echo \"$FZF_PREVIEW_LABEL\" | grep -q latest; then "
        f"echo \"change-preview-label([ ↑ newest first ])"
        f"+preview(python3 '{sp}' --preview {{1}} --reverse)\"; "
        f"elif echo \"$FZF_PREVIEW_LABEL\" | grep -q newest; then "
        f"echo \"change-preview-label([ ↓ chronological ])"
        f"+preview(python3 '{sp}' --preview {{1}})\"; "
        f"else "
        f"echo \"change-preview-label([ ◎ latest match ])"
        f"+preview(python3 '{sp}' --preview {{1}} --latest-match --query {{q}})\"; "
        "fi"
    )

    # ** focus: rebuild preview command from scratch using the label to detect
    # mode and {1}/{q} for the focused item. Runs on every navigation.
    focus_preview = (
        "focus:transform:"
        f"if echo \"$FZF_PREVIEW_LABEL\" | grep -q latest; then "
        f"echo \"preview(python3 '{sp}' --preview {{1}} --latest-match --query {{q}})\"; "
        f"elif echo \"$FZF_PREVIEW_LABEL\" | grep -q chronological; then "
        f"echo \"preview(python3 '{sp}' --preview {{1}})\"; "
        f"else "
        f"echo \"preview(python3 '{sp}' --preview {{1}} --reverse)\"; "
        f"fi"
    )

    export_html_bind = (
        f"alt-h:execute-silent(python3 '{sp}' --export-html {{1}})"
    )
    export_org_bind = (
        f"ctrl-i:execute-silent(python3 '{sp}' --export-org {{1}})"
    )
    export_md_bind = (
        f"alt-m:execute-silent(python3 '{sp}' --export-md {{1}})"
    )

    # ** alt-s: summarize selected sessions via headless agent
    summarize_bind = (
        f"alt-s:execute(python3 '{sp}' --summarize {{+1}})"
    )

    # ** ?: show help overlay in preview pane (browse mode only)
    help_bind = (
        f"?:preview(python3 '{sp}' --help-overlay)"
        "+change-preview-label([ ? help ])"
    )

    header = (
        f"{DIM}⏎ resume  Tab browse  x/u select  ctrl-g skills  alt-s summary  ? help{RESET}"
    )
    input_text = join_lines(search_lines)

    try:
        result = subprocess.run(
            [
                "fzf",
                "--ansi",
                "--multi",
                "--disabled",
                "--read0",
                "--delimiter", "\t",
                "--with-nth", "2",
                "--expect", "ctrl-g",
                "--preview", f"python3 '{sp}' --preview {{1}} --latest-match --query {{q}}",
                "--preview-window", "right:50%:wrap",
                "--preview-label", "[ ◎ latest match ]",
                "--header", header,
                "--layout", "reverse",
                "--no-sort",
                "--tiebreak", "index",
                "--info", "inline",
                "--prompt", "cursearch [all|mod]> ",
                "--bind", enter_bind,
                "--bind", esc_bind,
                "--bind", scope_toggle,
                "--bind", sort_toggle,
                "--bind", query_change,
                "--bind", order_toggle,
                "--bind", export_html_bind,
                "--bind", export_org_bind,
                "--bind", export_md_bind,
                "--bind", summarize_bind,
                "--bind", help_bind,
                "--bind", focus_preview,
                "--bind", "j:down,k:up",
                "--bind", "x:toggle",
                "--bind", "u:deselect+down",
                "--bind", slash_bind,
                "--bind", "start:unbind(j,k,/,x,u,?)",
                "--bind", "ctrl-n:down,ctrl-p:up,alt-j:down,alt-k:up",
                "--bind", tab_bind,
                "--color", "dark",
                "--color", "fg:-1,bg:-1,hl:#a6e3a1,hl+:#a6e3a1:bold:underline",
                "--color", "fg+:#cdd6f4,bg+:#313244",
                "--color", "pointer:#89b4fa,marker:#a6e3a1:bold,prompt:#89b4fa",
                "--color", "info:#6c7086,spinner:#89b4fa,header:#6c7086:italic",
                "--color", "border:#45475a,preview-border:#45475a",
                "--color", "preview-label:#89b4fa,label:#89b4fa",
                "--color", "gutter:-1",
            ],
            input=input_text,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        print("Error: fzf not found. Install with: brew install fzf", file=sys.stderr)
        sys.exit(1)

    if result.returncode == 0 and result.stdout.strip():
        raw_lines = [ln for ln in result.stdout.strip().splitlines() if ln.strip()]
        if not raw_lines:
            return None, []

        action = "resume"
        selected_lines = raw_lines
        if raw_lines[0] == "ctrl-g":
            action = "create_skills"
            selected_lines = raw_lines[1:]

        filepaths = []
        seen_paths = set()
        for ln in selected_lines:
            # Ignore wrapped/multiline display spillover lines; real entries
            # always include the hidden filepath column and a tab delimiter.
            if "\t" not in ln:
                continue
            filepath = ln.split("\t", 1)[0]
            if not filepath or filepath in seen_paths:
                continue
            seen_paths.add(filepath)
            filepaths.append(filepath)
        if filepaths:
            return action, filepaths
    elif result.returncode not in (0, 1, 130):
        print(f"fzf error (exit {result.returncode}): {result.stderr.strip()}", file=sys.stderr)
    return None, []


# * Main

def emit_lines(scope, sort_by="modified", query=""):
    """Print fzf lines to stdout (null-delimited) — used by fzf reload."""
    sessions = scan_all_transcripts()
    lines = build_search_lines(sessions, scope=scope, sort_by=sort_by, query=query)
    sys.stdout.write(join_lines(lines))
    sys.stdout.flush()


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--preview":
        reverse = "--reverse" in sys.argv[3:]
        latest_match = "--latest-match" in sys.argv[3:]
        query = ""
        if "--query" in sys.argv[3:]:
            idx = sys.argv.index("--query")
            if idx + 1 < len(sys.argv):
                query = sys.argv[idx + 1]
        preview_session(sys.argv[2], reverse=reverse, query=query, latest_match=latest_match)
        return

    if len(sys.argv) >= 3 and sys.argv[1] == "--resume":
        resume_session(sys.argv[2])
        return

    if len(sys.argv) >= 3 and sys.argv[1] == "--export-html":
        export_html(sys.argv[2])
        return

    if len(sys.argv) >= 3 and sys.argv[1] == "--export-org":
        export_org(sys.argv[2])
        return

    if len(sys.argv) >= 3 and sys.argv[1] == "--export-md":
        export_markdown(sys.argv[2])
        return

    if len(sys.argv) >= 3 and sys.argv[1] == "--summarize":
        summarize_sessions(sys.argv[2:])
        return

    if sys.argv[1:2] == ["--help-overlay"]:
        print_help_overlay()
        return

    if len(sys.argv) >= 3 and sys.argv[1] == "--lines":
        sort_by = "modified"
        query = ""
        if len(sys.argv) >= 5 and sys.argv[3] == "--sort":
            sort_by = "created" if sys.argv[4] == "cre" else "modified"
        if "--query" in sys.argv:
            idx = sys.argv.index("--query")
            if idx + 1 < len(sys.argv):
                query = sys.argv[idx + 1]
        emit_lines(sys.argv[2], sort_by=sort_by, query=query)
        return

    print("Scanning Cursor agent transcripts...", file=sys.stderr)
    sessions = scan_all_transcripts()
    print(f"Found {len(sessions)} sessions.", file=sys.stderr)

    if not sessions:
        print("No transcripts found.", file=sys.stderr)
        return

    print("Building search index...", file=sys.stderr)
    lines = build_search_lines(sessions, scope="all", sort_by="modified", query="")

    action, selected = run_fzf(lines)
    if not selected:
        return
    if action == "create_skills":
        launch_create_skills_agent(selected)
        return
    resume_session(selected[0])


main()
