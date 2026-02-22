#!/usr/bin/env python3
"""cursearch — fuzzy search through Cursor CLI agent transcripts."""

import html as html_mod
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import webbrowser
from datetime import datetime
from pathlib import Path

# * Configuration

CURSOR_PROJECTS_DIR = Path.home() / ".cursor" / "projects"
HOME_PREFIX = str(Path.home()).lstrip("/").replace("/", "-") + "-"
MAX_SUMMARY_QUERIES = 5
MAX_READ_BYTES = 50_000

# * ANSI colors

BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
CYAN = "\033[36m"
GREEN = "\033[32m"
BLUE = "\033[34m"
YELLOW = "\033[33m"
WHITE = "\033[37m"
GRAY = "\033[90m"


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


def build_search_lines(sessions, scope="all"):
    """Build lines for fzf — one per session.

    Each line is a single fzf entry with tab-separated fields:
      filepath \\t visible_card_with_hidden_search_text

    The visible card is ANSI-formatted. The searchable blob is appended
    with extreme dim so fzf can match it but users never see it.
    """
    lines = []
    for s in sessions:
        messages = parse_transcript(s["filepath"], max_bytes=MAX_READ_BYTES)

        if scope == "user":
            filtered = [(r, t) for r, t in messages if r == "user"]
        elif scope == "assistant":
            filtered = [(r, t) for r, t in messages if r == "assistant"]
        else:
            filtered = messages

        modified_str = s["modified"].strftime("%Y-%m-%d %H:%M")
        created_str = s["created"].strftime("%m-%d %H:%M")
        summary = make_summary(messages)

        searchable = " ".join(strip_tags(t).replace("\n", " ") for _, t in filtered)
        if len(searchable) > 4000:
            searchable = searchable[:4000]

        card = (
            f"{BOLD}{YELLOW}mod{RESET} {YELLOW}{modified_str}{RESET}"
            f"  {DIM}cre {created_str}{RESET}"
            f"  {CYAN}{BOLD}{s['project']}{RESET}\n"
            f"  {DIM}{summary}{RESET}"
            f" {GRAY}{DIM}{searchable}{RESET}"
        )

        lines.append(f"{s['filepath']}\t{card}")

    return lines


def join_lines(lines):
    """Join lines with null byte for fzf --read0."""
    return "\0".join(lines)


# * Preview (called by fzf subprocess)

def preview_session(filepath, reverse=False):
    """Print a colored preview of a session transcript."""
    messages = parse_transcript(filepath)
    project_dir = Path(filepath).parent.parent.name
    project = short_project_name(project_dir)
    created, modified = get_file_times(filepath)

    header = (
        f"{CYAN}{BOLD}Project:{RESET}  {WHITE}{project}{RESET}\n"
        f"{CYAN}{BOLD}Created:{RESET}  {WHITE}{created.strftime('%Y-%m-%d %H:%M')}{RESET}\n"
        f"{CYAN}{BOLD}Modified:{RESET} {WHITE}{modified.strftime('%Y-%m-%d %H:%M')}{RESET}\n"
        f"{CYAN}{BOLD}File:{RESET}     {DIM}{filepath}{RESET}\n"
        f"{CYAN}{BOLD}Messages:{RESET} {WHITE}{len(messages)}{RESET}\n"
        f"{GRAY}{'─' * 60}{RESET}"
    )

    display_msgs = list(reversed(messages)) if reverse else messages

    msg_lines = []
    for role, text in display_msgs:
        text = strip_tags(text)
        if not text:
            continue
        if role == "assistant" and len(text) > 400:
            text = text[:397] + "..."
        wrapped = textwrap.fill(text, width=72)
        if role == "user":
            msg_lines.append(f"\n{GREEN}{BOLD}[USER]{RESET}\n{wrapped}")
        else:
            msg_lines.append(f"\n{BLUE}[ASST]{RESET}\n{DIM}{wrapped}{RESET}")

    print(header)
    print("\n".join(msg_lines))


# * HTML export

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Cursor Session — {project} — {modified}</title>
<style>
  :root {{ --bg: #1e1e2e; --fg: #cdd6f4; --user-bg: #1e3a2e; --asst-bg: #1e2a3e;
           --user-label: #a6e3a1; --asst-label: #89b4fa; --meta: #6c7086; --border: #313244; }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; font-size: 14px;
          background: var(--bg); color: var(--fg); line-height: 1.6; padding: 0; }}
  .header {{ background: #11111b; padding: 20px 24px; border-bottom: 1px solid var(--border); }}
  .header h1 {{ font-size: 18px; color: #cba6f7; margin-bottom: 8px; }}
  .header .meta {{ color: var(--meta); font-size: 13px; }}
  .messages {{ max-width: 900px; margin: 0 auto; padding: 16px; }}
  .msg {{ padding: 12px 16px; margin: 8px 0; border-radius: 8px; white-space: pre-wrap;
          word-wrap: break-word; }}
  .msg.user {{ background: var(--user-bg); border-left: 3px solid var(--user-label); }}
  .msg.asst {{ background: var(--asst-bg); border-left: 3px solid var(--asst-label); }}
  .role {{ font-weight: 700; font-size: 12px; text-transform: uppercase; margin-bottom: 6px; }}
  .msg.user .role {{ color: var(--user-label); }}
  .msg.asst .role {{ color: var(--asst-label); }}
  .msg .body {{ color: var(--fg); }}
</style>
</head>
<body>
<div class="header">
  <h1>{project}</h1>
  <div class="meta">Created: {created} &nbsp;|&nbsp; Modified: {modified} &nbsp;|&nbsp; Messages: {count}</div>
</div>
<div class="messages">
{messages_html}
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
        escaped = html_mod.escape(text)
        msg_blocks.append(
            f'<div class="msg {css_class}">'
            f'<div class="role">{label}</div>'
            f'<div class="body">{escaped}</div>'
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


def export_org(filepath):
    """Export a session transcript to Org-mode and open in Emacs/default editor."""
    messages = parse_transcript(filepath)
    project_dir = Path(filepath).parent.parent.name
    project = short_project_name(project_dir)
    created, modified = get_file_times(filepath)

    lines = [
        f"#+title: Cursor Session — {project}",
        f"#+date: [{created.strftime('%Y-%m-%d %a')}]",
        "#+options: toc:nil num:nil ^:{}",
        "",
        f"- *Project:* {project}",
        f"- *Created:* [{created.strftime('%Y-%m-%d %a %H:%M')}]",
        f"- *Modified:* [{modified.strftime('%Y-%m-%d %a %H:%M')}]",
        f"- *Messages:* {len(messages)}",
        "",
    ]

    for role, text in messages:
        text = strip_tags(text)
        if not text:
            continue
        if role == "user":
            heading = make_heading_text(text)
            lines.append(f"* {heading}")
            if text.strip() != heading.rstrip("..."):
                lines.append("")
                lines.append(text)
        else:
            lines.append(f"** Assistant")
            lines.append("")
            lines.append(text)
        lines.append("")

    fd, tmp_path = tempfile.mkstemp(suffix=".org", prefix="cursearch_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    subprocess.Popen(["open", tmp_path])


# * FZF integration

def run_fzf(search_lines):
    """Launch fzf with session cards, preview pane, and keybindings.

    Column layout: 1=filepath  2=card_with_search_text
    fzf shows only column 2 (the card). Column 1 is hidden but extractable.
    """
    script_path = os.path.abspath(__file__)

    scope_toggle = (
        "`:transform:"
        f"if [[ $FZF_PROMPT =~ all ]]; then "
        f"echo \"reload(python3 '{script_path}' --lines user)"
        "+change-prompt(cursearch [user]> )\"; "
        f"else "
        f"echo \"reload(python3 '{script_path}' --lines all)"
        "+change-prompt(cursearch [all]> )\"; "
        "fi"
    )

    order_toggle = (
        "ctrl-o:transform:"
        f"if [[ $FZF_PREVIEW_LABEL =~ chronological ]]; then "
        f"echo \"change-preview(python3 '{script_path}' --preview {{1}} --reverse)"
        "+change-preview-label([ ↑ newest first ])\"; "
        f"else "
        f"echo \"change-preview(python3 '{script_path}' --preview {{1}})"
        "+change-preview-label([ ↓ chronological ])\"; "
        "fi"
    )

    export_html_bind = (
        f"alt-enter:execute-silent(python3 '{script_path}' --export-html {{1}})"
    )
    export_org_bind = (
        f"ctrl-i:execute-silent(python3 '{script_path}' --export-org {{1}})"
    )

    header = (
        f"{DIM}esc browse  / search  ` scope  ctrl-o order  "
        f"alt-⏎ html  ctrl-i org  ctrl-y copy{RESET}"
    )
    input_text = join_lines(search_lines)

    try:
        result = subprocess.run(
            [
                "fzf",
                "--ansi",
                "--exact",
                "--read0",
                "--delimiter", "\t",
                "--with-nth", "2",
                "--preview", f"python3 '{script_path}' --preview {{1}} --reverse",
                "--preview-window", "right:50%:wrap",
                "--preview-label", "[ ↑ newest first ]",
                "--header", header,
                "--layout", "reverse",
                "--info", "inline",
                "--prompt", "cursearch [all]> ",
                "--bind", scope_toggle,
                "--bind", order_toggle,
                "--bind", export_html_bind,
                "--bind", export_org_bind,
                "--bind", "j:down,k:up",
                "--bind", "start:unbind(j,k,/)",
                "--bind", "esc:rebind(j,k,/)+disable-search",
                "--bind", "/:unbind(j,k,/)+enable-search",
                "--bind", "ctrl-n:down,ctrl-p:up",
                "--bind", "ctrl-y:execute-silent(echo {1} | pbcopy)",
                "--color", "header:italic:dim",
            ],
            input=input_text,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        print("Error: fzf not found. Install with: brew install fzf", file=sys.stderr)
        sys.exit(1)

    if result.returncode == 0 and result.stdout.strip():
        parts = result.stdout.strip().split("\t")
        if parts:
            return parts[0]
    elif result.returncode not in (0, 1, 130):
        print(f"fzf error (exit {result.returncode}): {result.stderr.strip()}", file=sys.stderr)
    return None


# * Main

def emit_lines(scope):
    """Print fzf lines to stdout (null-delimited) — used by fzf reload."""
    sessions = scan_all_transcripts()
    lines = build_search_lines(sessions, scope=scope)
    sys.stdout.write(join_lines(lines))
    sys.stdout.flush()


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--preview":
        reverse = "--reverse" in sys.argv[3:]
        preview_session(sys.argv[2], reverse=reverse)
        return

    if len(sys.argv) >= 3 and sys.argv[1] == "--export-html":
        export_html(sys.argv[2])
        return

    if len(sys.argv) >= 3 and sys.argv[1] == "--export-org":
        export_org(sys.argv[2])
        return

    if len(sys.argv) >= 3 and sys.argv[1] == "--lines":
        emit_lines(sys.argv[2])
        return

    print("Scanning Cursor agent transcripts...", file=sys.stderr)
    sessions = scan_all_transcripts()
    print(f"Found {len(sessions)} sessions.", file=sys.stderr)

    if not sessions:
        print("No transcripts found.", file=sys.stderr)
        return

    print("Building search index...", file=sys.stderr)
    lines = build_search_lines(sessions, scope="all")

    selected = run_fzf(lines)
    if selected:
        print(f"\n{selected}")


main()
