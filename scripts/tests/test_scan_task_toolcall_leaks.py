"""Tests for scripts/scan_task_toolcall_leaks.py — the READ-ONLY,
detection-only sweep for leaked serialized tool-call XML fragments (e.g.
``</description>\\n<parameter name="priority">low``) in Taskmaster task text
columns.

Task 2939: builds the proactive sweep after the pattern recurred on ~32 live
tasks (first seen as an apparent one-off on task 2080, then recurring on
task 2865). This module — and its tests — never mutate task text; they only
detect and report.

Mirrors test_recon_busy_check.py: pure functions (detect_leak, scan_db,
discover_db_paths, format_report, format_json) get direct pytest coverage;
main() gets subprocess coverage (added alongside the CLI in a later step).

Fixture shapes below are modeled on the real live-DB leak shapes confirmed
while planning this task (tasks 992, 1068, 1067, 2691) and the real
false-positive prose mentions (tasks 2938/2939) — not invented shapes.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

import _task_db_scan
import scan_task_toolcall_leaks
from scan_task_toolcall_leaks import (
    LeakMatch,
    detect_leak,
    discover_db_paths,
    format_json,
    format_report,
    scan_db,
)

# Minimal reproduction of the live tasks table schema (columns + NOT NULL
# constraints only — see fused-memory's sqlite_task_backend.py _SCHEMA_SQL).
# Only the columns scan_db actually reads/needs are included.
_TASKS_SCHEMA = """
CREATE TABLE tasks (
    tag           TEXT NOT NULL DEFAULT 'master',
    id            INTEGER NOT NULL,
    title         TEXT NOT NULL,
    description   TEXT,
    details       TEXT,
    test_strategy TEXT,
    status        TEXT NOT NULL,
    metadata      TEXT,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (tag, id)
);
"""

# ---------------------------------------------------------------------------
# Genuine-leak fixtures: a stray closing tag, a REAL newline, then one or more
# serialized <parameter name="..."> fragments running to end-of-string.
# ---------------------------------------------------------------------------

# Shape (a) — task 992: a description leak ending in a bare priority param.
GENUINE_DESCRIPTION_LEAK_FRAGMENT = '</description>\n<parameter name="priority">low'
GENUINE_DESCRIPTION_LEAK = (
    "Add a new test that writes >limit records and asserts only the newest "
    "`limit` are materialised."
) + GENUINE_DESCRIPTION_LEAK_FRAGMENT

# Shape (b) — task 1068: a chained leak where the stray tag is itself
# </parameter> rather than </description>.
CHAINED_PARAMETER_LEAK_FRAGMENT = '</parameter>\n<parameter name="priority">high'
CHAINED_PARAMETER_LEAK = (
    "Root cause is documented in a code comment; a regression test exercises "
    "the orphaned-commit path."
) + CHAINED_PARAMETER_LEAK_FRAGMENT

# Shape (c) — task 1067: the leak lands in `details` (closing tag </details>).
DETAILS_LEAK_FRAGMENT = '</details>\n<parameter name="priority">polish'
DETAILS_LEAK_TEXT = (
    "Verify with `pytest fused-memory/tests/test_bulk_reset_guard.py -q`."
) + DETAILS_LEAK_FRAGMENT

# Shape (d) — task 2691: the "swallowed-details" variant, where an entire
# serialized <parameter name="details">...</parameter> tool-call payload
# leaked into `description` and runs all the way to end-of-string, while the
# real `details` column is left empty.
SWALLOWED_DETAILS_FRAGMENT = (
    '</description>\n<parameter name="details">Run fused-memory/scripts/'
    "audit_duplicate_memories.py or a direct Qdrant admin query against "
    "memory_id=028edb1f-299c-4755-9438-deadbeefcafe (null category, "
    "unretrievable content matching the fingerprint) to confirm-safe the "
    "deletion.\n\nClose out this one-off cleanup once resolved so it stops "
    "recurring in Stage 1/2 payloads every cycle."
)
SWALLOWED_DETAILS_LEAK = (
    "The dashboard shows an orphaned memory row with a null agent_id and "
    "unretrievable content matching the fingerprint."
) + SWALLOWED_DETAILS_FRAGMENT

# ---------------------------------------------------------------------------
# False-positive fixtures: prose that MENTIONS the fragment (quoting the
# ESCAPED literal backslash-n — two real characters, "\" then "n" — not an
# actual newline) and keeps going with trailing prose afterward. Exact
# 2938/2939 shape.
# ---------------------------------------------------------------------------

PROSE_MENTION_FALSE_POSITIVE = (
    "Stage 1 found that task 2865's description had a leaked plain-text "
    'tool-call/XML fragment appended: `</description>\\n<parameter '
    'name="priority">low`. Stage 2 (this run) verified the fragment '
    "verbatim via live get_task and stripped it from the description via "
    "update_task."
)

BARE_PARAMETER_MENTION = (
    'The bug report mentions <parameter name="priority"> appearing in raw '
    "form somewhere upstream, but this description itself is clean."
)

CLEAN_TEXT = "This is a perfectly normal task description with no XML leakage at all."

# Zero-whitespace adjacency: a closing tag immediately followed by
# `<parameter name=...>` with no real whitespace character in between. No
# live leak has ever been observed in this shape (the recon serialization
# bug always emits a real newline — see the module docstring); LEAK_TAIL
# requires `\s+` (one or more real whitespace chars), so this must NOT match.
ZERO_WHITESPACE_ADJACENT_TAG = (
    'Some clean lead-in text.</description><parameter name="priority">low'
)


# ---------------------------------------------------------------------------
# detect_leak(text) -> str | None
# ---------------------------------------------------------------------------

def test_detect_leak_returns_fragment_for_description_leak():
    assert detect_leak(GENUINE_DESCRIPTION_LEAK) == GENUINE_DESCRIPTION_LEAK_FRAGMENT


def test_detect_leak_returns_fragment_for_chained_parameter_leak():
    assert detect_leak(CHAINED_PARAMETER_LEAK) == CHAINED_PARAMETER_LEAK_FRAGMENT


def test_detect_leak_returns_fragment_for_details_leak():
    assert detect_leak(DETAILS_LEAK_TEXT) == DETAILS_LEAK_FRAGMENT


def test_detect_leak_returns_fragment_for_swallowed_details_leak():
    assert detect_leak(SWALLOWED_DETAILS_LEAK) == SWALLOWED_DETAILS_FRAGMENT


def test_detect_leak_returns_none_for_prose_mention_false_positive():
    """The exact 2938/2939 shape: escaped literal backslash-n (not a real
    newline) plus trailing prose after the fragment-looking substring."""
    assert detect_leak(PROSE_MENTION_FALSE_POSITIVE) is None


def test_detect_leak_returns_none_for_bare_parameter_mid_prose():
    assert detect_leak(BARE_PARAMETER_MENTION) is None


def test_detect_leak_returns_none_for_clean_text():
    assert detect_leak(CLEAN_TEXT) is None


def test_detect_leak_returns_none_for_empty_string():
    assert detect_leak("") is None


def test_detect_leak_returns_none_for_none():
    assert detect_leak(None) is None


def test_trailing_whitespace_after_fragment_is_tolerated():
    """detect_leak rstrips before matching, so trailing whitespace tacked on
    after the fragment (e.g. a task text with stray trailing blank lines)
    does not defeat detection."""
    text_with_trailing_ws = GENUINE_DESCRIPTION_LEAK + "\n\n   \n"
    assert detect_leak(text_with_trailing_ws) == GENUINE_DESCRIPTION_LEAK_FRAGMENT


def test_detect_leak_returns_none_for_zero_whitespace_adjacent_tag():
    """LEAK_TAIL requires >=1 real whitespace character between the closing
    tag and the parameter fragment (`\\s+`, not `\\s*`); a zero-whitespace
    adjacency — a shape no genuine leak has ever produced — must not match."""
    assert detect_leak(ZERO_WHITESPACE_ADJACENT_TAG) is None


# ---------------------------------------------------------------------------
# scan_db(db_path) -> list[LeakMatch]
# ---------------------------------------------------------------------------

def _make_db(tmp_path, rows):
    """Build a temp sqlite tasks.db at tmp_path/'tasks.db' seeded with *rows*.

    Each row is a dict of column -> value; title/status/updated_at (the
    NOT NULL columns) default to per-id placeholders when omitted.
    """
    db_path = tmp_path / "tasks.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_TASKS_SCHEMA)
        for row in rows:
            conn.execute(
                "INSERT INTO tasks (tag, id, title, description, details, "
                "test_strategy, status, metadata, updated_at) VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row.get("tag", "master"),
                    row["id"],
                    row.get("title", f"Task {row['id']}"),
                    row.get("description"),
                    row.get("details"),
                    row.get("test_strategy"),
                    row.get("status", "pending"),
                    row.get("metadata"),
                    row.get("updated_at", "2026-07-22T00:00:00+00:00"),
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return db_path


def test_scan_db_finds_only_genuine_leaks_and_is_read_only(tmp_path):
    rows = [
        {"id": 1, "description": "Clean task, nothing wrong here."},
        {"id": 2, "description": GENUINE_DESCRIPTION_LEAK},
        {"id": 3, "description": "Clean prose.", "details": DETAILS_LEAK_TEXT},
        {"id": 4, "description": SWALLOWED_DETAILS_LEAK, "details": ""},
        {"id": 5, "description": PROSE_MENTION_FALSE_POSITIVE},
        {
            "id": 6,
            "description": "Clean.",
            "status": "done",
            # Proves metadata is NOT scanned: this is exactly the shape task
            # 2865's remediation record carries, and must be excluded.
            "metadata": json.dumps(
                {
                    "stage2_description_corruption_fix": {
                        "stripped_fragment": GENUINE_DESCRIPTION_LEAK_FRAGMENT,
                    },
                },
            ),
        },
    ]
    db_path = _make_db(tmp_path, rows)
    before = db_path.read_bytes()

    matches = scan_db(str(db_path))

    after = db_path.read_bytes()
    assert after == before, "scan_db must not mutate the database file"

    by_task_id = {m.task_id: m for m in matches}
    assert set(by_task_id) == {2, 3, 4}, (
        "rows 1 (clean), 5 (prose mention), and 6 (metadata-only) must be excluded"
    )

    assert by_task_id[2].column == "description"
    assert by_task_id[2].fragment == GENUINE_DESCRIPTION_LEAK_FRAGMENT
    assert by_task_id[2].tag == "master"
    assert by_task_id[2].db_path == str(db_path)

    assert by_task_id[3].column == "details"
    assert by_task_id[3].fragment == DETAILS_LEAK_FRAGMENT

    assert by_task_id[4].column == "description"
    assert by_task_id[4].fragment == SWALLOWED_DETAILS_FRAGMENT


def test_scan_db_returns_empty_list_for_clean_db(tmp_path):
    db_path = _make_db(tmp_path, [{"id": 1, "description": "All clean here."}])
    assert scan_db(str(db_path)) == []


# ---------------------------------------------------------------------------
# discover_db_paths(explicit_dbs, project_roots, env) -> list[str]
# ---------------------------------------------------------------------------

def _touch_tasks_db(project_root):
    """Create an (empty-content) tasks.db under project_root/.taskmaster/tasks/."""
    db_dir = project_root / ".taskmaster" / "tasks"
    db_dir.mkdir(parents=True)
    db_file = db_dir / "tasks.db"
    db_file.write_text("")
    return db_file


def test_discover_db_paths_explicit_dbs_passed_through_existing_only(tmp_path):
    existing = tmp_path / "a.db"
    existing.write_text("")
    missing = tmp_path / "missing.db"

    result = discover_db_paths(explicit_dbs=[str(existing), str(missing)])

    assert result == [str(existing)]


def test_discover_db_paths_project_root_maps_to_taskmaster_tasks_db(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    db_file = _touch_tasks_db(root)

    result = discover_db_paths(project_roots=[str(root)])

    assert result == [str(db_file)]


def test_discover_db_paths_parses_dashboard_known_project_roots_env(tmp_path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    db_a = _touch_tasks_db(root_a)
    db_b = _touch_tasks_db(root_b)

    # Comma-separated, whitespace padded, with an empty entry (",,") that
    # must be dropped rather than mapped to a bogus db path.
    env = {"DASHBOARD_KNOWN_PROJECT_ROOTS": f" {root_a} , {root_b} ,, "}

    result = discover_db_paths(env=env)

    assert result == [str(db_a), str(db_b)]


def test_discover_db_paths_falls_back_to_dark_factory_default(monkeypatch):
    monkeypatch.delenv("DASHBOARD_KNOWN_PROJECT_ROOTS", raising=False)

    result = discover_db_paths()

    assert result == ["/home/leo/src/dark-factory/.taskmaster/tasks/tasks.db"]


def test_discover_db_paths_skips_project_root_without_tasks_db(tmp_path):
    root = tmp_path / "empty_proj"
    root.mkdir()

    result = discover_db_paths(project_roots=[str(root)])

    assert result == []


# ---------------------------------------------------------------------------
# format_report(matches) -> str / format_json(matches) -> str
# ---------------------------------------------------------------------------

def test_format_report_empty_list_yields_explicit_no_leaks_message():
    report = format_report([])
    assert "no leaked tool-call fragments found" in report


def test_format_report_groups_by_db_path_and_shows_match_fields():
    matches = [
        LeakMatch("/a/tasks.db", "master", 2, "description", GENUINE_DESCRIPTION_LEAK_FRAGMENT),
        LeakMatch("/a/tasks.db", "master", 3, "details", DETAILS_LEAK_FRAGMENT),
        LeakMatch("/b/tasks.db", "master", 10, "description", CHAINED_PARAMETER_LEAK_FRAGMENT),
    ]

    report = format_report(matches)

    assert "/a/tasks.db" in report
    assert "/b/tasks.db" in report
    for m in matches:
        assert str(m.task_id) in report
        assert m.tag in report
        assert m.column in report
    assert "3 leaked fragments across 3 tasks" in report


def test_format_report_summary_counts_distinct_tasks_not_fragments():
    """A task with leaks in two columns must count once toward "tasks", even
    though it contributes two lines/fragments to the report."""
    matches = [
        LeakMatch("/a/tasks.db", "master", 5, "description", GENUINE_DESCRIPTION_LEAK_FRAGMENT),
        LeakMatch("/a/tasks.db", "master", 5, "details", DETAILS_LEAK_FRAGMENT),
        LeakMatch("/a/tasks.db", "master", 6, "description", CHAINED_PARAMETER_LEAK_FRAGMENT),
    ]

    report = format_report(matches)

    assert "3 leaked fragments across 2 tasks" in report


def test_format_report_truncates_long_fragment():
    long_fragment = '</description>\n<parameter name="details">' + "x" * 500
    matches = [LeakMatch("/a/tasks.db", "master", 1, "description", long_fragment)]

    report = format_report(matches)

    assert "x" * 500 not in report, "the full fragment must be truncated, not dumped verbatim"


def test_format_json_is_parseable_and_carries_full_untruncated_fragment():
    matches = [
        LeakMatch("/a/tasks.db", "master", 2, "description", SWALLOWED_DETAILS_FRAGMENT),
    ]

    payload = format_json(matches)
    parsed = json.loads(payload)

    assert parsed == [
        {
            "db_path": "/a/tasks.db",
            "tag": "master",
            "task_id": 2,
            "column": "description",
            "fragment": SWALLOWED_DETAILS_FRAGMENT,
        },
    ]


def test_format_json_empty_list_is_empty_array():
    assert json.loads(format_json([])) == []


# ---------------------------------------------------------------------------
# CLI (main), driven via subprocess.run — mirrors test_recon_busy_check.py
# ---------------------------------------------------------------------------

SCRIPT = Path(__file__).parent.parent / "scan_task_toolcall_leaks.py"


def _run_cli(*args, timeout=10):
    return subprocess.run(
        ["python3", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_cli_leaky_db_exits_1_with_task_id_in_stdout_and_does_not_mutate(tmp_path):
    db_path = _make_db(tmp_path, [{"id": 992, "description": GENUINE_DESCRIPTION_LEAK}])
    before = db_path.read_bytes()

    result = _run_cli("--db", str(db_path))

    after = db_path.read_bytes()
    assert after == before, "CLI run must not mutate the database file"
    assert result.returncode == 1, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "992" in result.stdout


def test_cli_clean_db_exits_0_with_no_leaks_message(tmp_path):
    db_path = _make_db(tmp_path, [{"id": 1, "description": "Nothing wrong here."}])

    result = _run_cli("--db", str(db_path))

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "no leaked tool-call fragments found" in result.stdout


def test_cli_json_flag_exits_1_and_emits_parseable_json(tmp_path):
    db_path = _make_db(tmp_path, [{"id": 992, "description": GENUINE_DESCRIPTION_LEAK}])

    result = _run_cli("--db", str(db_path), "--json")

    assert result.returncode == 1, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    parsed = json.loads(result.stdout)
    assert len(parsed) == 1
    assert parsed[0]["task_id"] == 992
    assert parsed[0]["fragment"] == GENUINE_DESCRIPTION_LEAK_FRAGMENT


def test_cli_no_resolvable_db_exits_2():
    result = _run_cli("--db", "/nonexistent.db")

    assert result.returncode == 2, f"stdout={result.stdout!r} stderr={result.stderr!r}"


def test_cli_continues_past_unreadable_db_and_warns_on_stderr(tmp_path):
    """A corrupt/unreadable tasks.db (e.g. a stale WAL-less file, or a
    genuinely non-sqlite file at that path) must not abort the whole sweep:
    the CLI logs a warning naming the bad db to stderr and continues, still
    reporting leaks found in the other, readable database(s)."""
    corrupt_db = tmp_path / "corrupt.db"
    corrupt_db.write_bytes(b"not a sqlite database, just garbage bytes")
    leaky_db = _make_db(tmp_path, [{"id": 992, "description": GENUINE_DESCRIPTION_LEAK}])

    result = _run_cli("--db", str(corrupt_db), "--db", str(leaky_db))

    assert result.returncode == 1, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "992" in result.stdout
    assert str(corrupt_db) in result.stderr


# ---------------------------------------------------------------------------
# Shared-plumbing adoption contract (task 3336).
# ---------------------------------------------------------------------------

def test_module_sources_its_discovery_and_json_from_the_shared_module():
    """Task 3286's finding is that a PARTIAL extraction defeats the point: a
    private copy left behind keeps drifting while every other test still
    passes. So "this script sources its plumbing from _task_db_scan" is itself
    a behaviour worth pinning.

    Identity (`is`), not a source grep or a docstring match — a re-pasted copy
    with a byte-identical body satisfies every other assertion in this file but
    cannot satisfy this one.
    """
    assert scan_task_toolcall_leaks.discover_db_paths is _task_db_scan.discover_db_paths
    assert scan_task_toolcall_leaks.format_json is _task_db_scan.format_json


def test_the_private_discovery_copies_are_gone_not_merely_shadowed():
    """Catches the halfway state where the shared import lands on top of a
    still-present private copy: the module attribute would look right while
    the dead code sits underneath waiting to be re-wired."""
    assert not hasattr(scan_task_toolcall_leaks, "_DEFAULT_PROJECT_ROOTS")
    assert not hasattr(scan_task_toolcall_leaks, "_tasks_db_path")
