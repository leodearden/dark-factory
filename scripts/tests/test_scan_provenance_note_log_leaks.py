"""Tests for scripts/scan_provenance_note_log_leaks.py — the READ-ONLY,
detection-only sweep for leaked server-log lines in a task's
``metadata.done_provenance.note``.

Task 3286: ``DeterministicRunner._run_predicate`` stamped a predicate
script's raw 2000-char stdout tail straight into ``done_provenance.note``,
and fused-memory's ``_format_outcome_echo`` then appended that note to a Mem0
completion-summary write — so a chatty script's server-log noise was ingested
into memory. Task 2902 is the confirmed specimen (1 of 264 notes in the live
DB at the time of writing). The source is fixed; this scanner is the durable
recurrence guard, mirroring the 2939 precedent.

Mirrors test_scan_task_toolcall_leaks.py: pure functions (detect_log_leak,
scan_db, discover_db_paths, format_report, format_json) get direct pytest
coverage; main() gets subprocess coverage.

This module — and the scanner it tests — never mutate task text. Task 2902's
note is a preserved forensic specimen; the fixture below is a shape-faithful
ABRIDGEMENT reproduced from the plan's analysis, never a live re-read.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from scan_provenance_note_log_leaks import (
    NoteLeakMatch,
    detect_log_leak,
    discover_db_paths,
    format_json,
    format_report,
    scan_db,
)

# The tasks-table schema and the fake-db builder live in scripts/tests/
# conftest.py as the `tasks_table_schema` / `make_tasks_db` fixtures (task
# 3336) — they were previously copied near-identically into all three
# sweep-script test files.

# ---------------------------------------------------------------------------
# Genuine-leak fixture: the task-2902 shape, abridged.
#
# What makes it a leak is the LINE SHAPE — timestamp, logger name, level —
# not the presence of any particular logger name (see the false-positive
# guard below, which is exactly why the discriminator is structural).
# ---------------------------------------------------------------------------

POLLUTED_NOTE = (
    "_tariff_pence_per_kwh' in group 'my_solar_challenge' (exact-name "
    'identity gate should prevent this — investigate)\n'
    '2026-07-30 16:39:00,523 fused_memory.backends.graphiti_client WARNING '
    "identity scan found 3 candidate nodes for 'import_tariff_pence_per_kwh'\n"
    '2026-07-30 16:39:00,625 httpx INFO HTTP Request: GET '
    'http://localhost:6333 "HTTP/1.1 200 OK"\n'
    '{\n  "dry_run": true,\n  "orphan_count": 0\n}'
)

# ---------------------------------------------------------------------------
# Clean fixtures: every real `done_provenance.note` shape observed in the live
# DB, including the bounded verdict the fixed orchestrator now writes.
# ---------------------------------------------------------------------------

CLEAN_NOTES = [
    'predicate check passed (rc=0): check ok: 0 flakes',
    'predicate check passed (rc=0): {"dry_run":true,"orphan_count":0}',
    'resumed after gate resolution',
    'pure gate resolved',
    'covered by sibling task',
    'resumed after verified deploy (crash before done write)',
]

# FALSE-POSITIVE GUARD. Task 3286's own description quotes these marker names
# as PROSE while instructing the reader to grep for them. A naive substring
# scan flags exactly this text — which is why the discriminator requires full
# log-line shape, the same discipline LEAK_TAIL encodes in the 2939 precedent
# (whose own docstring records the tasks-2938/2939 over-reporting lesson).
PROSE_MENTIONING_MARKERS = (
    'Root-cause the leak: grep task metadata.done_provenance.note for '
    "server-log markers like 'fused_memory.backends.graphiti_client' / "
    "'httpx INFO HTTP Request' and report any task whose note carries them. "
    'A WARNING-level line from an unrelated project is the tell.'
)

# ---------------------------------------------------------------------------
# RECURRENCE fixtures: what a leak would look like in a note written by the
# FIXED orchestrator.
#
# These matter more than the historical POLLUTED_NOTE above. Every post-fix
# note is a SINGLE line prefixed `predicate check passed (rc=0): `, so a
# recurrence can only ever appear mid-line — a line-start-anchored
# discriminator would report such a note clean and the guard would fire only
# on legacy multi-line notes, i.e. never again.
# ---------------------------------------------------------------------------

PREFIXED_LEAK_NOTE = (
    'predicate check passed (rc=0): 2026-07-30 16:39:00,523 httpx INFO '
    'HTTP Request: GET http://localhost:6333 "HTTP/1.1 200 OK"'
)

# A tier-1 JSON payload whose string value embeds a log line: the note is one
# line and the embedded newline is the two-character escape `\n`, so the log
# line is neither at a line start nor preceded by real whitespace.
JSON_EMBEDDED_LEAK_NOTE = (
    'predicate check passed (rc=0): {"verdict":"clean","tail":"scan done'
    '\\n2026-07-30 16:39:00,523 fused_memory.backends.graphiti_client '
    'WARNING identity scan found 3 candidate nodes"}'
)


class TestDetectLogLeak:
    """The detector keys on log-LINE SHAPE, not on marker substrings."""

    def test_specimen_is_detected(self):
        leak = detect_log_leak(POLLUTED_NOTE)

        assert leak is not None
        assert isinstance(leak, str)

    def test_specimen_returns_the_matched_log_line(self):
        leak = detect_log_leak(POLLUTED_NOTE)

        assert leak is not None
        assert leak.startswith('2026-07-30 16:39:00,523'), leak
        assert 'fused_memory.backends.graphiti_client' in leak, leak
        # One line, not the whole blob.
        assert '\n' not in leak, leak

    def test_clean_notes_are_not_flagged(self):
        for note in CLEAN_NOTES:
            assert detect_log_leak(note) is None, note

    def test_leak_after_the_verdict_prefix_is_detected(self):
        """The recurrence case: a mid-line leak in a post-fix note shape.

        Anchoring at a line start would report this clean, leaving the guard
        able to fire only on legacy multi-line notes — the opposite of its
        stated purpose.
        """
        leak = detect_log_leak(PREFIXED_LEAK_NOTE)

        assert leak is not None, PREFIXED_LEAK_NOTE
        # The reported match is the log line itself, not the whole note: the
        # `predicate check passed` prefix is not part of the evidence.
        assert leak.startswith('2026-07-30 16:39:00,523 httpx INFO'), leak
        assert '\n' not in leak, leak

    def test_leak_embedded_in_a_json_payload_is_detected(self):
        """A tier-1 JSON payload can still smuggle a log line in a value."""
        leak = detect_log_leak(JSON_EMBEDDED_LEAK_NOTE)

        assert leak is not None, JSON_EMBEDDED_LEAK_NOTE
        assert 'fused_memory.backends.graphiti_client' in leak, leak

    def test_prose_mentioning_markers_is_not_flagged(self):
        """The guard against the naive-substring implementation.

        This text NAMES every marker but carries no log-line shape. A scanner
        that regressed to a substring check would flag task 3286 itself.
        """
        assert detect_log_leak(PROSE_MENTIONING_MARKERS) is None

    def test_degenerate_input_returns_none_without_raising(self):
        """Callers pass raw sqlite3 row values straight through."""
        for value in ('', None, 42, b'2026-07-30 16:39:00,523 x INFO y', []):
            assert detect_log_leak(value) is None, value


class TestNoteLeakMatch:
    """The match record carries enough to locate the note without re-reading."""

    def test_fields(self):
        match = NoteLeakMatch(
            db_path='/tmp/tasks.db',
            tag='master',
            task_id=2902,
            provenance_kind='deterministic-milestone',
            leak_line='2026-07-30 16:39:00,523 httpx INFO HTTP Request: GET /',
        )

        assert match.db_path == '/tmp/tasks.db'
        assert match.tag == 'master'
        assert match.task_id == 2902
        assert match.provenance_kind == 'deterministic-milestone'
        assert 'httpx' in match.leak_line


# ---------------------------------------------------------------------------
# Fixture-DB helpers
# ---------------------------------------------------------------------------

def _provenance(kind, **fields):
    return json.dumps({'done_provenance': {'kind': kind, **fields}})


# Every row shape scan_db must tolerate. Only task 2902 is a leak.
_MIXED_ROWS = [
    {'id': 2902, 'metadata': _provenance('deterministic-milestone', note=POLLUTED_NOTE)},
    {'id': 2903, 'metadata': _provenance('deterministic-milestone', note=CLEAN_NOTES[0])},
    # A merged provenance carries a commit but no note at all.
    {'id': 2904, 'metadata': _provenance('merged', commit='43b9439262')},
    # Valid JSON object, but no done_provenance key.
    {'id': 2905, 'metadata': json.dumps({'task_kind': 'deterministic'})},
    # Valid JSON, but a scalar / an array rather than an object.
    {'id': 2906, 'metadata': json.dumps('deterministic')},
    {'id': 2907, 'metadata': json.dumps([1, 2, 3])},
    # done_provenance present but not a dict.
    {'id': 2908, 'metadata': json.dumps({'done_provenance': 'deterministic-milestone'})},
    # Malformed, non-JSON metadata.
    {'id': 2909, 'metadata': '{not valid json'},
    # NULL metadata.
    {'id': 2910, 'metadata': None},
]


class TestScanDb:
    """scan_db reads metadata.done_provenance.note, read-only, tolerantly."""

    def test_finds_exactly_the_polluted_note(self, make_tasks_db):
        matches = scan_db(str(make_tasks_db(_MIXED_ROWS)))

        assert len(matches) == 1, matches
        match = matches[0]
        assert match.task_id == 2902
        assert match.tag == 'master'
        assert match.provenance_kind == 'deterministic-milestone'
        assert 'fused_memory.backends.graphiti_client' in match.leak_line

    def test_malformed_null_and_absent_rows_are_skipped_without_raising(self, make_tasks_db):
        """A single unparseable blob must not abort the sweep."""
        rows = [row for row in _MIXED_ROWS if row['id'] != 2902]

        assert scan_db(str(make_tasks_db(rows))) == []

    def test_clean_db_yields_no_matches(self, make_tasks_db):
        rows = [
            {'id': index, 'metadata': _provenance('deterministic-milestone', note=note)}
            for index, note in enumerate(CLEAN_NOTES, start=1)
        ]

        assert scan_db(str(make_tasks_db(rows))) == []

    def test_recurrence_in_a_post_fix_note_shape_is_found(self, make_tasks_db):
        """End-to-end on the shape the guard actually exists to catch.

        A note written by the FIXED orchestrator is one line prefixed
        `predicate check passed (rc=0): `; a leak recurring there sits
        mid-line, and must still be reported alongside the clean rows.
        """
        rows = [
            {'id': 3001, 'metadata': _provenance('deterministic-milestone', note=PREFIXED_LEAK_NOTE)},
            {'id': 3002, 'metadata': _provenance('deterministic-milestone', note=CLEAN_NOTES[0])},
            {'id': 3003, 'metadata': _provenance('deterministic-milestone', note=CLEAN_NOTES[1])},
        ]

        matches = scan_db(str(make_tasks_db(rows)))

        assert [m.task_id for m in matches] == [3001], matches
        assert matches[0].leak_line.startswith('2026-07-30'), matches[0]


class TestDiscoverDbPaths:
    """Same precedence contract as the 2939 precedent scanner."""

    def test_explicit_dbs_pass_through_existing_only(self, tmp_path):
        existing = tmp_path / 'a.db'
        existing.write_text('')
        missing = tmp_path / 'missing.db'

        result = discover_db_paths(explicit_dbs=[str(existing), str(missing)])

        assert result == [str(existing)]

    def test_project_root_maps_to_taskmaster_tasks_db(self, tmp_path, project_root_with_tasks_db):
        root = tmp_path / 'proj'
        root.mkdir()
        db_file = str(project_root_with_tasks_db(root))

        assert discover_db_paths(project_roots=[str(root)]) == [db_file]

    def test_parses_dashboard_known_project_roots_env(self, tmp_path, project_root_with_tasks_db):
        root_a, root_b = tmp_path / 'a', tmp_path / 'b'
        root_a.mkdir()
        root_b.mkdir()
        db_a, db_b = str(project_root_with_tasks_db(root_a)), str(project_root_with_tasks_db(root_b))

        # Whitespace padded, with an empty entry (",,") that must be dropped
        # rather than mapped to a bogus db path.
        env = {'DASHBOARD_KNOWN_PROJECT_ROOTS': f' {root_a} , {root_b} ,, '}

        assert discover_db_paths(env=env) == [db_a, db_b]

    def test_falls_back_to_dark_factory_default(self, monkeypatch):
        monkeypatch.delenv('DASHBOARD_KNOWN_PROJECT_ROOTS', raising=False)

        assert discover_db_paths() == [
            '/home/leo/src/dark-factory/.taskmaster/tasks/tasks.db'
        ]

    def test_skips_project_root_without_tasks_db(self, tmp_path):
        root = tmp_path / 'empty'
        root.mkdir()

        assert discover_db_paths(project_roots=[str(root)]) == []


class TestFormatting:
    """The human report truncates; --json carries the full line."""

    LONG_LINE = (
        '2026-07-30 16:39:00,523 fused_memory.backends.graphiti_client WARNING '
        'identity scan found 3 candidate nodes for a very long entity name '
        'that keeps going well past any sane report width'
    )

    def _matches(self):
        return [
            NoteLeakMatch('/a/tasks.db', 'master', 2902, 'deterministic-milestone',
                          self.LONG_LINE),
            NoteLeakMatch('/b/tasks.db', 'master', 3001, 'deterministic-deploy',
                          self.LONG_LINE),
        ]

    def test_empty_list_yields_explicit_no_leaks_message(self):
        report = format_report([])

        assert 'no ' in report.lower()
        assert 'leak' in report.lower()

    def test_report_groups_by_db_path_and_shows_match_fields(self):
        report = format_report(self._matches())

        assert '/a/tasks.db' in report
        assert '/b/tasks.db' in report
        for match in self._matches():
            assert str(match.task_id) in report
            assert match.provenance_kind in report

    def test_report_truncates_the_leak_line(self):
        report = format_report(self._matches(), max_line_len=40)

        assert self.LONG_LINE not in report
        assert '...' in report

    def test_format_json_round_trips_with_the_full_line(self):
        payload = json.loads(format_json(self._matches()))

        assert len(payload) == 2
        assert payload[0]['task_id'] == 2902
        assert payload[0]['leak_line'] == self.LONG_LINE
        assert payload[0]['provenance_kind'] == 'deterministic-milestone'

    def test_format_json_empty_list_is_empty_array(self):
        assert json.loads(format_json([])) == []


# ---------------------------------------------------------------------------
# CLI (main), driven via subprocess.run — mirrors the precedent suite.
# ---------------------------------------------------------------------------

SCRIPT = Path(__file__).parent.parent / 'scan_provenance_note_log_leaks.py'


def _run_cli(*args, env=None, timeout=30):
    child_env = {**os.environ, **(env or {})}
    return subprocess.run(
        ['python3', str(SCRIPT), *args],
        capture_output=True, text=True, timeout=timeout, env=child_env,
    )


class TestCli:
    """Exit codes: 0 = clean, 1 = leak found, 2 = no tasks.db resolvable."""

    def test_clean_db_exits_0_with_no_leaks_message(self, make_tasks_db):
        db_path = str(make_tasks_db([{'id': 1, 'metadata': _provenance('merged', commit='abc')}]))

        result = _run_cli('--db', db_path)

        assert result.returncode == 0, result.stderr
        assert 'no leaked log lines' in result.stdout

    def test_polluted_db_exits_1_and_names_the_task(self, make_tasks_db):
        db_path = str(make_tasks_db(_MIXED_ROWS))
        before = Path(db_path).read_bytes()

        result = _run_cli('--db', db_path)

        assert result.returncode == 1, result.stderr
        assert '2902' in result.stdout
        # Detection-only: the scan must not have touched the file.
        assert Path(db_path).read_bytes() == before

    def test_json_flag_emits_full_untruncated_leak_line(self, make_tasks_db):
        db_path = str(make_tasks_db(_MIXED_ROWS))

        result = _run_cli('--db', db_path, '--json')

        assert result.returncode == 1, result.stderr
        payload = json.loads(result.stdout)
        assert len(payload) == 1
        assert payload[0]['task_id'] == 2902
        # The FULL line, not the report's truncation.
        assert payload[0]['leak_line'] in POLLUTED_NOTE
        assert '...' not in payload[0]['leak_line']

    def test_no_resolvable_db_exits_2(self, tmp_path):
        result = _run_cli(
            '--db', str(tmp_path / 'nonexistent' / 'tasks.db'),
            env={'DASHBOARD_KNOWN_PROJECT_ROOTS': ''},
        )

        assert result.returncode == 2
        assert 'no tasks.db resolvable' in result.stderr

    def test_unreadable_db_is_warned_and_skipped_not_fatal(self, tmp_path, make_tasks_db):
        """One corrupt file must not abort the sweep over the others."""
        corrupt = tmp_path / 'corrupt.db'
        corrupt.write_text('this is not a sqlite database at all')
        good = str(make_tasks_db(_MIXED_ROWS, name='good.db'))

        result = _run_cli('--db', str(corrupt), '--db', good)

        # The good db was still scanned and its leak still reported.
        assert result.returncode == 1, result.stderr
        assert '2902' in result.stdout
        assert 'corrupt.db' in result.stderr
        assert 'incomplete' in result.stderr
