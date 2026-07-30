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
import sqlite3

from scan_provenance_note_log_leaks import (
    NoteLeakMatch,
    detect_log_leak,
    discover_db_paths,
    format_json,
    format_report,
    scan_db,
)

# Minimal reproduction of the live tasks table schema (columns + NOT NULL
# constraints only — see fused-memory's sqlite_task_backend.py _SCHEMA_SQL).
# `metadata TEXT` is the column that matters here; the precedent scanner
# deliberately never reads it.
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

def _make_db(tmp_path, rows, name='tasks.db'):
    """Build a fixture tasks.db. *rows* are (id, metadata_text) pairs."""
    db_path = tmp_path / name
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(_TASKS_SCHEMA)
        for task_id, metadata in rows:
            conn.execute(
                'INSERT INTO tasks (tag, id, title, status, metadata, updated_at) '
                "VALUES ('master', ?, ?, 'done', ?, '2026-07-30T00:00:00Z')",
                (task_id, f'Task {task_id}', metadata),
            )
        conn.commit()
    finally:
        conn.close()
    return str(db_path)


def _provenance(kind, **fields):
    return json.dumps({'done_provenance': {'kind': kind, **fields}})


# Every row shape scan_db must tolerate. Only task 2902 is a leak.
_MIXED_ROWS = [
    (2902, _provenance('deterministic-milestone', note=POLLUTED_NOTE)),
    (2903, _provenance('deterministic-milestone', note=CLEAN_NOTES[0])),
    # A merged provenance carries a commit but no note at all.
    (2904, _provenance('merged', commit='43b9439262')),
    # Valid JSON object, but no done_provenance key.
    (2905, json.dumps({'task_kind': 'deterministic'})),
    # Valid JSON, but a scalar / an array rather than an object.
    (2906, json.dumps('deterministic')),
    (2907, json.dumps([1, 2, 3])),
    # done_provenance present but not a dict.
    (2908, json.dumps({'done_provenance': 'deterministic-milestone'})),
    # Malformed, non-JSON metadata.
    (2909, '{not valid json'),
    # NULL metadata.
    (2910, None),
]


class TestScanDb:
    """scan_db reads metadata.done_provenance.note, read-only, tolerantly."""

    def test_finds_exactly_the_polluted_note(self, tmp_path):
        matches = scan_db(_make_db(tmp_path, _MIXED_ROWS))

        assert len(matches) == 1, matches
        match = matches[0]
        assert match.task_id == 2902
        assert match.tag == 'master'
        assert match.provenance_kind == 'deterministic-milestone'
        assert 'fused_memory.backends.graphiti_client' in match.leak_line

    def test_malformed_null_and_absent_rows_are_skipped_without_raising(self, tmp_path):
        """A single unparseable blob must not abort the sweep."""
        rows = [row for row in _MIXED_ROWS if row[0] != 2902]

        assert scan_db(_make_db(tmp_path, rows)) == []

    def test_clean_db_yields_no_matches(self, tmp_path):
        rows = [
            (index, _provenance('deterministic-milestone', note=note))
            for index, note in enumerate(CLEAN_NOTES, start=1)
        ]

        assert scan_db(_make_db(tmp_path, rows)) == []


class TestDiscoverDbPaths:
    """Same precedence contract as the 2939 precedent scanner."""

    def _touch(self, root):
        db = root / '.taskmaster' / 'tasks' / 'tasks.db'
        db.parent.mkdir(parents=True, exist_ok=True)
        db.write_text('')
        return str(db)

    def test_explicit_dbs_pass_through_existing_only(self, tmp_path):
        existing = tmp_path / 'a.db'
        existing.write_text('')
        missing = tmp_path / 'missing.db'

        result = discover_db_paths(explicit_dbs=[str(existing), str(missing)])

        assert result == [str(existing)]

    def test_project_root_maps_to_taskmaster_tasks_db(self, tmp_path):
        root = tmp_path / 'proj'
        root.mkdir()
        db_file = self._touch(root)

        assert discover_db_paths(project_roots=[str(root)]) == [db_file]

    def test_parses_dashboard_known_project_roots_env(self, tmp_path):
        root_a, root_b = tmp_path / 'a', tmp_path / 'b'
        root_a.mkdir()
        root_b.mkdir()
        db_a, db_b = self._touch(root_a), self._touch(root_b)

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
