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

from scan_provenance_note_log_leaks import (
    NoteLeakMatch,
    detect_log_leak,
)

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
