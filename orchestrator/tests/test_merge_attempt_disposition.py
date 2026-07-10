"""Merge-skew attribution — γ (mechanism M2, event/stats separation).

Proves that ``_emit_merge_attempt`` can carry an optional ``disposition``
payload key (the α ``MergeFailureDisposition`` enum, task 2381) that is
persisted into the ``events`` table of runs.db, and that omitting it keeps
existing callers' payloads byte-identical (no phantom ``'disposition': null``
key).  See plans/merge-skew-attribution-prd.md (task γ, boundary row 7).

γ proves this signal independently of β (task 2383, which threads
``MergeOutcome.disposition`` into the ~19 production call sites) via a
direct-emit unit test — it does not require β's forced-skew scenario.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from orchestrator.event_store import EventStore
from orchestrator.merge_disposition import MergeFailureDisposition
from orchestrator.merge_queue import _emit_merge_attempt
from orchestrator.merge_types import OutcomeKind


class TestEmitMergeAttemptDisposition:
    """Unit tests for the optional ``disposition`` kwarg of ``_emit_merge_attempt``."""

    def test_disposition_persisted_when_provided(self, tmp_path: Path) -> None:
        """A supplied disposition is persisted and readable via json_extract.

        Also proves the existing ``'outcome'`` key is still present alongside
        it (additive, not a replacement).
        """
        db_path = tmp_path / 'events.db'
        store = EventStore(db_path=db_path, run_id='test-run')

        _emit_merge_attempt(
            store, 'T1', OutcomeKind.verify_failed,
            disposition=MergeFailureDisposition.INTEGRATION_SKEW,
        )

        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute(
                "SELECT json_extract(data, '$.disposition'), "
                "       json_extract(data, '$.outcome') "
                "FROM events WHERE event_type = 'merge_attempt'"
            ).fetchall()
        finally:
            conn.close()

        assert rows == [('integration_skew', 'verify_failed')], f'rows={rows}'

    def test_disposition_absent_when_not_supplied(self, tmp_path: Path) -> None:
        """Omitting disposition leaves the key out of the payload entirely.

        Backward-compat: existing ~19 call sites that do not pass
        ``disposition`` must not gain a phantom ``'disposition': null`` key.
        """
        db_path = tmp_path / 'events.db'
        store = EventStore(db_path=db_path, run_id='test-run')

        _emit_merge_attempt(store, 'T2', OutcomeKind.verify_failed)

        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute(
                "SELECT json_extract(data, '$.disposition') FROM events "
                "WHERE event_type = 'merge_attempt'"
            ).fetchall()
        finally:
            conn.close()

        assert rows == [(None,)], f'rows={rows}'
