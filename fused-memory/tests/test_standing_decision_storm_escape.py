"""Tests for the standing-decision storm escapes (tasks 2896 and 2943).

The per-cycle escape and the cross-cycle streak escape both file through a
REAL ``EscalationQueue`` on tmp_path, and the streak state lives on a REAL
recon ledger: a fold, a ``dedupe_count`` and a value surviving from one cycle
to the next are what these contracts are about, and a mock can witness none
of them.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from fused_memory.reconciliation import standing_decision_storm_escape as storm_escape
from fused_memory.reconciliation.flag_dedup import EntityStandingSuppressionResult
from fused_memory.reconciliation.recon_ledger import ReconLedgerRecord, ReconLedgerStore
from fused_memory.reconciliation.standing_decision_constants import (
    CATEGORY_STANDING_DECISION_STORM,
    GROUNDS_STRUCTURAL_SIZE_CONFLATION,
    RECORD_KIND_ENTITY_SUPPRESSION_STREAK,
    STREAK_PAYLOAD_KEY,
    STREAK_WINDOW_PAYLOAD_KEY,
    SUPPRESSION_STORM_THRESHOLD_PER_CYCLE,
    SUPPRESSION_STREAK_THRESHOLD_CYCLES,
    SUPPRESSION_STREAK_VOLUME_THRESHOLD,
)

_ESD_U1 = 'b0057f3d-1234-4abc-8def-0123456789ab'
_ESD_U2 = 'a1b2c3d4-5678-4901-8234-567890abcdef'


@pytest_asyncio.fixture
async def ledger_memory_service(tmp_path):
    """A memory_service stand-in carrying a REAL initialized ReconLedgerStore
    as ``.recon_ledger``."""
    ledger = ReconLedgerStore(tmp_path / 'reconciliation.db')
    await ledger.initialize()
    service = AsyncMock()
    service.recon_ledger = ledger
    try:
        yield service
    finally:
        await ledger.close()


# ---------------------------------------------------------------------------
# maybe_escalate_suppression_storm (Hook A storm escape / γ, task 2896) — step-7
# ---------------------------------------------------------------------------


def _storm_result(
    *, entity_uuid: str = _ESD_U1, count: int | None = None, extra: dict | None = None
) -> EntityStandingSuppressionResult:
    """An EntityStandingSuppressionResult carrying *count* suppressions for one
    (or, via *extra*, several) decision(s).  Defaults to one over the threshold."""
    counts = {entity_uuid: SUPPRESSION_STORM_THRESHOLD_PER_CYCLE + 1 if count is None else count}
    counts.update(extra or {})
    return EntityStandingSuppressionResult(
        kept_flags=[],
        suppressed_by_decision=counts,
        grounds_by_decision={u: GROUNDS_STRUCTURAL_SIZE_CONFLATION for u in counts},
        suppression_evaluated=True,
    )


class TestMaybeEscalateSuppressionStorm:
    """Per-cycle, per-decision storm escape escalation (task 2896 step-7).

    Driven against a REAL ``EscalationQueue`` on tmp_path rather than a
    MagicMock: the filing path folds through ``submit_or_dedupe``, whose whole
    contract is what the queue does on the SECOND cycle, and a mock that answers
    every lookup with a mock cannot witness a fold, a dedupe_count, or the
    agreement between the key written and the key read back.
    """

    _PID = 'p'
    _RUN = 'run-1'

    @pytest.fixture
    def queue(self, tmp_path):
        from escalation.queue import EscalationQueue

        return EscalationQueue(tmp_path / 'escalations')

    @staticmethod
    def _pending(queue, entity_uuid: str = _ESD_U1) -> list:
        return queue.get_by_task(entity_uuid, status='pending', level=1)

    @pytest.mark.asyncio
    async def test_over_threshold_files_one_escalation(self, queue):
        """(a) count > threshold → exactly one L1 storm escalation for that uuid."""
        escalated = await storm_escape.maybe_escalate_suppression_storm(
            queue, self._PID, self._RUN, _storm_result()
        )
        assert escalated == [_ESD_U1]

        pending = self._pending(queue)
        assert len(pending) == 1
        esc = pending[0]
        assert esc.level == 1
        assert esc.severity == 'blocking'
        assert esc.category == CATEGORY_STANDING_DECISION_STORM
        assert esc.agent_role == 'reconciliation-stage1'
        # The entity is the record's subject: task_id is the key get_by_task /
        # has_open_l1 read a storm record back by, so a filing that left it ''
        # would be unfindable per entity.  Pinned explicitly, not merely implied
        # by the get_by_task lookup above, so the field cannot be repurposed
        # silently (reviewer finding test-coverage, amendment pass).
        assert esc.task_id == _ESD_U1
        blob = f'{esc.summary}\n{esc.detail}'
        assert _ESD_U1 in blob
        assert GROUNDS_STRUCTURAL_SIZE_CONFLATION in blob
        assert str(SUPPRESSION_STORM_THRESHOLD_PER_CYCLE + 1) in blob

    @pytest.mark.asyncio
    async def test_at_threshold_does_not_escalate(self, queue):
        """(b) count == threshold (strict >) → nothing filed, returns []."""
        escalated = await storm_escape.maybe_escalate_suppression_storm(
            queue, self._PID, self._RUN,
            _storm_result(count=SUPPRESSION_STORM_THRESHOLD_PER_CYCLE),
        )
        assert escalated == []
        assert self._pending(queue) == []

    @pytest.mark.asyncio
    async def test_below_threshold_does_not_escalate(self, queue):
        """(b') count well below threshold → nothing filed, returns []."""
        escalated = await storm_escape.maybe_escalate_suppression_storm(
            queue, self._PID, self._RUN, _storm_result(count=1)
        )
        assert escalated == []
        assert self._pending(queue) == []

    @pytest.mark.asyncio
    async def test_two_entities_over_threshold_each_file_once(self, queue):
        """Two decisions storming in ONE cycle each get their own record.

        The fold key is per-entity, so a second storming entity must not be
        mistaken for a recurrence of the first.
        """
        escalated = await storm_escape.maybe_escalate_suppression_storm(
            queue, self._PID, self._RUN,
            _storm_result(extra={_ESD_U2: SUPPRESSION_STORM_THRESHOLD_PER_CYCLE + 3}),
        )
        assert sorted(escalated) == sorted([_ESD_U1, _ESD_U2])
        assert len(self._pending(queue, _ESD_U1)) == 1
        assert len(self._pending(queue, _ESD_U2)) == 1
        first, second = self._pending(queue, _ESD_U1)[0], self._pending(queue, _ESD_U2)[0]
        assert first.dedupe_fingerprint != second.dedupe_fingerprint

    @pytest.mark.asyncio
    async def test_escalation_unavailable_returns_empty(self, queue, monkeypatch):
        """(d) Escalation package unavailable (None) → returns [], no raise, nothing filed."""
        monkeypatch.setattr(storm_escape, 'Escalation', None)
        escalated = await storm_escape.maybe_escalate_suppression_storm(
            queue, self._PID, self._RUN, _storm_result()
        )
        assert escalated == []
        assert self._pending(queue) == []

    @pytest.mark.asyncio
    async def test_submit_failure_logs_warning_and_excludes_entity(self, tmp_path, caplog):
        """A queue whose submit raises costs the entity its filing, not the cycle.

        The helper is best-effort by contract: the failure is logged WARNING
        naming the entity, the entity is absent from the returned list, and the
        exception never escapes into the Stage-1 run.
        """
        from escalation.queue import EscalationQueue

        class _BrokenQueue(EscalationQueue):
            def submit(self, escalation):
                raise RuntimeError('boom')

        queue = _BrokenQueue(tmp_path / 'escalations')
        with caplog.at_level(
            logging.WARNING, logger='fused_memory.reconciliation.standing_decision_storm_escape'
        ):
            escalated = await storm_escape.maybe_escalate_suppression_storm(
                queue, self._PID, self._RUN, _storm_result()
            )
        assert escalated == []
        assert any(
            rec.levelno == logging.WARNING and _ESD_U1 in rec.getMessage()
            for rec in caplog.records
        ), 'a WARNING naming the entity must be logged'

    @pytest.mark.asyncio
    async def test_recurring_storm_folds_into_one_parent(self, queue):
        """REGRESSION: a decision that keeps storming folds, it is not re-filed.

        Two things are pinned here, and only a real queue can pin either.  (1)
        The record is written under the key the reader queries — an earlier
        revision wrote ``task_id=''`` while looking up by entity_uuid, which made
        the guard a permanent no-op.  (2) Recurrence FOLDS rather than being
        silently skipped (task 3522): the second cycle mints no second record and
        increments ``dedupe_count`` on the first, which is the steward's
        triage-order signal.  A ``has_open_l1`` skip would leave that count
        pinned at 0 forever, making one storm indistinguishable from forty.
        """
        result = _storm_result()
        assert await storm_escape.maybe_escalate_suppression_storm(
            queue, self._PID, self._RUN, result
        ) == [_ESD_U1]
        assert queue.has_open_l1(_ESD_U1, category=CATEGORY_STANDING_DECISION_STORM)

        second = await storm_escape.maybe_escalate_suppression_storm(
            queue, self._PID, 'run-2', result
        )
        assert second == [], 'a folded recurrence is not a new filing'

        pending = self._pending(queue)
        assert len(pending) == 1, f'expected one storm escalation, got {len(pending)}'
        assert pending[0].dedupe_count == 1, 'the recurrence must be counted on the parent'


# ---------------------------------------------------------------------------
# update_suppression_streaks — cross-cycle streak state (task 2943)
# ---------------------------------------------------------------------------
# Successive calls stand in for successive full Stage-1 cycles.  Driven against
# the real ledger because the whole feature is a value surviving from one cycle
# to the next, which a mock cannot witness.

_STREAK_PID = 'p'
_STREAK_NOW = '2026-06-01T00:00:00+00:00'


def _suppressing_counts(counts: dict[str, int]) -> EntityStandingSuppressionResult:
    """A fully evaluated cycle in which each named decision suppressed its count of flags."""
    return EntityStandingSuppressionResult(
        kept_flags=[],
        suppressed_by_decision=dict(counts),
        grounds_by_decision={u: GROUNDS_STRUCTURAL_SIZE_CONFLATION for u in counts},
        suppression_evaluated=True,
    )


def _suppressing(*entity_uuids: str) -> EntityStandingSuppressionResult:
    """A fully evaluated cycle in which each named decision suppressed one flag."""
    return _suppressing_counts({u: 1 for u in entity_uuids})


async def _stored_streaks(ledger: ReconLedgerStore) -> dict[str, int]:
    return {row.entity_uuid: row.streak for row in await ledger.list_suppression_streaks(_STREAK_PID)}


async def _stored_windows(ledger: ReconLedgerStore) -> dict[str, list[int]]:
    return {
        row.entity_uuid: list(row.recent_counts)
        for row in await ledger.list_suppression_streaks(_STREAK_PID)
    }


async def _stored_streak_record(
    ledger: ReconLedgerStore, entity_uuid: str = _ESD_U1
) -> ReconLedgerRecord | None:
    """The raw streak record, for what the decoded row does not carry
    (write time, expiry)."""
    return await ledger.get_by_identity(
        _STREAK_PID,
        RECORD_KIND_ENTITY_SUPPRESSION_STREAK,
        task_id='',
        flag_type=GROUNDS_STRUCTURAL_SIZE_CONFLATION,
        run_id=entity_uuid,
    )


async def _seed_raw_streak_payload(ledger: ReconLedgerStore, payload: dict[str, Any]) -> None:
    """Write a streak row for _ESD_U1 past the ledger's validation, as a
    hand-edited or older-shaped row would arrive."""
    await ledger.upsert(
        ReconLedgerRecord(
            project_id=_STREAK_PID,
            record_kind=RECORD_KIND_ENTITY_SUPPRESSION_STREAK,
            payload_json=json.dumps(payload),
            state='active',
            created_at=_STREAK_NOW,
            flag_type=GROUNDS_STRUCTURAL_SIZE_CONFLATION,
            run_id=_ESD_U1,
            entity_uuid=_ESD_U1,
            expires_at='2099-01-01T00:00:00+00:00',
        )
    )


async def _run_cycle(
    memory_service: Any,
    run_id: str,
    result: EntityStandingSuppressionResult,
    *,
    now: str = _STREAK_NOW,
) -> list:
    return await storm_escape.update_suppression_streaks(
        memory_service, _STREAK_PID, run_id, result, now=now
    )


async def _run_consecutive_cycles(
    memory_service: Any, counts: list[int] | tuple[int, ...]
) -> list[storm_escape.SuppressionStreakUpdate]:
    """One cycle per entry of *counts*, runs run-1, run-2, ..., in which _ESD_U1's
    decision suppresses that many flags; returns its update from each cycle."""
    updates = []
    for n, count in enumerate(counts, start=1):
        (update,) = await _run_cycle(
            memory_service, f'run-{n}', _suppressing_counts({_ESD_U1: count})
        )
        updates.append(update)
    return updates


class TestUpdateSuppressionStreaks:
    """Increment, reset and replay rules of the streak state (task 2943 step-7)."""

    @pytest.mark.asyncio
    async def test_first_suppressing_cycle_persists_streak_one(self, ledger_memory_service):
        updates = await _run_cycle(ledger_memory_service, 'run-1', _suppressing(_ESD_U1))

        assert updates == [
            storm_escape.SuppressionStreakUpdate(
                entity_uuid=_ESD_U1,
                grounds=GROUNDS_STRUCTURAL_SIZE_CONFLATION,
                streak=1,
                window_suppressed=1,
                escalate=False,
            )
        ]
        ledger = ledger_memory_service.recon_ledger
        (row,) = await ledger.list_suppression_streaks(_STREAK_PID)
        assert (row.streak, row.recent_counts, row.last_run_id) == (1, (1,), 'run-1')
        record = await _stored_streak_record(ledger)
        assert record is not None
        assert record.expires_at == '2026-08-30T00:00:00+00:00', (
            'expires_at is the write time plus STANDING_DECISION_TTL_DAYS'
        )

    @pytest.mark.asyncio
    async def test_escalates_from_the_kth_consecutive_cycle_and_keeps_counting(
        self, ledger_memory_service
    ):
        """At two flags per cycle the window exceeds N from the Kth cycle
        (inclusive >= K), and escalating does not reset the counter."""
        assert SUPPRESSION_STREAK_THRESHOLD_CYCLES == 3
        updates = await _run_consecutive_cycles(ledger_memory_service, [2] * 4)

        assert [(u.streak, u.escalate) for u in updates] == [
            (1, False),
            (2, False),
            (3, True),
            (4, True),
        ]
        assert await _stored_streaks(ledger_memory_service.recon_ledger) == {_ESD_U1: 4}

    @pytest.mark.asyncio
    async def test_quiet_cycle_resets_and_the_next_streak_restarts_at_one(
        self, ledger_memory_service
    ):
        ledger = ledger_memory_service.recon_ledger
        await _run_cycle(ledger_memory_service, 'run-1', _suppressing(_ESD_U1))
        await _run_cycle(ledger_memory_service, 'run-2', _suppressing(_ESD_U1))

        updates = await _run_cycle(ledger_memory_service, 'run-3', _suppressing(_ESD_U2))

        assert await _stored_streaks(ledger) == {_ESD_U1: 0, _ESD_U2: 1}
        assert {u.entity_uuid: (u.streak, u.escalate) for u in updates} == {
            _ESD_U1: (0, False),
            _ESD_U2: (1, False),
        }

        (restart,) = [
            u
            for u in await _run_cycle(ledger_memory_service, 'run-4', _suppressing(_ESD_U1))
            if u.entity_uuid == _ESD_U1
        ]
        assert restart.streak == 1

    @pytest.mark.asyncio
    async def test_already_zero_row_is_not_rewritten_on_a_quiet_cycle(
        self, ledger_memory_service
    ):
        """No needless TTL refresh for a decision that has gone quiet."""
        ledger = ledger_memory_service.recon_ledger
        await _run_cycle(ledger_memory_service, 'run-1', _suppressing(_ESD_U1))
        await _run_cycle(
            ledger_memory_service,
            'run-2',
            EntityStandingSuppressionResult.empty_batch(),
            now='2026-06-02T00:00:00+00:00',
        )
        reset_record = await _stored_streak_record(ledger)

        updates = await _run_cycle(
            ledger_memory_service,
            'run-3',
            EntityStandingSuppressionResult.empty_batch(),
            now='2026-06-03T00:00:00+00:00',
        )

        assert updates == []
        assert reset_record is not None
        assert await _stored_streak_record(ledger) == reset_record

    @pytest.mark.asyncio
    async def test_replayed_run_id_does_not_increment(self, ledger_memory_service):
        await _run_consecutive_cycles(ledger_memory_service, [2, 2, 2])

        (replayed,) = await _run_cycle(
            ledger_memory_service, 'run-3', _suppressing_counts({_ESD_U1: 2})
        )

        assert (replayed.streak, replayed.escalate) == (3, True)
        assert await _stored_streaks(ledger_memory_service.recon_ledger) == {_ESD_U1: 3}

    @pytest.mark.asyncio
    async def test_two_entities_accumulate_independent_streaks(self, ledger_memory_service):
        both = {_ESD_U1: 2, _ESD_U2: 2}
        await _run_cycle(ledger_memory_service, 'run-1', _suppressing_counts({_ESD_U1: 2}))
        await _run_cycle(ledger_memory_service, 'run-2', _suppressing_counts(both))
        updates = await _run_cycle(ledger_memory_service, 'run-3', _suppressing_counts(both))

        assert {u.entity_uuid: (u.streak, u.escalate) for u in updates} == {
            _ESD_U1: (3, True),
            _ESD_U2: (2, False),
        }
        assert await _stored_streaks(ledger_memory_service.recon_ledger) == {
            _ESD_U1: 3,
            _ESD_U2: 2,
        }

    @pytest.mark.asyncio
    async def test_one_flag_per_cycle_never_escalates_however_long_the_streak(
        self, ledger_memory_service
    ):
        """The review-round-1 regression. A decision that works suppresses its
        re-derived complaint about once per cycle, indefinitely (PRD §Goal).
        Its window of the last K cycles sums to K <= N, so it never files."""
        updates = await _run_consecutive_cycles(ledger_memory_service, [1] * 6)

        assert [(u.streak, u.window_suppressed, u.escalate) for u in updates] == [
            (1, 1, False),
            (2, 2, False),
            (3, 3, False),
            (4, 3, False),
            (5, 3, False),
            (6, 3, False),
        ]
        assert await _stored_windows(ledger_memory_service.recon_ledger) == {
            _ESD_U1: [1, 1, 1]
        }

    @pytest.mark.asyncio
    async def test_sustained_volume_escalates_on_the_kth_cycle(self, ledger_memory_service):
        updates = await _run_consecutive_cycles(ledger_memory_service, [2, 2, 2])

        assert [(u.streak, u.window_suppressed, u.escalate) for u in updates] == [
            (1, 2, False),
            (2, 4, False),
            (3, 6, True),
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('counts', 'escalate'),
        [
            pytest.param((2, 2, 1), False, id='total-equal-to-n'),
            pytest.param((2, 2, 2), True, id='total-over-n'),
        ],
    )
    async def test_window_volume_must_strictly_exceed_n(
        self, ledger_memory_service, counts, escalate
    ):
        assert SUPPRESSION_STREAK_VOLUME_THRESHOLD == 5
        *_, last = await _run_consecutive_cycles(ledger_memory_service, counts)

        assert (last.window_suppressed, last.escalate) == (sum(counts), escalate)

    @pytest.mark.asyncio
    async def test_window_slides_back_under_n_without_a_reset(self, ledger_memory_service):
        """A decision that drops back toward its steady-state rate stops
        escalating on the next cycle, while its streak keeps counting."""
        updates = await _run_consecutive_cycles(ledger_memory_service, [2, 2, 2, 1])

        assert [(u.streak, u.window_suppressed, u.escalate) for u in updates[2:]] == [
            (3, 6, True),
            (4, 5, False),
        ]
        assert await _stored_windows(ledger_memory_service.recon_ledger) == {
            _ESD_U1: [2, 2, 1]
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('counts', 'window_suppressed', 'escalate'),
        [
            pytest.param((1, 1, 6), 2, False, id='flood-owned-by-the-per-cycle-escape'),
            pytest.param((6, 5, 5), 10, True, id='the-sub-n-cycles-are-a-drain-themselves'),
        ],
    )
    async def test_a_cycle_over_the_per_cycle_n_is_left_out_of_the_window_total(
        self, ledger_memory_service, counts, window_suppressed, escalate
    ):
        """One flood must not page both escapes in the same cycle, so a cycle
        the per-cycle escape already reported is not counted. The stored window
        still keeps that cycle's raw count."""
        assert SUPPRESSION_STORM_THRESHOLD_PER_CYCLE == 5
        *_, last = await _run_consecutive_cycles(ledger_memory_service, counts)

        assert (last.window_suppressed, last.escalate) == (window_suppressed, escalate)
        assert await _stored_windows(ledger_memory_service.recon_ledger) == {
            _ESD_U1: list(counts)
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('per_cycle_threshold', 'window_suppressed', 'escalate'),
        [
            pytest.param(SUPPRESSION_STORM_THRESHOLD_PER_CYCLE, 12, True, id='default-n'),
            pytest.param(3, 0, False, id='tuned-n-owns-every-cycle'),
        ],
    )
    async def test_the_per_cycle_cutoff_is_the_one_it_is_given(
        self, ledger_memory_service, per_cycle_threshold, window_suppressed, escalate
    ):
        """The cutoff must be the N the per-cycle escape actually runs with; a
        fixed cutoff would let one flood page both escapes once N is tuned."""
        for n in (1, 2, 3):
            (last,) = await storm_escape.update_suppression_streaks(
                ledger_memory_service,
                _STREAK_PID,
                f'run-{n}',
                _suppressing_counts({_ESD_U1: 4}),
                now=_STREAK_NOW,
                per_cycle_threshold=per_cycle_threshold,
            )

        assert (last.streak, last.window_suppressed, last.escalate) == (
            3,
            window_suppressed,
            escalate,
        )

    @pytest.mark.asyncio
    async def test_an_offset_now_is_written_in_utc(self, ledger_memory_service):
        """gc() compares expires_at as TEXT against a UTC now, so every stored
        timestamp is UTC whatever offset the caller's now carried."""
        await _run_cycle(
            ledger_memory_service,
            'run-1',
            _suppressing(_ESD_U1),
            now='2026-06-01T02:00:00+02:00',
        )

        record = await _stored_streak_record(ledger_memory_service.recon_ledger)
        assert record is not None
        assert (record.created_at, record.expires_at) == (
            '2026-06-01T00:00:00+00:00',
            '2026-08-30T00:00:00+00:00',
        )

    @pytest.mark.asyncio
    async def test_quiet_cycle_resets_the_window_with_the_streak(self, ledger_memory_service):
        ledger = ledger_memory_service.recon_ledger
        await _run_consecutive_cycles(ledger_memory_service, [2, 2])

        await _run_cycle(
            ledger_memory_service, 'run-3', EntityStandingSuppressionResult.empty_batch()
        )
        assert await _stored_windows(ledger) == {_ESD_U1: []}

        (restart,) = await _run_cycle(
            ledger_memory_service, 'run-4', _suppressing_counts({_ESD_U1: 3})
        )
        assert (restart.streak, restart.window_suppressed) == (1, 3)
        assert await _stored_windows(ledger) == {_ESD_U1: [3]}

    @pytest.mark.asyncio
    async def test_replayed_run_id_holds_the_window(self, ledger_memory_service):
        """A replay neither re-appends nor replaces: the stored window, and so
        the verdict, are the ones the original run wrote."""
        await _run_consecutive_cycles(ledger_memory_service, [2, 2, 2])

        (replayed,) = await _run_cycle(
            ledger_memory_service, 'run-3', _suppressing_counts({_ESD_U1: 1})
        )

        assert (replayed.streak, replayed.window_suppressed, replayed.escalate) == (3, 6, True)
        assert await _stored_windows(ledger_memory_service.recon_ledger) == {
            _ESD_U1: [2, 2, 2]
        }

    @pytest.mark.asyncio
    async def test_malformed_stored_window_under_counts(self, ledger_memory_service):
        """A lost window can only under-count: it may delay a filing, never cause
        one.  Which shapes read as lost is the ledger's decoding, pinned in
        test_recon_ledger.py."""
        await _seed_raw_streak_payload(
            ledger_memory_service.recon_ledger,
            {
                STREAK_PAYLOAD_KEY: 2,
                'last_run_id': 'run-seed',
                STREAK_WINDOW_PAYLOAD_KEY: [5, True],
            },
        )

        (update,) = await _run_cycle(
            ledger_memory_service, 'run-1', _suppressing_counts({_ESD_U1: 2})
        )

        assert (update.streak, update.window_suppressed) == (3, 2)

    @pytest.mark.asyncio
    async def test_window_on_a_zero_streak_row_is_ignored(self, ledger_memory_service):
        """A leftover window never carries across a reset."""
        ledger = ledger_memory_service.recon_ledger
        await _seed_raw_streak_payload(
            ledger,
            {
                STREAK_PAYLOAD_KEY: 0,
                'last_run_id': 'run-seed',
                STREAK_WINDOW_PAYLOAD_KEY: [5, 5, 5],
            },
        )

        (update,) = await _run_cycle(
            ledger_memory_service, 'run-1', _suppressing_counts({_ESD_U1: 1})
        )

        assert (update.streak, update.window_suppressed, update.escalate) == (1, 1, False)
        assert await _stored_windows(ledger) == {_ESD_U1: [1]}


async def _seed_streak(
    ledger: ReconLedgerStore,
    entity_uuid: str,
    streak: int,
    recent_counts: tuple[int, ...] | None = None,
) -> None:
    """Seed a streak row. The default window is the steady state the updater
    itself would have written: one flag in each of the last K cycles."""
    if recent_counts is None:
        recent_counts = (1,) * min(streak, SUPPRESSION_STREAK_THRESHOLD_CYCLES)
    await ledger.upsert_suppression_streak(
        project_id=_STREAK_PID,
        entity_uuid=entity_uuid,
        grounds=GROUNDS_STRUCTURAL_SIZE_CONFLATION,
        streak=streak,
        recent_counts=recent_counts,
        last_run_id='run-seed',
        updated_at=_STREAK_NOW,
        expires_at='2099-01-01T00:00:00+00:00',
    )


class TestUpdateSuppressionStreaksFailSafe:
    """An unread ledger is never read as a quiet cycle (task 2943 step-9)."""

    @pytest.mark.asyncio
    async def test_unevaluated_cycle_neither_increments_nor_resets(
        self, ledger_memory_service
    ):
        ledger = ledger_memory_service.recon_ledger
        await _seed_streak(ledger, _ESD_U1, 2)
        unevaluated = EntityStandingSuppressionResult(
            kept_flags=[],
            suppressed_by_decision={},
            grounds_by_decision={},
            suppression_evaluated=False,
        )

        assert await _run_cycle(ledger_memory_service, 'run-1', unevaluated) == []
        assert await _stored_streaks(ledger) == {_ESD_U1: 2}

    @pytest.mark.asyncio
    async def test_no_ledger_returns_empty_with_debug_log(self, caplog):
        memory_service = AsyncMock()
        memory_service.recon_ledger = None

        with caplog.at_level(logging.DEBUG, logger='fused_memory.reconciliation.standing_decision_storm_escape'):
            updates = await _run_cycle(memory_service, 'run-1', _suppressing(_ESD_U1))

        assert updates == []
        assert any(
            rec.levelno == logging.DEBUG and 'update_suppression_streaks' in rec.getMessage()
            for rec in caplog.records
        )

    @pytest.mark.asyncio
    async def test_read_failure_writes_nothing(self, ledger_memory_service, monkeypatch, caplog):
        """A transient read error leaves an established streak intact."""
        ledger = ledger_memory_service.recon_ledger
        await _seed_streak(ledger, _ESD_U1, 2)
        seeded = await _stored_streak_record(ledger)
        assert seeded is not None
        monkeypatch.setattr(
            ledger, 'list_suppression_streaks', AsyncMock(side_effect=RuntimeError('boom'))
        )

        with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.standing_decision_storm_escape'):
            updates = await _run_cycle(
                ledger_memory_service, 'run-1', _suppressing(_ESD_U1, _ESD_U2)
            )

        assert updates == []
        assert await _stored_streak_record(ledger) == seeded
        assert await _stored_streak_record(ledger, _ESD_U2) is None, (
            'no row may be written when the prior state could not be read'
        )
        assert any(rec.levelno == logging.WARNING for rec in caplog.records)

    @pytest.mark.asyncio
    async def test_malformed_stored_streak_restarts_at_one(self, ledger_memory_service):
        await _seed_raw_streak_payload(
            ledger_memory_service.recon_ledger, {STREAK_PAYLOAD_KEY: '2'}
        )

        (update,) = await _run_cycle(ledger_memory_service, 'run-1', _suppressing(_ESD_U1))

        assert update.streak == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'now',
        [
            pytest.param('2026-06-01T00:00:00', id='naive'),
            pytest.param('not a timestamp', id='malformed'),
        ],
    )
    async def test_a_now_without_a_utc_offset_raises_before_anything_is_written(
        self, ledger_memory_service, now
    ):
        """A caller-supplied now is a programmer's input, not a transient
        condition, so the best-effort contract does not cover it: an expires_at
        without an offset would silently break gc()'s TEXT comparison."""
        with pytest.raises(ValueError, match='now'):
            await _run_cycle(ledger_memory_service, 'run-1', _suppressing(_ESD_U1), now=now)

        assert await _stored_streaks(ledger_memory_service.recon_ledger) == {}

    @pytest.mark.asyncio
    async def test_one_entitys_write_failure_does_not_abort_the_others(
        self, ledger_memory_service, monkeypatch, caplog
    ):
        ledger = ledger_memory_service.recon_ledger
        real_upsert = ledger.upsert_suppression_streak

        async def _fail_for_u1(**kwargs):
            if kwargs['entity_uuid'] == _ESD_U1:
                raise RuntimeError('disk full')
            await real_upsert(**kwargs)

        monkeypatch.setattr(ledger, 'upsert_suppression_streak', _fail_for_u1)

        with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.standing_decision_storm_escape'):
            updates = await _run_cycle(
                ledger_memory_service, 'run-1', _suppressing(_ESD_U1, _ESD_U2)
            )

        assert [u.entity_uuid for u in updates] == [_ESD_U2]
        assert await _stored_streaks(ledger) == {_ESD_U2: 1}
        assert any(
            rec.levelno == logging.WARNING and _ESD_U1 in rec.getMessage()
            for rec in caplog.records
        ), 'a WARNING naming the failed entity must be logged'


# ---------------------------------------------------------------------------
# maybe_escalate_suppression_streak — the streak arm's filer (task 2943)
# ---------------------------------------------------------------------------


def _streak_update(
    entity_uuid: str = _ESD_U1,
    streak: int = SUPPRESSION_STREAK_THRESHOLD_CYCLES + 1,
    window_suppressed: int = SUPPRESSION_STREAK_VOLUME_THRESHOLD + 1,
) -> storm_escape.SuppressionStreakUpdate:
    """An update whose verdict follows update_suppression_streaks' rule, so a
    fixture cannot claim a verdict the updater would never produce."""
    return storm_escape.SuppressionStreakUpdate(
        entity_uuid=entity_uuid,
        grounds=GROUNDS_STRUCTURAL_SIZE_CONFLATION,
        streak=streak,
        window_suppressed=window_suppressed,
        escalate=(
            streak >= SUPPRESSION_STREAK_THRESHOLD_CYCLES
            and window_suppressed > SUPPRESSION_STREAK_VOLUME_THRESHOLD
        ),
    )


class TestMaybeEscalateSuppressionStreak:
    """Streak escalation filing, against a REAL EscalationQueue (task 2943 step-11).

    A mock cannot witness a fold, a dedupe_count, or whether two records share
    a fingerprint, and those are what this filer's contract is about.
    """

    _PID = 'p'
    _RUN = 'run-1'

    @pytest.fixture
    def queue(self, tmp_path):
        from escalation.queue import EscalationQueue

        return EscalationQueue(tmp_path / 'escalations')

    @staticmethod
    def _pending(queue, entity_uuid: str = _ESD_U1) -> list:
        return queue.get_by_task(entity_uuid, status='pending', level=1)

    @pytest.mark.asyncio
    async def test_escalating_update_files_one_storm_category_record(self, queue):
        update = _streak_update()
        escalated = await storm_escape.maybe_escalate_suppression_streak(
            queue, self._PID, self._RUN, [update]
        )
        assert escalated == [_ESD_U1]

        (esc,) = self._pending(queue)
        assert esc.level == 1
        assert esc.severity == 'blocking'
        assert esc.category == CATEGORY_STANDING_DECISION_STORM
        assert esc.agent_role == 'reconciliation-stage1'
        assert esc.task_id == _ESD_U1
        blob = f'{esc.summary}\n{esc.detail}'
        assert _ESD_U1 in blob
        assert GROUNDS_STRUCTURAL_SIZE_CONFLATION in blob
        assert f'streak: {update.streak}' in esc.detail
        assert f'threshold: {SUPPRESSION_STREAK_THRESHOLD_CYCLES}' in esc.detail
        assert f'suppressed_in_window: {update.window_suppressed}' in esc.detail
        assert f'volume_threshold: {SUPPRESSION_STREAK_VOLUME_THRESHOLD}' in esc.detail
        assert f'suppressed {update.window_suppressed} recon flag(s)' in esc.summary
        assert (
            f'last {SUPPRESSION_STREAK_THRESHOLD_CYCLES} consecutive cycles' in esc.summary
        )
        assert f'(> {SUPPRESSION_STREAK_VOLUME_THRESHOLD})' in esc.summary

    @pytest.mark.asyncio
    async def test_long_steady_state_streak_files_nothing(self, queue):
        """The filer keys off the verdict, never off the streak length alone:
        a decision suppressing one flag per cycle for ten cycles is working."""
        steady = _streak_update(streak=10, window_suppressed=SUPPRESSION_STREAK_THRESHOLD_CYCLES)
        assert steady.escalate is False

        escalated = await storm_escape.maybe_escalate_suppression_streak(
            queue, self._PID, self._RUN, [steady]
        )

        assert escalated == []
        assert self._pending(queue) == []

    @pytest.mark.asyncio
    async def test_record_names_the_volume_threshold_it_was_given(self, queue):
        custom_n = SUPPRESSION_STREAK_VOLUME_THRESHOLD + 2
        update = _streak_update(window_suppressed=custom_n + 1)

        await storm_escape.maybe_escalate_suppression_streak(
            queue, self._PID, self._RUN, [update], volume_threshold=custom_n
        )

        (esc,) = self._pending(queue)
        assert f'volume_threshold: {custom_n}' in esc.detail
        assert f'(> {custom_n})' in esc.summary

    @pytest.mark.asyncio
    async def test_update_below_threshold_files_nothing(self, queue):
        below = _streak_update(streak=SUPPRESSION_STREAK_THRESHOLD_CYCLES - 1)
        assert below.escalate is False
        escalated = await storm_escape.maybe_escalate_suppression_streak(
            queue, self._PID, self._RUN, [below]
        )
        assert escalated == []
        assert self._pending(queue) == []

    @pytest.mark.asyncio
    async def test_streak_and_per_cycle_storm_never_fold_into_each_other(self, queue):
        """Same entity, same cycle, both escapes: two records, two fingerprints.

        A flood in one cycle and a persistent low-grade drain are different
        diagnoses; folding either into the other would hide it for any entity
        that had ever tripped the other escape.
        """
        await storm_escape.maybe_escalate_suppression_storm(
            queue, self._PID, self._RUN, _storm_result()
        )
        await storm_escape.maybe_escalate_suppression_streak(
            queue, self._PID, self._RUN, [_streak_update()]
        )

        pending = self._pending(queue)
        assert len(pending) == 2
        assert pending[0].dedupe_fingerprint != pending[1].dedupe_fingerprint
        assert all(esc.dedupe_count == 0 for esc in pending)

    @pytest.mark.asyncio
    async def test_two_entities_each_get_their_own_record(self, queue):
        escalated = await storm_escape.maybe_escalate_suppression_streak(
            queue, self._PID, self._RUN, [_streak_update(_ESD_U1), _streak_update(_ESD_U2)]
        )
        assert sorted(escalated) == sorted([_ESD_U1, _ESD_U2])
        (first,) = self._pending(queue, _ESD_U1)
        (second,) = self._pending(queue, _ESD_U2)
        assert first.dedupe_fingerprint != second.dedupe_fingerprint

    @pytest.mark.asyncio
    async def test_recurring_streak_folds_into_one_parent(self, queue):
        """A streak keeps firing past K every suppressing cycle; each recurrence
        folds onto the first record rather than minting a new one."""
        assert await storm_escape.maybe_escalate_suppression_streak(
            queue, self._PID, 'run-3', [_streak_update(streak=3)]
        ) == [_ESD_U1]

        second = await storm_escape.maybe_escalate_suppression_streak(
            queue, self._PID, 'run-4', [_streak_update(streak=4)]
        )

        assert second == [], 'a folded recurrence is not a new filing'
        (parent,) = self._pending(queue)
        assert parent.dedupe_count == 1

    @pytest.mark.asyncio
    async def test_escalation_unavailable_returns_empty(self, queue, monkeypatch):
        monkeypatch.setattr(storm_escape, 'Escalation', None)
        escalated = await storm_escape.maybe_escalate_suppression_streak(
            queue, self._PID, self._RUN, [_streak_update()]
        )
        assert escalated == []
        assert self._pending(queue) == []

    @pytest.mark.asyncio
    async def test_submit_failure_logs_warning_and_excludes_entity(self, tmp_path, caplog):
        from escalation.queue import EscalationQueue

        class _BrokenQueue(EscalationQueue):
            def submit(self, escalation):
                raise RuntimeError('boom')

        queue = _BrokenQueue(tmp_path / 'escalations')
        with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.standing_decision_storm_escape'):
            escalated = await storm_escape.maybe_escalate_suppression_streak(
                queue, self._PID, self._RUN, [_streak_update()]
            )
        assert escalated == []
        assert any(
            rec.levelno == logging.WARNING and _ESD_U1 in rec.getMessage()
            for rec in caplog.records
        ), 'a WARNING naming the entity must be logged'
