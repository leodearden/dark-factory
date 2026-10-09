"""Tests for shared.task_claimant — is_stranded predicate (task 2182 step-9/10),
compose_claimant_run_id (task 2188 / omega1 step-1/2), and the is_stranded_blocked
/ has_live_claimant sibling predicates (task 2408 step-1/2).

PRD plans/task-status-authority-prd.md contract C4 / decision D4: "stranded" is
a queryable predicate over first-class claimant_run_id/heartbeat_at columns,
rather than plan.lock/owner_pid forensics.

Truth table covered:
  - in-progress + no live claimant (None or blank) -> stranded
  - in-progress + claimant + fresh heartbeat -> not stranded
  - in-progress + claimant + stale heartbeat -> stranded
  - in-progress + claimant + missing/unparseable heartbeat -> stranded
  - non-in-progress statuses (pending/done/infra-hold) -> never stranded
  - in-progress + legacy metadata.infra_hold=True overload -> not stranded
    (pre-omega4 migration-window safety net; D3 makes infra-hold a first-class
    status so the primary guard is the in-progress gate itself)
  - a naive ``now`` is tolerated (tz-normalized internally, no TypeError)

compose_claimant_run_id(run_id, session_id, owner_pid) is the producer half of
the claimant identity that is_stranded's ``claimant_run_id`` column consumes:
  - returns a non-empty str embedding all three components verbatim
  - deterministic for fixed inputs
  - distinct outputs for distinct (run_id | session_id | owner_pid)

is_stranded_blocked(task, now, ttl) (task 2408 mechanism 2) is is_stranded's
blocked-status sibling, built on the same extracted liveness core: identical
claimant/heartbeat/infra_hold truth table, but gated on status == 'blocked'
instead of 'in-progress' (so in-progress, pending, done, infra-hold all read
False).

has_live_claimant(task, now, ttl) (task 2408 mechanism 1) is status-agnostic —
no status gate, no infra_hold check — and is simply the negation of the
liveness core: True iff there is a live (non-blank claimant, fresh-heartbeat)
claim, regardless of the task's current status.

is_stranded_any_status(task, now, ttl) (C4-E6) is the status-agnostic read rule
"is anyone alive holding this row?": the legacy metadata.infra_hold carve-out
plus the liveness core. is_stranded / is_stranded_blocked are pinned, over a
grid, as status gates delegating to it. DEFAULT_CLAIMANT_HEARTBEAT_TTL carries
the single exact-value pin for the one shared heartbeat TTL.
"""

from __future__ import annotations

import itertools
from datetime import UTC, datetime, timedelta

import pytest

import shared.task_claimant
from shared.task_claimant import (
    DEFAULT_CLAIMANT_HEARTBEAT_TTL,
    compose_claimant_run_id,
    has_live_claimant,
    is_stranded,
    is_stranded_any_status,
    is_stranded_blocked,
)
from shared.task_statuses import TaskStatus

_TTL = timedelta(minutes=5)


def _now() -> datetime:
    return datetime.now(UTC)


class TestNoLiveClaimant:
    def test_in_progress_with_none_claimant_is_stranded(self):
        now = _now()
        task = {'status': 'in-progress', 'claimant_run_id': None, 'heartbeat_at': None}
        assert is_stranded(task, now, _TTL) is True

    def test_in_progress_with_blank_claimant_is_stranded(self):
        now = _now()
        task = {
            'status': 'in-progress',
            'claimant_run_id': '   ',
            'heartbeat_at': now.isoformat(),
        }
        assert is_stranded(task, now, _TTL) is True


class TestHeartbeatFreshness:
    def test_fresh_heartbeat_is_not_stranded(self):
        now = _now()
        heartbeat = (now - timedelta(minutes=1)).isoformat()
        task = {'status': 'in-progress', 'claimant_run_id': 'run-x', 'heartbeat_at': heartbeat}
        assert is_stranded(task, now, _TTL) is False

    def test_stale_heartbeat_is_stranded(self):
        now = _now()
        heartbeat = (now - timedelta(minutes=10)).isoformat()
        task = {'status': 'in-progress', 'claimant_run_id': 'run-x', 'heartbeat_at': heartbeat}
        assert is_stranded(task, now, _TTL) is True

    def test_missing_heartbeat_is_stranded(self):
        now = _now()
        task = {'status': 'in-progress', 'claimant_run_id': 'run-x'}
        assert is_stranded(task, now, _TTL) is True

    def test_none_heartbeat_is_stranded(self):
        now = _now()
        task = {'status': 'in-progress', 'claimant_run_id': 'run-x', 'heartbeat_at': None}
        assert is_stranded(task, now, _TTL) is True

    def test_unparseable_heartbeat_is_stranded(self):
        now = _now()
        task = {
            'status': 'in-progress',
            'claimant_run_id': 'run-x',
            'heartbeat_at': 'not-a-timestamp',
        }
        assert is_stranded(task, now, _TTL) is True


class TestNonInProgressStatusesNeverStranded:
    def test_pending_is_never_stranded(self):
        now = _now()
        task = {'status': 'pending', 'claimant_run_id': None, 'heartbeat_at': None}
        assert is_stranded(task, now, _TTL) is False

    def test_done_is_never_stranded(self):
        now = _now()
        task = {'status': 'done', 'claimant_run_id': None, 'heartbeat_at': None}
        assert is_stranded(task, now, _TTL) is False

    def test_infra_hold_is_never_stranded(self):
        now = _now()
        task = {'status': 'infra-hold', 'claimant_run_id': None, 'heartbeat_at': None}
        assert is_stranded(task, now, _TTL) is False


class TestLegacyInfraHoldMetadataOverload:
    def test_in_progress_with_metadata_infra_hold_is_not_stranded(self):
        """Pre-omega4 legacy overload: in-progress + metadata.infra_hold=True is
        never stranded, even with no live claimant at all — the defensive
        metadata check short-circuits before the claimant/heartbeat checks.
        """
        now = _now()
        task = {
            'status': 'in-progress',
            'claimant_run_id': None,
            'heartbeat_at': None,
            'metadata': {'infra_hold': True},
        }
        assert is_stranded(task, now, _TTL) is False


class TestNaiveNowTolerated:
    def test_naive_now_is_normalized_not_raised(self):
        now_aware = _now()
        now_naive = now_aware.replace(tzinfo=None)
        heartbeat = (now_aware - timedelta(minutes=1)).isoformat()
        task = {'status': 'in-progress', 'claimant_run_id': 'run-x', 'heartbeat_at': heartbeat}
        # Must not raise TypeError from comparing an offset-naive `now` against
        # the tz-aware heartbeat/ttl arithmetic performed internally.
        assert is_stranded(task, now_naive, _TTL) is False

    def test_naive_now_past_ttl_is_stranded(self):
        now_aware = _now()
        now_naive = now_aware.replace(tzinfo=None)
        heartbeat = (now_aware - timedelta(minutes=10)).isoformat()
        task = {'status': 'in-progress', 'claimant_run_id': 'run-x', 'heartbeat_at': heartbeat}
        assert is_stranded(task, now_naive, _TTL) is True


class TestIsStrandedBlockedNoLiveClaimant:
    def test_blocked_with_none_claimant_is_stranded(self):
        now = _now()
        task = {'status': 'blocked', 'claimant_run_id': None, 'heartbeat_at': None}
        assert is_stranded_blocked(task, now, _TTL) is True

    def test_blocked_with_blank_claimant_is_stranded(self):
        now = _now()
        task = {
            'status': 'blocked',
            'claimant_run_id': '   ',
            'heartbeat_at': now.isoformat(),
        }
        assert is_stranded_blocked(task, now, _TTL) is True


class TestIsStrandedBlockedHeartbeatFreshness:
    def test_fresh_heartbeat_is_not_stranded(self):
        now = _now()
        heartbeat = (now - timedelta(minutes=1)).isoformat()
        task = {'status': 'blocked', 'claimant_run_id': 'run-x', 'heartbeat_at': heartbeat}
        assert is_stranded_blocked(task, now, _TTL) is False

    def test_stale_heartbeat_is_stranded(self):
        now = _now()
        heartbeat = (now - timedelta(minutes=10)).isoformat()
        task = {'status': 'blocked', 'claimant_run_id': 'run-x', 'heartbeat_at': heartbeat}
        assert is_stranded_blocked(task, now, _TTL) is True

    def test_missing_heartbeat_is_stranded(self):
        now = _now()
        task = {'status': 'blocked', 'claimant_run_id': 'run-x'}
        assert is_stranded_blocked(task, now, _TTL) is True

    def test_none_heartbeat_is_stranded(self):
        now = _now()
        task = {'status': 'blocked', 'claimant_run_id': 'run-x', 'heartbeat_at': None}
        assert is_stranded_blocked(task, now, _TTL) is True

    def test_unparseable_heartbeat_is_stranded(self):
        now = _now()
        task = {
            'status': 'blocked',
            'claimant_run_id': 'run-x',
            'heartbeat_at': 'not-a-timestamp',
        }
        assert is_stranded_blocked(task, now, _TTL) is True


class TestIsStrandedBlockedLegacyInfraHoldMetadataOverload:
    def test_blocked_with_metadata_infra_hold_is_not_stranded(self):
        now = _now()
        task = {
            'status': 'blocked',
            'claimant_run_id': None,
            'heartbeat_at': None,
            'metadata': {'infra_hold': True},
        }
        assert is_stranded_blocked(task, now, _TTL) is False


class TestIsStrandedBlockedStatusGate:
    def test_in_progress_is_never_stranded_blocked(self):
        """Mirror image of is_stranded's in-progress gate: is_stranded_blocked
        is blocked-only, so an in-progress task — even with no live claimant
        at all — is never stranded-blocked.
        """
        now = _now()
        task = {'status': 'in-progress', 'claimant_run_id': None, 'heartbeat_at': None}
        assert is_stranded_blocked(task, now, _TTL) is False

    def test_pending_is_never_stranded_blocked(self):
        now = _now()
        task = {'status': 'pending', 'claimant_run_id': None, 'heartbeat_at': None}
        assert is_stranded_blocked(task, now, _TTL) is False

    def test_done_is_never_stranded_blocked(self):
        now = _now()
        task = {'status': 'done', 'claimant_run_id': None, 'heartbeat_at': None}
        assert is_stranded_blocked(task, now, _TTL) is False

    def test_infra_hold_status_is_never_stranded_blocked(self):
        now = _now()
        task = {'status': 'infra-hold', 'claimant_run_id': None, 'heartbeat_at': None}
        assert is_stranded_blocked(task, now, _TTL) is False


class TestIsStrandedBlockedNaiveNowTolerated:
    def test_naive_now_is_normalized_not_raised(self):
        now_aware = _now()
        now_naive = now_aware.replace(tzinfo=None)
        heartbeat = (now_aware - timedelta(minutes=1)).isoformat()
        task = {'status': 'blocked', 'claimant_run_id': 'run-x', 'heartbeat_at': heartbeat}
        assert is_stranded_blocked(task, now_naive, _TTL) is False

    def test_naive_now_past_ttl_is_stranded(self):
        now_aware = _now()
        now_naive = now_aware.replace(tzinfo=None)
        heartbeat = (now_aware - timedelta(minutes=10)).isoformat()
        task = {'status': 'blocked', 'claimant_run_id': 'run-x', 'heartbeat_at': heartbeat}
        assert is_stranded_blocked(task, now_naive, _TTL) is True


class TestHasLiveClaimantIsStatusAgnostic:
    """No status gate at all — proven by asserting the SAME True outcome for
    both a pending task and a blocked task given an identical live claim.
    """

    def test_fresh_heartbeat_is_live_for_pending_task(self):
        now = _now()
        heartbeat = (now - timedelta(minutes=1)).isoformat()
        task = {'status': 'pending', 'claimant_run_id': 'run-x', 'heartbeat_at': heartbeat}
        assert has_live_claimant(task, now, _TTL) is True

    def test_fresh_heartbeat_is_live_for_blocked_task(self):
        now = _now()
        heartbeat = (now - timedelta(minutes=1)).isoformat()
        task = {'status': 'blocked', 'claimant_run_id': 'run-x', 'heartbeat_at': heartbeat}
        assert has_live_claimant(task, now, _TTL) is True


class TestHasLiveClaimantNoLiveClaimant:
    def test_none_claimant_is_not_live(self):
        now = _now()
        task = {'status': 'pending', 'claimant_run_id': None, 'heartbeat_at': None}
        assert has_live_claimant(task, now, _TTL) is False

    def test_blank_claimant_is_not_live(self):
        now = _now()
        task = {
            'status': 'pending',
            'claimant_run_id': '   ',
            'heartbeat_at': now.isoformat(),
        }
        assert has_live_claimant(task, now, _TTL) is False


class TestHasLiveClaimantHeartbeatFreshness:
    def test_stale_heartbeat_is_not_live(self):
        now = _now()
        heartbeat = (now - timedelta(minutes=10)).isoformat()
        task = {'status': 'pending', 'claimant_run_id': 'run-x', 'heartbeat_at': heartbeat}
        assert has_live_claimant(task, now, _TTL) is False

    def test_missing_heartbeat_is_not_live(self):
        now = _now()
        task = {'status': 'pending', 'claimant_run_id': 'run-x'}
        assert has_live_claimant(task, now, _TTL) is False

    def test_unparseable_heartbeat_is_not_live(self):
        now = _now()
        task = {
            'status': 'pending',
            'claimant_run_id': 'run-x',
            'heartbeat_at': 'not-a-timestamp',
        }
        assert has_live_claimant(task, now, _TTL) is False


class TestHasLiveClaimantNaiveNowTolerated:
    def test_naive_now_is_normalized_not_raised(self):
        now_aware = _now()
        now_naive = now_aware.replace(tzinfo=None)
        heartbeat = (now_aware - timedelta(minutes=1)).isoformat()
        task = {'status': 'pending', 'claimant_run_id': 'run-x', 'heartbeat_at': heartbeat}
        assert has_live_claimant(task, now_naive, _TTL) is True

    def test_naive_now_past_ttl_is_not_live(self):
        now_aware = _now()
        now_naive = now_aware.replace(tzinfo=None)
        heartbeat = (now_aware - timedelta(minutes=10)).isoformat()
        task = {'status': 'pending', 'claimant_run_id': 'run-x', 'heartbeat_at': heartbeat}
        assert has_live_claimant(task, now_naive, _TTL) is False


class TestComposeClaimantRunId:
    """compose_claimant_run_id(run_id, session_id, owner_pid) — the W10-consumable
    claimant identity written into the claimant_run_id column at dispatch.
    """

    def test_returns_non_empty_str(self):
        result = compose_claimant_run_id('run-abc123', '2188-8b8e2ee4', 4242)
        assert isinstance(result, str)
        assert result != ''

    def test_embeds_all_three_components_verbatim(self):
        result = compose_claimant_run_id('run-abc123', '2188-8b8e2ee4', 4242)
        assert 'run-abc123' in result
        assert '2188-8b8e2ee4' in result
        assert '4242' in result

    def test_deterministic_for_fixed_inputs(self):
        first = compose_claimant_run_id('run-abc123', '2188-8b8e2ee4', 4242)
        second = compose_claimant_run_id('run-abc123', '2188-8b8e2ee4', 4242)
        assert first == second

    def test_distinct_run_id_yields_distinct_output(self):
        a = compose_claimant_run_id('run-aaa', '2188-8b8e2ee4', 4242)
        b = compose_claimant_run_id('run-bbb', '2188-8b8e2ee4', 4242)
        assert a != b

    def test_distinct_session_id_yields_distinct_output(self):
        a = compose_claimant_run_id('run-abc123', 'session-aaa', 4242)
        b = compose_claimant_run_id('run-abc123', 'session-bbb', 4242)
        assert a != b

    def test_distinct_owner_pid_yields_distinct_output(self):
        a = compose_claimant_run_id('run-abc123', '2188-8b8e2ee4', 1111)
        b = compose_claimant_run_id('run-abc123', '2188-8b8e2ee4', 2222)
        assert a != b


class TestDefaultClaimantHeartbeatTtl:
    def test_is_exactly_ten_minutes(self):
        """The one exact-value pin for the one claimant heartbeat TTL definition.

        Deliberately an exact-value pin: ``isinstance(..., timedelta)`` plus a
        positive-duration check is satisfied equally by 1 microsecond and by
        100 years, so it cannot catch a units slip (``seconds=10`` for
        ``minutes=10``) — the one error this constant is actually prone to.
        """
        assert isinstance(DEFAULT_CLAIMANT_HEARTBEAT_TTL, timedelta)
        assert timedelta(minutes=10) == DEFAULT_CLAIMANT_HEARTBEAT_TTL


def _row(status, claimant, heartbeat, metadata=None) -> dict:
    return {
        'status': status,
        'claimant_run_id': claimant,
        'heartbeat_at': heartbeat,
        'metadata': metadata,
    }


class TestIsStrandedAnyStatusIsStatusAgnostic:
    @pytest.mark.parametrize('status', list(TaskStatus))
    def test_stale_claimant_is_stranded(self, status):
        now = _now()
        stale = (now - timedelta(minutes=10)).isoformat()
        assert is_stranded_any_status(_row(status, 'run-x', stale), now, _TTL) is True

    @pytest.mark.parametrize('status', list(TaskStatus))
    def test_fresh_claimant_is_not_stranded(self, status):
        now = _now()
        fresh = (now - timedelta(minutes=1)).isoformat()
        assert is_stranded_any_status(_row(status, 'run-x', fresh), now, _TTL) is False

    @pytest.mark.parametrize('status', list(TaskStatus))
    def test_none_claimant_is_stranded(self, status):
        now = _now()
        assert is_stranded_any_status(_row(status, None, now.isoformat()), now, _TTL) is True

    @pytest.mark.parametrize('status', list(TaskStatus))
    def test_blank_claimant_is_stranded(self, status):
        now = _now()
        assert is_stranded_any_status(_row(status, '   ', now.isoformat()), now, _TTL) is True

    @pytest.mark.parametrize('status', list(TaskStatus))
    def test_missing_heartbeat_is_stranded(self, status):
        now = _now()
        assert is_stranded_any_status(_row(status, 'run-x', None), now, _TTL) is True

    @pytest.mark.parametrize('status', list(TaskStatus))
    def test_unparseable_heartbeat_is_stranded(self, status):
        now = _now()
        row = _row(status, 'run-x', 'not-a-timestamp')
        assert is_stranded_any_status(row, now, _TTL) is True


class TestIsStrandedAnyStatusInfraHoldCarveOut:
    @pytest.mark.parametrize('status', ['in-progress', 'blocked', 'pending'])
    def test_legacy_infra_hold_metadata_is_not_stranded(self, status):
        """Boundary case B9 — the carve-out has no status gate."""
        now = _now()
        stale = (now - timedelta(minutes=10)).isoformat()
        row = _row(status, 'run-x', stale, metadata={'infra_hold': True})
        assert is_stranded_any_status(row, now, _TTL) is False

    def test_non_mapping_metadata_does_not_trigger_carve_out(self):
        now = _now()
        stale = (now - timedelta(minutes=10)).isoformat()
        row = _row('in-progress', 'run-x', stale, metadata='infra_hold')
        assert is_stranded_any_status(row, now, _TTL) is True


class TestIsStrandedAnyStatusNaiveNow:
    def test_naive_now_is_normalized_not_raised(self):
        now_aware = _now()
        now_naive = now_aware.replace(tzinfo=None)
        fresh = (now_aware - timedelta(minutes=1)).isoformat()
        assert is_stranded_any_status(_row('pending', 'run-x', fresh), now_naive, _TTL) is False

    def test_naive_now_past_ttl_is_stranded(self):
        now_aware = _now()
        now_naive = now_aware.replace(tzinfo=None)
        stale = (now_aware - timedelta(minutes=10)).isoformat()
        assert is_stranded_any_status(_row('pending', 'run-x', stale), now_naive, _TTL) is True


_GRID_NOW = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
_GRID = [
    _row(status, claimant, heartbeat, metadata)
    for status, claimant, heartbeat, metadata in itertools.product(
        list(TaskStatus),
        [None, '   ', 'run-x'],
        [
            (_GRID_NOW - timedelta(minutes=1)).isoformat(),
            (_GRID_NOW - timedelta(minutes=10)).isoformat(),
            None,
            'not-a-timestamp',
        ],
        [None, {'infra_hold': True}],
    )
]


class TestSpecialisationsDelegate:
    """C4-E6: the status-gated predicates are status gates over is_stranded_any_status."""

    @pytest.mark.parametrize('row', _GRID)
    def test_is_stranded_is_in_progress_gate(self, row):
        expected = row['status'] == 'in-progress' and is_stranded_any_status(row, _GRID_NOW, _TTL)
        assert is_stranded(row, _GRID_NOW, _TTL) == expected

    @pytest.mark.parametrize('row', _GRID)
    def test_is_stranded_blocked_is_blocked_gate(self, row):
        expected = row['status'] == 'blocked' and is_stranded_any_status(row, _GRID_NOW, _TTL)
        assert is_stranded_blocked(row, _GRID_NOW, _TTL) == expected

    @pytest.mark.parametrize('row', [r for r in _GRID if r['metadata'] is None])
    def test_without_infra_hold_is_negated_has_live_claimant(self, row):
        assert is_stranded_any_status(row, _GRID_NOW, _TTL) == (
            not has_live_claimant(row, _GRID_NOW, _TTL)
        )


def test_public_api_surface():
    assert {'is_stranded_any_status', 'DEFAULT_CLAIMANT_HEARTBEAT_TTL'} <= set(
        shared.task_claimant.__all__
    )
