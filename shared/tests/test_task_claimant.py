"""Tests for shared.task_claimant — is_stranded predicate (task 2182 step-9/10).

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
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from shared.task_claimant import is_stranded

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
