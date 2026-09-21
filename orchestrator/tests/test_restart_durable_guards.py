"""A fleet redeploy must re-arm none of the orchestrator's one-strike guards.

Four guards, one property.  Each exists to say "never do this again" and each
used to hold that verdict in process RAM, so the ~8-15h fleet redeploy turned
every one of them into "never again *until the next redeploy*" — the prevented
thing recurred on a fixed cadence forever.  They are tested together, in one
file, because the guarantee is one guarantee: reading the four instances side
by side is what makes it legible, and no single owning suite could state it.

The restart is simulated the way it actually happens — a SECOND owner built
against the same ``project_root``, with fresh process memory and the same
files on disk.

Covers:
  step-7: the steward's capped set and its three per-escalation counters.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from escalation.models import Escalation

from orchestrator import guard_state

# A round, arbitrary instant.  Every TTL assertion is relative to it, so no
# test here depends on the wall clock.
_T0 = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def guard_clock(monkeypatch):
    """Freeze the clock every guard store defaults to, and hand back the dial.

    ``guard_state`` never reads the wall clock internally, so replacing this
    one module-level function is enough to drive every guard's TTL — no guard
    needs a test-only clock parameter threaded through its owner.
    """
    frozen = [_T0]
    monkeypatch.setattr(guard_state, '_utc_now', lambda: frozen[0])
    return frozen


def _escalation(**overrides) -> Escalation:
    defaults: dict = dict(
        id='esc-5352-1',
        task_id='42',
        agent_role='orchestrator',
        severity='blocking',
        category='limit_exhausted',
        summary='execute limit exhausted',
    )
    defaults.update(overrides)
    return Escalation(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# step-7 — the steward's give-up survives its own restart
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestStewardCapSurvivesRedeploy:
    """``make_steward`` roots every build at the same ``tmp_path/project``, so
    a second build IS the redeployed steward: same task, same project root,
    empty memory."""

    async def test_a_capped_escalation_is_not_re_handled_after_a_restart(
        self, make_steward, caplog, guard_clock,
    ):
        """The 1,183,854-lines-in-20.5h incident, one redeploy later.

        On main the fresh steward re-adopts the record from scratch, so a
        permanently unhandleable escalation is re-handled — and its ladder
        re-burnt — once per redeploy, forever.
        """
        first = make_steward(config_overrides={'steward_max_attempts': 1})
        first._mark_capped('esc-5352-1')

        redeployed = make_steward(config_overrides={'steward_max_attempts': 1})
        with patch(
            'orchestrator.steward.invoke_agent', new_callable=AsyncMock,
        ) as mock_invoke, caplog.at_level('INFO'):
            await redeployed._handle_escalation(_escalation())

        mock_invoke.assert_not_called()
        assert 'handling escalation' not in caplog.text, (
            'a capped record must produce no further log lines — silence is '
            'the signal that distinguishes a healthy idle steward from the spin'
        )

    async def test_a_capped_escalation_is_filtered_from_the_pending_read(
        self, make_steward, guard_clock,
    ):
        first = make_steward()
        first._mark_capped('esc-5352-1')

        redeployed = make_steward()
        redeployed.escalation_queue.get_by_task.return_value = [_escalation()]
        with patch.object(
            redeployed, '_watch_for_escalation', new_callable=AsyncMock,
        ) as mock_watch:
            mock_watch.return_value = None
            assert await redeployed._next_escalation() is None

        # The watcher must still be consulted, so the loop blocks on inotify
        # instead of hot-returning the capped record.  (Mock assertion methods
        # take no message argument — a trailing `, (...)` would build a
        # discarded tuple rather than attach an explanation.)
        mock_watch.assert_awaited_once()

    async def test_a_capped_escalation_is_excluded_from_the_watcher_argv(
        self, make_steward, guard_clock,
    ):
        """Filtering the pending read alone is not enough: the watcher's
        initial scan would emit the still-pending capped record immediately,
        turning every call into a subprocess respawn instead of an inotify
        block."""
        first = make_steward()
        first._mark_capped('esc-5352-1')

        redeployed = make_steward()
        with patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_exec:
            proc = AsyncMock()
            proc.returncode = 0
            proc.communicate.return_value = (b'', b'')
            mock_exec.return_value = proc
            await redeployed._watch_for_escalation()

        cmd = list(mock_exec.call_args[0])
        pairs = [
            (cmd[i], cmd[i + 1]) for i in range(len(cmd) - 1)
            if cmd[i] == '--exclude-id'
        ]
        assert pairs == [('--exclude-id', 'esc-5352-1')], f'got {cmd!r}'

    @pytest.mark.parametrize(
        ('counter', 'limit_field', 'limit'),
        [
            ('_retry_counts', 'steward_max_attempts', 1),
            ('_timeout_counts', 'steward_max_timeouts_per_escalation', 3),
            ('_empty_output_counts', 'steward_max_empty_outputs_per_escalation', 2),
        ],
    )
    async def test_an_exhausted_ladder_is_not_re_spent_after_a_restart(
        self, make_steward, guard_clock, counter, limit_field, limit,
    ):
        """The capped set is one layer; the counters beneath it are the other.

        Leaving these process-local would let a redeploy hand the same
        escalation a fresh full budget — the ladder re-spent once per
        redeploy, which is the cost this task is filed against.
        """
        overrides = {limit_field: limit, 'steward_max_attempts': 1}
        first = make_steward(config_overrides=overrides)
        getattr(first, counter)['esc-5352-1'] = limit

        redeployed = make_steward(config_overrides=overrides)
        with patch(
            'orchestrator.steward.invoke_agent', new_callable=AsyncMock,
        ) as mock_invoke:
            await redeployed._handle_escalation(_escalation())

        mock_invoke.assert_not_called()
        assert 'esc-5352-1' in redeployed._capped_escalations, (
            'the guard must fire on the FIRST look, not after another full budget'
        )

    async def test_a_new_escalation_still_gets_a_full_budget(
        self, make_steward, guard_clock,
    ):
        """The bound that makes persisting a give-up safe: the keys are
        escalation ids, so only the byte-identical record that already
        exhausted the ladder is remembered."""
        first = make_steward(config_overrides={'steward_max_attempts': 1})
        first._retry_counts['esc-5352-1'] = 1
        first._mark_capped('esc-5352-1')

        redeployed = make_steward(config_overrides={'steward_max_attempts': 1})
        assert redeployed._retry_counts.get('esc-5352-2', 0) == 0
        assert 'esc-5352-2' not in redeployed._capped_escalations

    async def test_resolution_clears_the_counters_durably(
        self, make_steward, guard_clock,
    ):
        """A resolved escalation's counters must not haunt a later record."""
        first = make_steward()
        esc = _escalation()
        first._retry_counts[esc.id] = 1
        first._timeout_counts[esc.id] = 2
        first._empty_output_counts[esc.id] = 1

        first._dismiss_capped_l0(esc, 'attempt_cap')

        redeployed = make_steward()
        assert redeployed._retry_counts.get(esc.id, 0) == 0
        assert redeployed._timeout_counts.get(esc.id, 0) == 0
        assert redeployed._empty_output_counts.get(esc.id, 0) == 0

    async def test_the_cap_re_arms_once_the_ttl_elapses(
        self, make_steward, guard_clock,
    ):
        """The state must not outlive its subject: past the TTL the record is
        uncapped again and its ladder is full."""
        first = make_steward(config_overrides={'steward_max_attempts': 1})
        first._mark_capped('esc-5352-1')
        first._retry_counts['esc-5352-1'] = 1

        guard_clock[0] = _T0 + timedelta(days=8)

        redeployed = make_steward(config_overrides={'steward_max_attempts': 1})
        assert 'esc-5352-1' not in redeployed._capped_escalations
        assert redeployed._retry_counts.get('esc-5352-1', 0) == 0

    async def test_two_tasks_stewards_do_not_clobber_each_other(
        self, make_steward, guard_clock,
    ):
        """Every steward in the process shares one file per counter, so the
        load-merge-write has to hold at the integration level too."""
        for_42 = make_steward(task={'id': '42', 'title': 't', 'description': 'd'})
        for_43 = make_steward(task={'id': '43', 'title': 't', 'description': 'd'})

        for_42._mark_capped('esc-42-7')
        for_43._mark_capped('esc-43-7')

        redeployed = make_steward(task={'id': '42', 'title': 't', 'description': 'd'})
        assert 'esc-42-7' in redeployed._capped_escalations
        assert 'esc-43-7' in redeployed._capped_escalations
