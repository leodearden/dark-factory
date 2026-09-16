"""The W9-δ steward auto-dismiss race (task 4495, esc-3902-1).

THE RACE.  On a timeout kill the steward returns early without re-reading the
record; the next loop iteration's ``_handle_escalation`` timeout guard awaits a
``git rev-list`` subprocess (the wip probe — tens to hundreds of ms) and only
then calls ``_dismiss_capped_l0``.  A ``resolve_issue`` already in flight from
the killed agent session lands somewhere inside that window.

``EscalationQueue.resolve`` is already a correct atomic check-and-set (its
status check is INSIDE ``escalation_id_lock``), so exactly one of the two wins.
This module pins BOTH halves of the outcome:

* the steward MUST OBSERVE which one won — today ``_dismiss_capped_l0``'s
  ``resolve()`` return value is discarded, so a lost race still logs
  ``gave up on <id> … dismissed the L0`` (a lie) and still publishes
  ``StewardInterrupted``, misreporting a genuine resolution as a benign
  interruption on the outcome channel.  The winner can arrive through either
  door: ``resume``/``restart`` leaves the record ``'resolved'``, while
  ``abandon``/``close_only`` leaves it ``'dismissed'`` — the same terminal
  state this steward's own dismissal produces, distinguished only by who is
  attributed;
* when the DISMISSAL wins, the late resolution must not be dropped on the floor
  — the queue captures it in ``late_resolutions`` and corrects the derived
  ``resolution_class='benign'`` stamp (group B, the end-to-end reproduction).

Prevention alone cannot close this: the steward cannot observe a tool call that
has left the agent process but not yet reached the escalation server.  So the
contract is "report truthfully and lose nothing", not "never race".
"""

from __future__ import annotations

import asyncio
import json
import logging
from unittest.mock import AsyncMock, patch

import pytest
from escalation.classify import effective_benign
from escalation.models import Escalation
from escalation.queue import EscalationQueue, ResolveOutcome

from orchestrator.workflow_types import StewardInterrupted, StewardResolved


@pytest.fixture
def steward(make_steward):
    """This module's mock-queue steward, built by the shared ``make_steward``.

    ``test_steward.py``'s fixture of the same name is module-local, so it is
    restated here rather than imported — and it is the same one-liner over the
    conftest factory (task 3514 made that factory the suite's single owner of
    steward construction).  ``steward_max_attempts=1`` matches the production
    default; the cap-arming helpers below set it explicitly anyway.

    ``make_id`` is stamped away from ``_make_escalation``'s default id for the
    same reason ``test_steward.py`` does it: conftest's default IS ``esc-42-1``,
    so a steward-minted escalation would otherwise silently collide with the
    escalation under test in the auto-escalation paths.
    """
    steward = make_steward(config_overrides={'steward_max_attempts': 1})
    steward.escalation_queue.make_id.return_value = 'esc-42-99'
    return steward


def _make_escalation(**overrides):  # type: ignore[no-untyped-def]
    """A minimal blocking L0, the same shape ``test_steward.py`` builds."""
    defaults: dict = dict(
        id='esc-42-1',
        task_id='42',
        agent_role='implementer',
        severity='blocking',
        category='infra_issue',
        summary='verify loop will not converge',
    )
    defaults.update(overrides)
    return Escalation(**defaults)  # type: ignore[arg-type]


def _resolve_returns(steward, **fields) -> Escalation:
    """Make the steward's mock queue's ``resolve()`` return a record with *fields*.

    The stock mock queue returns a bare ``MagicMock`` from ``resolve()``, which
    ``steward._dismissal_was_overtaken``'s status membership test rejects —
    which is exactly why the production branch reads the RETURNED RECORD rather
    than the ``ResolveOutcome`` out-param: the existing MagicMock-queue tests in
    ``test_steward.py`` keep their current behaviour with no churn.
    """
    rec = _make_escalation(**fields)
    steward.escalation_queue.resolve.return_value = rec
    return rec


def _arm_attempt_cap(steward, esc_id: str) -> None:
    """Drive *steward* to the attempt-cap guard (``steward.py`` :508) for *esc_id*."""
    steward.config.steward_max_attempts = 1
    steward._retry_counts[esc_id] = 1          # at cap
    steward._timeout_counts[esc_id] = 1        # stale sibling counter
    steward._empty_output_counts[esc_id] = 1   # stale sibling counter


def _arm_timeout_cap(steward, esc_id: str) -> None:
    """Drive *steward* to the timeout-kill guard (``steward.py`` :530) for *esc_id*.

    ``steward_max_attempts`` is raised above the seeded ``_retry_counts`` entry
    so the ATTEMPT cap — checked first — does not fire instead.
    """
    steward.config.steward_max_attempts = 5
    steward.config.steward_max_timeouts_per_escalation = 3
    steward._timeout_counts[esc_id] = 3        # at cap
    steward._retry_counts[esc_id] = 1          # stale sibling counter, below cap
    steward._empty_output_counts[esc_id] = 1   # stale sibling counter


def _assert_counters_popped(steward, esc_id: str) -> None:
    """All three per-escalation counters are popped, on BOTH race outcomes."""
    assert esc_id not in steward._retry_counts
    assert esc_id not in steward._timeout_counts
    assert esc_id not in steward._empty_output_counts


def _assert_dismissed_own_l0(steward, esc_id: str) -> None:
    """The converged give-up contract (task 3170) — unchanged by task 4495.

    Deliberately a byte-for-byte restatement of ``test_steward.py``'s helper of
    the same name: task 4495 must not weaken it on EITHER race outcome.  The
    steward still ATTEMPTS the dismissal in both cases; whether the queue
    applied it or refused it as an atomic no-op is what changes downstream.
    """
    steward.escalation_queue.resolve.assert_called_once()
    call_args = steward.escalation_queue.resolve.call_args
    assert call_args[0][0] == esc_id, (
        f'give-up must dismiss its OWN L0 ({esc_id}); '
        f'resolved {call_args[0][0]!r} instead'
    )
    assert call_args[1].get('dismiss') is True
    assert call_args[1].get('resolved_by') == 'auto-dismissed'


# ---------------------------------------------------------------------------
# Group A — unit, MagicMock queue: the steward OBSERVES the check-and-set
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDismissalWonTheRace:
    """``resolve()`` returned a DISMISSED record: today's behaviour, exactly.

    This is the overwhelmingly common case and the one the existing suite
    already pins.  It is restated here because task 4495 adds a branch directly
    above it — a regression that flipped the default would otherwise only show
    up as a change in ``test_steward.py``'s unrelated cap tests.
    """

    async def test_attempt_cap_publishes_interrupted(self, steward):
        channel = asyncio.Queue()
        steward.set_outcome_channel(channel)
        steward.set_wip_probe(AsyncMock(return_value=True))
        esc = _make_escalation()
        _arm_attempt_cap(steward, esc.id)
        _resolve_returns(steward, status='dismissed', resolved_by='auto-dismissed')

        with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            await steward._handle_escalation(esc)

        mock_invoke.assert_not_called()
        _assert_dismissed_own_l0(steward, esc.id)
        # Task-2060 resume-plan semantics: a dismissal is NOT an L1 hand-off.
        steward.escalation_queue.submit.assert_not_called()
        assert channel.get_nowait() == StewardInterrupted(
            reason='attempt_cap', wip_commits_present=True,
        )
        assert channel.empty(), 'exactly one outcome must be published'
        assert esc.id in steward._capped_escalations
        _assert_counters_popped(steward, esc.id)

    async def test_timeout_cap_publishes_interrupted(self, steward):
        channel = asyncio.Queue()
        steward.set_outcome_channel(channel)
        steward.set_wip_probe(AsyncMock(return_value=True))
        esc = _make_escalation()
        _arm_timeout_cap(steward, esc.id)
        _resolve_returns(steward, status='dismissed', resolved_by='auto-dismissed')

        with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            await steward._handle_escalation(esc)

        mock_invoke.assert_not_called()
        _assert_dismissed_own_l0(steward, esc.id)
        steward.escalation_queue.submit.assert_not_called()
        assert channel.get_nowait() == StewardInterrupted(
            reason='timeout', wip_commits_present=True,
        )
        assert channel.empty(), 'exactly one outcome must be published'
        assert esc.id in steward._capped_escalations
        _assert_counters_popped(steward, esc.id)

    async def test_the_stock_mock_queue_keeps_the_interrupted_contract(self, steward):
        """The premise the ~150 ``test_steward.py`` cases rest on, as BEHAVIOUR.

        Those cases never configure ``resolve()``'s return value, so the
        steward reads the stock queue's bare ``MagicMock``.  What has to hold
        is not something about ``unittest.mock`` — it is that THIS steward
        still publishes ``StewardInterrupted`` for that return, because if it
        ever flipped, every one of those cases would silently start asserting a
        different contract.  So the return value is deliberately left
        unconfigured here and the OUTCOME is what is pinned.
        """
        channel = asyncio.Queue()
        steward.set_outcome_channel(channel)
        steward.set_wip_probe(AsyncMock(return_value=True))
        esc = _make_escalation()
        _arm_timeout_cap(steward, esc.id)

        with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock):
            await steward._handle_escalation(esc)

        _assert_dismissed_own_l0(steward, esc.id)
        assert channel.get_nowait() == StewardInterrupted(
            reason='timeout', wip_commits_present=True,
        )
        assert channel.empty(), 'exactly one outcome must be published'


@pytest.mark.asyncio
class TestInFlightResolveWonTheRace:
    """``resolve()`` returned a RESOLVED record: a ``resolve_issue`` beat us.

    The queue's check-and-set already did the right thing — the dismissal was
    an atomic no-op and the record carries the agent's real finding.  What is
    wrong today is purely the steward's REPORTING: it publishes
    ``StewardInterrupted`` (telling the workflow "resume the plan, the steward
    was cut short") and logs that it dismissed the L0, both false.
    """

    async def test_attempt_cap_publishes_resolved(self, steward):
        channel = asyncio.Queue()
        steward.set_outcome_channel(channel)
        steward.set_wip_probe(AsyncMock(return_value=True))
        esc = _make_escalation()
        _arm_attempt_cap(steward, esc.id)
        _resolve_returns(
            steward, status='resolved', resolved_by='claude-task-42-implementer',
            resolution='root cause was a stale lockfile; removed it and verify is green',
        )

        with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            await steward._handle_escalation(esc)

        mock_invoke.assert_not_called()
        # The give-up contract is UNCHANGED: the dismissal is still attempted.
        _assert_dismissed_own_l0(steward, esc.id)
        steward.escalation_queue.submit.assert_not_called()
        assert channel.get_nowait() == StewardResolved(
            resolution_text=(
                'root cause was a stale lockfile; removed it and verify is green'
            ),
        ), 'a lost race is a RESOLUTION, not a benign interruption'
        assert channel.empty(), 'exactly one outcome must be published'
        # Still terminal for this steward, still cleaned up.
        assert esc.id in steward._capped_escalations
        _assert_counters_popped(steward, esc.id)

    async def test_timeout_cap_publishes_resolved(self, steward):
        """The timeout guard is the SECOND instance of the same shape.

        Fixing only the attempt-cap branch would leave the misreport reachable
        through a different door — which is precisely the drift the two guards'
        existing "so neither wip branch can drift from the other" comments
        exist to prevent.
        """
        channel = asyncio.Queue()
        steward.set_outcome_channel(channel)
        steward.set_wip_probe(AsyncMock(return_value=True))
        esc = _make_escalation()
        _arm_timeout_cap(steward, esc.id)
        _resolve_returns(
            steward, status='resolved', resolved_by='claude-task-42-implementer',
            resolution='fixed the flaky fixture',
        )

        with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            await steward._handle_escalation(esc)

        mock_invoke.assert_not_called()
        _assert_dismissed_own_l0(steward, esc.id)
        steward.escalation_queue.submit.assert_not_called()
        assert channel.get_nowait() == StewardResolved(
            resolution_text='fixed the flaky fixture',
        )
        assert channel.empty()
        assert esc.id in steward._capped_escalations
        _assert_counters_popped(steward, esc.id)

    async def test_resolution_text_falls_back_to_the_summary(self, steward):
        """A resolved record with no resolution text still publishes a resolution.

        ``StewardResolved.resolution_text`` is a plain ``str``, so an empty or
        absent ``resolution`` must fall back rather than publish ``None`` —
        the same fallback the success branch already uses.
        """
        channel = asyncio.Queue()
        steward.set_outcome_channel(channel)
        steward.set_wip_probe(AsyncMock(return_value=True))
        esc = _make_escalation(summary='verify loop will not converge')
        _arm_timeout_cap(steward, esc.id)
        _resolve_returns(steward, status='resolved', resolution=None)

        with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock):
            await steward._handle_escalation(esc)

        assert channel.get_nowait() == StewardResolved(
            resolution_text='verify loop will not converge',
        )

    async def test_lost_race_does_not_log_the_misleading_dismissal_warning(
        self, steward, caplog,
    ):
        """The ``gave up … dismissed the L0`` WARNING is FALSE on a lost race.

        It is also the line an operator greps for when reconstructing why a
        workflow resumed instead of completing, so leaving it in place would
        make the log actively mislead.  A warning naming the RACE takes its
        place — the loss must stay loud, just true.
        """
        steward.set_wip_probe(AsyncMock(return_value=True))
        esc = _make_escalation()
        _arm_timeout_cap(steward, esc.id)
        _resolve_returns(
            steward, status='resolved', resolved_by='claude-task-42-implementer',
            resolution='the real finding',
        )

        with caplog.at_level(logging.WARNING, logger='orchestrator.steward'), \
                patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock):
            await steward._handle_escalation(esc)

        text = '\n'.join(r.getMessage() for r in caplog.records)
        assert 'dismissed the L0' not in text, (
            'the dismissal was an atomic no-op — claiming it landed is the '
            'misreport this task exists to fix'
        )
        assert esc.id in text, 'the race warning must name the escalation'
        assert 'claude-task-42-implementer' in text, (
            'the race warning must name who actually resolved the record, so '
            'an operator can find the resolution without re-reading the queue'
        )

    async def test_won_race_still_logs_the_dismissal_warning(self, steward, caplog):
        """The common case keeps its existing, TRUE, give-up WARNING."""
        steward.set_wip_probe(AsyncMock(return_value=True))
        esc = _make_escalation()
        _arm_timeout_cap(steward, esc.id)
        _resolve_returns(steward, status='dismissed', resolved_by='auto-dismissed')

        with caplog.at_level(logging.WARNING, logger='orchestrator.steward'), \
                patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock):
            await steward._handle_escalation(esc)

        text = '\n'.join(r.getMessage() for r in caplog.records)
        assert 'dismissed the L0' in text
        assert esc.id in text


@pytest.mark.asyncio
class TestAnAgentDismissalWonTheRace:
    """``resolve()`` returned a DISMISSED record attributed to the AGENT.

    ``resolve_issue``'s dismissing actions (``abandon`` / ``close_only``, see
    ``server._DISMISS_ACTIONS``) close the record as ``'dismissed'`` — the SAME
    terminal state the steward's own auto-dismiss produces, so only the
    attribution tells the two apart.  Reading the status alone therefore missed
    exactly half the race: the steward logged that it had dismissed an L0 whose
    dismissal was an atomic no-op, and published ``StewardInterrupted``, telling
    the workflow to resume the plan for a record an agent had deliberately
    abandoned.
    """

    AGENT = 'claude-task-42-implementer'

    async def _drive(self, steward, esc, arm, **stored_fields):
        channel = asyncio.Queue()
        steward.set_outcome_channel(channel)
        steward.set_wip_probe(AsyncMock(return_value=True))
        arm(steward, esc.id)
        _resolve_returns(steward, **stored_fields)

        with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock):
            await steward._handle_escalation(esc)

        return channel

    @pytest.mark.parametrize(
        ('arm', 'reason'),
        [(_arm_attempt_cap, 'attempt_cap'), (_arm_timeout_cap, 'timeout')],
    )
    async def test_an_abandon_publishes_resolved_not_interrupted(
        self, steward, arm, reason: str,
    ):
        """Both wip-gated doors, because either can lose to the same abandon."""
        esc = _make_escalation()
        channel = await self._drive(
            steward, esc, arm,
            status='dismissed', resolved_by=self.AGENT,
            resolution='abandoning: the premise is false, this needs a new task',
        )

        # The give-up contract is UNCHANGED: the dismissal is still attempted.
        _assert_dismissed_own_l0(steward, esc.id)
        steward.escalation_queue.submit.assert_not_called()
        assert channel.get_nowait() == StewardResolved(
            resolution_text='abandoning: the premise is false, this needs a new task',
        ), (
            'an agent that deliberately abandoned the record did not leave the '
            'steward cut short — resuming the plan for it is the misreport this '
            'task exists to fix, arriving through the dismiss door'
        )
        assert channel.empty(), 'exactly one outcome must be published'
        assert esc.id in steward._capped_escalations
        _assert_counters_popped(steward, esc.id)

    async def test_the_warning_names_the_agent_that_actually_closed_it(
        self, steward, caplog,
    ):
        esc = _make_escalation()
        with caplog.at_level(logging.WARNING, logger='orchestrator.steward'):
            await self._drive(
                steward, esc, _arm_timeout_cap,
                status='dismissed', resolved_by=self.AGENT, resolution='abandoned',
            )

        text = '\n'.join(r.getMessage() for r in caplog.records)
        assert 'dismissed the L0' not in text, (
            'this steward dismissed nothing — its call was an atomic no-op'
        )
        assert self.AGENT in text, (
            'the race warning must name who actually closed the record'
        )
        assert esc.id in text

    async def test_another_automated_sweep_is_not_a_lost_race(self, steward, caplog):
        """A reaper-sweep resolver winning is NOT a finding to publish.

        ``harness-orphan-reaper`` closed the record the same way this steward
        was about to, carrying no agent resolution — so the outcome stays
        ``StewardInterrupted`` and the give-up WARNING stays true.  This is the
        conjunct that keeps the broadened check from reading every dismissal as
        a race.
        """
        esc = _make_escalation()
        with caplog.at_level(logging.WARNING, logger='orchestrator.steward'):
            channel = await self._drive(
                steward, esc, _arm_timeout_cap,
                status='dismissed', resolved_by='harness-orphan-reaper',
            )

        assert channel.get_nowait() == StewardInterrupted(
            reason='timeout', wip_commits_present=True,
        )
        assert channel.empty()
        text = '\n'.join(r.getMessage() for r in caplog.records)
        assert 'dismissed the L0' in text, (
            'the record WAS dismissed by an equivalent automated sweep, so the '
            'give-up warning is still the true one'
        )


# ---------------------------------------------------------------------------
# Group B — integration, a REAL EscalationQueue: the esc-3902-1 reproduction
# ---------------------------------------------------------------------------


@pytest.fixture
def steward_with_real_queue(make_steward, tmp_path):
    """``TaskSteward`` over a REAL filesystem-backed ``EscalationQueue``.

    An INDEPENDENT ``make_steward`` build — it deliberately does not request
    the ``steward`` fixture, so no second steward is constructed alongside it
    and its queue dir cannot collide with the mock queue's (which conftest
    derives from that steward's own worktree).  Mirrors
    ``test_steward.py::steward_with_real_queue``.
    """
    return make_steward(
        escalation_queue=EscalationQueue(tmp_path / 'dismiss-race-escalations'),
        config_overrides={'steward_max_attempts': 1},
    )


@pytest.mark.asyncio
class TestLateResolutionSurvivesTheAutoDismiss:
    """End-to-end: the auto-dismiss wins, and the late finding is NOT destroyed.

    The reported harm was that the steward's substantive finding "was recorded
    as ``resolution_class=benign`` instead of carrying the steward's actual
    finding".  Groups A above cover the half where the resolve wins; this is
    the half where it loses, driven through the REAL queue so the on-disk
    outcome — not a mock's call log — is what is asserted.
    """

    async def _drive_timeout_cap_dismissal(self, steward, esc: Escalation):
        """Seed *esc* pending, then drive the timeout cap to auto-dismiss it."""
        steward.escalation_queue.submit(esc)
        channel = asyncio.Queue()
        steward.set_outcome_channel(channel)
        steward.set_wip_probe(AsyncMock(return_value=True))
        _arm_timeout_cap(steward, esc.id)

        with patch('orchestrator.steward.invoke_agent', new_callable=AsyncMock):
            await steward._handle_escalation(esc)

        return channel

    async def test_auto_dismiss_lands_benign_then_late_resolve_corrects_it(
        self, steward_with_real_queue,
    ):
        steward = steward_with_real_queue
        queue = steward.escalation_queue
        esc = _make_escalation(id='esc-42-31')

        channel = await self._drive_timeout_cap_dismissal(steward, esc)

        # --- the dismissal won the race -------------------------------------
        stored = queue.get('esc-42-31')
        assert stored is not None
        assert stored.status == 'dismissed'
        assert stored.resolved_by == 'auto-dismissed'
        assert stored.resolution_class == 'benign', (
            "'auto-dismissed' is in classify._REAPER_SWEEP_RESOLVERS, so the "
            "stamp is DERIVED as benign — this is the state esc-3902-1 reported"
        )
        assert effective_benign(stored) == ('benign', 'stamped')
        assert channel.get_nowait() == StewardInterrupted(
            reason='timeout', wip_commits_present=True,
        )

        # --- the in-flight resolve_issue lands microseconds later ------------
        out: ResolveOutcome = {}  # type: ignore[typeddict-item]
        returned = queue.resolve(
            'esc-42-31', "steward's real finding",
            resolved_by='claude-task-4495-steward', outcome=out,
        )

        assert returned is not None
        assert out['applied'] is False, 'the check-and-set correctly refuses to re-close'
        assert out['late_resolution_captured'] is True
        assert out['resolution_class_corrected'] == 'actionable'

        rec = queue.get('esc-42-31')
        assert rec is not None
        # The record's OWN terminal state is untouched — downstream waiters
        # have already consumed it and the workflow has resumed.
        assert rec.status == 'dismissed'
        assert rec.resolved_by == 'auto-dismissed'
        assert rec.resolution == stored.resolution
        assert rec.resolved_at == stored.resolved_at
        # …but the finding survives, and the derived stamp is corrected.
        assert len(rec.late_resolutions) == 1
        entry = rec.late_resolutions[0]
        assert entry['resolution'] == "steward's real finding"
        assert entry['resolved_by'] == 'claude-task-4495-steward'
        assert entry['prior_resolution_class'] == 'benign'
        assert entry['timestamp'], 'the queue stamps its own clock at write time'
        assert rec.resolution_class == 'actionable'
        assert effective_benign(rec) == ('actionable', 'stamped'), (
            'the dashboard aggregation must now read the truth'
        )

    async def test_capture_writes_in_place_and_never_resurrects_the_record(
        self, steward_with_real_queue,
    ):
        """The archived record stays archived — one copy, no queue-root ghost.

        ``TestResolveIdempotent`` pins this no-orphan-archive invariant at the
        queue level; it is re-asserted through the real steward path because
        that path is what actually archives the record, and a capture that
        wrote via ``_atomic_write`` rather than ``_atomic_write_path`` would
        resurrect it into the root where ``_next_escalation`` would re-read it
        as pending.
        """
        steward = steward_with_real_queue
        queue = steward.escalation_queue
        esc = _make_escalation(id='esc-42-32')

        await self._drive_timeout_cap_dismissal(steward, esc)
        assert not (queue.queue_dir / 'esc-42-32.json').exists()

        queue.resolve(
            'esc-42-32', "steward's real finding",
            resolved_by='claude-task-4495-steward',
        )

        assert not (queue.queue_dir / 'esc-42-32.json').exists(), (
            'the capture must write IN PLACE at _locate_path, never into the '
            'queue root — a resurrected record is re-read as pending'
        )
        archived = [
            p for p in queue.queue_dir.rglob('esc-42-32.json')
            if p.parent != queue.queue_dir
        ]
        assert len(archived) == 1, 'exactly one archive copy must exist'
        on_disk = json.loads(archived[0].read_text())
        assert on_disk['status'] == 'dismissed'
        assert on_disk['resolution_class'] == 'actionable'
        assert len(on_disk['late_resolutions']) == 1
        assert on_disk['late_resolutions'][0]['resolution'] == "steward's real finding"
