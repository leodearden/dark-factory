"""Contract tests for the claimant-liveness resume fork (task 3540 / PRD
``plans/task-escalation-state-graph-prd.md`` D8, spec E9).

The change under test replaces two LEVEL/STATUS codifications with a single
CLAIMANT-LIVENESS one:

  - ``Harness._cascade_unblock_member`` no longer gates on
    ``status == 'blocked'``.  It gates on an allow-list
    (``{blocked, in-progress}``) plus "does this row have a LIVE claimant?".
    No live claimant → re-pend to Table B's ``resume`` target; live claimant
    → skip, because the synchronous wake path owns the re-pend.
  - ``Harness._on_escalation_resolved`` no longer gates the resume branch on
    ``escalation.level >= 1``.  An L0 whose workflow crashed between filing
    and exit has had its ``_escalation_events`` entry popped and MUST reach
    the re-pend path.

and adds one delivery: ``granted_files`` are folded into plan.json /
``metadata.files`` on the re-pend, so a scope grant resolved against a task
with no live workflow is actually delivered.

This module owns the shared fixtures every later step consumes.  Two
MagicMock traps are closed here deliberately — see ``harness`` below.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.artifacts import TaskArtifacts
from orchestrator.harness import _REBLOCK_GUARD_THRESHOLD, Harness

# The claimant-liveness TTL these fixtures pin onto ``config.claimant_liveness_
# ttl_secs``.  ``mock_orch_config`` is a spec_set MagicMock and does NOT supply
# a real numeric default for this field, so ``timedelta(seconds=...)`` against
# it would raise TypeError.  Mirrors the production default
# (``OrchestratorConfig.claimant_liveness_ttl_secs`` = 300.0).
TTL_SECS: float = 300.0

# A well-formed claimant identity in ``compose_claimant_run_id`` shape
# (``f'{run_id}/{session_id}/pid={owner_pid}'``).
LIVE_CLAIMANT: str = 'run-abc/sess-def/pid=4242'


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def harness(tmp_path: Path, mock_orch_config) -> Harness:
    """Harness with mocked internals for resume/claimant-liveness unit testing.

    Modelled on ``test_cascade_unblock.harness``, with the MagicMock traps the
    new liveness fork introduces closed explicitly:

    - ``scheduler.is_actively_held`` — a bare MagicMock attribute returns a
      TRUTHY child mock, which would make EVERY row read as having a live
      claimant and silently skip EVERY flip (a green suite that tests
      nothing).  Stubbed to ``False``.
    - ``config.claimant_liveness_ttl_secs`` — ``mock_orch_config`` is
      ``spec_set`` against ``OrchestratorConfig`` but supplies no real numeric
      value for this field; ``timedelta(seconds=<MagicMock>)`` raises
      TypeError.  Pinned to :data:`TTL_SECS`.
    - ``scheduler.get_task`` — returns a FULL row (id/status/claimant_run_id/
      heartbeat_at/metadata), not the ``{'id': 'x', 'metadata': {}}`` stub the
      older suite used: the liveness read and the write-time corroborating read
      both need ``status`` and the claimant columns on the same snapshot.
    """
    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.OverrideStore'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        h = Harness(mock_orch_config)

    h.config.claimant_liveness_ttl_secs = TTL_SECS

    h.scheduler = MagicMock()
    # Default row: a stranded `blocked` task — no claimant, no heartbeat — so
    # the default case flips, matching the older suite's default expectation.
    h.scheduler.get_task = AsyncMock(
        return_value=_row('blocked', claimant=None, heartbeat=None)
    )
    h.scheduler.get_status = AsyncMock(return_value='blocked')
    h.scheduler.set_task_status = AsyncMock()
    h.scheduler.update_task = AsyncMock(return_value=True)
    # THE trap: must be an explicit False, never a bare MagicMock attribute.
    h.scheduler.is_actively_held = MagicMock(return_value=False)

    # _merge_worker stays None — the unhalt branch is skipped in all tests here.
    return h


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(
    status: str,
    *,
    task_id: str = '3438',
    claimant: str | None = LIVE_CLAIMANT,
    heartbeat: str | None = 'fresh',
    metadata: dict | None = None,
    ttl_secs: float = TTL_SECS,
) -> dict:
    """Build a fused-memory task row for the liveness oracle.

    ``heartbeat`` is a symbolic age, derived from the real config knob rather
    than a magic literal so a TTL change can never silently invert a fixture:

    - ``'fresh'`` → ``now`` (well inside the TTL → live, given a claimant)
    - ``'stale'`` → ``now - 2 * ttl_secs`` (outside the TTL → not live)
    - ``None``    → no ``heartbeat_at`` at all (unparseable → not live)

    ``claimant=None`` means "no claimant at all", which reads as not-live
    regardless of the heartbeat.
    """
    now = datetime.now(UTC)
    if heartbeat == 'fresh':
        heartbeat_at: str | None = now.isoformat()
    elif heartbeat == 'stale':
        heartbeat_at = (now - timedelta(seconds=2 * ttl_secs)).isoformat()
    elif heartbeat is None:
        heartbeat_at = None
    else:  # pragma: no cover — fixture misuse
        raise ValueError(f'unknown heartbeat age {heartbeat!r}')

    return {
        'id': task_id,
        'status': status,
        'claimant_run_id': claimant,
        'heartbeat_at': heartbeat_at,
        'metadata': dict(metadata or {}),
    }


def _esc(
    *,
    task_id: str = '3438',
    level: int = 1,
    resolved_by: str = 'l2-cascade:esc-4000-39',
    status: str = 'resolved',
    granted_files: list[str] | None = None,
    category: str = 'infra_issue',
    summary: str = 'x',
    esc_id: str | None = None,
) -> Escalation:
    """Build a resolved Escalation — mirrors ``test_cascade_unblock._make_l1_esc``
    with the level, granted_files and category slots opened up."""
    return Escalation(
        id=esc_id or f'esc-{task_id}-1',
        task_id=task_id,
        agent_role='workflow',
        severity='blocking',
        category=category,
        summary=summary,
        level=level,
        status=status,
        resolved_by=resolved_by,
        granted_files=list(granted_files or []),
    )


# ---------------------------------------------------------------------------
# Fixture self-checks — these guard the traps above, not production behaviour.
# ---------------------------------------------------------------------------

class TestFixtureWiring:
    """If these fail, every liveness assertion in this module is vacuous."""

    def test_is_actively_held_is_explicitly_false(self, harness: Harness):
        """A bare MagicMock attribute would be TRUTHY and suppress every flip."""
        assert harness.scheduler.is_actively_held('anything') is False

    def test_ttl_is_a_real_number(self, harness: Harness):
        """``timedelta(seconds=<MagicMock>)`` raises TypeError — pin a float."""
        assert timedelta(seconds=harness.config.claimant_liveness_ttl_secs) == (
            timedelta(seconds=TTL_SECS)
        )

    def test_row_ages_derive_from_the_ttl_knob(self):
        """'stale' must land outside the TTL and 'fresh' inside it, by
        construction rather than by a hardcoded literal."""
        from shared.task_claimant import has_live_claimant

        now = datetime.now(UTC)
        ttl = timedelta(seconds=TTL_SECS)

        assert has_live_claimant(_row('blocked', heartbeat='fresh'), now, ttl)
        assert not has_live_claimant(_row('blocked', heartbeat='stale'), now, ttl)
        assert not has_live_claimant(_row('blocked', heartbeat=None), now, ttl)
        assert not has_live_claimant(
            _row('blocked', claimant=None, heartbeat='fresh'), now, ttl
        )

    def test_default_fixture_row_is_a_stranded_blocked_task(self, harness: Harness):
        """The fixture's default ``get_task`` row must read as NOT live, so a
        test that does not care about liveness still exercises the flip."""
        from shared.task_claimant import has_live_claimant

        row = harness.scheduler.get_task.return_value  # type: ignore[attr-defined]
        assert row['status'] == 'blocked'
        assert not has_live_claimant(
            row, datetime.now(UTC), timedelta(seconds=TTL_SECS)
        )

    def test_esc_builder_carries_granted_files(self):
        esc = _esc(task_id='7', level=0, granted_files=['pkg/b.py'])
        assert esc.level == 0
        assert esc.task_id == '7'
        assert esc.granted_files == ['pkg/b.py']

    def test_worktree_base_is_under_tmp_path(self, harness: Harness, tmp_path: Path):
        """Step-8's on-disk plan.json fixtures resolve through
        ``git_ops.worktree_base``; it must be inside tmp_path, not the repo."""
        assert harness.git_ops.worktree_base.is_relative_to(tmp_path)


# ---------------------------------------------------------------------------
# step-1 — the _cascade_unblock_member claimant-liveness fork
# ---------------------------------------------------------------------------

def _resume_target(esc: Escalation) -> str:
    """The Table B target for this escalation's ``resume``.

    Asserted against the SAME authority the production code consults so a
    future Table B edit propagates into these tests instead of drifting from
    them — never the literal ``'pending'``.
    """
    from escalation.action_effects import effect_for

    effect = effect_for('resume', esc.level, esc.category)
    assert effect is not None and effect.target_status is not None
    return effect.target_status


async def _drive(harness: Harness, esc: Escalation) -> None:
    """Fire the resolve callback and drain the scheduled background work.

    The existing cascade-suite drain idiom — ``_on_escalation_resolved`` is
    synchronous and schedules ``_cascade_unblock_member`` via
    ``_schedule_coro_threadsafe``.
    """
    harness._on_escalation_resolved(esc)
    await asyncio.gather(*list(harness._background_tasks))


def _wire(harness: Harness, row: dict) -> None:
    """Point BOTH reads at the same row.

    ``get_status`` is the gate's status source and ``get_task`` the liveness
    source; production reads them independently (see ``_cascade_unblock_
    member``'s "Efficiency note"), so a fixture that sets only one of them
    silently tests a row that cannot exist.
    """
    harness.scheduler.get_status = AsyncMock(return_value=row['status'])
    harness.scheduler.get_task = AsyncMock(return_value=row)


def _guard_charges(harness: Harness) -> list:
    """The ``update_task`` calls that charged the re-block guard."""
    return [
        c for c in harness.scheduler.update_task.await_args_list  # type: ignore[attr-defined]
        if 'reblock_guard' in (c.args[1] if len(c.args) > 1 else c.kwargs.get('updates', {}))
    ]


@pytest.mark.asyncio
class TestCascadeMemberLivenessFork:
    """``_cascade_unblock_member`` gates on claimant liveness, not ``status ==
    'blocked'`` (task 3540 / PRD D8, spec E9)."""

    async def test_in_progress_with_no_claimant_is_repended(
        self, harness: Harness, caplog
    ):
        """[boundary #11 — THE red assertion] A stranded in-progress row
        re-pends.

        Today this DEBUG-skips on ``status != 'blocked'``, leaving a task
        in-progress with nobody heartbeating it and its escalation now closed.
        """
        esc = _esc(task_id='3438')
        _wire(harness, _row('in-progress', claimant=None, heartbeat=None))

        with caplog.at_level(logging.INFO):
            await _drive(harness, esc)

        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            '3438', _resume_target(esc)
        )

    async def test_in_progress_with_stale_heartbeat_is_repended(
        self, harness: Harness
    ):
        """A claimant whose heartbeat aged past ``claimant_liveness_ttl_secs``
        is not a live claimant — the row re-pends."""
        esc = _esc(task_id='3438')
        _wire(harness, _row('in-progress', claimant=LIVE_CLAIMANT, heartbeat='stale'))

        await _drive(harness, esc)

        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            '3438', _resume_target(esc)
        )

    async def test_in_progress_with_fresh_heartbeat_is_not_flipped(
        self, harness: Harness, caplog
    ):
        """A LIVE claimant owns its own re-pend (woken by ``event.set()``) —
        flipping here would race a running workflow."""
        esc = _esc(task_id='3438')
        _wire(harness, _row('in-progress', claimant=LIVE_CLAIMANT, heartbeat='fresh'))

        with caplog.at_level(logging.DEBUG):
            await _drive(harness, esc)

        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]
        assert any(
            r.levelno == logging.DEBUG and 'live claimant' in r.getMessage()
            for r in caplog.records
        ), "Expected a DEBUG record naming the live claimant as the reason for the skip"

    async def test_is_actively_held_alone_suppresses_the_flip(
        self, harness: Harness, caplog
    ):
        """The in-memory signal is consulted FIRST and is sufficient on its own.

        Mirrors ``TaskGroundTruth._resolve_live_claimant``'s priority order: a
        workflow can hold the slot and the module locks before it has stamped
        a claimant row, which a DB-only oracle reads as stranded.
        """
        esc = _esc(task_id='3438')
        _wire(harness, _row('in-progress', claimant=None, heartbeat=None))
        harness.scheduler.is_actively_held = MagicMock(return_value=True)

        with caplog.at_level(logging.DEBUG):
            await _drive(harness, esc)

        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]
        harness.scheduler.is_actively_held.assert_called_with('3438')

    async def test_blocked_with_no_claimant_still_flips(self, harness: Harness):
        """[regression] Today's behaviour for the ordinary blocked orphan is
        unchanged — this is the common production shape."""
        esc = _esc(task_id='3438')
        _wire(harness, _row('blocked', claimant=None, heartbeat=None))

        await _drive(harness, esc)

        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            '3438', _resume_target(esc)
        )

    async def test_blocked_with_fresh_heartbeat_is_not_flipped(
        self, harness: Harness, caplog
    ):
        """[the second red assertion] Today ``blocked`` flips UNCONDITIONALLY.

        The rule is status-agnostic: a live claimant means the wake path owns
        the re-pend at every allowed status, blocked included.  The recovery
        edge for the skipped row is the scheduler's stranded-blocked-redispatch
        sweep, which re-owns exactly "blocked, no live claimant, no open
        escalation" once the claimant goes stale.
        """
        esc = _esc(task_id='3438')
        _wire(harness, _row('blocked', claimant=LIVE_CLAIMANT, heartbeat='fresh'))

        with caplog.at_level(logging.DEBUG):
            await _drive(harness, esc)

        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]
        assert any(
            r.levelno == logging.DEBUG and 'live claimant' in r.getMessage()
            for r in caplog.records
        ), "Expected a DEBUG record naming the live claimant"

    @pytest.mark.parametrize(
        'status',
        ['done', 'cancelled', 'deferred', 'merge-deferred', 'pending', 'review'],
    )
    async def test_status_outside_the_allow_list_is_never_flipped(
        self, harness: Harness, caplog, status: str
    ):
        """The allow-list is ``{blocked, in-progress}`` — an ALLOW-list, not a
        deny-list, so a future status is skipped by construction rather than
        silently acquired.

        A NULL claimant makes every one of these read as "not live", so only
        the status gate can be what withholds the flip.
        """
        esc = _esc(task_id='3438')
        _wire(harness, _row(status, claimant=None, heartbeat=None))

        with caplog.at_level(logging.DEBUG):
            await _drive(harness, esc)

        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]
        assert any(
            r.levelno == logging.DEBUG and status in r.getMessage()
            for r in caplog.records
        ), f"Expected a DEBUG record naming the skipped status {status!r}"


@pytest.mark.asyncio
class TestLivenessForkReblockGuardSemantics:
    """"Re-block guard semantics unchanged: the resolution-driven flip still
    charges it" — including for the newly-eligible in-progress origin."""

    async def test_stranded_in_progress_flip_charges_the_guard(
        self, harness: Harness
    ):
        esc = _esc(task_id='3438')
        _wire(harness, _row('in-progress', claimant=None, heartbeat=None))

        await _drive(harness, esc)

        charges = _guard_charges(harness)
        assert len(charges) == 1, f'expected exactly one guard charge, got {charges}'
        assert charges[0].kwargs.get('metadata_mode') == 'merge'
        harness.scheduler.set_task_status.assert_awaited_once()  # type: ignore[attr-defined]

    async def test_live_claimant_skip_does_not_charge_the_guard(
        self, harness: Harness
    ):
        """The liveness gate sits BEFORE the guard, so a skipped row never
        burns a re-pend budget it did not spend."""
        esc = _esc(task_id='3438')
        _wire(harness, _row('in-progress', claimant=LIVE_CLAIMANT, heartbeat='fresh'))

        await _drive(harness, esc)

        assert _guard_charges(harness) == []
        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]

    async def test_guard_at_threshold_withholds_the_in_progress_flip_too(
        self, harness: Harness
    ):
        """A same-signature guard already at the threshold withholds the flip
        regardless of whether the row originated blocked or in-progress."""
        from orchestrator.harness import _REBLOCK_GUARD_THRESHOLD

        esc = _esc(task_id='3438')
        signature = harness._reblock_signature(esc)
        _wire(
            harness,
            _row(
                'in-progress',
                claimant=None,
                heartbeat=None,
                metadata={
                    'reblock_guard': {
                        'count': _REBLOCK_GUARD_THRESHOLD,
                        'signature': signature,
                    }
                },
            ),
        )

        await _drive(harness, esc)

        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]
        assert _guard_charges(harness) == []


# ---------------------------------------------------------------------------
# step-4 — the INV-3 write-time corroborating read
# ---------------------------------------------------------------------------

def _wire_two_snapshots(
    harness: Harness, early: dict, late: dict | None
) -> list[str]:
    """``get_task`` returns *early* until the re-block guard's ``update_task``
    lands, then *late* — and every scheduler call is appended to one ordered log.

    Modelling the skew this way, rather than as a positional ``side_effect``
    list, keeps the fixture independent of HOW MANY ``get_task`` round-trips the
    gate path happens to make (today: the infra pre-gate's, then the guard's).
    The only thing it pins is the thing INV-3 is about — that the snapshot the
    WRITE is authorised against is read after the last gate step, not before it.

    The returned list is the call-order assertion surface the happy-path case
    uses to prove the corroborating read is the LAST read before the write.
    ``get_status`` is deliberately NOT logged: it is the gate's status source
    and is unchanged by this step.
    """
    order: list[str] = []
    guard_written = False

    async def _get_task(_tid, *args, **kwargs):
        order.append('get_task')
        return late if guard_written else early

    async def _update_task(*args, **kwargs):
        nonlocal guard_written
        order.append('update_task')
        guard_written = True
        return True

    async def _set_task_status(*args, **kwargs):
        order.append('set_task_status')

    harness.scheduler.get_status = AsyncMock(return_value=early['status'])
    harness.scheduler.get_task = AsyncMock(side_effect=_get_task)
    harness.scheduler.update_task = AsyncMock(side_effect=_update_task)
    harness.scheduler.set_task_status = AsyncMock(side_effect=_set_task_status)
    return order


@pytest.mark.asyncio
class TestCorroboratingReadBeforeTheWrite:
    """INV-3: the flip is authorised against a snapshot read immediately
    before the write, not against the (by then staler) gate snapshot.

    ``_cascade_unblock_member`` is a chain of separate MCP round-trips with no
    atomic compare-and-set, so the TOCTOU window cannot be CLOSED — only
    narrowed. Before this step the window spanned
    ``get_status → guard get_task → guard update_task → set_task_status``:
    a task could reach a terminal status, or be claimed by a fresh workflow,
    anywhere inside it and still be flipped back to ``pending``. After it the
    window is one hop, and status AND claimant liveness are re-derived from a
    single snapshot rather than two skewed reads.
    """

    async def test_late_terminal_done_aborts_the_write(
        self, harness: Harness, caplog
    ):
        """The row completed between the gate and the write — never resurrect it.

        A ``pending`` write onto a ``done`` row is the worst outcome on this
        path: it re-dispatches finished work, and the task's own escalation is
        already closed, so nothing upstream would notice.
        """
        esc = _esc(task_id='3438')
        _wire_two_snapshots(
            harness,
            _row('in-progress', claimant=None, heartbeat=None),
            _row('done', claimant=None, heartbeat=None),
        )

        with caplog.at_level(logging.DEBUG):
            await _drive(harness, esc)

        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]
        assert any(
            'done' in r.getMessage() and r.levelno >= logging.DEBUG
            for r in caplog.records
        ), 'Expected a record naming the observed late status'

    async def test_late_terminal_cancelled_aborts_the_write(
        self, harness: Harness
    ):
        """Same rule for the other terminal status — the gate is the allow-list,
        re-applied, not a hardcoded ``done`` special case."""
        esc = _esc(task_id='3438')
        _wire_two_snapshots(
            harness,
            _row('in-progress', claimant=None, heartbeat=None),
            _row('cancelled', claimant=None, heartbeat=None),
        )

        await _drive(harness, esc)

        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]

    async def test_late_fresh_claimant_aborts_the_write(
        self, harness: Harness, caplog
    ):
        """A workflow claimed the row between the gate and the write.

        The status is still re-pendable, so the allow-list alone would let the
        write through — it is the LIVENESS half of the corroboration that has
        to catch this. Flipping now would race a workflow that has already
        stamped a claimant and is heartbeating it.
        """
        esc = _esc(task_id='3438')
        _wire_two_snapshots(
            harness,
            _row('blocked', claimant=None, heartbeat=None),
            _row('blocked', claimant=LIVE_CLAIMANT, heartbeat='fresh'),
        )

        with caplog.at_level(logging.DEBUG):
            await _drive(harness, esc)

        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]
        assert any(
            'claimant' in r.getMessage() for r in caplog.records
        ), 'Expected a record naming the claimant that appeared'

    async def test_unreadable_late_row_aborts_the_write(
        self, harness: Harness, caplog
    ):
        """Fail-SAFE, not fail-open: a row we could not read is never re-pended.

        Note this is the OPPOSITE default from ``_resume_repend_liveness``,
        where an absent row reads as "no live claimant" (so an unreadable row
        never SUPPRESSES a flip). The asymmetry is deliberate: there, a missing
        row means missing claimant columns and the conservative reading is
        "not live"; here, a missing row means no evidence at all that the write
        is still correct, and the conservative action is to do nothing. The
        next resolution, sweep or dispatch re-derives it.
        """
        esc = _esc(task_id='3438')
        _wire_two_snapshots(
            harness, _row('blocked', claimant=None, heartbeat=None), None
        )

        with caplog.at_level(logging.DEBUG):
            await _drive(harness, esc)

        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]
        assert any(
            r.levelno == logging.WARNING for r in caplog.records
        ), 'An unreadable row must be operator-visible — WARNING, not DEBUG'

    async def test_agreeing_snapshots_flip_and_the_read_is_last(
        self, harness: Harness
    ):
        """[the ordering assertion] Happy path, plus WHERE the read sits.

        Asserting only "it flips" would pass against an implementation that
        corroborated against the gate's own stale snapshot — which is the bug.
        So the call-order log is asserted directly: the last two scheduler
        calls must be ``get_task`` then ``set_task_status``, and the guard's
        ``update_task`` must already have happened. That makes the
        corroborating read provably the final read before the write.
        """
        esc = _esc(task_id='3438')
        row = _row('in-progress', claimant=None, heartbeat=None)
        order = _wire_two_snapshots(harness, row, dict(row))

        await _drive(harness, esc)

        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            '3438', _resume_target(esc)
        )
        assert order[-2:] == ['get_task', 'set_task_status'], (
            f'The corroborating read must be the LAST read before the write; '
            f'observed call order {order}'
        )
        assert 'update_task' in order[:-2], (
            f"The corroborating read must come AFTER the re-block guard's "
            f'persist, not before it; observed call order {order}'
        )


# ---------------------------------------------------------------------------
# step-6 — the `level >= 1` gate removal in _on_escalation_resolved
# ---------------------------------------------------------------------------

def _record_scheduling(harness: Harness) -> list[str]:
    """Start recording the labels ``_on_escalation_resolved`` schedules under.

    Call BEFORE the drive; the returned list fills in as coroutines are
    scheduled. Asserting on WHICH coroutine was scheduled — not only on the
    eventual ``set_task_status`` — is what separates "the routing let it
    through" from "the routing dropped it and something else wrote the
    status". The labels are the harness's own, produced at the two scheduling
    sites inside the ``WORKFLOW_RESUME`` branch (``cascade-unblock task ...``
    / ``orphan-unblock task ...``).

    ``_schedule_coro_threadsafe`` does NOT put its ``label`` on the asyncio
    task name, so the label is observable only by wrapping the method. This
    WRAPS rather than replaces: the coroutine is still really scheduled, so
    the ``await asyncio.gather(*_background_tasks)`` drain idiom keeps working
    and the assertions below stay end-to-end.
    """
    labels: list[str] = []
    original = harness._schedule_coro_threadsafe

    def _spy(coro, *, label: str):
        labels.append(label)
        return original(coro, label=label)

    harness._schedule_coro_threadsafe = _spy  # type: ignore[method-assign]
    return labels


@pytest.mark.asyncio
class TestResumeLevelGateRemoved:
    """``_on_escalation_resolved`` routes the ``resume`` effect on LIVENESS,
    not on ``escalation.level`` (task 3540 / PRD D8, spec E9).

    The old ``level >= 1`` wrapper rested on "every L0 has a live workflow
    waiting on ``event.set()``". That is false for exactly the workflows this
    task exists to rescue: one that died between filing its escalation and
    exiting has already had its ``_escalation_events`` entry popped, so the
    synchronous wake sets nothing and the gate then dropped the re-pend in
    silence. The local ``_escalation_events`` check below is the process-local
    half of the liveness test and is unchanged; the store-side claimant oracle
    inside ``_cascade_unblock_member`` is the authoritative half, and is the
    one that can see a claimant held by ANOTHER orchestrator.
    """

    async def test_level0_orphan_reaches_the_repend(self, harness: Harness):
        """[THE red assertion] An L0 with no live workflow re-pends.

        Today the ``level >= 1`` gate returns before either scheduling site,
        so a stranded in-progress row with a closed escalation has nothing
        left that will ever advance it.
        """
        esc = _esc(task_id='3438', level=0, resolved_by='escalation-watcher-auto')
        _wire(harness, _row('in-progress', claimant=None, heartbeat=None))
        harness._escalation_events.pop('3438', None)

        labels = _record_scheduling(harness)
        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            '3438', _resume_target(esc)
        )
        assert any('orphan-unblock task 3438' in label for label in labels), (
            f'Expected the orphan-unblock coro to be scheduled; got {labels}'
        )

    async def test_level0_with_a_live_workflow_is_woken_not_flipped(
        self, harness: Harness
    ):
        """The half of the old L0 rule that SURVIVES, and why it survives.

        A registered ``_escalation_events`` entry IS the process-local live
        signal. The workflow is woken synchronously and owns its own re-pend,
        so nothing is scheduled and nothing is written. The event assertion is
        the regression half: removing the level gate must not disturb the wake
        path that made the gate defensible in the first place.
        """
        esc = _esc(task_id='3438', level=0, resolved_by='escalation-watcher-auto')
        _wire(harness, _row('in-progress', claimant=None, heartbeat=None))
        event = asyncio.Event()
        harness._escalation_events['3438'] = event

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        assert event.is_set(), 'The synchronous wake path must still fire'
        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]

    async def test_level0_with_a_cross_host_claimant_is_scheduled_then_skipped(
        self, harness: Harness, caplog
    ):
        """The two liveness halves are BOTH needed, and they are not redundant.

        This row has no ``_escalation_events`` entry — process-locally it looks
        orphaned, so the routing above schedules it — but its claimant is
        heartbeating, which is what a workflow running under ANOTHER
        orchestrator looks like from here. Only the store-side oracle inside
        ``_cascade_unblock_member`` can see that, and it withholds the write.
        """
        esc = _esc(task_id='3438', level=0, resolved_by='escalation-watcher-auto')
        _wire(harness, _row('in-progress', claimant=LIVE_CLAIMANT, heartbeat='fresh'))
        harness._escalation_events.pop('3438', None)

        labels = _record_scheduling(harness)
        with caplog.at_level(logging.DEBUG):
            harness._on_escalation_resolved(esc)
            await asyncio.gather(*list(harness._background_tasks))

        assert any('orphan-unblock task 3438' in label for label in labels), (
            f'The routing must still schedule it — the local half sees no '
            f'live workflow; got {labels}'
        )
        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]
        assert any(
            r.levelno == logging.DEBUG and 'live claimant' in r.getMessage()
            for r in caplog.records
        ), 'Expected the store-side oracle to name the live claimant'

    @pytest.mark.parametrize(
        ('resolved_by', 'expected_label'),
        [
            ('l2-cascade:esc-4000-39', 'cascade-unblock task 3438'),
            ('steward', 'orphan-unblock task 3438'),
        ],
    )
    async def test_level1_routing_is_unchanged(
        self, harness: Harness, resolved_by: str, expected_label: str
    ):
        """[regression] Removing the wrapper must not disturb the discrimination
        INSIDE it: an l2-cascade member still routes to the cascade label and a
        non-cascade orphan to the orphan label, at L1 exactly as today."""
        esc = _esc(task_id='3438', level=1, resolved_by=resolved_by)
        _wire(harness, _row('blocked', claimant=None, heartbeat=None))
        harness._escalation_events.pop('3438', None)

        labels = _record_scheduling(harness)
        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        assert any(expected_label in label for label in labels), (
            f'Expected {expected_label!r} to be scheduled; got {labels}'
        )
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            '3438', _resume_target(esc)
        )

    async def test_dismissed_level0_is_close_only(self, harness: Harness):
        """[regression] The gate removal must widen only the ``resume`` effect.

        A DISMISSED escalation maps to ``close_only`` -> ``WORKFLOW_NONE``,
        which returns before the resume branch entirely. Without this, "L0 now
        reaches the re-pend" could quietly mean "an abandoned L0 re-pends too",
        which is the opposite of what dismissal means.
        """
        esc = _esc(task_id='3438', level=0, status='dismissed', resolved_by='steward')
        _wire(harness, _row('in-progress', claimant=None, heartbeat=None))
        harness._escalation_events.pop('3438', None)

        labels = _record_scheduling(harness)
        harness._on_escalation_resolved(esc)
        assert labels == [], f'close_only must schedule nothing; got {labels}'
        await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]

    async def test_scheduler_pause_sentinel_returns_before_dispatch(
        self, harness: Harness
    ):
        """[regression] The synthetic pause-sentinel id is not a real task and
        still returns before action dispatch — it has its own auto-resume
        handling above, and a re-pend of ``__scheduler__`` would be nonsense."""
        sentinel = Harness._SCHEDULER_PAUSE_SENTINEL
        esc = _esc(task_id=sentinel, level=0, resolved_by='steward')
        _wire(harness, _row('blocked', task_id=sentinel, claimant=None, heartbeat=None))
        harness.scheduler.is_paused = False

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# step-8 — the granted_files fold on the re-pend path
# ---------------------------------------------------------------------------

def _plan_root(harness: Harness, task_id: str, *, legacy: bool = False) -> Path:
    """The directory the fold must resolve for *task_id*'s plan.json.

    Mirrors the harness's own new-then-old artifact resolution
    (``Harness._resolve_recovery_artifact``): the W11 ``.task-meta`` SIBLING of
    the worktree first, the legacy ``<worktree>/.task`` second.  Derived
    through ``TaskArtifacts.meta_root_for`` / ``Harness._resolve_task_worktree``
    rather than by joining the path shape by hand, so a relocation moves the
    fixture with the production resolver instead of silently pointing the test
    at a directory nothing writes.
    """
    wt = harness._resolve_task_worktree(task_id)
    if legacy:
        return wt / '.task'
    return TaskArtifacts.meta_root_for(harness.git_ops.worktree_base, wt.name)


def _seed_plan(
    harness: Harness,
    task_id: str = '3438',
    *,
    files: list[str],
    legacy: bool = False,
    session_id: str | None = 'sess-planner',
    revalidated_by: str | None = None,
) -> Path:
    """Write a REAL plan.json on disk and return its path.

    A real file rather than a mocked ``TaskArtifacts``: plan.json is the
    durable half of the grant (on redispatch ``workflow.py::TaskWorkflow.
    _apply_revalidation_skip`` re-derives the module set from ``plan['files']``
    and ``_reconcile_scope_locks`` persists ``metadata.files = plan_files``,
    which would silently NARROW a metadata-only grant back away), so the
    assertion that matters is what is on disk afterwards.

    ``_session_id`` / ``_revalidated_by_session`` are the plan's OWN provenance
    — the fold must pass one of them back to ``set_plan_files`` so its
    ``already_owner`` branch is taken and nothing is re-stamped.
    """
    root = _plan_root(harness, task_id, legacy=legacy)
    root.mkdir(parents=True, exist_ok=True)
    plan: dict = {
        'task_id': task_id,
        'title': 'a task with a scope',
        'analysis': 'x',
        'files': list(files),
        'steps': [{'id': 'step-1', 'description': 'x', 'status': 'pending'}],
        '_created_at': '2020-01-01T00:00:00+00:00',
    }
    if session_id is not None:
        plan['_session_id'] = session_id
    if revalidated_by is not None:
        plan['_revalidated_by_session'] = revalidated_by
    path = root / 'plan.json'
    path.write_text(json.dumps(plan, indent=2) + '\n')
    return path


def _wire_queue(
    harness: Harness, tmp_path: Path, *escalations: Escalation
) -> EscalationQueue:
    """Wire a REAL ``EscalationQueue`` holding *escalations* onto the harness.

    Real rather than mocked because the fold's union is defined over
    ``get_by_task``'s ACTUAL filtering (root + archive scan, per-record
    ``status``), exactly as ``workflow.py::TaskWorkflow._collect_granted_files``
    reads it — a stubbed list would not exercise the ``status != 'resolved'``
    skip that mirror has to reproduce.
    """
    queue = EscalationQueue(tmp_path / 'esc-fold')
    for esc in escalations:
        queue.submit(esc)
    harness._escalation_queue = queue
    return queue


def _files_updates(harness: Harness) -> list:
    """The ``update_task`` calls carrying a ``files`` payload — i.e. the fold's
    ``metadata.files`` write, as distinct from the re-block guard's
    ``reblock_guard`` charge, which shares the same mock."""
    out = []
    for c in harness.scheduler.update_task.await_args_list:  # type: ignore[attr-defined]
        updates = c.args[1] if len(c.args) > 1 else c.kwargs.get('updates')
        if isinstance(updates, dict) and 'files' in updates:
            out.append(c)
    return out


@pytest.mark.asyncio
class TestGrantedFilesFoldOnRepend:
    """A scope grant resolved against a task with NO live workflow is actually
    DELIVERED before the re-pend (task 3540 / PRD D8, spec E9 — boundary #11).

    ``granted_files`` is written to the escalation RECORD by
    ``escalation/queue.py::EscalationQueue.resolve``, and its only production
    reader was ``workflow.py::TaskWorkflow._collect_granted_files`` — reached
    from exactly one site, the LIVE in-workflow L0 resume loop.  A steward who
    resolved a ``scope_violation`` with ``granted_files=[...]`` against a
    blocked/stranded task therefore had the grant recorded and never applied:
    the task re-pended against its ORIGINAL scope and the agent re-escalated
    for the same files.  This is the re-pend-path twin of that reader.
    """

    async def test_grant_is_folded_into_plan_and_metadata(
        self, harness: Harness, tmp_path: Path
    ):
        """[boundary #11 — THE red assertion] The grant reaches BOTH halves.

        plan.json is the durable half (survives redispatch's re-derivation);
        ``metadata.files`` is the half the next dispatch derives its module
        LOCKS from.  Writing only one of them leaves the two diverged, which
        is also what the MERGE-entry ``_check_scope_invariant`` tripwire fires
        on — so both are asserted here, not just the one that is easier to
        observe.
        """
        esc = _esc(
            task_id='3438', category='scope_violation',
            granted_files=['pkg/b.py'], resolved_by='steward',
        )
        _wire_queue(harness, tmp_path, esc)
        plan_path = _seed_plan(harness, files=['pkg/a.py'])
        _wire(harness, _row('in-progress', claimant=None, heartbeat=None))
        harness._escalation_events.pop('3438', None)

        await _drive(harness, esc)

        assert json.loads(plan_path.read_text())['files'] == [
            'pkg/a.py', 'pkg/b.py',
        ], 'plan.json must be widened by the grant, order-preserving'
        writes = _files_updates(harness)
        assert len(writes) == 1, f'Expected exactly one files write; got {writes}'
        assert writes[0].args[0] == '3438'
        assert writes[0].args[1] == {'files': ['pkg/a.py', 'pkg/b.py']}
        assert writes[0].kwargs.get('metadata_mode') == 'merge', (
            "metadata_mode='merge' is required — 'additive' resolves scalar "
            'conflicts OLD-wins and would not replace the files list'
        )
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            '3438', _resume_target(esc)
        )

    async def test_legacy_task_dir_plan_is_folded(
        self, harness: Harness, tmp_path: Path
    ):
        """A task whose plan still lives at the legacy ``<worktree>/.task``
        must be widened too — the resolution is new-then-old, not new-only.

        Without the fallback arm, every task predating the W11 ``.task-meta``
        relocation would silently take the "no plan, nothing to widen" exit.
        """
        esc = _esc(
            task_id='3438', category='scope_violation',
            granted_files=['pkg/b.py'], resolved_by='steward',
        )
        _wire_queue(harness, tmp_path, esc)
        plan_path = _seed_plan(harness, files=['pkg/a.py'], legacy=True)
        _wire(harness, _row('blocked', claimant=None, heartbeat=None))
        harness._escalation_events.pop('3438', None)

        await _drive(harness, esc)

        assert json.loads(plan_path.read_text())['files'] == [
            'pkg/a.py', 'pkg/b.py',
        ]
        harness.scheduler.set_task_status.assert_awaited_once()  # type: ignore[attr-defined]

    async def test_the_fold_lands_before_the_status_write(
        self, harness: Harness, tmp_path: Path
    ):
        """Ordering, not just outcome: the widened scope must already be
        visible when the row goes ``pending``.

        The status write is what makes the task dispatchable again, and
        dispatch derives its locks from ``metadata.files``.  A fold that
        landed AFTER it would race the scheduler: the redispatched agent could
        observe the un-widened scope and re-escalate for exactly the files the
        steward just granted.  Both halves are sampled from inside
        ``set_task_status`` itself, so this cannot pass on end-state alone.
        """
        esc = _esc(
            task_id='3438', category='scope_violation',
            granted_files=['pkg/b.py'], resolved_by='steward',
        )
        _wire_queue(harness, tmp_path, esc)
        plan_path = _seed_plan(harness, files=['pkg/a.py'])
        _wire(harness, _row('in-progress', claimant=None, heartbeat=None))
        harness._escalation_events.pop('3438', None)

        observed: dict = {}

        async def _sample(*_args, **_kwargs):
            observed['plan_files'] = json.loads(plan_path.read_text()).get('files')
            observed['metadata_written'] = bool(_files_updates(harness))

        harness.scheduler.set_task_status = AsyncMock(side_effect=_sample)

        await _drive(harness, esc)

        assert observed.get('plan_files') == ['pkg/a.py', 'pkg/b.py'], (
            'plan.json must already be widened when the row is re-pended; '
            f'observed {observed.get("plan_files")!r}'
        )
        assert observed.get('metadata_written') is True, (
            'metadata.files must already be widened when the row is re-pended'
        )

    async def test_union_spans_the_whole_resolved_history_and_skips_unresolved(
        self, harness: Harness, tmp_path: Path
    ):
        """The union is over EVERY resolved record for the task, and only
        those — a verbatim structural mirror of ``_collect_granted_files``.

        Two reasons it cannot be "just the resolving record": a grant from an
        earlier resolution in this task's history would be dropped on the next
        re-pend (silently narrowing a scope the steward already widened), and
        a still-PENDING record's proposed files are not a grant at all — the
        steward has not agreed to them yet.

        Order across records is deliberately NOT asserted: ``get_by_task``
        scans via ``glob``, whose order is filesystem-dependent.  What IS
        asserted is the part the contract fixes — the pre-existing entries keep
        their order at the front, the grants are appended, and nothing is
        duplicated.
        """
        older = _esc(
            task_id='3438', esc_id='esc-3438-1', category='scope_violation',
            granted_files=['pkg/c.py'], resolved_by='steward',
        )
        pending = _esc(
            task_id='3438', esc_id='esc-3438-2', category='scope_violation',
            status='pending', granted_files=['pkg/never.py'], resolved_by='steward',
        )
        resolving = _esc(
            task_id='3438', esc_id='esc-3438-3', category='scope_violation',
            # 'pkg/c.py' repeats the older grant — de-duped, not appended twice.
            granted_files=['pkg/b.py', 'pkg/c.py'], resolved_by='steward',
        )
        _wire_queue(harness, tmp_path, older, pending, resolving)
        plan_path = _seed_plan(harness, files=['pkg/a.py'])
        _wire(harness, _row('in-progress', claimant=None, heartbeat=None))
        harness._escalation_events.pop('3438', None)

        await _drive(harness, resolving)

        files = json.loads(plan_path.read_text())['files']
        assert files[0] == 'pkg/a.py', 'existing entries keep their order, in front'
        assert sorted(files) == ['pkg/a.py', 'pkg/b.py', 'pkg/c.py'], (
            f'Expected the union of every RESOLVED grant; got {files}'
        )
        assert 'pkg/never.py' not in files, (
            'a still-pending record is a request, not a grant — it must not widen'
        )
        assert len(files) == len(set(files)), f'duplicates in {files}'

    @pytest.mark.parametrize(
        ('session_id', 'revalidated_by'),
        [
            ('sess-planner', None),
            (None, 'sess-revalidator'),
        ],
    )
    async def test_plan_provenance_is_not_restamped(
        self, harness: Harness, tmp_path: Path,
        session_id: str | None, revalidated_by: str | None,
    ):
        """The fold must take ``set_plan_files``' ``already_owner`` branch.

        ``set_plan_files`` STAMPS ``_session_id`` when the caller does not
        already own the plan.  Passing a harness-invented id would therefore
        rewrite the plan's provenance to a session that never planned
        anything, destroying the audit trail and moving the owner out from
        under ``validate_plan_owner`` — which is what
        ``_escalate_plan_overwrite`` exists to catch.  Passing the plan's OWN
        owner id back makes the write provenance-neutral.
        """
        esc = _esc(
            task_id='3438', category='scope_violation',
            granted_files=['pkg/b.py'], resolved_by='steward',
        )
        _wire_queue(harness, tmp_path, esc)
        plan_path = _seed_plan(
            harness, files=['pkg/a.py'],
            session_id=session_id, revalidated_by=revalidated_by,
        )
        before = json.loads(plan_path.read_text())
        _wire(harness, _row('in-progress', claimant=None, heartbeat=None))
        harness._escalation_events.pop('3438', None)

        await _drive(harness, esc)

        after = json.loads(plan_path.read_text())
        assert after['files'] == ['pkg/a.py', 'pkg/b.py'], 'the widen must land'
        assert after.get('_session_id') == before.get('_session_id')
        assert after.get('_revalidated_by_session') == before.get(
            '_revalidated_by_session'
        )
        assert after.get('_created_at') == before.get('_created_at')

    @pytest.mark.parametrize(
        ('granted', 'why'),
        [
            ([], 'a resolution with no grant'),
            (['pkg/a.py'], 'a grant already covered by plan.files'),
        ],
    )
    async def test_a_no_op_grant_writes_nothing(
        self, harness: Harness, tmp_path: Path, granted: list[str], why: str
    ):
        """No widen → no write at all, and the flip is unaffected.

        The overwhelmingly common case is a resume with no grant whatsoever;
        it must not cost a plan.json rewrite or an ``update_task`` round-trip
        per re-pend, and must not touch a plan this path has no business
        editing.
        """
        esc = _esc(
            task_id='3438', category='scope_violation',
            granted_files=granted, resolved_by='steward',
        )
        _wire_queue(harness, tmp_path, esc)
        plan_path = _seed_plan(harness, files=['pkg/a.py'])
        before = plan_path.read_text()
        _wire(harness, _row('in-progress', claimant=None, heartbeat=None))
        harness._escalation_events.pop('3438', None)

        await _drive(harness, esc)

        assert plan_path.read_text() == before, f'plan.json rewritten for {why}'
        assert _files_updates(harness) == [], f'metadata.files written for {why}'
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            '3438', _resume_target(esc)
        )

    async def test_missing_plan_does_not_block_the_flip(
        self, harness: Harness, tmp_path: Path, caplog
    ):
        """No plan.json in EITHER location → skip the fold, still re-pend.

        A task can legitimately have no plan (it never reached the architect,
        or its worktree was reclaimed).  There is no scope to widen, and the
        re-pend is still the right outcome — the grant is not a precondition
        of the resume.
        """
        esc = _esc(
            task_id='3438', category='scope_violation',
            granted_files=['pkg/b.py'], resolved_by='steward',
        )
        _wire_queue(harness, tmp_path, esc)
        _wire(harness, _row('in-progress', claimant=None, heartbeat=None))
        harness._escalation_events.pop('3438', None)

        with caplog.at_level(logging.DEBUG):
            await _drive(harness, esc)

        assert _files_updates(harness) == [], (
            'metadata.files must not be widened when plan.files could not be — '
            'the two would diverge and trip _check_scope_invariant at merge'
        )
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            '3438', _resume_target(esc)
        )
        assert any(
            'plan' in r.getMessage() and '3438' in r.getMessage()
            for r in caplog.records
        ), 'the skipped fold must leave a trace naming the task and the plan'

    async def test_plan_write_failure_does_not_block_the_flip(
        self, harness: Harness, tmp_path: Path, caplog
    ):
        """A failed fold degrades to today's status quo, never to a hold.

        Warn-and-continue is the deliberate choice here (INV-4's
        escalate-on-failure rule does not apply): the grant is ADDITIVE, so a
        failed fold re-pends the task against its original scope and the agent
        re-escalates — annoying, self-healing, and observable.  Withholding
        the re-pend instead would leave the task parked with its escalation
        already closed and nothing left to advance it, which is a permanent
        silent hold.
        """
        esc = _esc(
            task_id='3438', category='scope_violation',
            granted_files=['pkg/b.py'], resolved_by='steward',
        )
        _wire_queue(harness, tmp_path, esc)
        _seed_plan(harness, files=['pkg/a.py'])
        _wire(harness, _row('in-progress', claimant=None, heartbeat=None))
        harness._escalation_events.pop('3438', None)

        with (
            patch.object(
                TaskArtifacts, 'set_plan_files', side_effect=OSError('disk full')
            ),
            caplog.at_level(logging.DEBUG),
        ):
            await _drive(harness, esc)

        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            '3438', _resume_target(esc)
        )
        assert _files_updates(harness) == [], (
            'metadata.files must not be widened once the plan write failed — '
            'a metadata-only widen is silently narrowed back on the next '
            'redispatch and diverges from plan.files in the meantime'
        )
        assert any(
            r.levelno == logging.WARNING and '3438' in r.getMessage()
            for r in caplog.records
        ), 'a dropped grant must be operator-visible — WARNING, not DEBUG'

    async def test_metadata_write_failure_does_not_block_the_flip(
        self, harness: Harness, tmp_path: Path, caplog
    ):
        """The other half of the same rule, on the other write.

        plan.json is already widened at this point, and that is the DURABLE
        half: the next dispatch's ``_apply_revalidation_skip`` re-derives the
        module set from ``plan['files']`` and ``_reconcile_scope_locks``
        persists it back to ``metadata.files``, so the grant still lands.
        """
        esc = _esc(
            task_id='3438', category='scope_violation',
            granted_files=['pkg/b.py'], resolved_by='steward',
        )
        _wire_queue(harness, tmp_path, esc)
        plan_path = _seed_plan(harness, files=['pkg/a.py'])
        _wire(harness, _row('in-progress', claimant=None, heartbeat=None))
        harness._escalation_events.pop('3438', None)

        async def _update(_tid, updates=None, **_kwargs):
            if isinstance(updates, dict) and 'files' in updates:
                raise RuntimeError('fused-memory unreachable')
            return True

        harness.scheduler.update_task = AsyncMock(side_effect=_update)

        with caplog.at_level(logging.DEBUG):
            await _drive(harness, esc)

        assert json.loads(plan_path.read_text())['files'] == [
            'pkg/a.py', 'pkg/b.py',
        ], 'the durable half must survive the failed metadata write'
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            '3438', _resume_target(esc)
        )
        assert any(
            r.levelno == logging.WARNING and '3438' in r.getMessage()
            for r in caplog.records
        )

    async def test_a_withheld_flip_performs_no_fold(
        self, harness: Harness, tmp_path: Path
    ):
        """Nothing was re-pended, so nothing may be widened.

        The re-block guard withholding the flip means this task is NOT going
        back to dispatch on this resolution.  Widening its scope anyway would
        persist a grant against a task parked for human attention, and the
        widened ``metadata.files`` would change the locks it competes for the
        moment an operator does resume it.
        """
        esc = _esc(
            task_id='3438', category='scope_violation',
            granted_files=['pkg/b.py'], resolved_by='steward',
        )
        _wire_queue(harness, tmp_path, esc)
        plan_path = _seed_plan(harness, files=['pkg/a.py'])
        before = plan_path.read_text()
        # Drive the guard to its threshold end-to-end (same signature, count
        # already at the limit) rather than stubbing _check_reblock_guard —
        # the ordering contract is "fold AFTER the guard returns True".
        _wire(harness, _row(
            'in-progress', claimant=None, heartbeat=None,
            metadata={'reblock_guard': {
                'count': _REBLOCK_GUARD_THRESHOLD,
                'signature': Harness._reblock_signature(esc),
            }},
        ))
        harness._escalation_events.pop('3438', None)

        await _drive(harness, esc)

        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]
        assert plan_path.read_text() == before, 'a withheld flip must not widen'
        assert _files_updates(harness) == []

    async def test_no_escalation_queue_is_a_no_op(self, harness: Harness):
        """Bare-harness path (``_escalation_queue is None``): no fold, no crash.

        Mirrors ``_collect_granted_files``' own eval-mode guard — with no
        queue wired there is nothing to read, and the re-pend proceeds
        unchanged.
        """
        esc = _esc(
            task_id='3438', category='scope_violation',
            granted_files=['pkg/b.py'], resolved_by='steward',
        )
        assert harness._escalation_queue is None
        plan_path = _seed_plan(harness, files=['pkg/a.py'])
        before = plan_path.read_text()
        _wire(harness, _row('in-progress', claimant=None, heartbeat=None))
        harness._escalation_events.pop('3438', None)

        await _drive(harness, esc)

        assert plan_path.read_text() == before
        assert _files_updates(harness) == []
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            '3438', _resume_target(esc)
        )
