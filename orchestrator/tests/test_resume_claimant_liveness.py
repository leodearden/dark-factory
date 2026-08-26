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
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from escalation.models import Escalation

from orchestrator.harness import Harness

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
