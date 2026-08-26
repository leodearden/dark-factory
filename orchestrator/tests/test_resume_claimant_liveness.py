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
