"""Tests for task 1926: Warm-lane auto-GC cadence loop.

Covers:
  step-03 RED  — _run_warm_lane_gc_pass() delegates to
                 git_ops._run_warm_lane_gc_reclaim() and is fail-soft.
  step-05 RED  — Loop lifecycle: _start/_stop_warm_lane_gc kill-switch +
                 dedup + cancel/clear.
  amend-1      — _warm_lane_gc_loop() body: exception path is swallowed and
                 loop survives (failure-resilience contract).
  amend-3      — Startup/shutdown wiring: _warm_lane_gc_task is None before
                 startup, live after _start_warm_lane_gc(), None after
                 _stop_warm_lane_gc() (mirrors run()/shutdown contract).
"""
from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.harness import Harness
from orchestrator.run_store import RunStore

# ---------------------------------------------------------------------------
# Test factories
# ---------------------------------------------------------------------------


def _make_harness(tmp_path: Path) -> tuple[Harness, MagicMock]:
    """Bare Harness with a real config and a spy RunStore.

    Mirrors test_harness_no_landings_breaker._make_harness.
    Returns (harness, mock_run_store).
    """
    config = OrchestratorConfig(project_root=tmp_path)
    harness = Harness(config)
    mock_run_store = MagicMock(spec=RunStore)
    harness._run_store = mock_run_store
    harness._run_id = 'run-warm-lane-gc-0001'
    harness.event_store = EventStore(tmp_path / 'events.db', 'run-warm-lane-gc-0001')
    return harness, mock_run_store


# ---------------------------------------------------------------------------
# step-03: _run_warm_lane_gc_pass delegates to _run_warm_lane_gc_reclaim
# ---------------------------------------------------------------------------


class TestWarmLaneGcPass:
    """Harness._run_warm_lane_gc_pass() delegates to git_ops._run_warm_lane_gc_reclaim().

    RED until step-4 GREEN adds _run_warm_lane_gc_pass to Harness.
    """

    @pytest.mark.asyncio
    async def test_pass_delegates_to_reclaim(self, tmp_path: Path) -> None:
        """_run_warm_lane_gc_pass() awaits git_ops._run_warm_lane_gc_reclaim() once."""
        harness, _rs = _make_harness(tmp_path)
        mock_reclaim = AsyncMock(return_value=0)
        harness.git_ops._run_warm_lane_gc_reclaim = mock_reclaim

        await harness._run_warm_lane_gc_pass()

        mock_reclaim.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_pass_swallows_nonzero_rc(self, tmp_path: Path) -> None:
        """_run_warm_lane_gc_pass() does not raise when reclaim returns non-zero rc.

        rc=127 is the fail-soft sentinel (script absent); other non-zero values
        indicate a script error. Neither should propagate as an exception.
        """
        harness, _rs = _make_harness(tmp_path)
        for rc in (127, 1, 2):
            mock_reclaim = AsyncMock(return_value=rc)
            harness.git_ops._run_warm_lane_gc_reclaim = mock_reclaim
            # Must not raise
            await harness._run_warm_lane_gc_pass()
            mock_reclaim.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_pass_delegates_to_terminal_lane_record_reclaim(
        self, tmp_path: Path
    ) -> None:
        """_run_warm_lane_gc_pass() awaits self._reclaim_terminal_lane_records() once.

        Leaf γ (task 2891): the durable-record terminal-lane reclaim rides the
        existing warm-lane GC cadence tick — no new timer/loop.
        """
        harness, _rs = _make_harness(tmp_path)
        harness.git_ops._run_warm_lane_gc_reclaim = AsyncMock(return_value=0)
        mock_reclaim = AsyncMock(return_value=0)
        harness._reclaim_terminal_lane_records = mock_reclaim

        await harness._run_warm_lane_gc_pass()

        mock_reclaim.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_pass_swallows_terminal_lane_reclaim_raise(
        self, tmp_path: Path
    ) -> None:
        """A raise from _reclaim_terminal_lane_records must NOT break the GC cadence.

        Belt-and-suspenders fail-soft, mirroring the interactive-worktree reaper
        delegate: a fault in the reclaim delegate cannot propagate out of
        _run_warm_lane_gc_pass (never-raise contract).
        """
        harness, _rs = _make_harness(tmp_path)
        harness.git_ops._run_warm_lane_gc_reclaim = AsyncMock(return_value=0)
        harness._reclaim_terminal_lane_records = AsyncMock(
            side_effect=RuntimeError('boom')
        )

        # Must not raise
        await harness._run_warm_lane_gc_pass()

        harness._reclaim_terminal_lane_records.assert_awaited_once()



# ---------------------------------------------------------------------------
# task 5504: the gc record-age bound may not undercut the digest census
# ---------------------------------------------------------------------------


#: The shipped bash artefact whose default this gate reads.  Resolved from
#: ``__file__``, not the process CWD, for the same reason
#: ``test_lane_state_lib.py`` does it: the merge-verify harness and a plain
#: ``pytest orchestrator/tests`` run from different directories.
_GC_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / 'scripts' / 'warm-lane' / 'warm-lane-gc.sh'
)

#: ``[ -n "$MAX_RECORD_AGE_DAYS" ] || MAX_RECORD_AGE_DAYS=14`` — the ONE line
#: that establishes the effective default.  Narrow on purpose: a looser pattern
#: would also match the header prose that documents the number, and a gate that
#: reads its subject out of a comment is not reading the code.
_GC_DEFAULT_RE = re.compile(
    r'^\[ -n "\$MAX_RECORD_AGE_DAYS" \] \|\| MAX_RECORD_AGE_DAYS=(\d+)\s*$',
    re.MULTILINE,
)


def _gc_default_max_record_age_days() -> int:
    """The default ``--max-record-age-days`` the shipped script actually applies.

    Raises rather than returning a sentinel when the line cannot be found: a
    silently-unparsed default would make every assertion below vacuous, which is
    the failure mode a drift gate most needs to avoid — it would advertise a
    guarantee it had stopped providing.
    """
    text = _GC_SCRIPT.read_text()
    match = _GC_DEFAULT_RE.search(text)
    if match is None:
        raise AssertionError(
            f'could not find the MAX_RECORD_AGE_DAYS default in {_GC_SCRIPT}. '
            f'The apply-defaults line was renamed or reshaped; update '
            f'_GC_DEFAULT_RE to match, or this gate silently stops gating.'
        )
    return int(match.group(1))


def _census_reports_before_gc_acts(census_days: float, gc_days: int) -> bool:
    """The ordering the two numbers must satisfy.

    Factored out so the hostile-pair case below can drive the SAME predicate the
    real assertion uses — a prove-it-can-fail check against a reimplementation
    would prove nothing about the gate that ships.
    """
    return gc_days >= census_days


class TestGcRecordAgeBoundDoesNotUndercutTheCensus:
    """warm-lane-gc.sh's staleness bound may not act before the digest reports.

    The two numbers are RELATED but not the same knob, and the relationship is
    an ORDERING rather than an equality:

    * ``OrchestratorConfig.lane_stale_report_days`` (7.0) decides when a
      non-terminal ASSIGNED record starts appearing in the digest's
      ``## Stale lane assignments`` section.  That pass is report-only by
      explicit design: it never releases the lane assignment, because the
      WIP-preserving invariant says a live task's lane is its own.
    * ``--max-record-age-days`` (14) decides when warm-lane-gc.sh stops
      believing that same record and lets the lane fall through to the
      live-reference gate.  What it ultimately deletes is only ``target/`` —
      the source tree, ``refs/heads/task/NNNN`` and the record itself all
      survive, and ``acquire_lane`` re-seeds from base on the next acquire.

    So they are not two spellings of one threshold and must not be collapsed
    into one knob.  What MUST hold is report-before-act: an operator gets to see
    a lane in the digest before a sweep acts on it.  Equality would forbid a
    deployment widening its gc margin for no reason; the inequality is the real
    contract.

    The two cannot be kept in sync mechanically — project-agnostic bash on one
    side, a pydantic field on the other — which is exactly the situation
    ``lib_lane_state.sh`` solved for ``PROTECTED_PREFIXES`` with a machine-checked
    drift gate rather than a comment.  Same remedy here.
    """

    def test_the_gc_bound_is_a_non_negative_integer(self) -> None:
        # The script validates its own flag at runtime; this pins that the
        # DEFAULT it falls back to would itself survive that validation.
        assert _gc_default_max_record_age_days() >= 0

    def test_the_gc_bound_does_not_act_before_the_census_reports(self) -> None:
        census_days = OrchestratorConfig.model_fields[
            'lane_stale_report_days'
        ].default
        gc_days = _gc_default_max_record_age_days()
        assert _census_reports_before_gc_acts(census_days, gc_days), (
            f'warm-lane-gc.sh would reclaim a stale lane after {gc_days}d while '
            f'the digest census only starts reporting it at {census_days}d — the '
            f'sweep would act on lanes an operator has never been shown. Either '
            f'lower lane_stale_report_days (orchestrator/src/orchestrator/'
            f'config.py) or raise the MAX_RECORD_AGE_DAYS default '
            f'(orchestrator/scripts/warm-lane/warm-lane-gc.sh).'
        )

    def test_the_drift_guard_actually_fires(self) -> None:
        """The gate must go RED on a hostile pair, not pass vacuously.

        Without this, the assertion above could be green because the comparison
        is broken rather than because the ordering holds — and a drift gate that
        cannot fail is worse than none.
        """
        gc_days = _gc_default_max_record_age_days()
        hostile_census_days = float(gc_days) + 1.0
        assert not _census_reports_before_gc_acts(hostile_census_days, gc_days), (
            'the ordering check accepted a census threshold ABOVE the gc bound — '
            'it is not actually gating anything'
        )
