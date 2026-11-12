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

import logging
import re
from collections.abc import Callable
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
# task 7857: the reclaim pass record, end to end through the production entry
# ---------------------------------------------------------------------------


_SUMMARY_LOGGER = 'orchestrator.warm_lane_gc_summary'
_GIT_OPS_LOGGER = 'orchestrator.git_ops'
_PASS_RECORD = '_run_warm_lane_gc_reclaim: reclaim pass'
_REIFY_SUMMARY = 'reclaim: reset=2 removed=0 preserved=6 preserved_live_ref=1 reset_cold=1'
_DF_SUMMARY = (
    'reclaim: reset=3 removed=1 preserved=5 preserved_live_ref=2 '
    'preserved_assigned=4 downgraded_assigned=0 reset_cold=1 skipped_unchanged=7'
)


def _write_gc_stub(path: Path, body: str, mode: int = 0o755) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/usr/bin/env bash\n' + body)
    path.chmod(mode)
    return path


def _kv(message: str) -> dict[str, str]:
    return dict(re.findall(r'(\w+)=(\S+)', message))


def _messages(caplog: pytest.LogCaptureFixture, logger_name: str, level: int) -> list[str]:
    return [
        r.getMessage() for r in caplog.records
        if r.name == logger_name and r.levelno == level
    ]


class TestWarmLaneGcPassRecord:
    """A real GitOps runs a stub gc.sh; the pass record is read back from the log.

    The project copy wins resolution when present; the autouse conftest pin
    keeps dark-factory's copy absent unless ``df_warm_lane_script_dir`` opts in.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('origin', 'summary'),
        [('project', _REIFY_SUMMARY), ('dark-factory', _DF_SUMMARY)],
        ids=['project-reify-shaped', 'dark-factory-df-shaped'],
    )
    async def test_rc0_logs_one_pass_record_with_origin_and_counts(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        df_warm_lane_script_dir: Callable[..., Path],
        origin: str,
        summary: str,
    ) -> None:
        stub_dir = tmp_path / 'scripts' if origin == 'project' else df_warm_lane_script_dir()
        stub = _write_gc_stub(stub_dir / 'warm-lane-gc.sh', f"printf '%s\\n' '{summary}'\n")
        harness, _rs = _make_harness(tmp_path)
        caplog.set_level(logging.DEBUG)

        await harness._run_warm_lane_gc_pass()

        records = [
            m for m in _messages(caplog, _SUMMARY_LOGGER, logging.INFO)
            if m.startswith(_PASS_RECORD)
        ]
        assert len(records) == 1, caplog.text
        fields = _kv(records[0])
        expected = _kv(summary)
        assert fields['origin'] == origin
        assert fields['script'] == str(stub)
        assert float(fields['duration_s']) >= 0
        for name in ('reset', 'removed', 'preserved'):
            assert fields[name] == expected[name]

    @pytest.mark.asyncio
    async def test_rc0_without_a_summary_warns_uncounted(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        stub = _write_gc_stub(tmp_path / 'scripts' / 'warm-lane-gc.sh', 'exit 0\n')
        harness, _rs = _make_harness(tmp_path)
        caplog.set_level(logging.DEBUG)

        await harness._run_warm_lane_gc_pass()

        warnings = _messages(caplog, _SUMMARY_LOGGER, logging.WARNING)
        assert len(warnings) == 1, caplog.text
        assert str(stub) in warnings[0]
        assert 'uncounted' in warnings[0]
        assert not [
            m for m in _messages(caplog, _SUMMARY_LOGGER, logging.INFO)
            if m.startswith(_PASS_RECORD)
        ]

    @pytest.mark.asyncio
    async def test_a_spawn_that_raises_is_warned_as_an_unexpected_error(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A present but non-executable project copy reaches the spawn, which raises.

        Resolution keys on existence, not the execute bit, so the 0o644 stub is
        chosen and ``PermissionError`` from the spawn lands in git_ops' exception
        arm (rc 127), which had no test before task 7857.
        """
        _write_gc_stub(
            tmp_path / 'scripts' / 'warm-lane-gc.sh',
            f"printf '%s\\n' '{_REIFY_SUMMARY}'\n",
            mode=0o644,
        )
        harness, _rs = _make_harness(tmp_path)
        caplog.set_level(logging.DEBUG)

        await harness._run_warm_lane_gc_pass()

        assert any(
            m.startswith('_run_warm_lane_gc_reclaim: unexpected error')
            for m in _messages(caplog, _GIT_OPS_LOGGER, logging.WARNING)
        ), caplog.text
        assert not _messages(caplog, _SUMMARY_LOGGER, logging.INFO)


_HARNESS_LOGGER = 'orchestrator.harness'
_RECLAIM_LINE = '_run_warm_lane_gc_reclaim'


def _gc_leg_only_harness(tmp_path: Path) -> Harness:
    """A harness whose own pass logs come from the GC leg alone."""
    harness, _rs = _make_harness(tmp_path)
    harness._run_interactive_worktree_reaper_pass = AsyncMock()
    harness._reclaim_terminal_lane_records = AsyncMock(return_value=0)
    return harness


async def _tick_lines(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, rc: int,
) -> list[logging.LogRecord]:
    """The harness's own records for one pass whose reclaim returned ``rc``."""
    harness = _gc_leg_only_harness(tmp_path)
    harness.git_ops._run_warm_lane_gc_reclaim = AsyncMock(return_value=rc)
    caplog.set_level(logging.DEBUG)

    await harness._run_warm_lane_gc_pass()

    return [r for r in caplog.records if r.name == _HARNESS_LOGGER]


class TestWarmLaneGcPassTickVisibility:
    """A tick on which no reclaim ran is visible at default verbosity."""

    @pytest.mark.asyncio
    async def test_rc127_tick_is_logged_at_info_or_above(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        records = await _tick_lines(tmp_path, caplog, rc=127)

        assert [
            r for r in records if r.levelno >= logging.INFO and 'rc=127' in r.getMessage()
        ], caplog.text

    @pytest.mark.asyncio
    async def test_rc127_tick_points_at_the_preceding_reclaim_line(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """127 also means a lease deferral, a pool refusal or a spawn failure."""
        records = await _tick_lines(tmp_path, caplog, rc=127)

        tick = [r.getMessage() for r in records if 'rc=127' in r.getMessage()]
        assert tick, caplog.text
        assert all(_RECLAIM_LINE in m for m in tick), tick

    @pytest.mark.asyncio
    async def test_rc0_tick_is_debug_and_points_at_the_preceding_reclaim_line(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The pass's INFO record, or its uncounted WARNING, comes from git_ops."""
        records = await _tick_lines(tmp_path, caplog, rc=0)

        tick = [r for r in records if 'exited 0' in r.getMessage()]
        assert tick, caplog.text
        assert all(r.levelno == logging.DEBUG for r in tick), tick
        assert all(_RECLAIM_LINE in r.getMessage() for r in tick), tick

    @pytest.mark.asyncio
    async def test_other_nonzero_rc_still_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        records = await _tick_lines(tmp_path, caplog, rc=1)

        assert [
            r for r in records
            if r.levelno == logging.WARNING and 'non-zero rc=1' in r.getMessage()
        ], caplog.text


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

    TWO LIMITS ON WHAT THIS GATE CLAIMS, both deliberate:

    * It compares the pydantic field's DEFAULT against the bash default, so it
      pins the STOCK pair only.  A deployment that raises
      ``lane_stale_report_days`` past 14 in its own YAML is outside this gate's
      reach and silently loses the ordering — reading an effective config here
      would mean this test depended on whichever YAML the runner happened to
      find, which is not a drift gate at all.
    * ``_stale_lane_assignment_census`` skips any record whose backing task is
      TERMINAL or unknown, so report-before-act is a claim about NON-TERMINAL
      assignments.  A record stranded by an ENOSPC at release belongs to a task
      that has since finished and is never censused; the gc bound acting on it
      unreported is the intent, not a hole this ordering was meant to cover.
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
