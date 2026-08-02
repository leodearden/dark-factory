"""Tests for the SEGMENTED fallback verify — task 3338 / esc-3062-2.

The fallback verify config's ``test_command`` is an `&&` chain across every
subproject in the fleet, and ``_run_or_skip_timed`` hands it to a single
``/bin/bash -c``. The `&&` short-circuit is the SHELL's, so an unrelated
earlier subproject's red means a task's OWN assigned-file tests are never
executed at all — and the orchestrator sees exactly one rc, with no way to
tell "skipped" from "passed". That inversion is what esc-3062-2 reports: the
triaging agent's job becomes proving an unrelated red is unrelated, instead of
reading its own result.

``verify._run_segmented`` is the fix. It runs EVERY segment — no short-circuit
— aggregating to one ``(rc, output, timed_out, segments)``, under ONE shared
wall-clock deadline so the total wall-clock contract ``run_verification``
already has is preserved exactly.

``run_one`` is injected rather than importing ``_run_cmd`` directly, so this
module exercises the aggregator with a recording fake and spawns no
subprocesses.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

from orchestrator.verify_cmd import ChainSegment, split_and_chain_segments

# Loaded from the COMMITTED config rather than hand-copied a third time.
# ``test_verify_cmd.TestSplitAndChainSegmentsLiveConfigDrift`` already pins
# this same file against its ``_ROOT_TEST_COMMAND`` corpus constant, so
# reading it here keeps the two suites describing one chain instead of two
# copies free to drift apart.
_DF_CONFIG_PATH = pathlib.Path(__file__).resolve().parents[2] / 'dark-factory-orchestrator.yaml'
_FLEET_TEST_COMMAND = yaml.safe_load(_DF_CONFIG_PATH.read_text(encoding='utf-8'))['test_command']

_SEGMENT_KEYS = {
    'index',
    'label',
    'cwd',
    'cmd',
    'status',
    'rc',
    'timed_out',
    'duration_secs',
    'skip_reason',
}


def _fleet_segments() -> list[ChainSegment]:
    """The committed fleet chain's 8 runnable segments."""
    segments = split_and_chain_segments(_FLEET_TEST_COMMAND)
    assert segments is not None, 'the committed fleet chain must be segmentable'
    return segments


def _test_run_entry(runs_capture: list) -> dict:
    """The `test` entry of the runs dict `run_verification` hands downstream.

    Captured off `_persist_attempt_logs`'s argument rather than reconstructed,
    because that list — `[c.to_dict() for c in attempt.checks]` — is exactly
    what `_build_summary_payload` and the review/merge tooling read, so it is
    what the segment facts have to arrive on.

    This seam stops at the ARGUMENT. That the facts also survive the whitelist
    rebuild into the persisted `attempt-N[.<prefix>].summary.json` is pinned
    separately, by reading the written file, in
    `TestSegmentsReachThePersistedSummaryJson`.
    """
    assert runs_capture, 'run_verification recorded no runs dict'
    return next(r for r in runs_capture[-1] if r['label'] == 'test')


class _Clock:
    """A monotonic clock the FAKE ``run_one`` advances, not the test body.

    Advancing on each simulated run (rather than on each ``now()`` read) keeps
    these tests decoupled from how many times the implementation happens to
    read the clock, so a refactor that reads it once more does not have to
    rewrite the assertions.
    """

    def __init__(self, cost_per_segment: float = 0.0) -> None:
        self.t = 0.0
        self.cost_per_segment = cost_per_segment

    def __call__(self) -> float:
        return self.t

    def tick(self) -> None:
        self.t += self.cost_per_segment


class _RecordingRunOne:
    """An async fake for ``run_one`` that records every call and scripts results.

    *results* maps a 0-based segment index to ``(rc, output, timed_out)``;
    anything unlisted is a pass. The recorded call list is what proves segments
    AFTER a red one still ran — the regression esc-3062-2 is about.
    """

    def __init__(self, clock: _Clock, results: dict[int, tuple[int, str, bool]] | None = None):
        self.clock = clock
        self.results = results or {}
        self.calls: list[tuple[str, pathlib.Path, float, str]] = []

    async def __call__(self, cmd: str, cwd, timeout: float, label: str):
        index = len(self.calls)
        self.calls.append((cmd, cwd, timeout, label))
        self.clock.tick()
        return self.results.get(index, (0, f'ok {label}', False))


class TestRunSegmentedRunsEverySegment:
    """The no-short-circuit contract: every segment runs, whatever came before.

    This is the whole product of task 3338. A future "optimisation" that stops
    at the first red is not a speed-up — it is a re-introduction of the exact
    bug, and this class is what catches it.
    """

    @pytest.mark.asyncio
    async def test_all_green_reports_pass_and_runs_each_segment_in_its_own_cwd(self, tmp_path):
        clock = _Clock(cost_per_segment=5.0)
        run_one = _RecordingRunOne(clock)
        segments = _fleet_segments()
        from orchestrator.verify import _run_segmented  # noqa: PLC0415

        rc, output, timed_out, seg_dicts = await _run_segmented(
            segments,
            run_one=run_one,
            worktree=tmp_path,
            budget_secs=10_000.0,
            now=clock,
        )

        assert rc == 0
        assert timed_out is False
        assert len(seg_dicts) == len(segments) == 8
        assert [d['status'] for d in seg_dicts] == ['passed'] * 8
        assert len(run_one.calls) == 8
        # Each segment ran in the cwd the shell's own `cd` folding would have
        # put it in — that is the whole reason `cd` clauses are folded rather
        # than executed.
        assert [call[1] for call in run_one.calls] == [
            tmp_path / segment.cwd_rel for segment in segments
        ]
        assert [call[0] for call in run_one.calls] == [segment.command for segment in segments]
        assert output != ''

    @pytest.mark.asyncio
    async def test_segment_dicts_carry_the_full_per_segment_shape(self, tmp_path):
        """The facts a triaging agent reads, and that land in the summary JSON."""
        clock = _Clock(cost_per_segment=5.0)
        run_one = _RecordingRunOne(clock, results={2: (1, 'boom', False)})
        segments = _fleet_segments()
        from orchestrator.verify import _run_segmented  # noqa: PLC0415

        _rc, _output, _timed_out, seg_dicts = await _run_segmented(
            segments,
            run_one=run_one,
            worktree=tmp_path,
            budget_secs=10_000.0,
            now=clock,
        )

        for position, (entry, segment) in enumerate(
            zip(seg_dicts, segments, strict=True), start=1,
        ):
            assert set(entry) == _SEGMENT_KEYS
            assert entry['index'] == position
            assert entry['label'] == segment.label
            assert entry['cwd'] == segment.cwd_rel
            assert entry['cmd'] == segment.command
            assert entry['duration_secs'] == 5.0
        assert seg_dicts[2]['status'] == 'failed'
        assert seg_dicts[2]['rc'] == 1
        assert seg_dicts[2]['timed_out'] is False
        assert seg_dicts[2]['skip_reason'] is None

    @pytest.mark.asyncio
    async def test_a_red_third_segment_does_not_skip_the_five_after_it(self, tmp_path):
        """THE regression test for esc-3062-2.

        With one `/bin/bash -c '<chain>'`, a red `orchestrator` clause means
        every later subproject — including the final `tests/scripts/` clause a
        task's own assigned files may live in — is never executed, and the
        single rc cannot say so. Here they all run, and the final segment
        carries its OWN recorded result.
        """
        clock = _Clock(cost_per_segment=1.0)
        run_one = _RecordingRunOne(clock, results={2: (1, 'orchestrator tests failed', False)})
        segments = _fleet_segments()
        assert segments[2].cwd_rel == 'orchestrator'
        from orchestrator.verify import _run_segmented  # noqa: PLC0415

        rc, _output, timed_out, seg_dicts = await _run_segmented(
            segments,
            run_one=run_one,
            worktree=tmp_path,
            budget_secs=10_000.0,
            now=clock,
        )

        assert len(run_one.calls) == 8, 'segments after the red one were skipped'
        assert [call[3] for call in run_one.calls] == [s.label for s in segments]
        # The clause esc-3062-2 is about, with its own verdict rather than
        # silence.
        assert 'tests/scripts/' in seg_dicts[-1]['cmd']
        assert seg_dicts[-1]['status'] == 'passed'
        assert seg_dicts[-1]['rc'] == 0
        # ...and the overall verdict is still RED. Running the later segments
        # buys information, never leniency.
        assert rc == 1
        assert timed_out is False
        assert [d['status'] for d in seg_dicts] == (
            ['passed'] * 2 + ['failed'] + ['passed'] * 5
        )

    @pytest.mark.asyncio
    async def test_multiple_reds_aggregate_to_the_first_nonzero_rc(self, tmp_path):
        """rc is the FIRST non-zero, matching what the `&&` chain would have returned.

        The shell reports the rc of the clause that stopped it; preserving that
        keeps every downstream rc consumer reading the same number as before,
        while the per-segment dicts carry the reds it used to hide.
        """
        clock = _Clock(cost_per_segment=1.0)
        run_one = _RecordingRunOne(
            clock,
            results={1: (3, 'escalation red', False), 4: (2, 'dashboard red', False)},
        )
        segments = _fleet_segments()
        from orchestrator.verify import _run_segmented  # noqa: PLC0415

        rc, _output, _timed_out, seg_dicts = await _run_segmented(
            segments,
            run_one=run_one,
            worktree=tmp_path,
            budget_secs=10_000.0,
            now=clock,
        )

        assert rc == 3
        assert len(run_one.calls) == 8
        assert seg_dicts[1]['status'] == 'failed'
        assert seg_dicts[1]['rc'] == 3
        assert seg_dicts[4]['status'] == 'failed'
        assert seg_dicts[4]['rc'] == 2
        assert [d['index'] for d in seg_dicts if d['status'] == 'failed'] == [2, 5]

    @pytest.mark.asyncio
    async def test_a_timed_out_segment_is_recorded_and_does_not_stop_the_chain(self, tmp_path):
        """A segment that blew its own timeout is `timed_out`, not `not_run`.

        It RAN and produced no verdict; the distinction matters because
        `not_run` is reserved for segments the shared deadline never reached
        (step-9), which must never be conflated with one that executed.
        """
        clock = _Clock(cost_per_segment=1.0)
        run_one = _RecordingRunOne(clock, results={0: (124, 'timed out', True)})
        segments = _fleet_segments()
        from orchestrator.verify import _run_segmented  # noqa: PLC0415

        rc, _output, timed_out, seg_dicts = await _run_segmented(
            segments,
            run_one=run_one,
            worktree=tmp_path,
            budget_secs=10_000.0,
            now=clock,
        )

        assert len(run_one.calls) == 8
        assert seg_dicts[0]['status'] == 'timed_out'
        assert seg_dicts[0]['timed_out'] is True
        assert seg_dicts[0]['rc'] == 124
        assert seg_dicts[0]['skip_reason'] is None
        assert timed_out is True
        assert rc == 124


class TestRunSegmentedSharedDeadline:
    """ONE wall-clock deadline across all segments, and the `not_run` encoding.

    ``run_verification`` resolves a SINGLE timeout for the test leg today
    (``_resolve_verify_timeout``), and that total wall-clock contract is
    preserved exactly: each segment gets ``deadline - now()``, not a fresh full
    budget. A per-segment full timeout would silently multiply the worst case
    by the segment count and could wedge the merge queue.

    This path is load-bearing, not theoretical. The committed config's own
    measured table records five of seven segments already costing 1838.60s
    (orchestrator alone 1366.23s), a hard lower bound because that run timed
    out before dashboard started — so a real chain can exhaust even the raised
    3600s ceiling mid-run.

    A segment the deadline never reached is `not_run` with ``rc=None``: the
    UNCONFLATABLE encoding. `rc=0` would read as a pass, which is precisely the
    silent-fail-soft this whole task exists to remove.
    """

    @staticmethod
    async def _exhaust_after_two(tmp_path):
        """8 segments, a 10s budget, and 5s per segment — 2 run, 6 do not."""
        clock = _Clock(cost_per_segment=5.0)
        run_one = _RecordingRunOne(clock)
        segments = _fleet_segments()
        from orchestrator.verify import _run_segmented  # noqa: PLC0415

        rc, output, timed_out, seg_dicts = await _run_segmented(
            segments,
            run_one=run_one,
            worktree=tmp_path,
            budget_secs=10.0,
            now=clock,
        )
        return run_one, rc, output, timed_out, seg_dicts

    @pytest.mark.asyncio
    async def test_no_subprocess_is_spawned_once_the_budget_is_gone(self, tmp_path):
        """The deadline is a real stop, not a label applied after the fact."""
        run_one, _rc, _output, _timed_out, seg_dicts = await self._exhaust_after_two(tmp_path)

        assert len(run_one.calls) == 2
        assert [d['status'] for d in seg_dicts] == ['passed'] * 2 + ['not_run'] * 6

    @pytest.mark.asyncio
    async def test_not_run_segments_carry_rc_none_and_a_skip_reason(self, tmp_path):
        """`rc is None`, never 0 — the anti-conflation assertion.

        A reader (human or agent) must be structurally unable to mistake a
        segment that never ran for one that passed.
        """
        _run_one, _rc, _output, _timed_out, seg_dicts = await self._exhaust_after_two(tmp_path)

        for entry in seg_dicts[2:]:
            assert entry['status'] == 'not_run'
            assert entry['rc'] is None
            assert entry['rc'] != 0  # explicit: `not_run` is not a pass
            assert entry['skip_reason']
            assert entry['duration_secs'] == 0.0

    @pytest.mark.asyncio
    async def test_a_green_so_far_run_with_unrun_segments_is_never_green(self, tmp_path):
        """Every segment that ACTUALLY ran passed — and the verdict is still red.

        Reporting 0 here would claim a fleet-wide pass on the strength of two
        subprojects. ``timed_out=True`` additionally keeps the failure
        classified as `infra_timeout` (retryable), exactly as a single-chain
        timeout is today, so run_verification's retry/env-recovery machinery
        needs no change.
        """
        _run_one, rc, _output, timed_out, seg_dicts = await self._exhaust_after_two(tmp_path)

        assert all(d['status'] == 'passed' for d in seg_dicts[:2])
        assert rc != 0
        assert timed_out is True

    @pytest.mark.asyncio
    async def test_each_executed_segment_gets_the_remaining_budget(self, tmp_path):
        """`deadline - now()`, not a fresh `budget_secs` per segment."""
        run_one, _rc, _output, _timed_out, _seg_dicts = await self._exhaust_after_two(tmp_path)

        assert [call[2] for call in run_one.calls] == [10.0, 5.0]

    @pytest.mark.asyncio
    async def test_not_run_output_says_unknown_rather_than_pass(self, tmp_path):
        """The output blob is what a triaging agent reads; it must not imply a pass."""
        _run_one, _rc, output, _timed_out, seg_dicts = await self._exhaust_after_two(tmp_path)

        assert 'NOT RUN' in output
        for entry in seg_dicts[2:]:
            assert entry['label'] in output
        lowered = output.lower()
        assert 'unknown' in lowered
        assert entry['skip_reason'] in output

    @pytest.mark.asyncio
    async def test_a_red_segment_before_the_deadline_still_wins_the_rc(self, tmp_path):
        """A real failure outranks the synthetic not-run rc — AND its flag.

        The cause_hint a triaging agent needs is the genuine red, so a
        budget-exhausted tail must not overwrite it.

        The same now holds for the timeout FLAG, and it has to:
        ``verify_classify.classify_failure``'s guard 2 (``if timed_out: return
        FailureCategory.INFRA_TIMEOUT``) wins over EVERY output pattern, so a
        SYNTHESISED ``timed_out`` relabels a genuine red as a timeout no matter
        what the output says. Downstream that also makes
        ``VerifyAttempt.pure_timeout_failure`` true (the test check has rc!=0
        AND timed_out=True, satisfying its ``all(c.rc == 0 or c.timed_out ...)``
        clause), so the run is re-run ``max_retries`` times at a full budget
        each and routed to infra-hold instead of the debugger.

        That would be a REGRESSION against the pre-change baseline: under the
        `&&` chain the shell short-circuited at the red, so the check finished
        fast with rc=1, timed_out=False and a clean `test_failure`. Nothing is
        hidden by dropping the flag — the unrun tail stays visible via the
        segment dicts (`rc=None`, non-empty `skip_reason`), the NOT RUN output
        blocks and roster lines, and the `| segments not run:` cause_hint
        suffix. Only the CLASSIFICATION moves, and only toward the truth.
        """
        clock = _Clock(cost_per_segment=5.0)
        run_one = _RecordingRunOne(clock, results={1: (7, 'escalation red', False)})
        segments = _fleet_segments()
        from orchestrator.verify import _run_segmented  # noqa: PLC0415

        rc, _output, timed_out, seg_dicts = await _run_segmented(
            segments,
            run_one=run_one,
            worktree=tmp_path,
            budget_secs=10.0,
            now=clock,
        )

        assert rc == 7
        assert timed_out is False
        assert [d['status'] for d in seg_dicts] == (
            ['passed', 'failed'] + ['not_run'] * 6
        )

    @pytest.mark.asyncio
    async def test_a_green_so_far_run_whose_tail_is_unrun_still_reports_timed_out(
        self, tmp_path,
    ):
        """The contrasting half of the rule the test above states.

        Making the synthetic flag conditional must NOT weaken the not-run
        contract — it only stops it firing when a REAL red is already there to
        report. With no red anywhere, a truncated chain is exactly the case the
        synthetic flag exists for: rc!=0 AND timed_out=True, so it classifies
        as `infra_timeout` (retryable) precisely as a single-chain timeout does
        today. Read this and the test above as one rule, not two.
        """
        clock = _Clock(cost_per_segment=5.0)
        run_one = _RecordingRunOne(clock)  # every executed segment passes
        segments = _fleet_segments()
        from orchestrator.verify import _run_segmented  # noqa: PLC0415

        rc, _output, timed_out, seg_dicts = await _run_segmented(
            segments,
            run_one=run_one,
            worktree=tmp_path,
            budget_secs=10.0,
            now=clock,
        )

        assert [d['status'] for d in seg_dicts] == ['passed'] * 2 + ['not_run'] * 6
        assert rc != 0
        assert timed_out is True

    @pytest.mark.asyncio
    async def test_a_red_segment_then_a_deadline_bound_timeout_is_a_test_failure(
        self, tmp_path,
    ):
        """The MID-FLIGHT half of the budget-exhaustion rule, via the real flag.

        `test_a_red_segment_before_the_deadline_still_wins_the_rc` covers the
        boundary-exact case: the budget runs out cleanly BETWEEN two segments,
        so the tail is `not_run` and only the SYNTHETIC flag is in play. That
        is the rarer shape. The common one is mid-flight: a segment is handed
        `remaining = deadline - now()` — never the full budget once anything
        ran before it — straddles the deadline and comes back
        `timed_out=True`. That sets `any_timed_out`, which used to be
        UNCONDITIONAL, so the same `infra_timeout` relabelling the sibling test
        forbids came back through the other door.

        With a genuine red already recorded, the run must still classify as
        `test_failure`: under the old `&&` chain the shell short-circuited at
        the red and the later segment never ran at all, so nothing it reports
        may upgrade the category. Reporting `infra_timeout` here would re-run
        the whole fleet `max_retries` times at a full budget each
        (`VerifyAttempt.pure_timeout_failure`) and route it to infra-hold
        instead of the debugger.
        """
        clock = _Clock(cost_per_segment=2.0)
        run_one = _RecordingRunOne(
            clock,
            results={
                1: (7, _PYTEST_RED, False),          # genuine red, segment 2
                3: (124, 'timed out', True),         # straddles the deadline
            },
        )
        segments = _fleet_segments()
        from orchestrator.verify import _run_segmented  # noqa: PLC0415
        from orchestrator.verify_categories import FailureCategory  # noqa: PLC0415
        from orchestrator.verify_classify import classify_failure  # noqa: PLC0415
        from orchestrator.verify_cmd import ToolKind  # noqa: PLC0415

        rc, output, timed_out, seg_dicts = await _run_segmented(
            segments,
            run_one=run_one,
            worktree=tmp_path,
            budget_secs=10_000.0,
            now=clock,
        )

        # The timeout is still recorded as a FACT on its own segment — only the
        # aggregate flag is withheld. Nothing is hidden, only reclassified.
        assert seg_dicts[3]['status'] == 'timed_out'
        assert seg_dicts[3]['timed_out'] is True
        assert rc == 7
        assert timed_out is False
        assert classify_failure(
            ToolKind.PYTEST, rc, output, timed_out,
        ) is FailureCategory.TEST_FAILURE

    @pytest.mark.asyncio
    async def test_a_lone_hang_with_no_earlier_red_is_still_an_infra_timeout(
        self, tmp_path,
    ):
        """The contrasting half: withholding the flag must not swallow a hang.

        With nothing red before it, a segment that blows its wall clock is the
        genuine article — the `&&` chain would have reached it and reported
        exactly this — so the flag stays unconditional there and the category
        stays `infra_timeout` (retryable). Read this and the test above as one
        rule: a timeout counts iff the shell would have gotten that far.
        """
        clock = _Clock(cost_per_segment=2.0)
        run_one = _RecordingRunOne(clock, results={2: (124, 'timed out', True)})
        segments = _fleet_segments()
        from orchestrator.verify import _run_segmented  # noqa: PLC0415
        from orchestrator.verify_categories import FailureCategory  # noqa: PLC0415
        from orchestrator.verify_classify import classify_failure  # noqa: PLC0415
        from orchestrator.verify_cmd import ToolKind  # noqa: PLC0415

        rc, output, timed_out, _seg_dicts = await _run_segmented(
            segments,
            run_one=run_one,
            worktree=tmp_path,
            budget_secs=10_000.0,
            now=clock,
        )

        assert rc == 124
        assert timed_out is True
        assert classify_failure(
            ToolKind.PYTEST, rc, output, timed_out,
        ) is FailureCategory.INFRA_TIMEOUT


_PYTEST_RED = (
    'collected 412 items\n'
    '.......F\n'
    'FAILED tests/test_scheduler.py::test_claims_are_exclusive - AssertionError\n'
    '===== 1 failed, 411 passed in 88.20s ====='
)


class TestRunSegmentedRoster:
    """The roster block: index/label/status for every segment, at the TOP.

    This is what turns the triaging agent's job from "prove this unrelated red
    is unrelated" into "read your own segment's line" — the human-time cost
    esc-3062-2 is actually about. It leads the output so a reader hits it
    first rather than after scrolling six subprojects of pytest chatter.

    But ``_summarize_checks`` derives cause_hint and category by
    PATTERN-SCANNING that same output, so a roster carrying tokens like
    `FAILED` or `error:` would shadow the genuine failure line. The `#` prefix
    plus a neutral status vocabulary keeps it inert to those scanners, and
    that is pinned here rather than assumed.
    """

    @staticmethod
    async def _mixed_run(tmp_path, red_output: str = 'segment blew up'):
        """One red, one not_run, the rest passed — all three statuses present."""
        clock = _Clock(cost_per_segment=5.0)
        run_one = _RecordingRunOne(clock, results={2: (1, red_output, False)})
        segments = _fleet_segments()
        from orchestrator.verify import _run_segmented  # noqa: PLC0415

        # 8 segments at 5s each needs 40s; a 35s budget is exhausted exactly as
        # the 8th is reached, leaving it unrun.
        return await _run_segmented(
            segments,
            run_one=run_one,
            worktree=tmp_path,
            budget_secs=35.0,
            now=clock,
        )

    @pytest.mark.asyncio
    async def test_output_starts_with_a_hash_prefixed_roster_of_every_segment(self, tmp_path):
        _rc, output, _timed_out, seg_dicts = await self._mixed_run(tmp_path)

        lines = output.splitlines()
        roster = [line for line in lines[: len(seg_dicts)]]
        assert len(roster) == len(seg_dicts) == 8
        for line in roster:
            assert line.startswith('#'), f'roster line is not `#`-prefixed: {line!r}'
        for entry, line in zip(seg_dicts, roster, strict=True):
            assert f'{entry["index"]}/8' in line
            assert entry['label'] in line
        # The roster leads; per-segment output blocks come after it.
        assert lines[len(seg_dicts)].startswith('=====') or lines[len(seg_dicts)] == ''

    @pytest.mark.asyncio
    async def test_roster_shows_all_three_statuses_distinctly(self, tmp_path):
        _rc, output, _timed_out, seg_dicts = await self._mixed_run(tmp_path)

        assert [d['status'] for d in seg_dicts] == (
            ['passed'] * 2 + ['failed'] + ['passed'] * 4 + ['not_run']
        )
        roster = output.splitlines()[:8]
        assert 'NOT RUN' in roster[7]
        # A reader must be able to tell the red from the greens and from the
        # one that never ran, without decoding rc numbers.
        assert roster[2] != roster[0]
        assert roster[7] != roster[2]
        assert len({roster[0], roster[2], roster[7]}) == 3

    @pytest.mark.asyncio
    async def test_roster_is_inert_to_the_cause_hint_and_category_scanners(self, tmp_path):
        """The SAME output with and without the roster must classify identically.

        If the roster could move either answer, it would be shadowing the
        genuine failing segment — turning a reporting improvement into a
        diagnosis regression.
        """
        _rc, output, _timed_out, seg_dicts = await self._mixed_run(tmp_path, _PYTEST_RED)
        from orchestrator.verify import _summarize_checks  # noqa: PLC0415

        roster_lines = len(seg_dicts)
        without_roster = '\n'.join(output.splitlines()[roster_lines:])
        assert without_roster != output

        def _classify(test_out: str):
            passed, category, cause_hint, _summary = _summarize_checks(
                1, test_out, False, 'uv run pytest tests/',
                0, '', False, 'uv run ruff check src/',
                0, '', False, 'npx pyright',
            )
            return passed, category, cause_hint

        assert _classify(output) == _classify(without_roster)

    @pytest.mark.asyncio
    async def test_the_real_pytest_failure_line_surfaces_not_a_roster_line(self, tmp_path):
        """The hint is the red segment's own output, not the report about it."""
        _rc, output, _timed_out, _seg_dicts = await self._mixed_run(tmp_path, _PYTEST_RED)
        from orchestrator.verify import _summarize_checks  # noqa: PLC0415

        _passed, _category, cause_hint, _summary = _summarize_checks(
            1, output, False, 'uv run pytest tests/',
            0, '', False, 'uv run ruff check src/',
            0, '', False, 'npx pyright',
        )

        assert cause_hint == (
            'FAILED tests/test_scheduler.py::test_claims_are_exclusive - AssertionError'
        )
        assert not cause_hint.startswith('#')


class TestRunVerificationSegmentChainedTestWiring:
    """`run_verification`'s opt-in flag — default OFF, byte-identical to today.

    Making run_verification segment any chain it is handed would silently
    change the global tail, the cargo-scoped path,
    `merge_queue._run_unscoped_typechecks` and every module_configs run — a
    wide, unrequested change in a function dozens of tests stub. The reported
    defect is the FALLBACK path, so the flag defaults False and only that call
    site passes True (step-14 / test_verify.py).
    """

    @staticmethod
    def _fallback_config(test_command: str):
        from orchestrator.config import ModuleConfig  # noqa: PLC0415

        return ModuleConfig(
            prefix='__fallback__',
            test_command=test_command,
            lint_command='uv run ruff check src/',
            type_check_command='npx pyright',
        )

    @staticmethod
    async def _run(tmp_path, test_command: str, **kwargs):
        """Drive run_verification with _run_cmd stubbed; return (result, calls)."""
        from unittest.mock import patch  # noqa: PLC0415

        from orchestrator.config import OrchestratorConfig  # noqa: PLC0415
        from orchestrator.verify import run_verification  # noqa: PLC0415

        (tmp_path / '.task').mkdir(parents=True, exist_ok=True)
        calls: list[dict] = []

        async def fake_run_cmd(cmd, cwd, timeout, env=None, log_path=None, **_kw):
            calls.append({'cmd': cmd, 'cwd': cwd, 'timeout': timeout, 'log_path': log_path})
            return 0, 'ok', False

        config = OrchestratorConfig(project_root=tmp_path, verify_admission_enabled=False)
        runs_capture: list = []

        def spy_persist(*args, **kwargs):
            for candidate in (*args, *kwargs.values()):
                if isinstance(candidate, list) and candidate and isinstance(candidate[0], dict):
                    runs_capture.append(candidate)
            return []

        with patch('orchestrator.verify._run_cmd', side_effect=fake_run_cmd), \
             patch('orchestrator.verify._persist_attempt_logs', side_effect=spy_persist):
            result = await run_verification(
                tmp_path,
                config,
                TestRunVerificationSegmentChainedTestWiring._fallback_config(test_command),
                attempt_id=1,
                # task_id is load-bearing, not decoration: run_verification
                # persists the runs dict only when attempt_id AND task_id are
                # both set (the task path's `elif attempt_id is not None and
                # task_id is not None`), so omitting it leaves `spy_persist`
                # never called and the capture seam silently empty — on a
                # green run as much as a red one.
                task_id='3338',
                max_retries=0,
                **kwargs,
            )
        return result, calls, runs_capture

    @pytest.mark.asyncio
    async def test_default_off_issues_exactly_one_run_cmd_for_the_whole_chain(self, tmp_path):
        """Every non-fallback caller keeps today's behaviour, byte for byte."""
        result, calls, _runs = await self._run(tmp_path, _FLEET_TEST_COMMAND)

        test_calls = [c for c in calls if 'pytest' in c['cmd']]
        assert len(test_calls) == 1
        assert test_calls[0]['cmd'] == _FLEET_TEST_COMMAND
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_flag_on_issues_one_run_cmd_per_segment_in_its_own_cwd(self, tmp_path):
        _result, calls, _runs = await self._run(
            tmp_path, _FLEET_TEST_COMMAND, segment_chained_test=True,
        )
        segments = _fleet_segments()

        test_calls = [c for c in calls if 'pytest' in c['cmd']]
        assert len(test_calls) == 8
        assert [c['cmd'] for c in test_calls] == [s.command for s in segments]
        assert [c['cwd'] for c in test_calls] == [tmp_path / s.cwd_rel for s in segments]

    @pytest.mark.asyncio
    async def test_check_run_keeps_the_original_chain_as_cmd_and_carries_segments(self, tmp_path):
        """Persisted-log compatibility: CheckRun.cmd is still the WHOLE chain.

        It feeds _persist_attempt_logs/_build_summary_payload/_summarize_checks,
        so preserving it keeps every persisted artifact and every failure
        classification identical. The per-segment facts ride alongside it on
        the new `segments` field instead of replacing it.
        """
        result, _calls, runs = await self._run(
            tmp_path, _FLEET_TEST_COMMAND, segment_chained_test=True,
        )

        assert result.passed is True
        test_run = _test_run_entry(runs)
        assert test_run['cmd'] == _FLEET_TEST_COMMAND
        assert test_run['segments'] is not None
        assert len(test_run['segments']) == 8
        assert [s['status'] for s in test_run['segments']] == ['passed'] * 8

    @pytest.mark.asyncio
    async def test_each_segment_streams_to_its_own_log_path(self, tmp_path):
        """Two segments share cwd `.`, so index-suffixed labels are load-bearing."""
        _result, calls, _runs = await self._run(
            tmp_path, _FLEET_TEST_COMMAND, segment_chained_test=True,
        )

        test_paths = [c['log_path'] for c in calls if 'pytest' in c['cmd']]
        assert len(test_paths) == 8
        assert all(p is not None for p in test_paths)
        assert len(set(test_paths)) == 8
        for path, segment in zip(test_paths, _fleet_segments(), strict=True):
            assert path.name == f'attempt-1.__fallback__.test.{segment.label}.log'

    @pytest.mark.asyncio
    async def test_a_refused_chain_falls_back_to_one_unsegmented_run(self, tmp_path):
        """REFUSE => byte-identical to today, with `segments is None` saying so."""
        opaque = 'uv run pytest tests/ --timeout=300 || true'
        result, calls, runs = await self._run(tmp_path, opaque, segment_chained_test=True)

        test_calls = [c for c in calls if 'pytest' in c['cmd']]
        assert len(test_calls) == 1
        assert test_calls[0]['cmd'] == opaque
        assert result.passed is True
        assert _test_run_entry(runs)['segments'] is None

    @pytest.mark.asyncio
    async def test_lint_and_type_legs_are_untouched_by_the_flag(self, tmp_path):
        """Only the test leg is wired; the others stay one call each."""
        _result, calls, _runs = await self._run(
            tmp_path, _FLEET_TEST_COMMAND, segment_chained_test=True,
        )

        assert len([c for c in calls if 'ruff check' in c['cmd']]) == 1
        assert len([c for c in calls if 'pyright' in c['cmd']]) == 1


class _SegmentedBudgetClock:
    """A monotonic clock that blows ``_run_segmented``'s budget after N segments.

    The budget is whatever ``_resolve_verify_timeout`` resolves for this config
    — deliberately NOT hard-coded here. The clock reads it off
    ``_run_segmented``'s own ``budget_secs`` argument and sizes each tick as a
    fraction of it, so "segments 3..8 never ran" holds whatever the configured
    ceiling happens to be (it has already moved 1800 -> 3600 -> 5400 across
    tasks 3348/3350; a hard-coded number here would rot on the next bump).

    ``blow_after=None`` means "never exhaust" — the clock stays frozen at 0 and
    every segment gets the full remaining budget.
    """

    def __init__(self, blow_after: int | None = None) -> None:
        self.t = 0.0
        self.blow_after = blow_after
        self.per_tick = 0.0

    def arm(self, budget_secs: float) -> None:
        if self.blow_after is not None:
            # After exactly `blow_after` ticks the elapsed time must EXCEED the
            # budget: `blow_after` segments run, every later one is not_run.
            self.per_tick = budget_secs / self.blow_after * 1.01

    def __call__(self) -> float:
        return self.t

    def tick(self) -> None:
        self.t += self.per_tick


class TestRunVerificationSegmentedAcceptance:
    """End-to-end through ``run_verification`` — the acceptance test for esc-3062-2.

    REVALIDATION NOTE. Task 3350 landed ``tests/scripts/orchestrator.yaml``, so
    a diff confined to ``tests/scripts/`` NO LONGER reaches the fallback:
    ``_build_fallback_config`` is consulted only when ``module_configs`` is
    EMPTY, so that diff now takes the module-config path. Reproducing the
    incident by scoping a ``tests/scripts/``-only diff would therefore take a
    path that has nothing to do with the defect and assert nothing about it —
    a green test proving the wrong thing. These tests construct the fallback
    ``ModuleConfig`` DIRECTLY instead.

    3350 removed one TRIGGER of the defect, not the defect: any diff that still
    falls through to ``__fallback__`` (repo-root files, ``plans/``, ``hooks/``,
    ``.github/``, mixes with no single subproject) short-circuits exactly as
    reported.
    """

    @staticmethod
    async def _run(tmp_path, *, red_labels=(), blow_budget_after=None):
        """Drive the fallback path with per-segment rc scripted by label.

        ``_run_cmd`` is stubbed and segments are identified by their streamed
        log path (``attempt-1.__fallback__.test.<label>.log``), which is the
        same per-segment namespacing production uses — so the script keys are
        the real labels rather than a positional index that would silently
        re-target if the committed chain gained a clause.
        """
        from unittest.mock import patch  # noqa: PLC0415

        from orchestrator import verify as verify_mod  # noqa: PLC0415
        from orchestrator.config import OrchestratorConfig  # noqa: PLC0415

        (tmp_path / '.task').mkdir(parents=True, exist_ok=True)
        calls: list[dict] = []
        clock = _SegmentedBudgetClock(blow_budget_after)

        def _segment_label(log_path) -> str | None:
            if log_path is None or '.test.' not in log_path.name:
                return None
            return log_path.name.split('.test.', 1)[1].removesuffix('.log')

        async def fake_run_cmd(cmd, cwd, timeout, env=None, log_path=None, **_kw):
            label = _segment_label(log_path)
            calls.append({
                'cmd': cmd, 'cwd': cwd, 'timeout': timeout,
                'log_path': log_path, 'segment': label,
            })
            if label is None:  # the unsegmented lint / type legs
                return 0, 'ok', False
            clock.tick()
            if label in red_labels:
                return 1, f'FAILED tests/test_thing.py::test_thing in {label}', False
            return 0, f'ok {label}', False

        real_run_segmented = verify_mod._run_segmented

        async def clocked_run_segmented(*args, **kwargs):
            # `now` exists on _run_segmented precisely so the shared deadline is
            # controllable; injecting it here keeps the REAL aggregator (and the
            # real run_verification wiring) under test and fakes only the clock.
            clock.arm(kwargs['budget_secs'])
            kwargs['now'] = clock
            return await real_run_segmented(*args, **kwargs)

        config = OrchestratorConfig(project_root=tmp_path, verify_admission_enabled=False)
        runs_capture: list = []

        def spy_persist(*args, **kwargs):
            for candidate in (*args, *kwargs.values()):
                if isinstance(candidate, list) and candidate and isinstance(candidate[0], dict):
                    runs_capture.append(candidate)
            return []

        with patch('orchestrator.verify._run_cmd', side_effect=fake_run_cmd), \
             patch('orchestrator.verify._run_segmented', side_effect=clocked_run_segmented), \
             patch('orchestrator.verify._persist_attempt_logs', side_effect=spy_persist):
            result = await verify_mod.run_verification(
                tmp_path,
                config,
                TestRunVerificationSegmentChainedTestWiring._fallback_config(
                    _FLEET_TEST_COMMAND,
                ),
                attempt_id=1,
                task_id='3338',
                max_retries=0,
                segment_chained_test=True,
            )
        return result, calls, runs_capture

    @pytest.mark.asyncio
    async def test_an_unrelated_red_no_longer_skips_the_last_segment(self, tmp_path):
        """THE acceptance test for esc-3062-2.

        `orchestrator` goes red three clauses in. Under the old single-shell
        `&&` chain the shell short-circuited there and `tests/scripts/` — the
        clause a task's OWN assigned files can live in — was never executed,
        with one rc that could not say so. Here it runs, and is recorded
        `passed` on its own evidence.
        """
        result, _calls, runs = await self._run(tmp_path, red_labels={'orchestrator-3'})

        test_run = _test_run_entry(runs)
        assert test_run['cmd'] == _FLEET_TEST_COMMAND, 'the persisted cmd is still the whole chain'
        segments = test_run['segments']
        assert segments is not None
        assert len(segments) == 8

        by_label = {s['label']: s for s in segments}
        # The fact the old &&-chain could NEVER produce.
        assert by_label['root-8']['status'] == 'passed'
        assert by_label['root-8']['rc'] == 0
        # Substring, not a full literal: this segment's command comes from the
        # LIVE committed yaml, whose target LIST grows as the fleet gains
        # script suites (it gained `scripts/tests/` on 2026-08-02). What
        # esc-3062-2 is about is that the `tests/scripts/` clause RAN, not how
        # many targets it happens to carry today — pinning the whole string
        # here just re-breaks this test on every unrelated config edit.
        # `test_verify_cmd.TestSplitAndChainSegmentsLiveConfigDrift` is the
        # single place that guards the exact text.
        assert 'tests/scripts/' in by_label['root-8']['cmd']
        # Nothing after the red was skipped.
        assert [s['status'] for s in segments] == [
            'passed', 'passed', 'failed', 'passed', 'passed', 'passed', 'passed', 'passed',
        ]
        # ...and the unrelated red is still a red overall. Running the later
        # segments buys information, never leniency.
        assert by_label['orchestrator-3']['status'] == 'failed'
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_a_not_run_segment_is_never_green_and_never_silent(self, tmp_path):
        """Every segment that RAN passed — and the verdict still names the rest.

        A green-so-far run with unrun segments must not report green, and the
        one-line verdict a triaging agent reads must SAY which segments have no
        result, not merely be non-green.
        """
        result, _calls, runs = await self._run(tmp_path, blow_budget_after=2)

        segments = _test_run_entry(runs)['segments']
        not_run = [s for s in segments if s['status'] == 'not_run']
        assert [s['label'] for s in not_run] == [
            'orchestrator-3', 'fused-memory-4', 'dashboard-5',
            'sampler-6', 'root-7', 'root-8',
        ]
        # Every segment that actually executed passed...
        assert all(s['status'] == 'passed' for s in segments if s['status'] != 'not_run')
        # ...and `rc is None`, never 0, is what makes "no result" unconflatable
        # with "passed" for every downstream consumer.
        assert all(s['rc'] is None for s in not_run)
        assert all(s['skip_reason'] for s in not_run)

        assert result.passed is False, 'a run with unrun segments is never green'
        for segment in not_run:
            assert segment['label'] in result.cause_hint, (
                f'cause_hint must NAME the unrun segment {segment["label"]!r}; '
                f'got {result.cause_hint!r}'
            )

    @pytest.mark.asyncio
    async def test_a_not_run_segment_keeps_the_existing_timeout_classification(self, tmp_path):
        """`timed_out` stays True so the timeout-retry path is reached unchanged.

        Forcing it is what keeps a budget-exhausted chain classified as
        `infra_timeout` (retryable) exactly as a single-chain timeout is today,
        leaving run_verification's retry/env-recovery machinery untouched.

        Pinned as an explicit PAIR with
        ``test_a_genuine_red_survives_budget_exhaustion_as_a_test_failure``
        below: with no red present the run IS a timeout, and with one present it
        is not. Asserting the category on both sides keeps the two outcomes
        stated rather than one asserted and the other assumed.
        """
        result, _calls, runs = await self._run(tmp_path, blow_budget_after=2)

        assert any(s['status'] == 'not_run' for s in _test_run_entry(runs)['segments'])
        assert result.timed_out is True
        assert result.category == 'infra_timeout'

    @pytest.mark.asyncio
    async def test_a_genuine_red_survives_budget_exhaustion_as_a_test_failure(self, tmp_path):
        """A real red plus an exhausted budget is a `test_failure`, not a timeout.

        This is the COMMON shape of a red fallback verify, not a corner case:
        removing the `&&` short-circuit — the whole point of task 3338 — makes
        budget exhaustion strictly MORE likely, because all 8 segments now
        always run where the shell previously stopped at the first red. The
        committed config's own measured table already records five of seven
        segments costing 1838.60s.

        Synthesising `timed_out` on exhaustion would relabel this genuine red as
        an `infra_timeout` (``classify_failure`` guard 2 wins over every output
        pattern), making ``VerifyAttempt.pure_timeout_failure`` true and routing
        the run to infra-hold — re-run ``max_retries`` times at a full budget
        each — instead of to the debugger. Under the old `&&` chain the shell
        short-circuited at the red and the check finished fast with
        rc=1/timed_out=False/`test_failure`, so without this the segmentation
        would be a regression against its own baseline.

        And no information is lost in exchange: the hint must still carry BOTH
        the genuine failure line AND the names of every segment with no result.
        """
        result, _calls, runs = await self._run(
            tmp_path, red_labels={'escalation-2'}, blow_budget_after=3,
        )

        segments = _test_run_entry(runs)['segments']
        assert [s['status'] for s in segments] == (
            ['passed', 'failed', 'passed'] + ['not_run'] * 5
        )

        assert result.passed is False
        assert result.timed_out is False, 'a genuine red is not a timeout'
        assert result.category == 'test_failure', (
            'the run must reach the debugger/test_failure path, not infra-hold; '
            f'got {result.category!r} with cause_hint {result.cause_hint!r}'
        )
        # Nothing is silently greened: the genuine red AND every unrun segment
        # are both still named in the one line a triaging agent reads.
        assert 'FAILED tests/test_thing.py::test_thing in escalation-2' in result.cause_hint
        assert 'segments not run:' in result.cause_hint
        for entry in segments:
            if entry['status'] == 'not_run':
                assert entry['label'] in result.cause_hint, (
                    f'cause_hint must still NAME the unrun segment {entry["label"]!r}; '
                    f'got {result.cause_hint!r}'
                )


class TestSegmentsReachThePersistedSummaryJson:
    """The segment facts must survive into the FILE, not just the runs dict.

    Every other test in this module spies on ``_persist_attempt_logs``'
    ARGUMENT. That seam cannot see what happens next: ``_build_summary_payload``
    rebuilds each ``commands`` entry from an explicit key WHITELIST rather than
    passing the run dict through, so a new key on ``CheckRun`` reaches the
    argument and is then silently dropped on the way to disk. ``segments`` was
    dropped exactly that way until this amendment, leaving the one STRUCTURED
    record of which segments never ran surviving only as free text inside the
    aggregated ``.log`` blob — the docstrings claimed the JSON, and the tests
    asserting the JSON were asserting the argument.

    So these tests read the written
    ``.task/verify/attempt-1.__fallback__.summary.json`` and run
    ``_persist_attempt_logs`` for real.
    """

    @staticmethod
    async def _run_persisting(tmp_path, *, red_labels=(), blow_budget_after=None):
        """Drive the fallback path with ``_persist_attempt_logs`` UNPATCHED."""
        import json  # noqa: PLC0415
        from unittest.mock import patch  # noqa: PLC0415

        from orchestrator import verify as verify_mod  # noqa: PLC0415
        from orchestrator.config import OrchestratorConfig  # noqa: PLC0415

        (tmp_path / '.task').mkdir(parents=True, exist_ok=True)
        clock = _SegmentedBudgetClock(blow_budget_after)

        def _segment_label(log_path) -> str | None:
            if log_path is None or '.test.' not in log_path.name:
                return None
            return log_path.name.split('.test.', 1)[1].removesuffix('.log')

        async def fake_run_cmd(cmd, cwd, timeout, env=None, log_path=None, **_kw):
            label = _segment_label(log_path)
            if label is None:  # the unsegmented lint / type legs
                return 0, 'ok', False
            clock.tick()
            if label in red_labels:
                return 1, f'FAILED tests/test_thing.py::test_thing in {label}', False
            return 0, f'ok {label}', False

        real_run_segmented = verify_mod._run_segmented

        async def clocked_run_segmented(*args, **kwargs):
            clock.arm(kwargs['budget_secs'])
            kwargs['now'] = clock
            return await real_run_segmented(*args, **kwargs)

        config = OrchestratorConfig(project_root=tmp_path, verify_admission_enabled=False)
        with patch('orchestrator.verify._run_cmd', side_effect=fake_run_cmd), \
             patch('orchestrator.verify._run_segmented', side_effect=clocked_run_segmented):
            result = await verify_mod.run_verification(
                tmp_path,
                config,
                TestRunVerificationSegmentChainedTestWiring._fallback_config(
                    _FLEET_TEST_COMMAND,
                ),
                attempt_id=1,
                task_id='3338',
                max_retries=0,
                segment_chained_test=True,
            )
        # The fallback ModuleConfig's `__fallback__` prefix is the filename
        # infix _persist_attempt_logs inserts to keep concurrent per-subproject
        # runs from clobbering one another.
        summary_path = tmp_path / '.task' / 'verify' / 'attempt-1.__fallback__.summary.json'
        assert summary_path.exists(), (
            f'run_verification persisted no summary JSON at {summary_path}; '
            f'wrote {sorted(p.name for p in (tmp_path / ".task" / "verify").glob("*"))}'
        )
        return result, json.loads(summary_path.read_text(encoding='utf-8'))

    @staticmethod
    def _command_entry(summary: dict, label: str) -> dict:
        return next(c for c in summary['commands'] if c['label'] == label)

    @pytest.mark.asyncio
    async def test_the_written_summary_json_carries_every_segment(self, tmp_path):
        """A triaging agent reading the persisted artifact sees the per-segment facts."""
        _result, summary = await self._run_persisting(tmp_path, red_labels={'orchestrator-3'})

        segments = self._command_entry(summary, 'test')['segments']
        assert segments is not None, (
            '_build_summary_payload rebuilds each command entry from a key '
            'whitelist; `segments` must be in it or the facts never reach disk'
        )
        assert [s['label'] for s in segments] == [s.label for s in _fleet_segments()]
        by_label = {s['label']: s for s in segments}
        assert by_label['orchestrator-3']['status'] == 'failed'
        # The fact the old &&-chain could never write down: the LAST clause ran
        # anyway, and the persisted artifact says so on its own evidence.
        assert by_label['root-8']['status'] == 'passed'

    @pytest.mark.asyncio
    async def test_unrun_segments_are_readable_from_the_file_not_only_the_log_text(
        self, tmp_path,
    ):
        """`rc=None` + a skip_reason is the unconflatable encoding — structurally, on disk."""
        _result, summary = await self._run_persisting(tmp_path, blow_budget_after=3)

        segments = self._command_entry(summary, 'test')['segments']
        assert [s['status'] for s in segments] == ['passed'] * 3 + ['not_run'] * 5
        for entry in segments:
            if entry['status'] == 'not_run':
                assert entry['rc'] is None, 'a segment that never ran must not read as rc 0'
                assert entry['skip_reason']

    @pytest.mark.asyncio
    async def test_unsegmented_checks_persist_segments_as_null(self, tmp_path):
        """`None`, not an absent key — the same absent-vs-null contract `to_dict` keeps.

        A conditionally-present key would leave a consumer unable to tell "this
        artifact predates segments" from "this check was not segmented".
        """
        _result, summary = await self._run_persisting(tmp_path)

        for label in ('lint', 'type'):
            entry = self._command_entry(summary, label)
            assert 'segments' in entry, f'{label}: key must be present unconditionally'
            assert entry['segments'] is None
