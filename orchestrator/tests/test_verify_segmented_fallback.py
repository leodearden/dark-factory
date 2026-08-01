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
        """The facts a triaging agent reads, and that land in attempt-N.json."""
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
