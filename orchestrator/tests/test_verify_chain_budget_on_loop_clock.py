"""verify.py::_run_segmented measures its shared chain budget on the running
event loop's clock, the clock verify.py::_run_cmd measures each segment's
budget on — so the deadline a segment is handed and the deadline the chain
checks before the next one are read off one clock."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from shared.testing_virtual_clock import virtual_clock_test

from orchestrator.verify import _run_segmented
from orchestrator.verify_cmd import ChainSegment


async def _spend_the_whole_segment_budget(
    cmd: str, cwd: Path, timeout: float, label: str,
) -> tuple[int, str, bool]:
    await asyncio.sleep(timeout)
    return 0, f'ok {label}', False


@pytest.mark.timeout(15)
@virtual_clock_test
async def test_a_segment_spending_the_chain_budget_on_the_loop_clock_leaves_the_next_unrun(
    tmp_path: Path,
):
    segments = [
        ChainSegment(cwd_rel='.', command='true', label='first'),
        ChainSegment(cwd_rel='.', command='true', label='second'),
    ]

    rc, out, timed_out, seg_dicts = await _run_segmented(
        segments,
        run_one=_spend_the_whole_segment_budget,
        worktree=tmp_path,
        budget_secs=1.0,
    )

    assert [seg['status'] for seg in seg_dicts] == ['passed', 'not_run'], (
        f'The first segment spent the 1.0s chain budget on the loop clock, so '
        f'the second must not run; got {seg_dicts!r}, rc={rc}, '
        f'timed_out={timed_out}, out={out!r}'
    )
