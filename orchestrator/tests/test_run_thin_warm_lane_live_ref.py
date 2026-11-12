"""``GitOps.thin_warm_lane`` against the REAL shipped thin-warm-lane.sh.

Task 5307.  The script now refuses to thin a lane a live process references
(its T4 gate; ``orchestrator/scripts/warm-lane/README.md`` "Delta 17") with the
same exit 75 it uses for a held flock.  This pins the wrapper's side of that
contract end to end: the refusal stays a benign DEBUG skip (never WARNING,
inv.11), dark-factory's log can still tell WHICH gate refused, because the
script's ``LANE_LIVE_REF:`` token reaches it, and the same lane thins once the
reference clears.  A successful thin's log line carries the script's stderr
too, because that is where a fail-open scan's ``live-ref scan DEGRADED``
warning lands.
"""
from __future__ import annotations

import logging
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from test_warm_lane_scripts_shipped import WARM_LANE_SCRIPT_DIR

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps

pytestmark = pytest.mark.asyncio


@pytest.fixture
def git_ops(tmp_path: Path, monkeypatch, df_warm_lane_script_dir) -> GitOps:
    """A pool-less GitOps whose thin-warm-lane.sh resolves to dark-factory's copy.

    No ``<project>/scripts/`` exists, so resolution falls through to the REAL
    shipped directory, and being pool-less keeps the pool-storage guard quiet.
    ``REIFY_WARM_LANE_MOUNT`` is cleared so an ambient mount cannot trip the
    script's under-mount guard.
    """
    df_warm_lane_script_dir(WARM_LANE_SCRIPT_DIR)
    monkeypatch.delenv('REIFY_WARM_LANE_MOUNT', raising=False)
    project = tmp_path / 'proj'
    project.mkdir()
    return GitOps(GitConfig(), project)


@pytest.fixture
def lane(tmp_path: Path) -> Path:
    lane_dir = tmp_path / 'pool' / '_lane-1'
    (lane_dir / 'target').mkdir(parents=True)
    (lane_dir / 'target' / 'DIVERGENT_MARKER').touch()
    return lane_dir


@pytest.fixture
def holder(lane: Path) -> Iterator[subprocess.Popen[bytes]]:
    """A live process whose cwd is ``<lane>/target``.

    ``Popen`` returns only after the child's chdir + exec, so the reference is
    causally established before the test proceeds — no sleep.
    """
    proc = subprocess.Popen(['sleep', '300'], cwd=lane / 'target')
    try:
        yield proc
    finally:
        proc.kill()
        proc.wait()


async def test_a_live_reference_is_an_attributed_benign_skip_until_it_clears(
    git_ops: GitOps, lane: Path, holder: subprocess.Popen[bytes], caplog,
) -> None:
    with caplog.at_level(logging.DEBUG, logger='orchestrator.git_ops'):
        refused_rc = await git_ops.thin_warm_lane(lane)

    assert refused_rc == 75
    assert (lane / 'target' / 'DIVERGENT_MARKER').exists()
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == [], f'a live-reference skip must never WARN: {warnings}'
    debug = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    assert any('LANE_LIVE_REF:' in message for message in debug), (
        'the rc=75 DEBUG line does not attribute the skip to the live-reference '
        f'gate, so it reads as a flock refusal: {debug}'
    )

    holder.kill()
    holder.wait()
    thinned_rc = await git_ops.thin_warm_lane(lane)

    assert thinned_rc == 0
    assert not (lane / 'target').exists()


async def test_a_successful_thin_logs_the_scripts_stderr(
    git_ops: GitOps, lane: Path, caplog,
) -> None:
    with caplog.at_level(logging.INFO, logger='orchestrator.git_ops'):
        rc = await git_ops.thin_warm_lane(lane)

    assert rc == 0
    info = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert any('Thinned lane:' in message for message in info), (
        'the success line drops the script stderr, so a degraded /proc that let '
        f'the T4 gate fail open leaves no trace in dark-factory\'s log: {info}'
    )
