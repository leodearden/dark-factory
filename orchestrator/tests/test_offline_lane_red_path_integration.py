"""Integration gate for the offline lane's red path: PRD ``plans/offline-lane-red-path-prd.md`` §6 scenarios 1-23.

STRATEGY: every scenario drives one REAL ``OfflineLaneWorker`` through its public
surface, ``on_post_merge`` and ``run``, over a real tmp repo and its real
``_offline-deep`` worktree.  The lane command is a stub ``pytest`` on PATH that plays
back one scripted sub-run per launch, so each pass's detect and confirm are real
``sh -c`` processes with real output files, junit reports, exits and signals; the
task store is fused-memory's task tools as the production typed client reaches them;
the escalation queue is real; the worker's clock is injected so passes are spaced
without sleeping.  ``_offline_lane_red_path_fixtures.py`` builds and steps the worker.

This module adds tests only: no production module changes with it.
"""

from __future__ import annotations

import signal
from pathlib import Path

import pytest
from _offline_lane_red_path_fixtures import A, C, RedPathLane, Run, lane_test_timeout

pytestmark = [pytest.mark.usefixtures('code_default_config')]


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=2))
async def test_scenario_01_a_red_and_a_green_pass_each_leave_a_record(tmp_path, monkeypatch):
    lane = await RedPathLane.open(tmp_path, monkeypatch)
    red_run = Run(failed={A: 'AssertionError: a'}, errors={C: 'fixture c missing'})

    red = await lane.advance(red_run, red_run)
    red_head = await lane.head()
    green = await lane.advance(Run(passed=(A, C)))

    assert red.head == red_head
    assert green.head == await lane.head()
    for record in (red, green):
        directory = lane.project_root / 'data' / 'orchestrator' / 'offline-lane' / 'lane-fixture' / record.pass_id
        assert (directory / 'record.json').is_file()
        assert record.detect is not None
        assert record.detect.ended_by == 'exit'
        assert record.detect.started_at >= record.started_at
    assert red.detect is not None and green.detect is not None
    assert red.detect.returncode == 1
    assert green.detect.returncode == 0
    assert [(f.node, f.kind, f.first_line) for f in red.detect.failures] == [
        (A, 'failed', 'AssertionError: a'),
        (C, 'error', 'fixture c missing'),
    ]
    output = Path(red.detect.output_path)
    assert output.is_file()
    assert f'FAILED {A} - AssertionError: a' in output.read_text()
    assert green.detect.failures == ()
    assert green.outcome is None


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=1))
@pytest.mark.parametrize('phase', ['detect', 'confirm'])
async def test_scenario_02_a_signal_ended_sub_run_is_unconfirmable(tmp_path, monkeypatch, phase):
    lane = await RedPathLane.open(tmp_path, monkeypatch)
    killed = Run(failed={A: 'AssertionError: a'}, signal=signal.SIGKILL, junit=False, complete=False)
    runs = (killed,) if phase == 'detect' else (Run(failed={A: 'AssertionError: a'}), killed)

    record = await lane.advance(*runs)

    run = record.detect if phase == 'detect' else record.confirm
    assert run is not None
    assert run.ended_by == 'signal'
    assert run.returncode in (-signal.SIGKILL, 128 + signal.SIGKILL)
    assert record.outcome == {
        'arm': 'unconfirmable',
        'reason': f'{phase}_signal_ended',
        'detail': str(run.returncode),
    }
    if phase == 'detect':
        assert record.confirm is None
