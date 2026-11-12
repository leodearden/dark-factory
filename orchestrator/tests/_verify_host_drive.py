"""Builders and doubles for tests that drive a merge's verify through the merge worker's verify-host seam."""
from __future__ import annotations

import asyncio
import dataclasses
import logging
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from _merge_lane_fakes import FakeVerifier
from _merge_worker_drive import make_merge_request
from _verify_result_doubles import real_verify_result

from orchestrator.config import GitConfig, OrchestratorConfig, VerifyRunnerConfig
from orchestrator.merge_lane.types import (
    InflightVerifyResult,
    MergeRequest,
    RealMergeItem,
)
from orchestrator.merge_lane.worker import SpeculativeMergeWorker
from orchestrator.verify import VerifyResult
from orchestrator.verify_runner import HostLease, RunnerUnavailable


def probe_free_orch_config(
    *,
    verify_runners: list[VerifyRunnerConfig] | None = None,
    main_branch: str = 'main',
    drift_every_n: int = 20,
) -> OrchestratorConfig:
    """Build an OrchestratorConfig with optional verify_runners.

    ``verify_env_probes`` is pinned EMPTY (Lever C off, task 4537).
    conftest.py::_isolate_orch_config is autouse and points ORCH_CONFIG_PATH at
    this repo's own dark-factory-orchestrator.yaml, which declares probes — so
    without this a config built here inherits them, and
    orchestrator/src/orchestrator/merge_lane/drift.py::_run_drift_check would
    capture the LOCAL fingerprint
    through verify_runner.py::env_probe_local_run: real, untimed
    subprocesses (npx/uv/node/python3 plus rustc/cargo/sccache) spawned by
    tests that are about runner-pool wiring, not about env probing.  Any test
    that wants Lever C on should say so explicitly.
    """
    kwargs: dict[str, Any] = {
        'git': GitConfig(main_branch=main_branch),
        'verify_drift_check_every_n_lands': drift_every_n,
        'verify_env_probes': [],
    }
    if verify_runners is not None:
        kwargs['verify_runners'] = verify_runners
    return OrchestratorConfig(**kwargs)


def merge_request_for_verify(
    config: OrchestratorConfig,
    *,
    task_files: list[str] | None = None,
    worktree: Path | None = None,
) -> MergeRequest:
    """Build task-42's MergeRequest, scoped to *task_files*, for a post-merge verify."""
    request = make_merge_request('task-42', 'task/42', worktree or Path('/repo/task-42'), config)
    return dataclasses.replace(request, task_files=task_files)


def git_ops_mock_for_verify() -> MagicMock:
    """Build a minimal async mock for GitOps."""
    mock = MagicMock()
    mock.get_main_sha = AsyncMock(return_value='main-sha')
    mock.get_free_disk_bytes = AsyncMock(return_value=100 * 1024 ** 3)  # plenty of disk
    mock.cleanup_merge_worktree = AsyncMock()
    mock.create_ephemeral_verify_worktree = AsyncMock(return_value='/repo/_throwaway')
    return mock


def runner_double(
    name: str,
    *,
    is_local: bool,
    result: VerifyResult | None = None,
    error: BaseException | None = None,
) -> MagicMock:
    """A verify host that reports *result*, or raises *error* when dispatched.

    The pool and the drift detector ask a runner only for ``name``,
    ``is_local``, ``health()`` and ``run_merge_verify()`` — so a host is an
    injectable collaborator and no module attribute has to be substituted to
    decide what a verify on it does.
    """
    runner = MagicMock()
    runner.name = name
    runner.is_local = is_local
    runner.health = AsyncMock(return_value=True)
    if error is not None:
        runner.run_merge_verify = AsyncMock(side_effect=error)
    else:
        runner.run_merge_verify = AsyncMock(return_value=result or real_verify_result(True))
    return runner


#: Host name used by the RunnerUnavailable drives below.
RUNNER_UNAVAILABLE_HOST = 'leo-laptop'


async def drive_runner_unavailable(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    reason: str,
    *,
    error: BaseException | None = None,
) -> tuple[InflightVerifyResult, list[logging.LogRecord]]:
    """Raise RunnerUnavailable(*reason*) from a REMOTE lease; return (result, warning records).

    *error* replaces the raised RunnerUnavailable, to drive a kind other than
    'transport'.

    The shared drive for the ``except RunnerUnavailable`` handler's stored
    ``reason`` (task 1795) and its WARNING's host/rc/trigger (task 4194): the
    one reach into ``_run_inflight_verify`` those assertions need is spelled
    once, here.
    """
    git_ops = git_ops_mock_for_verify()
    q: asyncio.Queue = asyncio.Queue()
    worker = SpeculativeMergeWorker(
        git_ops=git_ops, queue=q, verifier=FakeVerifier(),
    )

    merge_result = MagicMock()
    merge_result.merge_commit = 'abc123def456789abc1'
    config = probe_free_orch_config()
    item = RealMergeItem(
        request=merge_request_for_verify(config, task_files=[], worktree=tmp_path),
        merge_result=merge_result,
        merge_wt=tmp_path / 'merge-wt',
        base_sha='base123',
        speculative=False,
    )

    # REMOTE lease whose host is genuinely unreachable: a remote-lease
    # dispatch builds a single-runner pool, so the RunnerUnavailable the
    # runner raises propagates through the real verify path exactly as a
    # dead laptop does in production — no module attribute substituted.
    fake_runner = runner_double(
        RUNNER_UNAVAILABLE_HOST, is_local=False, error=error or RunnerUnavailable(reason),
    )
    lease = HostLease(name=RUNNER_UNAVAILABLE_HOST, runner=fake_runner, is_local=False)

    with caplog.at_level(logging.WARNING, logger='orchestrator.merge_lane'):
        result = await worker._run_inflight_verify(item, lease)

    records = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and 'remote runner unavailable' in r.message
    ]
    return result, records
