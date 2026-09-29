"""``merge_request`` carries lane intent all the way to the enqueued request.

Task 4888.  Two gaps, one branch.  ``merge_request`` had no way for a caller to
say which merge lane a submission belongs in, and the ``metadata.merge_lane``
key the orchestrator's own submit path honours was INERT for every
MCP-submitted merge — ``MergeRequest(...)`` omitted ``lane=`` and took the
dataclass default ``'normal'``.

These are WIRING tests: they assert which lane reaches the enqueued
``MergeRequest``, and what the caller is told.  The precedence rule itself is
pinned as pure functions in ``test_merge_lane_resolution.py``; neither module
reaches into the other's internals.

Harness patterns are taken wholesale from the existing suite rather than
hand-rolled — ``call_merge_request`` from ``_merge_tool_calls``, the
``captured_req`` background worker from ``test_server_chokepoint.py``, and the
``scheduler.get_task`` AsyncMock (including its ``scheduler_raises`` fault arm)
from ``test_merge_status_git_authority.py``.  The AsyncMock is also what makes
the COST contract executable: ``await_count`` / ``assert_not_awaited`` pin that
the lane fallback never adds a second Taskmaster round-trip to the submit path.
"""
from __future__ import annotations

import asyncio
import types
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _merge_tool_calls import call_merge_request

from escalation.queue import EscalationQueue
from escalation.server import create_server

# ---------------------------------------------------------------------------
# Cross-package orchestrator imports — mirrors test_merge_state_wiring.py's guard.
# ---------------------------------------------------------------------------
try:
    from orchestrator.config import OrchestratorConfig  # type: ignore[reportMissingImports]
    from orchestrator.merge_queue import (  # type: ignore[reportMissingImports]
        InFlightMergeRegistry,
    )
    from orchestrator.merge_types import MergeOutcome  # type: ignore[reportMissingImports]
    _ORCHESTRATOR_AVAILABLE = True
except ImportError:
    _ORCHESTRATOR_AVAILABLE = False
    OrchestratorConfig: Any = None  # type: ignore[assignment,misc]
    InFlightMergeRegistry: Any = None  # type: ignore[assignment,misc]
    MergeOutcome: Any = None  # type: ignore[assignment,misc]


pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not _ORCHESTRATOR_AVAILABLE, reason='orchestrator package not installed'
    ),
]


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _stub_git_ops(**overrides) -> types.SimpleNamespace:
    """git_ops stub for the ORDINARY submission — neither fast-path arm hits.

    ``resolve_branch_sha`` answers None by default, which short-circuits both
    the ancestor arm and the patch-id backstop before ``_declined()`` is ever
    awaited.  That is the overwhelmingly common submission, and the one on
    which the lane fallback's cost is actually paid.
    """
    stub = types.SimpleNamespace(
        resolve_branch_sha=AsyncMock(return_value=None),
        is_ancestor=AsyncMock(return_value=False),
        find_inflight_merge_worktree=AsyncMock(return_value=None),
    )
    for name, fn in overrides.items():
        setattr(stub, name, fn)
    return stub


def _make_harness(
    *,
    metadata: dict[str, Any] | None = None,
    git_ops: types.SimpleNamespace | None = None,
    scheduler_raises: bool = False,
    no_scheduler: bool = False,
) -> types.SimpleNamespace:
    """Harness with a ``scheduler.get_task`` AsyncMock the tests can interrogate."""
    harness = types.SimpleNamespace(git_ops=git_ops or _stub_git_ops())
    if not no_scheduler:
        harness.scheduler = types.SimpleNamespace(
            get_task=(
                AsyncMock(side_effect=RuntimeError('scheduler unreachable'))
                if scheduler_raises
                else AsyncMock(return_value={'metadata': metadata or {}})
            ),
        )
    return harness


def _make_server(tmp_path: Path, harness, mq: asyncio.Queue, registry=None):
    """``registry`` is only passed when a test needs it PRE-SEEDED.

    A pre-seeded entry is what makes a submission coalesce instead of
    dispatching, which is the only way to reach the ``'attached'`` response
    shape.
    """
    return create_server(
        EscalationQueue(tmp_path / 'esc'),
        merge_queue=mq,
        orch_config=OrchestratorConfig(project_root=tmp_path / 'repo'),
        harness=harness,
        merge_inflight_registry=registry if registry is not None else InFlightMergeRegistry(),
    )


async def _submit(tmp_path: Path, harness, **kwargs) -> tuple[dict[str, Any], list]:
    """Submit one merge request and capture the ``MergeRequest`` the queue saw.

    A background worker dequeues and resolves the future so the ``wait_secs>0``
    call returns a terminal shape rather than timing out; ``captured_req``
    observes the enqueued request's ``lane`` without touching worker internals.
    """
    mq: asyncio.Queue = asyncio.Queue()
    server = _make_server(tmp_path, harness, mq)
    captured_req: list = []

    async def _worker():
        req = await mq.get()
        captured_req.append(req)
        req.result.set_result(MergeOutcome('done', reason='test worker'))

    worker_task = asyncio.create_task(_worker())
    try:
        result = await asyncio.wait_for(
            call_merge_request(
                server,
                task_id='4888',
                branch='4888',
                worktree=str(tmp_path / 'wt'),
                wait_secs=5,
                **kwargs,
            ),
            timeout=5.0,
        )
    finally:
        if not captured_req:
            worker_task.cancel()
    if captured_req:
        await worker_task
    return result, captured_req


async def _submit_nonblocking(
    tmp_path: Path, harness, *, registry=None, **kwargs
) -> dict[str, Any]:
    """Submit with ``wait_secs=0`` and return the immediate non-blocking shape.

    Deliberately runs with NO worker.  The non-blocking path answers before
    anything dequeues, so this response is the whole of what the submitter
    learns at submit time — which is what makes it the audit surface.
    """
    mq: asyncio.Queue = asyncio.Queue()
    server = _make_server(tmp_path, harness, mq, registry=registry)
    return await asyncio.wait_for(
        call_merge_request(
            server,
            task_id='4888',
            branch='4888',
            worktree=str(tmp_path / 'wt'),
            wait_secs=0,
            **kwargs,
        ),
        timeout=5.0,
    )


# ---------------------------------------------------------------------------
# The explicit parameter
# ---------------------------------------------------------------------------


class TestExplicitLaneArgument:
    """A caller can say which lane a submission belongs in."""

    @pytest.mark.parametrize('lane', ['high', 'normal'])
    async def test_explicit_lane_reaches_the_enqueued_request(
        self, tmp_path: Path, lane: str
    ):
        _, captured = await _submit(tmp_path, _make_harness(), lane=lane)
        assert captured, 'nothing was enqueued'
        assert captured[0].lane == lane


class TestInvalidLaneIsRejectedLoudly:
    """A typo'd lane is refused at the door, not silently downgraded."""

    async def test_invalid_lane_returns_the_structured_reject(self, tmp_path: Path):
        """Assert on the machine-branchable ``code``, never on the prose."""
        mq: asyncio.Queue = asyncio.Queue()
        server = _make_server(tmp_path, _make_harness(), mq)

        result = await call_merge_request(
            server,
            task_id='4888',
            branch='4888',
            worktree=str(tmp_path / 'wt'),
            wait_secs=5,
            lane='higgh',
        )

        assert result.get('code') == 'invalid_lane', result
        assert isinstance(result.get('error'), str) and result['error']
        assert isinstance(result.get('hint'), str) and result['hint']
        assert 'status' not in result, (
            f'a reject must not look like a submission outcome: {result}'
        )

    async def test_invalid_lane_enqueues_nothing(self, tmp_path: Path):
        mq: asyncio.Queue = asyncio.Queue()
        server = _make_server(tmp_path, _make_harness(), mq)

        await call_merge_request(
            server,
            task_id='4888',
            branch='4888',
            worktree=str(tmp_path / 'wt'),
            wait_secs=5,
            lane='higgh',
        )

        assert mq.empty(), 'a rejected lane must leave nothing on the queue'

    async def test_invalid_lane_costs_no_git_and_no_scheduler_round_trip(
        self, tmp_path: Path
    ):
        """The ORDERING contract: validation runs before any git or metadata work.

        Validation is pure, so a typo can never pay a ``git`` ref read or a
        Taskmaster ``get_task`` dispatch (whose internal timeout is 15s).
        """
        git_ops = _stub_git_ops()
        harness = _make_harness(git_ops=git_ops, metadata={'merge_lane': 'high'})
        mq: asyncio.Queue = asyncio.Queue()
        server = _make_server(tmp_path, harness, mq)

        await call_merge_request(
            server,
            task_id='4888',
            branch='4888',
            worktree=str(tmp_path / 'wt'),
            wait_secs=5,
            lane='higgh',
        )

        harness.scheduler.get_task.assert_not_awaited()
        git_ops.resolve_branch_sha.assert_not_awaited()


# ---------------------------------------------------------------------------
# The metadata fallback — the headline regression
# ---------------------------------------------------------------------------


class TestMetadataMergeLaneIsHonoured:
    """``metadata.merge_lane`` was INERT for every MCP-submitted merge."""

    async def test_metadata_lane_reaches_the_enqueued_request(self, tmp_path: Path):
        """THE SIGNAL. No ``lane`` argument, so the task's own lane is inherited.

        RED before task 4888: ``MergeRequest(...)`` omitted ``lane=`` and took
        the dataclass default, so a main-health fix task carrying
        ``metadata.merge_lane='high'`` was silently enqueued behind every
        routine merge.
        """
        harness = _make_harness(metadata={'merge_lane': 'high'})
        _, captured = await _submit(tmp_path, harness)
        assert captured, 'nothing was enqueued'
        assert captured[0].lane == 'high'

    async def test_explicit_normal_argument_beats_high_metadata(self, tmp_path: Path):
        """PRECEDENCE. The argument wins even when it equals the default.

        A truthiness-based implementation passes the test above and fails this
        one — and holding a ``'high'`` task back to the normal lane is a thing
        an operator legitimately wants to do.
        """
        harness = _make_harness(metadata={'merge_lane': 'high'})
        _, captured = await _submit(tmp_path, harness, lane='normal')
        assert captured and captured[0].lane == 'normal'

    async def test_no_metadata_lane_is_normal(self, tmp_path: Path):
        _, captured = await _submit(tmp_path, _make_harness(metadata={}))
        assert captured and captured[0].lane == 'normal'

    async def test_inherited_typo_normalises_silently_and_still_submits(
        self, tmp_path: Path
    ):
        """The deliberate asymmetry against an invalid ``lane`` ARGUMENT.

        An inherited value was written by another actor at another time, so it
        fails OPEN to ``'normal'`` and the submission succeeds — where the same
        spelling passed as ``lane=`` is rejected outright.
        """
        harness = _make_harness(metadata={'merge_lane': 'higgh'})
        result, captured = await _submit(tmp_path, harness)
        assert 'code' not in result, f'an inherited typo must not reject: {result}'
        assert captured and captured[0].lane == 'normal'


class TestLaneResolutionFailsOpen:
    """A lane resolution must never be able to FAIL a merge submission."""

    async def test_raising_scheduler_still_submits_at_the_normal_lane(
        self, tmp_path: Path
    ):
        harness = _make_harness(scheduler_raises=True)
        result, captured = await _submit(tmp_path, harness)
        assert result.get('status') == 'done', result
        assert captured and captured[0].lane == 'normal'

    async def test_harness_without_a_scheduler_still_submits(self, tmp_path: Path):
        harness = _make_harness(no_scheduler=True)
        result, captured = await _submit(tmp_path, harness)
        assert result.get('status') == 'done', result
        assert captured and captured[0].lane == 'normal'


class TestMetadataReadIsPaidAtMostOnce:
    """The COST contract, made executable (design decision 5).

    ``git_authority.task_metadata`` is uncached: every call is a fresh
    ``scheduler.get_task`` → Taskmaster MCP dispatch with an internal
    ``timeout=15``.  Without these two tests the memoization is unpinned and a
    later edit can silently reintroduce a second round-trip on the submit path.
    """

    @staticmethod
    def _degenerate_branch_git_ops(tmp_path: Path, tip: str) -> types.SimpleNamespace:
        """git_ops for a branch that reaches the degeneracy probe AND enqueues.

        ``is_ancestor`` is True so the ancestor arm runs ``_declined()``; the
        task metadata records ``branch_base_sha == tip`` so the probe answers
        DEGENERATE, which declines both fast-path arms and lets the request
        fall through to the queue.  Both metadata consumers are therefore live
        in one call — which is the only configuration in which a second
        round-trip could hide.
        """
        repo = tmp_path / 'repo'
        # `git cherry` runs with cwd=git_ops.project_root, and git_ops raises
        # WorktreeMissing on a cwd that does not exist.  The directory need only
        # EXIST: it is not a git repo, so `git cherry` exits non-zero and
        # patch_content_contained fails open to False — which is what lets the
        # request fall through to the queue.
        repo.mkdir(parents=True, exist_ok=True)
        return _stub_git_ops(
            resolve_branch_sha=AsyncMock(return_value=tip),
            is_ancestor=AsyncMock(return_value=True),
            project_root=repo,
        )

    async def test_at_most_one_get_task_when_both_consumers_are_live(
        self, tmp_path: Path
    ):
        tip = 'b' * 40
        harness = _make_harness(
            metadata={'branch_base_sha': tip, 'merge_lane': 'high'},
            git_ops=self._degenerate_branch_git_ops(tmp_path, tip),
        )
        _, captured = await _submit(tmp_path, harness)

        assert captured, 'the degenerate branch should fall through to the queue'
        assert captured[0].lane == 'high'
        assert harness.scheduler.get_task.await_count <= 1, (
            'the degeneracy probe and the lane fallback must share ONE metadata '
            f'read; got {harness.scheduler.get_task.await_count} awaits'
        )

    async def test_explicit_lane_pays_no_metadata_read_at_all(self, tmp_path: Path):
        """An explicit argument makes the metadata irrelevant by precedence."""
        harness = _make_harness(metadata={'merge_lane': 'normal'})
        _, captured = await _submit(tmp_path, harness, lane='high')

        assert captured and captured[0].lane == 'high'
        harness.scheduler.get_task.assert_not_awaited()


# ---------------------------------------------------------------------------
# The audit echo — the submitter can see which lane it got
# ---------------------------------------------------------------------------


class TestSubmitResponseEchoesTheResolvedLane:
    """The submit response reports the resolved lane AND which source won.

    Design decision 4 puts the audit trail here rather than on the
    ``merge_queued`` event.  ``get_merge_queue`` already carries ``lane`` per
    queue item; what no existing surface can show is which SOURCE won, or
    anything at all about a submission that coalesced and so never became a
    queue item of its own.  Without this echo a caller that passed no ``lane``
    has no way to learn whether its task's ``metadata.merge_lane`` was
    honoured — the very blindness that let the key sit inert.
    """

    async def test_metadata_lane_is_echoed_with_its_source(self, tmp_path: Path):
        result = await _submit_nonblocking(
            tmp_path, _make_harness(metadata={'merge_lane': 'high'})
        )
        assert result.get('status') == 'queued', result
        assert result.get('lane') == 'high', result
        assert result.get('lane_source') == 'task_metadata', result

    async def test_explicit_argument_is_named_as_the_source(self, tmp_path: Path):
        """The metadata says ``'normal'`` and loses, so ``source`` is the tell.

        Both sources would otherwise be indistinguishable from the lane alone
        on a submission whose argument and metadata agree.
        """
        result = await _submit_nonblocking(
            tmp_path, _make_harness(metadata={'merge_lane': 'normal'}), lane='high'
        )
        assert result.get('lane') == 'high', result
        assert result.get('lane_source') == 'argument', result

    async def test_neither_source_reports_the_default(self, tmp_path: Path):
        """``'normal'`` alone cannot say whether anything asked for it."""
        result = await _submit_nonblocking(tmp_path, _make_harness(metadata={}))
        assert result.get('lane') == 'normal', result
        assert result.get('lane_source') == 'default', result

    async def test_queued_response_reports_the_lane_as_applied(self, tmp_path: Path):
        """On the dispatched arm the echo IS the effective lane."""
        result = await _submit_nonblocking(
            tmp_path, _make_harness(metadata={'merge_lane': 'high'})
        )
        assert result.get('status') == 'queued', result
        assert result.get('lane_applied') is True, result

    async def test_attached_response_reports_the_lane_as_not_applied(
        self, tmp_path: Path
    ):
        """A coalesced submission never becomes its own queue item.

        So this response is the only place its lane resolution is ever
        observable: ``get_merge_queue`` lists the in-flight entry, not the
        submission that attached to it.  The keys report what THIS submission
        resolved to; attaching does not move the in-flight entry between
        lanes, and ``lane_applied`` is the MACHINE-READABLE statement of that
        — without it a steward reading ``{status:'attached', lane:'high'}``
        is told their hotfix is high-lane when it is in fact riding a
        possibly-``'normal'`` in-flight entry.
        """
        registry = InFlightMergeRegistry()
        never: asyncio.Future = asyncio.get_running_loop().create_future()
        assert registry.acquire(
            '4888', 'other-task', never, request_id='mr-existing'
        ), 'prerequisite: the registry must accept the first acquire'
        try:
            result = await _submit_nonblocking(
                tmp_path,
                _make_harness(metadata={'merge_lane': 'high'}),
                registry=registry,
                lane='high',
            )
        finally:
            never.cancel()

        assert result.get('status') == 'attached', result
        assert result.get('lane') == 'high', result
        assert result.get('lane_source') == 'argument', result
        assert result.get('lane_applied') is False, result
