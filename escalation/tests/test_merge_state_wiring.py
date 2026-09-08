"""The escalation server emits the SHARED merge-state vocabulary.

Task 4829 (PRD ``plans/merge-status-durable-non-landed-prd.md``, D3).  This is
a BEHAVIOURAL test, not a source scan: it drives the real ``merge_status`` and
``merge_cancel`` MCP tools and asserts that the ``state`` they return is an
actual ``shared.merge_state.MergeState`` member drawn from the right partition.
A source scan could be satisfied by an import the code never reaches.

``shared.merge_state`` IS importable here even though ``escalation/tests/
conftest.py`` stubs ``sys.modules['shared']``: the stub sets ``__path__`` to
the real ``shared/src/shared`` directory, so submodules still load from disk —
only ``shared/__init__.py`` (which drags in aiosqlite) is skipped.  That works
precisely because this module is NOT re-exported from the package ``__init__``.

Harness patterns are taken wholesale from the existing suite: ``_stub_git_ops``
/ ``_make_config`` (test_merge_status_git_authority.py), ``_make_orch_config`` /
``_make_registry`` (test_server_chokepoint.py).  The three merge-tool invocation
wrappers, which the suite had re-typed verbatim in four modules, are imported
from ``_merge_tool_calls`` instead — see that module for why the copies were not
migrated here too.
"""
from __future__ import annotations

import asyncio
import json
import time
import types
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _merge_tool_calls import call_merge_cancel, call_merge_request, call_merge_status
from shared.merge_state import (
    CANCEL_STATES,
    EPISTEMIC_STATES,
    LIVE_STATES,
    POLL_STOP_STATES,
    TERMINAL_STATES,
    MergeState,
)

from escalation.queue import EscalationQueue
from escalation.server import create_server

# ---------------------------------------------------------------------------
# Cross-package orchestrator imports — mirrors test_server.py's guard.
# ---------------------------------------------------------------------------
try:
    from orchestrator.config import (  # type: ignore[reportMissingImports]
        GitConfig,
        OrchestratorConfig,
    )
    from orchestrator.merge_queue import (  # type: ignore[reportMissingImports]
        InFlightMergeRegistry,
    )
    from orchestrator.merge_types import MergeOutcome  # type: ignore[reportMissingImports]
    _ORCHESTRATOR_AVAILABLE = True
except ImportError:
    _ORCHESTRATOR_AVAILABLE = False
    GitConfig: Any = None  # type: ignore[assignment,misc]
    OrchestratorConfig: Any = None  # type: ignore[assignment,misc]
    InFlightMergeRegistry: Any = None  # type: ignore[assignment,misc]
    MergeOutcome: Any = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _live_worker(raw_state: str, request_id: str = 'mr-live') -> types.SimpleNamespace:
    """A duck-typed merge worker whose snapshot holds one entry in *raw_state*."""
    return types.SimpleNamespace(
        snapshot=lambda: {
            'entries': [
                {
                    'request_id': request_id,
                    'branch': 'b-live',
                    'task_id': 'T-live',
                    'state': raw_state,
                    'position': 0,
                    'enqueued_at': 123.0,
                }
            ]
        }
    )


def _ring(raw_state: str, request_id: str = 'mr-ring') -> types.SimpleNamespace:
    """A duck-typed terminal-retention ring holding one record in *raw_state*.

    ``finished_at`` must be an epoch FLOAT — ``_durable_terminal_state`` runs it
    through ``_epoch_to_iso8601``, which would raise on a string.
    """
    rec = types.SimpleNamespace(
        request_id=request_id, state=raw_state, finished_at=time.time() - 5.0
    )
    return types.SimpleNamespace(
        get=lambda rid: rec if rid == request_id else None,
        get_by_branch=lambda b: None,
        get_by_task=lambda t: None,
    )


def _server(tmp_path: Path, harness: Any = None, **kwargs: Any):
    return create_server(EscalationQueue(tmp_path / 'esc'), harness=harness, **kwargs)


def _assert_is_member(value: Any, partition: frozenset, label: str) -> None:
    """The core wiring assertion: a real enum member from the right partition."""
    assert isinstance(value, MergeState), (
        f'{label}: expected a MergeState member, got {type(value).__name__} {value!r}. '
        'The server must emit shared.merge_state.MergeState, not a bare str literal.'
    )
    assert value in partition, f'{label}: {value!r} is not in the expected partition'


# ---------------------------------------------------------------------------
# merge_status — Tier 1 (live snapshot)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMergeStatusTier1EmitsLiveStates:
    @pytest.mark.parametrize(
        ('raw', 'expected'),
        [
            ('queued', MergeState.queued),
            ('merging', MergeState.verifying),
            ('awaiting_verify', MergeState.verifying),
            ('verifying', MergeState.verifying),
            ('gate_reverify', MergeState.gate),
            ('finalizing', MergeState.finalizing),
        ],
    )
    async def test_recognised_raw_maps_to_a_live_member(
        self, tmp_path: Path, raw: str, expected: MergeState
    ) -> None:
        harness = types.SimpleNamespace(
            _merge_worker=_live_worker(raw), _terminal_retention=None
        )
        result = await call_merge_status(_server(tmp_path, harness), request_id='mr-live')

        _assert_is_member(result['state'], LIVE_STATES, f'Tier-1 raw={raw!r}')
        assert result['state'] == expected

    async def test_unrecognised_raw_still_passes_through(self, tmp_path: Path) -> None:
        """The deliberate fail-open — NOT a missed call site.

        ``_map_live_state`` ends ``return raw`` for worker states the server does
        not yet know.  Coercing that into a hard ``MergeState(raw)`` would turn a
        forward-compatible passthrough into a ValueError on a read-only probe
        path.  Task 4887 owns merge_status's degradation semantics; this test
        pins the passthrough so a later reader reads the surviving ``str``
        branch as deliberate.
        """
        harness = types.SimpleNamespace(
            _merge_worker=_live_worker('some_future_worker_state'), _terminal_retention=None
        )
        result = await call_merge_status(_server(tmp_path, harness), request_id='mr-live')

        assert result['state'] == 'some_future_worker_state'
        assert not isinstance(result['state'], MergeState)


# ---------------------------------------------------------------------------
# merge_status — Tiers 2-3 (durable) + Tier 4
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMergeStatusDurableTiersEmitTerminalStates:
    @pytest.mark.parametrize(
        ('raw', 'expected'),
        [
            ('done', MergeState.done),
            ('done_wip_recovery', MergeState.done),
            ('already_merged', MergeState.done),
            ('conflict', MergeState.conflict),
            ('abandoned', MergeState.abandoned),
            ('superseded', MergeState.superseded),
            ('wip_halted', MergeState.blocked),
            ('error', MergeState.blocked),
        ],
    )
    async def test_raw_collapses_to_a_terminal_member(
        self, tmp_path: Path, raw: str, expected: MergeState
    ) -> None:
        harness = types.SimpleNamespace(_merge_worker=None, _terminal_retention=_ring(raw))
        result = await call_merge_status(_server(tmp_path, harness), request_id='mr-ring')

        _assert_is_member(result['state'], TERMINAL_STATES, f'durable raw={raw!r}')
        # Collapse behaviour is UNCHANGED by the switchover.
        assert result['state'] == expected
        # The raw value survives verbatim on `outcome` — the collapse is lossy
        # by design and this is where the detail lives.
        assert result['outcome'] == raw


@pytest.mark.asyncio
class TestMergeStatusTier4EmitsUnknown:
    async def test_no_record_anywhere_returns_unknown_member(self, tmp_path: Path) -> None:
        """Tier 4's `unknown` is pinned to the POLL partitions, not the cancel one.

        ``POLL_STOP_STATES`` and ``CANCEL_STATES`` hold the same members today, so
        pinning this to ``CANCEL_STATES`` — as it did before this amendment —
        could not fail.  That equality is incidental: they are the stop sets of two
        different call sites, and PRD beta widens what ``merge_status`` may return
        without touching ``merge_cancel``.  In a file whose whole purpose is
        asserting the right partition per call site, a poll assertion checked
        against the cancel vocabulary would silently start checking the wrong
        contract the moment the two diverge.

        ``EPISTEMIC_STATES`` is the tighter of the two claims and is asserted as
        well: Tier 4 fires when the server has NO RECORD of the request, so what it
        reports is a state of knowledge, not an outcome the merge reached.
        """
        result = await call_merge_status(_server(tmp_path), request_id='mr-nothing')

        _assert_is_member(result['state'], POLL_STOP_STATES, 'Tier-4')
        assert result['state'] in EPISTEMIC_STATES, (
            f"Tier-4 returned {result['state']!r}, an OUTCOME state — a poll that "
            'found no record anywhere must report what the server knows, not what '
            'the merge did.'
        )
        assert result['state'] == MergeState.unknown


@pytest.mark.asyncio
@pytest.mark.skipif(not _ORCHESTRATOR_AVAILABLE, reason='orchestrator package not installed')
class TestMergeStatusTier35EmitsDone:
    async def test_found_on_main_returns_done_member(self, tmp_path: Path) -> None:
        tip, main_sha, citation = 'a' * 40, 'm' * 40, 'c' * 40
        stub_git = types.SimpleNamespace(
            resolve_branch_sha=AsyncMock(
                side_effect=lambda b: tip if b == 'task/123' else main_sha
            ),
            is_ancestor=AsyncMock(return_value=True),
            find_merge_marker=AsyncMock(return_value=None),
            find_task_citation_commit=AsyncMock(return_value=citation),
            commit_effect_present_in_main=AsyncMock(return_value=True),
        )
        harness = types.SimpleNamespace(
            _merge_worker=None, _terminal_retention=None, git_ops=stub_git
        )
        config = OrchestratorConfig(
            project_root=tmp_path,
            max_concurrent_tasks=1,
            git=GitConfig(
                main_branch='main',
                branch_prefix='task/',
                remote='origin',
                worktree_dir='.worktrees',
            ),
        )
        result = await call_merge_status(
            _server(tmp_path, harness, orch_config=config), task_id='123'
        )

        assert result.get('kind') == 'found_on_main', f'Tier 3.5 did not fire: {result}'
        _assert_is_member(result['state'], TERMINAL_STATES, 'Tier-3.5')
        assert result['state'] == MergeState.done


# ---------------------------------------------------------------------------
# merge_cancel — every branch-order path returns a CANCEL_STATES member
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.skipif(not _ORCHESTRATOR_AVAILABLE, reason='orchestrator package not installed')
class TestMergeCancelEmitsCancelStates:
    """``_waiters`` is a closure-local dict inside ``create_server`` — it cannot
    be injected from a test, so every path that needs a live waiter goes through
    ``merge_request`` first, as the existing suite does.

    Branches 2 and 3 depend on the ``_waiters.pop`` done-callback NOT having run
    yet, so the tool is fetched UP FRONT: any ``await server.get_tool(...)``
    between setup and the call yields the loop and destroys the window.
    """

    def _queue_server(self, tmp_path: Path):
        return create_server(
            EscalationQueue(tmp_path / 'esc'),
            merge_queue=asyncio.Queue(),
            orch_config=OrchestratorConfig(project_root=tmp_path / 'repo'),
            merge_inflight_registry=InFlightMergeRegistry(),
        )

    async def test_no_waiter_miss_returns_unknown_member(self, tmp_path: Path) -> None:
        result = await call_merge_cancel(_server(tmp_path), request_id='mr-never-seen')

        assert result['cancelled'] is False
        _assert_is_member(result['state'], CANCEL_STATES, 'merge_cancel no-waiter miss')
        assert result['state'] == MergeState.unknown

    async def test_no_waiter_durable_hit_returns_terminal_member(self, tmp_path: Path) -> None:
        harness = types.SimpleNamespace(
            _merge_worker=None, _terminal_retention=_ring('superseded', 'mr-gone')
        )
        result = await call_merge_cancel(_server(tmp_path, harness), request_id='mr-gone')

        assert result['cancelled'] is False
        _assert_is_member(result['state'], CANCEL_STATES, 'merge_cancel durable hit')
        # superseded is precisely the member DEFECT 1's docstring omitted.
        assert result['state'] == MergeState.superseded

    async def test_pending_waiter_returns_abandoned_member(self, tmp_path: Path) -> None:
        server = self._queue_server(tmp_path)
        submitted = await call_merge_request(
            server, task_id='c1', branch='c1', worktree=str(tmp_path / 'wt-c1'), wait_secs=0
        )
        result = await call_merge_cancel(server, request_id=submitted['request_id'])

        assert result['cancelled'] is True
        _assert_is_member(result['state'], CANCEL_STATES, 'merge_cancel pending')
        assert result['state'] == MergeState.abandoned

    async def test_mid_finalize_window_returns_coarse_terminal_member(
        self, tmp_path: Path
    ) -> None:
        mq = asyncio.Queue()
        server = create_server(
            EscalationQueue(tmp_path / 'esc2'),
            merge_queue=mq,
            orch_config=OrchestratorConfig(project_root=tmp_path / 'repo'),
            merge_inflight_registry=InFlightMergeRegistry(),
        )
        submitted = await call_merge_request(
            server, task_id='c2', branch='c2', worktree=str(tmp_path / 'wt-c2'), wait_secs=0
        )
        cancel_tool = await server.get_tool('merge_cancel')  # fetch BEFORE resolving
        req = mq.get_nowait()
        req.result.set_result(MergeOutcome(status='already_merged', reason='late resolve'))

        result = await cancel_tool.fn(request_id=submitted['request_id'])  # type: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]

        assert result['cancelled'] is False
        _assert_is_member(result['state'], CANCEL_STATES, 'merge_cancel mid-finalize')
        # already_merged is SUBMIT-only; it must collapse to the poll member done.
        assert result['state'] == MergeState.done

    async def test_excepted_future_returns_blocked_member(self, tmp_path: Path) -> None:
        mq = asyncio.Queue()
        server = create_server(
            EscalationQueue(tmp_path / 'esc'),
            merge_queue=mq,
            orch_config=OrchestratorConfig(project_root=tmp_path / 'repo'),
            merge_inflight_registry=InFlightMergeRegistry(),
        )
        submitted = await call_merge_request(
            server, task_id='c3', branch='c3', worktree=str(tmp_path / 'wt-c3'), wait_secs=0
        )
        cancel_tool = await server.get_tool('merge_cancel')  # fetch BEFORE resolving
        req = mq.get_nowait()
        req.result.set_exception(RuntimeError('worker blew up'))

        result = await cancel_tool.fn(request_id=submitted['request_id'])  # type: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]

        assert result['cancelled'] is False
        _assert_is_member(result['state'], CANCEL_STATES, 'merge_cancel excepted future')
        assert result['state'] == MergeState.blocked


# ---------------------------------------------------------------------------
# Wire compatibility — the switchover must be observationally invisible
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestWireCompatibility:
    """A StrEnum member serialises and compares as its plain str value, so no
    existing consumer — in-process or over the MCP wire — can tell the
    difference.  Asserted explicitly rather than assumed.
    """

    async def test_response_json_round_trips_to_a_plain_string(self, tmp_path: Path) -> None:
        harness = types.SimpleNamespace(_merge_worker=None, _terminal_retention=_ring('done'))
        result = await call_merge_status(_server(tmp_path, harness), request_id='mr-ring')

        round_tripped = json.loads(json.dumps(result))
        assert round_tripped['state'] == 'done'
        assert type(round_tripped['state']) is str

    async def test_plain_str_comparison_still_holds(self, tmp_path: Path) -> None:
        harness = types.SimpleNamespace(_merge_worker=None, _terminal_retention=_ring('conflict'))
        result = await call_merge_status(_server(tmp_path, harness), request_id='mr-ring')

        # The exact shape of every pre-existing assertion in the suite.
        assert result['state'] == 'conflict'
        assert result['state'] in ('conflict', 'blocked')
