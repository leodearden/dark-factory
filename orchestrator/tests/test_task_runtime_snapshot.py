"""B+H integration gate — producer leg (B1-B4).

Task 2638 (epsilon), PRD ``plans/dashboard-task-runtime-endpoint-prd.md``.
Drives ``orchestrator.task_runtime.build_task_runtime_snapshot`` (task alpha,
2634) END-TO-END through the REAL escalation ``get_task_runtime_state`` MCP
tool (task beta, 2635) — never a ``SimpleNamespace`` stub of the accessor's
output, and never the raw accessor called in isolation from the tool.

This closes a seam neither leg's own unit tests exercise alone:

- ``escalation/tests/test_task_runtime_state_tool.py`` stubs the harness
  with ``types.SimpleNamespace`` stand-ins for ``TaskRuntimeState`` and never
  touches real disk.
- ``orchestrator/tests/test_task_runtime.py`` calls
  ``build_task_runtime_snapshot`` directly and never goes through the tool.

Every test below builds a REAL tmp git repo with real ``.lane-state`` /
``.task-meta`` artifacts (reusing ``test_task_runtime.py``'s fixture
builders — the established cross-test-module reuse pattern
``test_harness_task_runtime.py`` already uses), wraps
``build_task_runtime_snapshot`` in a fixture harness, and drives the actual
``get_task_runtime_state`` MCP tool end-to-end — asserting on the wire-JSON
dict it emits.

No product code (alpha/beta/gamma/delta) is modified — all four are already
merged and green; this is a pure characterization/integration suite (every
test passes on arrival).
"""

from __future__ import annotations

import asyncio
import types
from pathlib import Path

import pytest
from escalation.queue import EscalationQueue
from escalation.server import create_server
from test_task_runtime import _init_repo, _make_task_artifacts, _warm_config

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps
from orchestrator.lane_lifecycle import LaneState
from orchestrator.task_runtime import build_task_runtime_snapshot

# ---------------------------------------------------------------------------
# Repo fixture — mirrors test_task_runtime.py's git_repo fixture (redefined
# rather than cross-module-imported: conftest.py provides no shared git_repo
# fixture, matching that module's own self-contained convention).
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Temporary git repo with an initial commit + a resolvable warm base."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    default_base = repo / '.worktrees' / '_merge-verify' / 'target'
    default_base.mkdir(parents=True, exist_ok=True)
    (default_base / '.keep').write_text('warm base sentinel\n')
    (repo / '.worktrees' / '.pool-root').touch()
    return repo


# ---------------------------------------------------------------------------
# The alpha -> beta wire rope, reused by every B1-B4 test below.
# ---------------------------------------------------------------------------


def _tool_runtime_tasks(git_ops: GitOps, tmp_path: Path) -> tuple[dict[int, dict], bool]:
    """Drive the REAL get_task_runtime_state MCP tool end-to-end against *git_ops*.

    Wires a fixture harness whose ``task_runtime_snapshot()`` calls the real
    ``build_task_runtime_snapshot(git_ops=...)`` — so this ropes alpha's
    real-disk read (``LaneLifecycle`` + ``TaskArtifacts``) THROUGH beta's
    tool projection (``_project_task_runtime_entry`` +
    ``TaskRuntimeSnapshot.model_dump``) onto the wire: the seam neither leg's
    own unit tests exercise alone.

    Returns ``(by_task_id, offline)``: *by_task_id* maps each wire entry's
    ``task_id`` to its full projected dict (``has_worktree``/``loops``/
    ``attempts``/``started``/``lane``/``phase``/``lane_state``/``error``);
    *offline* is the envelope's top-level ``offline`` flag (always ``False``
    as emitted server-side — the dashboard synthesizes ``True`` client-side
    when the escalation server itself is unreachable).
    """
    fixture = types.SimpleNamespace(
        task_runtime_snapshot=lambda: build_task_runtime_snapshot(git_ops=git_ops, event_store=None),
    )
    server = create_server(EscalationQueue(tmp_path / 'esc'), harness=fixture)
    tool = asyncio.run(server.get_tool('get_task_runtime_state'))
    result = tool.fn()
    return {t['task_id']: t for t in result['tasks']}, result['offline']


# ---------------------------------------------------------------------------
# B1 — Pooled task round-trip
# ---------------------------------------------------------------------------


class TestB1PooledRoundtrip:
    def test_pooled_task_round_trips_through_the_tool(self, git_repo: Path, tmp_path: Path):
        """A pooled, ASSIGNED lane's real .lane-state + .task-meta artifacts
        round-trip through the real tool onto the wire.
        """
        git_ops = GitOps(_warm_config(), git_repo, warm_lane_pool_size=1)
        assert git_ops.pool_in_use()
        worktree_base = git_ops.worktree_base
        lifecycle = git_ops._lane_lifecycle

        lifecycle.note_assigned('_lane-3', task_id='42')
        ta = _make_task_artifacts(worktree_base, '_lane-3', '42')
        ta.write_plan({'steps': [{'id': 's1', 'status': 'done'}]})
        ta.append_iteration_log({'note': 'iter-1'})
        ta.append_iteration_log({'note': 'iter-2'})
        ta.append_iteration_log({'note': 'iter-3'})
        ta.write_review('reviewer-a', {'verdict': 'PASS', 'issues': []})

        by_task, offline = _tool_runtime_tasks(git_ops, tmp_path)

        assert offline is False
        entry = by_task[42]
        assert entry['loops'] == 3
        assert entry['attempts'] == 1
        assert entry['lane'] == '_lane-3'
        assert entry['lane_state'] == 'assigned'
        assert entry['has_worktree'] is True
        assert entry['error'] is None


# ---------------------------------------------------------------------------
# B2 — Non-pooled task
# ---------------------------------------------------------------------------


class TestB2NonPooled:
    def test_non_pooled_task_derives_phase_with_null_lane(self, git_repo: Path, tmp_path: Path):
        """A non-pooled (per-task-worktree) layout round-trips with lane/
        lane_state null and phase derived from plan.json's mixed steps — the
        accessor abstracts pooled vs non-pooled and the tool passes that
        through unchanged.
        """
        git_ops = GitOps(GitConfig(), git_repo)
        assert not git_ops.pool_in_use()
        worktree_base = git_ops.worktree_base

        ta = _make_task_artifacts(worktree_base, '42', '42')
        ta.write_plan({'steps': [
            {'id': 's1', 'status': 'done'},
            {'id': 's2', 'status': 'pending'},
        ]})
        ta.append_iteration_log({'note': 'iter-1'})
        ta.append_iteration_log({'note': 'iter-2'})

        by_task, offline = _tool_runtime_tasks(git_ops, tmp_path)

        assert offline is False
        entry = by_task[42]
        assert entry['loops'] == 2
        assert entry['lane'] is None
        assert entry['lane_state'] is None
        assert entry['phase'] == 'EXECUTE'


# ---------------------------------------------------------------------------
# B3 — Quarantined lane surfaced
# ---------------------------------------------------------------------------


class TestB3Quarantined:
    def test_quarantined_lane_surfaces_lane_state(self, git_repo: Path, tmp_path: Path):
        """A lane transitioned to QUARANTINED surfaces lane_state=='quarantined'
        end-to-end — the state the OLD dashboard hand reader (pre-2636) was
        blind to.
        """
        git_ops = GitOps(_warm_config(), git_repo, warm_lane_pool_size=1)
        lifecycle = git_ops._lane_lifecycle
        lifecycle.note_assigned('_lane-5', task_id='43')
        lifecycle.transition('_lane-5', LaneState.QUARANTINED)

        by_task, offline = _tool_runtime_tasks(git_ops, tmp_path)

        assert offline is False
        entry = by_task[43]
        assert entry['lane_state'] == 'quarantined'
        assert entry['lane'] == '_lane-5'


# ---------------------------------------------------------------------------
# B4 — Honest zero
# ---------------------------------------------------------------------------


class TestB4HonestZero:
    def test_empty_iteration_log_is_honest_zero_not_missing(self, git_repo: Path, tmp_path: Path):
        """A genuinely-not-iterated task (empty iterations.jsonl, no read
        failure) reports an honest integer 0 end-to-end — never a missing/
        None/dropped entry (the structured-facts boundary).
        """
        git_ops = GitOps(_warm_config(), git_repo, warm_lane_pool_size=1)
        worktree_base = git_ops.worktree_base
        lifecycle = git_ops._lane_lifecycle
        lifecycle.note_assigned('_lane-7', task_id='44')
        _make_task_artifacts(worktree_base, '_lane-7', '44')
        empty_log = TaskArtifacts.meta_root_for(worktree_base, '_lane-7') / 'iterations.jsonl'
        empty_log.write_text('')

        by_task, offline = _tool_runtime_tasks(git_ops, tmp_path)

        assert offline is False
        entry = by_task[44]
        assert entry['loops'] == 0
        assert entry['loops'] is not None
        assert entry['attempts'] == 0
        assert entry['error'] is None
