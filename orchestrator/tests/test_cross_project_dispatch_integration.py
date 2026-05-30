"""Integration gate: cross-project dispatch behavior end-to-end (task 1581).

All boundary rows are exercised through acquire_next()'s returned TaskAssignment
(dispatch) or None (no dispatch) — never by peeking at task storage. This matches
the user-observable signal the scheduler exposes.

Test organisation (TDD steps S1-S12):
  S1/S2   — Resolver contract: TwoProjectMcpSession faithfully mirrors α's
             get_external_statuses (sentinels, normalization, batching).
  S3/S4   — Headline sequencing: dependent dispatched ONLY on tick AFTER
             upstream reaches 'done'; pending upstream → no dispatch/escalation.
  S5/S6   — Row 3: cancelled upstream → immediate human escalation + no dispatch.
  S7/S8   — Rows 4+5: unknown_project/unknown_task → grace-then-escalate at
             harness.config.max_external_dep_unresolved_cycles ticks.
  S9/S10  — Row 9: exactly one get_external_statuses call per tick regardless
             of task/dep fan-out (invariant 5).
  S11/S12 — Row 11: transient resolver error → fail-safe wait, no counter
             increment, no escalation; retry dispatches on next tick.

Design decisions (abbreviated):
  - Stage at Scheduler._mcp_session.call_tool (the injectable dispatch seam).
  - Observe dispatch via acquire_next()'s TaskAssignment, not by reading storage.
  - Drive real Harness with a real EscalationQueue so 'escalation raised' is a
    genuinely-filed, read-back L1 (not a mock assertion).
  - Read grace threshold from harness.config.max_external_dep_unresolved_cycles.
  - No production code is modified; all code is additive test-only.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from escalation.queue import EscalationQueue
from orchestrator.config import OrchestratorConfig
from orchestrator.harness import Harness


# ─────────────────────────────────────────────────────────────────────────────
# TwoProjectMcpSession
# ─────────────────────────────────────────────────────────────────────────────

class TwoProjectMcpSession:
    """In-process MCP session double for two-project integration tests.

    Reproduces α's get_external_statuses contract (PRD Contract
    §get_external_statuses) and provides get_tasks / set_task_status against
    the dependent project's live task tree.

    Modelled on _StubMcpSession (orchestrator/src/orchestrator/evals/runner.py:530)
    for the JSON-RPC envelope shape and call_tool signature.

    Attributes
    ----------
    dependent_tasks   list[dict]     Dependent project's live task tree.
    upstream_statuses dict[str,str]  Foreign project's task_id → status ('DB').
    known_projects    set[str]       Canonical underscore project IDs this double
                                     recognises (e.g. {'dark_factory', 'proj_a'}).
    ext_call_count    int            Cumulative count of get_external_statuses calls.
    raise_on_external bool           When True get_external_statuses raises
                                     RuntimeError to simulate a transient failure.
    """

    def __init__(self, known_projects: set[str]) -> None:
        self.dependent_tasks: list[dict] = []
        self.upstream_statuses: dict[str, str] = {}
        self.known_projects: set[str] = known_projects
        self.ext_call_count: int = 0
        self.raise_on_external: bool = False
        self._request_id: int = 0

    # ── envelope helpers ─────────────────────────────────────────────────────

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _envelope(self, text: str) -> dict:
        """Return a JSON-RPC 2.0 envelope with a single text content block."""
        return {
            'jsonrpc': '2.0',
            'id': self._next_id(),
            'result': {
                'content': [{'type': 'text', 'text': text}],
            },
        }

    # ── call_tool dispatcher ──────────────────────────────────────────────────

    async def call_tool(
        self, name: str, arguments: dict, timeout: float = 30
    ) -> dict:
        """Route a tool call and return a JSON-RPC 2.0 envelope.

        Supported tools:
        - ``get_tasks``            → live dependent_tasks list.
        - ``set_task_status``      → mutate matching dependent task in-place.
        - ``get_external_statuses``→ NOT YET IMPLEMENTED (raises NotImplementedError;
                                      implemented in step S2).
        """
        if name == 'get_tasks':
            return self._envelope(json.dumps({'tasks': self.dependent_tasks}))

        if name == 'set_task_status':
            task_id = arguments['id']
            status = arguments['status']
            for t in self.dependent_tasks:
                if str(t.get('id')) == task_id:
                    t['status'] = status
                    break
            return self._envelope(json.dumps({'id': task_id, 'status': status}))

        if name == 'get_external_statuses':
            raise NotImplementedError(
                'TwoProjectMcpSession.get_external_statuses not yet implemented '
                '— will be implemented in step S2 to make S1 GREEN'
            )

        raise NotImplementedError(
            f'TwoProjectMcpSession: unknown tool {name!r} — '
            'add a branch in call_tool if this tool is needed by the test'
        )


# ─────────────────────────────────────────────────────────────────────────────
# Shared constants
# ─────────────────────────────────────────────────────────────────────────────

# Canonical underscore-form project IDs recognised by the session double.
# The dependent project is the one whose task tree lives in dependent_tasks;
# the upstream project holds the foreign tasks referenced by external_deps.
_UPSTREAM_PROJECT = 'upstream_proj'
_KNOWN_PROJECTS = {'dep_proj', _UPSTREAM_PROJECT}


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures and helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_harness(tmp_path: Path) -> tuple[Harness, TwoProjectMcpSession]:
    """Build a Harness wired with a real EscalationQueue and a TwoProjectMcpSession.

    Injection points:
    - harness.scheduler._mcp_session = session  (routes all tool calls through it)
    - harness._escalation_queue = EscalationQueue(tmp_path/'escalations')
    """
    config = OrchestratorConfig(project_root=tmp_path)
    session = TwoProjectMcpSession(known_projects=_KNOWN_PROJECTS)
    harness = Harness(config)
    # Inject our in-process session into the scheduler dispatch seam.
    harness.scheduler._mcp_session = session
    # Wire a real EscalationQueue so escalations can be filed and read back.
    harness._escalation_queue = EscalationQueue(tmp_path / 'escalations')
    return harness, session


def register_pending_dependent_task(
    session: TwoProjectMcpSession,
    task_id: str,
    external_deps: list[str],
) -> dict:
    """Append a pending, dispatchable dependent task to session.dependent_tasks.

    The task is shaped so that:
    - status = 'pending'
    - dependencies = [] (no local deps — only external deps can block dispatch)
    - metadata.files = [f'cross_project/{task_id}.py'] (unique file per task so
      ModuleLockTable.try_acquire succeeds for each task independently)
    - metadata.external_deps = external_deps

    Returns the newly-added task dict.
    """
    task: dict = {
        'id': task_id,
        'title': f'Cross-project integration task {task_id}',
        'status': 'pending',
        'dependencies': [],
        'metadata': {
            'files': [f'cross_project/{task_id}.py'],
            'external_deps': list(external_deps),
        },
    }
    session.dependent_tasks.append(task)
    return task


async def run_tick(harness: Harness) -> str | None:
    """Run one acquire_next() tick; return dispatched task_id or None.

    Dispatch is OBSERVED through the returned TaskAssignment — never by reading
    task storage. This matches the task's 'dispatch path' requirement.
    """
    assignment = await harness.scheduler.acquire_next()
    return assignment.task_id if assignment is not None else None


def set_upstream_status(
    session: TwoProjectMcpSession, task_id: str, status: str
) -> None:
    """Set the status of a foreign (upstream) task in the session lookup store."""
    session.upstream_statuses[task_id] = status


def escalations_for(harness: Harness, task_id: str) -> list:
    """Return pending L1 escalations for task_id from the real EscalationQueue."""
    q = harness._escalation_queue
    if q is None:
        return []
    return q.get_by_task(task_id, status='pending')
