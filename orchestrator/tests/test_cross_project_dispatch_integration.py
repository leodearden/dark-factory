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
            return await self._get_external_statuses(arguments)

        raise NotImplementedError(
            f'TwoProjectMcpSession: unknown tool {name!r} — '
            'add a branch in call_tool if this tool is needed by the test'
        )

    async def _get_external_statuses(self, arguments: dict) -> dict:
        """Implement α's get_external_statuses contract (PRD Contract §get_external_statuses).

        Resolution matrix (mirrored from fused-memory/tests/test_get_external_statuses.py):
        - dep not parseable as '<project_id>:<int>' → 'malformed'
        - project_id (after hyphen→underscore normalization) not in known_projects
          → 'unknown_project'
        - project_id known but task_id absent from upstream_statuses → 'unknown_task'
        - otherwise → upstream_statuses[task_id]

        Increments ext_call_count exactly once per call (invariant 5).
        Raises RuntimeError when raise_on_external=True (transient-error simulation).
        """
        if self.raise_on_external:
            raise RuntimeError(
                'TwoProjectMcpSession: simulated transient resolver error'
            )

        self.ext_call_count += 1

        deps: list[str] = arguments.get('deps', [])
        statuses: dict[str, str] = {}

        for dep in deps:
            # Parse '<project_id>:<task_id>' — must have exactly one colon,
            # non-empty sides, and an integer task_id.
            if ':' not in dep:
                statuses[dep] = 'malformed'
                continue

            colon_idx = dep.index(':')
            raw_project = dep[:colon_idx]
            raw_task = dep[colon_idx + 1:]

            # Validate both sides are non-empty and task_id is a plain integer.
            if not raw_project or not raw_task:
                statuses[dep] = 'malformed'
                continue
            try:
                int(raw_task)   # must be a plain integer (no dots, no letters)
            except ValueError:
                statuses[dep] = 'malformed'
                continue

            # Normalize project_id: hyphen → underscore for registry lookup.
            normalized_project = raw_project.replace('-', '_')
            if normalized_project not in self.known_projects:
                statuses[dep] = 'unknown_project'
                continue

            # Look up task in the upstream (foreign) status store.
            task_status = self.upstream_statuses.get(raw_task)
            if task_status is None:
                statuses[dep] = 'unknown_task'
            else:
                statuses[dep] = task_status

        return self._envelope(json.dumps({'statuses': statuses}))


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


# ─────────────────────────────────────────────────────────────────────────────
# Helper: extract statuses from call_tool envelope
# ─────────────────────────────────────────────────────────────────────────────

async def _call_get_external_statuses(
    session: TwoProjectMcpSession, deps: list[str]
) -> dict:
    """Call session.call_tool('get_external_statuses', ...) and extract statuses.

    Parses the JSON-RPC text-content envelope, returns the inner 'statuses' dict.
    Raises NotImplementedError when the branch is not yet implemented (S1 RED phase).
    """
    result = await session.call_tool(
        'get_external_statuses', {'deps': deps}
    )
    text = result['result']['content'][0]['text']
    payload = json.loads(text)
    return payload['statuses']


# ─────────────────────────────────────────────────────────────────────────────
# S1 RED — TwoProjectMcpSession resolver CONTRACT (fidelity to α)
# ─────────────────────────────────────────────────────────────────────────────

class TestResolverContract:
    """Pin TwoProjectMcpSession.get_external_statuses to α's documented contract.

    All tests here call session.call_tool('get_external_statuses', ...) directly.
    They fail RED because get_external_statuses raises NotImplementedError until S2.

    Contract (mirrored from fused-memory/tests/test_get_external_statuses.py and
    PRD Contract §get_external_statuses):
    - Real status passed through verbatim (done / pending / cancelled).
    - 'nope:1' → 'unknown_project' (unrecognised project_id).
    - '<known>:999999' → 'unknown_task' (task absent from upstream_statuses).
    - Malformed dep strings → 'malformed'.
    - Hyphen-form project id resolves like underscore form; key is verbatim.
    - A single multi-dep call increments ext_call_count by exactly 1.
    """

    def _make_session(self) -> TwoProjectMcpSession:
        session = TwoProjectMcpSession(known_projects={_UPSTREAM_PROJECT})
        session.upstream_statuses = {
            '10': 'done',
            '20': 'pending',
            '30': 'cancelled',
        }
        return session

    @pytest.mark.asyncio
    async def test_real_statuses_passed_through(self) -> None:
        """Known project + known task → real status keyed by verbatim dep string."""
        session = self._make_session()
        statuses = await _call_get_external_statuses(
            session,
            [
                f'{_UPSTREAM_PROJECT}:10',
                f'{_UPSTREAM_PROJECT}:20',
                f'{_UPSTREAM_PROJECT}:30',
            ],
        )
        assert statuses[f'{_UPSTREAM_PROJECT}:10'] == 'done'
        assert statuses[f'{_UPSTREAM_PROJECT}:20'] == 'pending'
        assert statuses[f'{_UPSTREAM_PROJECT}:30'] == 'cancelled'

    @pytest.mark.asyncio
    async def test_unknown_project_sentinel(self) -> None:
        """Project not in known_projects → 'unknown_project' sentinel."""
        session = self._make_session()
        statuses = await _call_get_external_statuses(session, ['nope:1'])
        assert statuses['nope:1'] == 'unknown_project'

    @pytest.mark.asyncio
    async def test_unknown_task_sentinel(self) -> None:
        """Known project but absent task_id → 'unknown_task' sentinel."""
        session = self._make_session()
        statuses = await _call_get_external_statuses(
            session, [f'{_UPSTREAM_PROJECT}:999999']
        )
        assert statuses[f'{_UPSTREAM_PROJECT}:999999'] == 'unknown_task'

    @pytest.mark.asyncio
    @pytest.mark.parametrize('dep', [
        'garbage',                              # no colon at all
        f'{_UPSTREAM_PROJECT}:',                # empty task_id
        ':10',                                  # empty project_id
        f'{_UPSTREAM_PROJECT}:abc',             # non-numeric task_id
    ])
    async def test_malformed_dep_sentinel(self, dep: str) -> None:
        """Unparseable dep string → 'malformed' sentinel."""
        session = self._make_session()
        statuses = await _call_get_external_statuses(session, [dep])
        assert statuses[dep] == 'malformed', (
            f'Expected malformed sentinel for {dep!r}; got {statuses!r}'
        )

    @pytest.mark.asyncio
    async def test_hyphen_project_id_normalized(self) -> None:
        """Hyphen form of project_id resolves identically to underscore form.

        Key in the result must be the verbatim hyphen string.
        """
        hyphen_project = _UPSTREAM_PROJECT.replace('_', '-')
        session = self._make_session()
        statuses = await _call_get_external_statuses(
            session, [f'{hyphen_project}:10']
        )
        # Key is verbatim (hyphen form); value is the real status.
        assert statuses[f'{hyphen_project}:10'] == 'done', (
            f'Hyphen-form dep should resolve to done; got {statuses!r}'
        )

    @pytest.mark.asyncio
    async def test_single_call_increments_ext_call_count_by_one(self) -> None:
        """A single multi-dep call increments ext_call_count by exactly 1."""
        session = self._make_session()
        assert session.ext_call_count == 0
        await _call_get_external_statuses(
            session,
            [
                f'{_UPSTREAM_PROJECT}:10',
                f'{_UPSTREAM_PROJECT}:20',
                'nope:1',
            ],
        )
        assert session.ext_call_count == 1, (
            f'Expected ext_call_count==1 after one call; got {session.ext_call_count}'
        )


# ─────────────────────────────────────────────────────────────────────────────
# S3 RED — rows 1+2 headline sequencing
# ─────────────────────────────────────────────────────────────────────────────

class TestHeadlineSequencing:
    """Prove that a dependent task dispatches only AFTER its upstream goes 'done'.

    Boundary rows exercised:
    - Row 1 (done → dispatch): after upstream becomes done, acquire_next returns
      a TaskAssignment with the dependent task_id.
    - Row 2 (pending → wait): while upstream is pending, acquire_next returns None
      AND no escalation is filed (no action taken for a live but not-done upstream).

    Dispatch is observed via acquire_next()'s returned TaskAssignment, never by
    reading the task storage directly.
    """

    @pytest.mark.asyncio
    async def test_dependent_dispatched_only_tick_after_upstream_done(
        self, tmp_path: Path
    ) -> None:
        """Dependent task T dispatches only on the tick AFTER upstream N → done.

        Tick A: upstream N='pending' → acquire_next returns None, no escalation.
        set N='done'.
        Tick B: acquire_next returns a TaskAssignment with task_id == T.
        """
        harness, session = build_harness(tmp_path)
        upstream_task_id = '5'
        dep_string = f'{_UPSTREAM_PROJECT}:{upstream_task_id}'

        # Register a pending dependent task T with one external dep.
        register_pending_dependent_task(session, 'T', external_deps=[dep_string])
        # Upstream is initially pending.
        set_upstream_status(session, upstream_task_id, 'pending')

        # Tick A: upstream pending → no dispatch, no escalation (row 2).
        tick_a_result = await run_tick(harness)
        assert tick_a_result is None, (
            f'Tick A: upstream pending → must NOT dispatch; got {tick_a_result!r}'
        )
        assert escalations_for(harness, 'T') == [], (
            'Tick A: pending upstream must not produce any escalation (row 2)'
        )

        # Transition upstream to done.
        set_upstream_status(session, upstream_task_id, 'done')

        # Tick B: upstream done → T dispatched (row 1).
        tick_b_result = await run_tick(harness)
        assert tick_b_result == 'T', (
            f'Tick B: upstream done → must dispatch T; got {tick_b_result!r}'
        )


# ─────────────────────────────────────────────────────────────────────────────
# S5 RED — row 3: cancelled upstream → human escalation, no dispatch
# ─────────────────────────────────────────────────────────────────────────────

class TestCancelledUpstreamEscalation:
    """Row 3: cancelled upstream dep → immediate L1 escalation + task blocked.

    When an upstream dep reaches 'cancelled', the scheduler must:
    1. NOT dispatch the dependent task.
    2. Set the dependent task to 'blocked' (via set_task_status through the session).
    3. File exactly one pending L1 escalation whose summary starts with
       'EXTERNAL_DEP_CANCELLED'.
    This exercises the full chain:
      scheduler gate → _apply_external_dep_policy → _on_external_dep_block
      → harness._block_and_escalate_external_dep → real EscalationQueue.
    """

    @pytest.mark.asyncio
    async def test_cancelled_upstream_escalates_not_dispatched(
        self, tmp_path: Path
    ) -> None:
        """Cancelled external dep → T not dispatched, L1 filed, task blocked."""
        harness, session = build_harness(tmp_path)
        upstream_task_id = '7'
        dep_string = f'{_UPSTREAM_PROJECT}:{upstream_task_id}'

        register_pending_dependent_task(session, 'T', external_deps=[dep_string])
        set_upstream_status(session, upstream_task_id, 'cancelled')

        # One tick: cancelled dep must not dispatch T but must escalate.
        tick_result = await run_tick(harness)

        # (a) T must NOT be dispatched.
        assert tick_result is None, (
            f'Cancelled upstream → T must NOT be dispatched; got {tick_result!r}'
        )

        # (b) Exactly one pending L1 escalation for T.
        escs = escalations_for(harness, 'T')
        assert len(escs) == 1, (
            f'Expected exactly 1 pending L1 for T; got {len(escs)}: {escs!r}'
        )
        esc = escs[0]
        assert esc.level == 1, f'Expected level==1; got {esc.level}'
        assert 'EXTERNAL_DEP_CANCELLED' in esc.summary, (
            f'Expected EXTERNAL_DEP_CANCELLED prefix in summary; got {esc.summary!r}'
        )

        # (c) T's status in the dependent_tasks list must now be 'blocked'.
        t_status = next(
            (t['status'] for t in session.dependent_tasks if str(t.get('id')) == 'T'),
            None,
        )
        assert t_status == 'blocked', (
            f"Expected T.status=='blocked' after cancelled-dep policy; got {t_status!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# S7 RED — rows 4+5: unknown_project / unknown_task → grace-then-escalate
# ─────────────────────────────────────────────────────────────────────────────

class TestUnknownDepGraceThenEscalate:
    """Rows 4+5: sentinel deps escalate at the threshold, NOT before.

    Parametrized over:
    - 'unknown_project': dep references a project not in known_projects.
    - 'unknown_task': dep references a known project but an absent task_id.

    For ticks 1..threshold-1: acquire_next returns None AND no escalation is filed.
    On tick == threshold: exactly one pending L1 with 'EXTERNAL_DEP_UNRESOLVED' prefix.
    T is never dispatched.

    The threshold is read from harness.config.max_external_dep_unresolved_cycles
    (production default = 3, set in config.py:647) — not hard-coded.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize('scenario,dep_string', [
        ('unknown_project', 'nope:1'),
        ('unknown_task',    f'{_UPSTREAM_PROJECT}:999999'),
    ])
    async def test_unknown_dep_grace_then_escalate(
        self, tmp_path: Path, scenario: str, dep_string: str
    ) -> None:
        """Grace ticks elapse silently; escalation fires on tick == threshold."""
        harness, session = build_harness(tmp_path)
        threshold = harness.config.max_external_dep_unresolved_cycles

        register_pending_dependent_task(session, 'T', external_deps=[dep_string])
        # (No upstream_statuses entry — dep resolves to the sentinel.)

        # Ticks 1..threshold-1: no dispatch, no escalation (grace period).
        for tick in range(1, threshold):
            tick_result = await run_tick(harness)
            escs = escalations_for(harness, 'T')
            assert tick_result is None, (
                f'Scenario={scenario!r} tick {tick}: must NOT dispatch; got {tick_result!r}'
            )
            assert escs == [], (
                f'Scenario={scenario!r} tick {tick}: must NOT escalate before threshold '
                f'({tick}/{threshold - 1}); got {escs!r}'
            )

        # Tick == threshold: escalation fires.
        tick_result = await run_tick(harness)
        escs = escalations_for(harness, 'T')
        assert tick_result is None, (
            f'Scenario={scenario!r} tick {threshold}: T must NOT be dispatched; '
            f'got {tick_result!r}'
        )
        assert len(escs) == 1, (
            f'Scenario={scenario!r} tick {threshold}: expected 1 L1; '
            f'got {len(escs)}: {escs!r}'
        )
        esc = escs[0]
        assert esc.level == 1, f'Expected level==1; got {esc.level}'
        assert 'EXTERNAL_DEP_UNRESOLVED' in esc.summary, (
            f'Expected EXTERNAL_DEP_UNRESOLVED prefix; got {esc.summary!r}'
        )


# ─────────────────────────────────────────────────────────────────────────────
# S9 RED — row 9: one get_external_statuses call per tick (invariant 5)
# ─────────────────────────────────────────────────────────────────────────────

class TestOneExternalCallPerTick:
    """Invariant 5: exactly one get_external_statuses call per tick regardless of fan-out.

    Three pending dependent tasks whose external_deps collectively reference
    five DISTINCT foreign deps (mix of done/pending/unknown across one or more
    known upstream projects). A single acquire_next tick must issue exactly ONE
    batched get_external_statuses call.
    """

    @pytest.mark.asyncio
    async def test_one_external_call_per_tick_regardless_of_fanout(
        self, tmp_path: Path
    ) -> None:
        """One get_external_statuses call covers all five distinct deps in one tick."""
        harness, session = build_harness(tmp_path)

        # Set up upstream statuses: 3 real, 1 pending, 1 unknown (absent).
        set_upstream_status(session, '1', 'done')
        set_upstream_status(session, '2', 'done')
        set_upstream_status(session, '3', 'pending')

        # Three tasks whose deps collectively span five distinct strings.
        register_pending_dependent_task(
            session, 'T1',
            external_deps=[
                f'{_UPSTREAM_PROJECT}:1',   # done
                f'{_UPSTREAM_PROJECT}:2',   # done
            ],
        )
        register_pending_dependent_task(
            session, 'T2',
            external_deps=[
                f'{_UPSTREAM_PROJECT}:3',   # pending
                f'{_UPSTREAM_PROJECT}:4',   # unknown_task
            ],
        )
        register_pending_dependent_task(
            session, 'T3',
            external_deps=[
                'nope:9',                   # unknown_project
            ],
        )

        # Reset counter just before the tick being measured.
        session.ext_call_count = 0

        await run_tick(harness)

        assert session.ext_call_count == 1, (
            f'Invariant 5: expected exactly 1 get_external_statuses call per tick; '
            f'got {session.ext_call_count}'
        )
