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
from _recording_event_store import _RecordingEventStore
from escalation.queue import EscalationQueue

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventType
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
        self.malformed_external_result: bool = False
        self._request_id: int = 0
        # Payload of the most-recent get_external_statuses call (for assertion).
        self.last_ext_call_deps: list[str] | None = None

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

        if self.malformed_external_result:
            # Return an envelope whose payload is NOT a usable 'statuses' dict.
            # This simulates the non-dict/unparseable path in get_external_statuses
            # (the silent-empty-cache fall-through bug this task fixes).
            return self._envelope(json.dumps({'data': 'not-a-dict'}))

        deps: list[str] = arguments.get('deps', [])
        # Record the deps payload so tests can assert the full batched set.
        self.last_ext_call_deps = list(deps)
        self.ext_call_count += 1

        statuses: dict[str, str] = {}

        for dep in deps:
            # Parse '<project_id>:<task_id>' — must have exactly one colon,
            # non-empty sides, and an integer task_id.
            if ':' not in dep:
                statuses[dep] = 'malformed'
                continue

            # Split on the FIRST colon only (index, not split).
            # Multi-colon deps (e.g. 'proj:1:2') yield raw_task='1:2', which
            # fails int() → 'malformed'. This edge case is intentionally out
            # of scope for the double; α's own tests cover any divergence.
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

        # Flat shape: producer returns a bare {dep: status} dict — no 'statuses' wrapper.
        # Mirrors fused-memory tools.py get_external_statuses: `return result` (line 2231).
        return self._envelope(json.dumps(statuses))


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
    """Call session.call_tool('get_external_statuses', ...) and return the status dict.

    Parses the JSON-RPC text-content envelope.  The producer returns a FLAT
    {dep: status} dict — no 'statuses' wrapper (mirrors the real tool contract;
    see TwoProjectMcpSession._get_external_statuses).
    Raises NotImplementedError when the branch is not yet implemented (S1 RED phase).
    """
    result = await session.call_tool(
        'get_external_statuses', {'deps': deps}
    )
    text = result['result']['content'][0]['text']
    # Flat shape: payload IS the {dep: status} dict (no 'statuses' wrapper).
    return json.loads(text)


# ─────────────────────────────────────────────────────────────────────────────
# S1 RED — TwoProjectMcpSession resolver CONTRACT (fidelity to α)
# ─────────────────────────────────────────────────────────────────────────────

class TestResolverContract:
    """Smoke-test: TwoProjectMcpSession.get_external_statuses wires through correctly.

    These are minimal 'double wires through' checks — not an exhaustive copy of
    α's contract. Exhaustive sentinel, normalization, and edge-case coverage lives
    in fused-memory/tests/test_get_external_statuses.py (the real α tool). A full
    parallel copy here would silently drift from production if α's contract changes.
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
    async def test_sentinel_wiring_and_call_count(self) -> None:
        """Sentinel routing wires through and ext_call_count increments once per call.

        Checks one passthrough (real status) and one sentinel (unknown_project)
        in a single call to confirm the double routes both branches correctly.
        Exhaustive sentinel cases (unknown_task, malformed, hyphen-normalization)
        are covered by fused-memory/tests/test_get_external_statuses.py.
        """
        session = self._make_session()
        assert session.ext_call_count == 0
        statuses = await _call_get_external_statuses(
            session,
            [f'{_UPSTREAM_PROJECT}:10', 'nope:1'],
        )
        assert statuses[f'{_UPSTREAM_PROJECT}:10'] == 'done'
        assert statuses['nope:1'] == 'unknown_project'
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

        # Reset counter and payload capture just before the tick being measured.
        session.ext_call_count = 0
        session.last_ext_call_deps = None

        await run_tick(harness)

        assert session.ext_call_count == 1, (
            f'Invariant 5: expected exactly 1 get_external_statuses call per tick; '
            f'got {session.ext_call_count}'
        )

        # The single call must have queried ALL five distinct deps — not just the
        # deps from one task or a partial subset. A regression where production
        # batches incorrectly (e.g. only collects deps from the first task) would
        # still emit one call but with an incomplete payload.
        expected_deps = {
            f'{_UPSTREAM_PROJECT}:1',
            f'{_UPSTREAM_PROJECT}:2',
            f'{_UPSTREAM_PROJECT}:3',
            f'{_UPSTREAM_PROJECT}:4',
            'nope:9',
        }
        assert session.last_ext_call_deps is not None, (
            'last_ext_call_deps is None — get_external_statuses was never called'
        )
        assert set(session.last_ext_call_deps) == expected_deps, (
            f'Invariant 5: the single call must cover all 5 distinct deps; '
            f'got {set(session.last_ext_call_deps)!r}, expected {expected_deps!r}'
        )


# ─────────────────────────────────────────────────────────────────────────────
# S11 RED — row 11: transient resolver error → fail-safe wait, no counter,
#           no escalation; retry dispatches on next tick (invariant 6)
# ─────────────────────────────────────────────────────────────────────────────

class TestTransientResolverError:
    """Row 11 / invariant 6: transient get_external_statuses error → fail-safe wait.

    When the resolver raises (session.raise_on_external=True):
    - T is NOT dispatched (acquire_next returns None).
    - No escalation is filed (fail-safe wait, not a policy decision).
    - scheduler._external_unresolved_counts has NO entry for (T, dep) — the grace
      counter must NOT be incremented on a transient error tick.

    After clearing raise_on_external (Tick B), the upstream is already 'done', so
    acquire_next must dispatch T immediately (no wasted grace ticks).
    """

    @pytest.mark.asyncio
    async def test_transient_resolver_error_waits_no_counter_then_retries(
        self, tmp_path: Path
    ) -> None:
        """Transient resolver error: no dispatch, no counter, no escalation; retry dispatches."""
        harness, session = build_harness(tmp_path)
        upstream_task_id = '42'
        dep_string = f'{_UPSTREAM_PROJECT}:{upstream_task_id}'

        register_pending_dependent_task(session, 'T', external_deps=[dep_string])
        # Upstream is already done — only the transient error blocks dispatch.
        set_upstream_status(session, upstream_task_id, 'done')

        # Inject transient resolver failure.
        session.raise_on_external = True

        # Tick A: resolver raises → fail-safe wait.
        tick_a_result = await run_tick(harness)

        # (a) T must NOT be dispatched.
        assert tick_a_result is None, (
            f'Transient error: T must NOT be dispatched on Tick A; got {tick_a_result!r}'
        )

        # (b) No escalation filed (fail-safe wait, not a policy escalation).
        assert escalations_for(harness, 'T') == [], (
            'Transient error: must NOT file any escalation on Tick A'
        )

        # (c) Grace counter must NOT be incremented on a transient error tick.
        count_key = ('T', dep_string)
        assert count_key not in harness.scheduler._external_unresolved_counts, (
            f'Transient error: _external_unresolved_counts must NOT have an entry '
            f'for {count_key!r} after a transient error tick; '
            f'got {harness.scheduler._external_unresolved_counts!r}'
        )

        # Clear the transient error → resolver will answer on next call.
        session.raise_on_external = False

        # Tick B: resolver answers, upstream is done → T dispatches immediately.
        tick_b_result = await run_tick(harness)
        assert tick_b_result == 'T', (
            f'After clearing transient error: upstream done → T must dispatch; '
            f'got {tick_b_result!r}'
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestMalformedExternalResult (task 1799 — step-9 regression)
# ─────────────────────────────────────────────────────────────────────────────

class TestMalformedExternalResult:
    """Regression: non-dict/malformed resolver result → fail-safe wait, NOT silent strand.

    Proves the combined behavior of tasks 1799 and 1855:
    - When the resolver returns a non-dict payload, the scheduler fails LOUD
      (WARNING logged, ExternalResolverError raised internally) and fails SAFE
      (task NOT dispatched even though upstream dep is actually 'done').
    - No sentinel counter entry, no escalation on early (pre-threshold) ticks.
    - EXTERNAL_DEP_RESOLVER_DEGRADED L1 escalation filed after threshold
      consecutive malformed ticks; task blocked. No external_dep_gate_held event
      emitted on this path (task 1855 intentionally suppresses it when a real
      _on_external_dep_block callback is installed).
    - Recovery: re-pending T + healthy resolver → T dispatches immediately.

    This is the exact failure mode that stranded reify task 4635 for hours — the
    old code fell through to ``return {}, None`` and the task was silently stranded
    with zero events/escalations/logs.
    """

    @pytest.mark.asyncio
    async def test_malformed_resolver_fails_safe_and_recovers(
        self, tmp_path: Path, caplog
    ) -> None:
        """Non-dict resolver: fail-safe wait + WARNING + escalation-blocked + recovery dispatch."""
        import logging

        harness, session = build_harness(tmp_path)
        # Inject a recording event store so we can assert ABSENCE of gate_held events.
        rec = _RecordingEventStore()
        harness.scheduler.event_store = rec  # type: ignore[assignment]

        upstream_task_id = '42'
        dep_string = f'{_UPSTREAM_PROJECT}:{upstream_task_id}'

        register_pending_dependent_task(session, 'T', external_deps=[dep_string])
        # Upstream is actually done — only the malformed resolver blocks dispatch.
        set_upstream_status(session, upstream_task_id, 'done')

        threshold = harness.scheduler.config.max_external_dep_unresolved_cycles

        # ── Tick A: malformed resolver result ────────────────────────────────
        session.malformed_external_result = True

        with caplog.at_level(logging.WARNING, logger='orchestrator.scheduler'):
            tick_a_result = await run_tick(harness)

        # (a) Task T must NOT be dispatched — fail-safe wait, not silent strand.
        assert tick_a_result is None, (
            f'Malformed resolver: T must NOT be dispatched on Tick A (fail-safe wait); '
            f'got {tick_a_result!r}'
        )

        # (b) A WARNING must have been emitted from get_external_statuses.
        scheduler_warnings = [
            r.message for r in caplog.records
            if r.levelno >= logging.WARNING
        ]
        assert scheduler_warnings, (
            'Malformed resolver: at least one WARNING must be logged on Tick A; '
            f'got records: {[r.message for r in caplog.records]!r}'
        )

        # (c) Sentinel counter must NOT be bumped — degraded ticks are fail-safe.
        count_key = ('T', dep_string)
        assert count_key not in harness.scheduler._external_unresolved_counts, (
            f'Malformed resolver: _external_unresolved_counts must NOT have entry '
            f'{count_key!r}; got {harness.scheduler._external_unresolved_counts!r}'
        )

        # (d) No escalation filed yet — a degraded resolver is not a policy escalation
        #     until the threshold is reached.
        assert escalations_for(harness, 'T') == [], (
            'Malformed resolver: must NOT file any escalation on Tick A'
        )

        # ── Ticks B…(threshold): accumulate degraded count → escalation on threshold tick ──
        for _ in range(threshold - 1):
            tick_result = await run_tick(harness)
            assert tick_result is None, (
                'Malformed resolver: T must remain un-dispatched during degraded ticks'
            )

        # ── After threshold malformed ticks: escalation filed, T blocked ────────────
        # (e) Exactly one pending L1 escalation with infra_issue / EXTERNAL_DEP_RESOLVER_DEGRADED.
        escs = escalations_for(harness, 'T')
        assert len(escs) == 1, (
            f'After {threshold} malformed ticks: expected exactly 1 pending L1 for T; '
            f'got {len(escs)}: {escs!r}'
        )
        esc = escs[0]
        assert esc.level == 1, f'Expected level==1; got {esc.level}'
        assert esc.category == 'infra_issue', (
            f'Expected category=="infra_issue"; got {esc.category!r}'
        )
        assert 'EXTERNAL_DEP_RESOLVER_DEGRADED' in esc.summary, (
            f'Expected EXTERNAL_DEP_RESOLVER_DEGRADED in summary; got {esc.summary!r}'
        )

        # (f) T's status must be 'blocked' after the threshold-escalation path.
        t_status = next(
            (t['status'] for t in session.dependent_tasks if str(t.get('id')) == 'T'),
            None,
        )
        assert t_status == 'blocked', (
            f"Expected T.status=='blocked' after threshold degraded-resolver policy; "
            f"got {t_status!r}"
        )

        # (g) Inverse guard: NO external_dep_gate_held event recorded.
        #     Task 1855 deliberately suppresses gate_held on the terminal-escalation path
        #     when a real _on_external_dep_block callback is installed (scheduler.py:1969-1999).
        #     This assertion encodes that product decision as a regression guard.
        gate_held_events = [
            e for e in rec.events
            if e[0] == str(EventType.external_dep_gate_held)
        ]
        assert gate_held_events == [], (
            f'After {threshold} malformed ticks with real callback: external_dep_gate_held '
            f'must NOT be emitted (task 1855 suppresses it on the escalation path); '
            f'got {gate_held_events!r}'
        )

        # ── Recovery: re-pend T manually, fix resolver → T dispatches ───────────────
        # Task 1855 sets T to 'blocked'; re-pending mirrors the documented recovery path
        # ("manually set the task back to pending" in _block_and_escalate_external_dep).
        # has_open_l1 is NOT consulted for dispatch eligibility, so T dispatches even
        # with the open L1 still queued.
        for t in session.dependent_tasks:
            if str(t.get('id')) == 'T':
                t['status'] = 'pending'
                break
        session.malformed_external_result = False

        tick_recovery = await run_tick(harness)
        assert tick_recovery == 'T', (
            f'After re-pending T and resolver recovery: upstream done → T must dispatch; '
            f'got {tick_recovery!r}'
        )
