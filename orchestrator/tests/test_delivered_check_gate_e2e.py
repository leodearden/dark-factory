"""End-to-end integration gate for the delivered-check dep-gate
(capability-delivered-checks PRD, ``plans/capability-delivered-checks-prd.md``,
task zeta) driven through the REAL (unmocked) git runner.

Complements ``fused-memory/tests/test_delivered_checks_e2e.py`` (the
cross-cutting stamp -> gate headline, which needs the fused-memory backend).
This file cannot import ``fused_memory`` (orchestrator/pyproject.toml does
not add ../fused-memory/src to pythonpath) — it drives delta/epsilon
(``orchestrator.scheduler``/``orchestrator.harness``) directly against a
real temp git repo with hand-authored producer metadata (the same
``metadata.delivered_checks`` shape gamma's ``commit_planning`` stamps),
using the UNMOCKED ``orchestrator.delivered_checks.run_delivered_check`` +
``Scheduler._resolve_main_sha`` (real ``git grep``/``git rev-parse``
subprocess calls) — new coverage vs. ``test_delivered_check_gate.py``,
whose unit tests fake both.

Boundary rows owned by this file (see the PRD's Boundary-test sketch):
- rows 3 + 4 (real git grep: token absent -> withhold + held event; token
  committed to main -> dispatch)
- row 7 (script-kind runner ERROR via a real missing-script OSError ->
  fail-safe wait, then real recovery once the script exists)
- row 10 (a CANCELLED producer carrying a failing check is gated exactly
  like a done one)

Rig pattern adapted from
``orchestrator/tests/test_cross_project_dispatch_integration.py``: a
backend-free MCP session double (``_LocalDepMcpSession``) serving
``get_tasks``/``set_task_status`` over an in-memory local-dep task list,
a real ``Harness`` with a real ``EscalationQueue``, and dispatch observed
exclusively through ``acquire_next()``'s returned ``TaskAssignment`` / None
-- never by reading scheduler internals for the primary assertions.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from _recording_event_store import _RecordingEventStore
from escalation.queue import EscalationQueue

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.harness import Harness

# ─────────────────────────────────────────────────────────────────────────────
# Import-resolution smoke test (prerequisite pre-1)
# ─────────────────────────────────────────────────────────────────────────────


def test_imports_resolve_without_fused_memory():
    """Prerequisite pre-1: this file imports orchestrator + escalation only
    (no fused_memory — orchestrator/pyproject.toml has no path to it)."""
    assert EscalationQueue is not None
    assert OrchestratorConfig is not None
    assert EventType is not None
    assert Harness is not None


# ─────────────────────────────────────────────────────────────────────────────
# Row 3+4 fixture constants (real `git grep` gate)
# ─────────────────────────────────────────────────────────────────────────────

_CAP_NAME_34 = 'row34_cap'
_CAP_TOKEN_34 = 'ROW34_CAPABILITY_TOKEN_V1'
_MARKER_REL_PATH_34 = 'src/row34_marker.py'


# ─────────────────────────────────────────────────────────────────────────────
# Rig helpers (backend-free: no fused_memory import)
# ─────────────────────────────────────────────────────────────────────────────


def _run_git(project_root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ['git', '-C', str(project_root), *args],
        check=True, capture_output=True, text=True,
    )


def _init_git_repo(root: Path, *, marker_rel_path: str | None = None) -> Path:
    """Initialize a real git repo at *root* on branch main with an initial
    commit, so ``main`` resolves before any row-specific fixture files are
    added — the delivered-check gate's git runner
    (``Scheduler._resolve_main_sha`` / ``run_delivered_check``) shells out
    against *root*, so a missing/unresolvable ``main`` would silently
    fail-safe the whole sweep to an empty cache (no held event, no streak
    — indistinguishable from "nothing checked").

    When *marker_rel_path* is given, the initial commit ALSO seeds that
    path with placeholder content (no capability token yet). This keeps a
    later grep-kind check's "token absent" state a clean no-match
    (``git grep`` rc==1 -> FAILED, the row-4 withhold) rather than a
    pathspec-not-found error (rc>=2 -> ERRORED, row 7's fail-safe wait) —
    mirrors ``fused-memory/tests/test_delivered_checks_e2e.py``'s
    ``_init_git_repo``. Omit for script-kind fixtures (row 7), which
    intentionally want the real ERRORED path from a genuinely missing
    script.
    """
    subprocess.run(
        ['git', 'init', '-b', 'main', str(root)],
        check=True, capture_output=True, text=True,
    )
    _run_git(root, 'config', 'user.email', 'e2e-test@example.com')
    _run_git(root, 'config', 'user.name', 'E2E Test')
    seeded_rel = marker_rel_path or 'README.md'
    seeded = root / seeded_rel
    seeded.parent.mkdir(parents=True, exist_ok=True)
    seeded.write_text(
        '# marker file -- capability token lands here later\n'
        if marker_rel_path else '# e2e fixture repo\n',
        encoding='utf-8',
    )
    _run_git(root, 'add', seeded_rel)
    _run_git(root, 'commit', '-m', 'initial commit')
    return root


def _commit_marker(project_root: Path, rel_path: str, token: str) -> str:
    """Append *token* to the marker file at *rel_path* (seeded by
    ``_init_git_repo``) and commit it to branch main, advancing the SHA so
    the delivered-check gate's stale-cache prune self-heals the withheld
    dependent on the very next tick. Returns the new main SHA.
    """
    target = project_root / rel_path
    target.write_text(
        target.read_text(encoding='utf-8') + f'{token}\n', encoding='utf-8',
    )
    _run_git(project_root, 'add', rel_path)
    _run_git(project_root, 'commit', '-m', f'land capability token {token}')
    return _run_git(project_root, 'rev-parse', 'main').stdout.strip()


def _write_exec_script(project_root: Path, rel_path: str, body: str) -> None:
    """Write *body* to *rel_path* under *project_root* and mark it
    executable (0o755) — the real recovery leg of row 7: a missing script
    produces a genuine ``FileNotFoundError`` from the subprocess spawn
    (-> ``DeliveredCheckResult.ERRORED``); writing + chmod-ing it here makes
    the real (unmocked) runner succeed on the very next tick. Deliberately
    NOT committed to git — the script kind is evaluated against the WORKING
    CHECKOUT, not the committed ``main`` tree (unlike the grep kind; see
    ``orchestrator.delivered_checks``'s module docstring).
    """
    target = project_root / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding='utf-8')
    os.chmod(target, 0o755)


def _grep_check(name: str, token: str, paths: list[str]) -> dict:
    """Build a grep-kind ``delivered_checks`` entry — the same shape
    gamma's ``commit_planning`` stamps from a capability-manifest sidecar
    (``fused-memory/tests/test_manifest_stamping.py``)."""
    return {'name': name, 'kind': 'grep', 'pattern': token, 'expect': 'present', 'paths': paths}


def _script_check(name: str, script_rel_path: str, *, timeout_secs: float = 5) -> dict:
    """Build a script-kind ``delivered_checks`` entry — the same shape
    gamma's ``commit_planning`` stamps from a capability-manifest sidecar's
    script capability."""
    return {
        'name': name, 'kind': 'script', 'script': script_rel_path,
        'args': [], 'timeout_secs': timeout_secs,
    }


class _LocalDepMcpSession:
    """Backend-free MCP session double serving a SINGLE project's local-dep
    task list (no fused_memory import — orchestrator/pyproject.toml has no
    path to it). Modelled on ``TwoProjectMcpSession``
    (test_cross_project_dispatch_integration.py) for the JSON-RPC envelope
    shape and call_tool signature, but simplified to local (not external)
    deps.

    Supports the four tools ``acquire_next``'s tick pipeline actually
    issues against a project_root with terminal local deps carrying
    ``metadata.delivered_checks``:

    - ``get_tasks``: honours the server-side ``statuses`` filter exactly
      like the real backend, so a 'done'/'cancelled' producer is excluded
      from the ACTIVE-only fetch — forcing the real
      ``_phase_backfill_terminal_dep_records`` / ``get_task`` fallback to
      run, exactly as it would against a live backend.
    - ``get_task``: unfiltered single-task fetch by id (the backfill
      fallback's primitive).
    - ``get_statuses``: unfiltered ``{id: status}`` projection (the
      dep-status backfill primitive).
    - ``set_task_status``: mutates the matching task dict in place.
    """

    def __init__(self) -> None:
        self.tasks: list[dict] = []
        self._request_id = 0

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _envelope(self, text: str) -> dict:
        return {
            'jsonrpc': '2.0',
            'id': self._next_id(),
            'result': {'content': [{'type': 'text', 'text': text}]},
        }

    async def call_tool(self, name: str, arguments: dict, timeout: float = 30) -> dict:
        if name == 'get_tasks':
            statuses = arguments.get('statuses')
            tasks = self.tasks
            if statuses is not None:
                allowed = set(statuses)
                tasks = [t for t in tasks if t.get('status') in allowed]
            return self._envelope(json.dumps({'tasks': tasks}))

        if name == 'get_task':
            task_id = str(arguments['id'])
            task = next((t for t in self.tasks if str(t.get('id')) == task_id), None)
            return self._envelope(json.dumps({'data': task}))

        if name == 'get_statuses':
            ids = arguments.get('ids')
            tasks = self.tasks
            if ids is not None:
                id_set = {str(i) for i in ids}
                tasks = [t for t in tasks if str(t.get('id')) in id_set]
            statuses = {str(t['id']): t['status'] for t in tasks}
            return self._envelope(json.dumps({'statuses': statuses}))

        if name == 'set_task_status':
            task_id = str(arguments['id'])
            status = arguments['status']
            for t in self.tasks:
                if str(t.get('id')) == task_id:
                    t['status'] = status
                    break
            return self._envelope(json.dumps({'id': task_id, 'status': status}))

        raise NotImplementedError(
            f'_LocalDepMcpSession: unknown tool {name!r} — add a branch in '
            'call_tool if this tool is needed by the test'
        )


def _register_producer(
    session: _LocalDepMcpSession, task_id: str, *, status: str, checks: list[dict],
) -> dict:
    """Append a producer task carrying ``metadata.delivered_checks`` to the
    session's in-memory task list. *status* is 'done' or 'cancelled' —
    both TERMINAL statuses the delivered-check gate treats identically
    (the gate trusts main, not the status label)."""
    task: dict = {
        'id': task_id,
        'title': f'Producer task {task_id}',
        'status': status,
        'dependencies': [],
        'metadata': {
            'files': [f'local_dep/producer_{task_id}.py'],
            'delivered_checks': checks,
        },
    }
    session.tasks.append(task)
    return task


def _register_dependent(session: _LocalDepMcpSession, task_id: str, *, dep_id: str) -> dict:
    """Append a pending dependent task with a single LOCAL dependency on
    *dep_id* to the session's in-memory task list. ``metadata.files`` is a
    unique plain file path so ``ModuleLockTable.try_acquire`` succeeds."""
    task: dict = {
        'id': task_id,
        'title': f'Dependent task {task_id}',
        'status': 'pending',
        'dependencies': [{'id': dep_id}],
        'metadata': {'files': [f'local_dep/dependent_{task_id}.py']},
    }
    session.tasks.append(task)
    return task


def _build_harness(
    project_root: Path,
    session: _LocalDepMcpSession,
    escalation_dir: Path,
    *,
    grace_cycles: int | None = None,
) -> Harness:
    """Build a real orchestrator Harness/Scheduler wired to *session* (the
    dispatch seam) plus a real EscalationQueue — so the born-at-L2
    grace-streak escalation (``Harness._block_and_escalate_delivered_check``)
    is genuinely filed and read back — and a recording EventStore so the
    hold-visibility event (``EventType.delivered_check_gate_held``) is
    observable. Mirrors ``build_harness()`` in
    test_cross_project_dispatch_integration.py.

    *grace_cycles*, when given, is set explicitly on
    ``config.delivered_checks.grace_cycles`` before the Harness is built —
    for tick-count determinism in tests that walk the withhold->grace->L2
    arc (row 10). Omitted (the default) leaves ``DeliveredChecksConfig``'s
    own default in place, byte-identical to before this parameter existed
    — rows 3/4/7 never reach grace_cycles regardless of its value.
    """
    config = OrchestratorConfig(project_root=project_root)
    if grace_cycles is not None:
        config.delivered_checks.grace_cycles = grace_cycles
    harness = Harness(config)
    harness.scheduler._mcp_session = session
    harness.scheduler.event_store = _RecordingEventStore()  # type: ignore[assignment]
    harness._escalation_queue = EscalationQueue(escalation_dir)
    # run_tick() drives harness.scheduler.acquire_next() directly, bypassing
    # harness.run()'s startup sweeps — satisfy the startup gate (task 2235)
    # the same way those sweeps would.
    harness.scheduler.finish_startup()
    return harness


async def _run_tick(harness: Harness) -> str | None:
    """Run one acquire_next() tick; return the dispatched task_id or None.

    Dispatch is OBSERVED through the returned TaskAssignment — never by
    reading task storage. This matches the task's "dispatch path"
    requirement.
    """
    assignment = await harness.scheduler.acquire_next()
    return assignment.task_id if assignment is not None else None


async def _flip_status(session: _LocalDepMcpSession, task_id: str, status: str) -> None:
    """Flip *task_id*'s status via the session's own ``set_task_status``
    tool — the same seam ``Scheduler.set_task_status`` calls through —
    rather than mutating ``session.tasks`` directly, so a manual re-pend
    goes through the same dispatch-facing write path a real operator
    action would use."""
    await session.call_tool('set_task_status', {'id': task_id, 'status': status})


def _held_events(harness: Harness) -> list[tuple[str, dict]]:
    """Return recorded ``delivered_check_gate_held`` events from *harness*'s
    injected recording EventStore (secondary corroboration of a withhold —
    the primary signal is ``_run_tick`` returning None)."""
    store = harness.scheduler.event_store
    if store is None:
        return []
    return [
        (evt, data) for evt, data in store.events  # type: ignore[attr-defined]
        if evt == str(EventType.delivered_check_gate_held)
    ]


def _l2_for(harness: Harness, task_id: str) -> list:
    """Return pending born-at-L2 delivered-check escalations for *task_id*.

    Scoped exactly like ``Harness._block_and_escalate_delivered_check``'s
    own dedupe read (``level=2``, ``agent_role='orchestrator-scheduler'``)
    so an unrelated open L2 for the same task never masks (or is masked
    by) this check.
    """
    queue = harness._escalation_queue
    if queue is None:
        return []
    return queue.get_by_task(
        task_id, status='pending', level=2, agent_role='orchestrator-scheduler',
    )


# ─────────────────────────────────────────────────────────────────────────────
# TestRealGitGrepGate — rows 3+4: real `git grep` withhold, then dispatch
# ─────────────────────────────────────────────────────────────────────────────


class TestRealGitGrepGate:
    """Rows 3+4, driven through the REAL (unmocked) ``run_delivered_check``
    + ``Scheduler._resolve_main_sha`` — new coverage vs.
    ``test_delivered_check_gate.py``, whose unit tests fake both.

    A 'done' producer carries a grep-kind ``metadata.delivered_checks``
    entry (the same shape gamma's ``commit_planning`` stamps) whose token
    is initially absent from branch main; a pending dependent depends on
    it via a LOCAL dependency. Row 4: while the token is absent, a real
    ``git grep`` finds no match (rc==1 -> FAILED) so ``acquire_next()``
    withholds dispatch and the hold is visible (event + streak). Row 3:
    once the token is committed to main, the very next tick dispatches
    the dependent (real ``git grep`` rc==0 -> DELIVERED).
    """

    @pytest.mark.asyncio
    async def test_token_absent_withholds_then_landed_token_dispatches(
        self, tmp_path: Path
    ) -> None:
        project_root = _init_git_repo(tmp_path, marker_rel_path=_MARKER_REL_PATH_34)
        session = _LocalDepMcpSession()
        _register_producer(
            session, 'P34', status='done',
            checks=[_grep_check(_CAP_NAME_34, _CAP_TOKEN_34, [_MARKER_REL_PATH_34])],
        )
        _register_dependent(session, 'D34', dep_id='P34')

        harness = _build_harness(project_root, session, tmp_path / 'escalations')

        # --- row 4: token absent from main -> withhold, held event/streak ---
        result = await _run_tick(harness)
        assert result is None, (
            f'row 4: token absent from main must withhold dispatch of the '
            f'dependent; got {result!r}'
        )
        assert _held_events(harness), (
            'row 4: a delivered_check_gate_held event must be recorded for '
            'the withheld dependent'
        )
        assert harness.scheduler._streak_delivered_hold.value('D34') >= 1, (
            'row 4: _streak_delivered_hold must bump for the withheld dependent'
        )

        # --- row 3: land the token on main -> dispatch the very next tick ---
        _commit_marker(project_root, _MARKER_REL_PATH_34, _CAP_TOKEN_34)

        result = await _run_tick(harness)
        assert result == 'D34', (
            f'row 3: once the real `git grep` finds the token on main, the '
            f'dependent must dispatch; got {result!r}'
        )


# ─────────────────────────────────────────────────────────────────────────────
# Row 7 fixture constants (real script-kind runner ERROR -> fail-safe, recover)
# ─────────────────────────────────────────────────────────────────────────────

_CAP_NAME_7 = 'row7_cap'
_SCRIPT_REL_PATH_7 = 'scripts/row7_check.sh'


# ─────────────────────────────────────────────────────────────────────────────
# TestScriptRunnerErrorFailSafe — row 7: real ERRORED -> fail-safe, then recover
# ─────────────────────────────────────────────────────────────────────────────


class TestScriptRunnerErrorFailSafe:
    """Row 7, driven through the REAL (unmocked) script-kind runner: a
    'done' producer carries a script-kind ``metadata.delivered_checks``
    entry whose script file is genuinely MISSING, so
    ``orchestrator.delivered_checks.run_delivered_check`` hits a real
    ``FileNotFoundError`` from the subprocess spawn and maps it to
    ``DeliveredCheckResult.ERRORED`` -- never a definitive FAILED.

    The gate's fail-safe contract for ERRORED is distinct from rows 4/5:
    the dependent is withheld tick after tick with NO
    ``delivered_check_gate_held`` event, NO ``_streak_delivered_fail``
    bump, and NO born-at-L2 escalation -- an ERRORED check must never be
    treated as evidence the capability is missing, only that it could not
    be evaluated. Once the script is created as an executable exit-0 file,
    the very next tick dispatches the dependent -- real recovery, not a
    mock swap.
    """

    @pytest.mark.asyncio
    async def test_missing_script_fails_safe_then_recovers(self, tmp_path: Path) -> None:
        project_root = _init_git_repo(tmp_path)
        session = _LocalDepMcpSession()
        _register_producer(
            session, 'P7', status='done',
            checks=[_script_check(_CAP_NAME_7, _SCRIPT_REL_PATH_7)],
        )
        _register_dependent(session, 'D7', dep_id='P7')

        harness = _build_harness(project_root, session, tmp_path / 'escalations')

        # --- row 7: script missing -> real FileNotFoundError -> ERRORED;
        # fail-safe wait across several ticks, with NONE of the withhold
        # visibility a definitive FAILED would produce.
        for tick in range(1, 4):
            result = await _run_tick(harness)
            assert result is None, (
                f'tick {tick}: a script-kind check whose script is missing '
                f'must withhold dispatch (fail-safe); got {result!r}'
            )
            assert _held_events(harness) == [], (
                f'tick {tick}: an ERRORED check must NOT emit a '
                f'delivered_check_gate_held event (row 7 fail-safe contract)'
            )
            assert harness.scheduler._streak_delivered_fail.value(('D7', 'P7')) == 0, (
                f'tick {tick}: an ERRORED check must NOT bump _streak_delivered_fail'
            )
            assert _l2_for(harness, 'D7') == [], (
                f'tick {tick}: an ERRORED check must NOT file a born-at-L2 escalation'
            )

        # --- real recovery: create the script as an executable exit-0 file ---
        _write_exec_script(project_root, _SCRIPT_REL_PATH_7, '#!/bin/sh\nexit 0\n')

        result = await _run_tick(harness)
        assert result == 'D7', (
            f'once the script exists and exits 0, the dependent must '
            f'dispatch; got {result!r}'
        )


# ─────────────────────────────────────────────────────────────────────────────
# Row 10 fixture constants (cancelled producer, gated exactly like done)
# ─────────────────────────────────────────────────────────────────────────────

_CAP_NAME_10 = 'row10_cap'
_CAP_TOKEN_10 = 'ROW10_CAPABILITY_TOKEN_V1'
_MARKER_REL_PATH_10 = 'src/row10_marker.py'
_GRACE_CYCLES_10 = 3


# ─────────────────────────────────────────────────────────────────────────────
# TestCancelledProducerSameLaneAsDone — row 10: cancelled dep gated like done
# ─────────────────────────────────────────────────────────────────────────────


class TestCancelledProducerSameLaneAsDone:
    """Row 10: a CANCELLED producer carrying a failing ``delivered_checks``
    entry is gated EXACTLY like a 'done' one — the gate trusts what's
    committed on ``main``, not the dep's status label. Ticks 1..G-1
    withhold with no escalation; tick G files the born-at-L2 naming the
    check and the 'cancelled' status, and the dependent goes 'blocked'.
    Landing the capability token on main and manually re-pending the
    dependent then dispatches it — a cancelled dep whose checks now PASS
    satisfies the gate just as a done one would.
    """

    @pytest.mark.asyncio
    async def test_cancelled_dep_gated_like_done_then_heals(self, tmp_path: Path) -> None:
        project_root = _init_git_repo(tmp_path, marker_rel_path=_MARKER_REL_PATH_10)
        session = _LocalDepMcpSession()
        _register_producer(
            session, 'P10', status='cancelled',
            checks=[_grep_check(_CAP_NAME_10, _CAP_TOKEN_10, [_MARKER_REL_PATH_10])],
        )
        _register_dependent(session, 'D10', dep_id='P10')

        harness = _build_harness(
            project_root, session, tmp_path / 'escalations',
            grace_cycles=_GRACE_CYCLES_10,
        )

        # --- ticks 1..G-1: withhold, no escalation yet ---
        for tick in range(1, _GRACE_CYCLES_10):
            result = await _run_tick(harness)
            assert result is None, (
                f'tick {tick}: a cancelled dep with a failing check must '
                f'withhold dispatch exactly like a done one; got {result!r}'
            )
            assert _l2_for(harness, 'D10') == [], (
                f'tick {tick}: no L2 escalation must exist before grace_cycles '
                f'({_GRACE_CYCLES_10}) is reached'
            )

        # --- tick G: born-at-L2 escalation names the check + the
        # 'cancelled' status; the dependent is blocked ---
        result = await _run_tick(harness)
        assert result is None, (
            f'tick {_GRACE_CYCLES_10}: dependent must still not be dispatched; '
            f'got {result!r}'
        )

        escs = _l2_for(harness, 'D10')
        assert len(escs) == 1, (
            f'expected exactly one pending born-at-L2 escalation for the '
            f'dependent on tick {_GRACE_CYCLES_10}; got {escs!r}'
        )
        esc = escs[0]
        assert 'DEP_CAPABILITY_NOT_DELIVERED' in esc.summary, esc.summary
        assert _CAP_NAME_10 in esc.summary, esc.summary
        assert 'P10' in esc.summary, esc.summary
        assert 'cancelled' in esc.summary, esc.summary

        d10_status = next(
            (t['status'] for t in session.tasks if str(t.get('id')) == 'D10'), None,
        )
        assert d10_status == 'blocked', (
            f"expected D10.status=='blocked' after the grace-streak escalation; "
            f"got {d10_status!r}"
        )

        # --- land the capability + re-pend -> dispatches (trusts main, not
        # the status label) ---
        _commit_marker(project_root, _MARKER_REL_PATH_10, _CAP_TOKEN_10)
        await _flip_status(session, 'D10', 'pending')

        result = await _run_tick(harness)
        assert result == 'D10', (
            f'once the capability lands on main, a cancelled dep must '
            f'satisfy the gate exactly like a done one; got {result!r}'
        )
