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
    (no fused_memory — orchestrator/pyproject.toml has no path to it).

    The module-level imports above already enforce this at collection
    time — a broken pythonpath fails `pytest --collect-only`, not an
    in-body assert on an already-imported symbol. This documents the
    contract with a direct import round-trip per package.
    """
    import importlib

    for module_name in ('orchestrator.harness', 'escalation.queue'):
        importlib.import_module(module_name)


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


# ─────────────────────────────────────────────────────────────────────────────
# TestAuthoringDefectIsDistinguishable — task 3500's headline complaint
# ─────────────────────────────────────────────────────────────────────────────

_GRACE_CYCLES_3500 = 2

# MIS-AUTHORED (the measured task-3536 MODE 3 shape): the pattern names a
# tracked FILE whose contents never mention it. A grep check reads file
# CONTENTS, so this one can never go green however much work lands — and a
# test module famously does not mention its own name.
_MISAUTHORED_REL_PATH = 'src/row3500_strand.py'
_MISAUTHORED_CAP = 'row3500_filename_cap'
_MISAUTHORED_PATTERN = 'row3500_strand'

# GENUINELY UNDELIVERED: a well-formed forward-looking check on a token that
# simply has not landed yet. Identical gate behaviour, entirely different
# remedy — which is the whole point.
_UNDELIVERED_REL_PATH = 'src/row3500_marker.py'
_UNDELIVERED_CAP = 'row3500_token_cap'
_UNDELIVERED_TOKEN = 'ROW3500_CAPABILITY_TOKEN_V1'


def _drive_to_l2(
    root: Path,
    escalation_dir: Path,
    *,
    marker_rel_path: str,
    check: dict,
    producer_id: str,
    dependent_id: str,
) -> tuple[Harness, list]:
    """Build one producer/dependent pair on its own real git repo.

    Returns ``(harness, session)``; the caller drives the grace arc with
    :func:`_tick_through_grace` and reads the dependent's status back off the
    session, so each specimen gets an independent repo, escalation queue and
    tick clock rather than sharing one and cross-contaminating.
    """
    root.mkdir(parents=True, exist_ok=True)
    project_root = _init_git_repo(root, marker_rel_path=marker_rel_path)
    session = _LocalDepMcpSession()
    _register_producer(session, producer_id, status='done', checks=[check])
    _register_dependent(session, dependent_id, dep_id=producer_id)
    harness = _build_harness(
        project_root, session, escalation_dir, grace_cycles=_GRACE_CYCLES_3500,
    )
    return harness, session


async def _tick_through_grace(harness: Harness, dependent_id: str) -> list:
    """Tick until grace is exhausted; assert the withhold arc; return the L2s."""
    for tick in range(1, _GRACE_CYCLES_3500):
        assert await _run_tick(harness) is None, f'tick {tick} must withhold'
        assert _l2_for(harness, dependent_id) == [], (
            f'tick {tick}: no L2 before grace_cycles is reached'
        )
    assert await _run_tick(harness) is None
    return _l2_for(harness, dependent_id)


class TestAuthoringDefectIsDistinguishable:
    """Task 3500's OPENING COMPLAINT, closed end to end.

    Before this, a mis-authored check and a genuinely undelivered capability
    produced IDENTICAL critical escalations: both said
    ``DEP_CAPABILITY_NOT_DELIVERED``, both named the check, and nothing in
    either told a reader which one they were looking at. The two need
    opposite responses — one is repaired by editing the producer's metadata,
    the other by waiting for (or chasing) real work — so a reader who cannot
    tell them apart either waits on a check that can never go green or
    "fixes" a descriptor that was correct all along.
    """

    @pytest.mark.asyncio
    async def test_misauthored_and_undelivered_escalations_differ(
        self, tmp_path: Path
    ) -> None:
        """(a)+(b) THE DISTINGUISHABILITY PROPERTY. Two producers, identical
        gate behaviour (grace -> born-at-L2 -> dependent blocked), different
        escalation detail: only the mis-authored one carries an authoring
        diagnosis naming the code and a remedy."""
        # --- (a) MIS-AUTHORED: pattern matches a FILENAME, never contents ---
        bad_harness, bad_session = _drive_to_l2(
            tmp_path / 'misauthored', tmp_path / 'esc-bad',
            marker_rel_path=_MISAUTHORED_REL_PATH,
            check=_grep_check(
                _MISAUTHORED_CAP, _MISAUTHORED_PATTERN, [_MISAUTHORED_REL_PATH],
            ),
            producer_id='P3500A', dependent_id='D3500A',
        )
        bad_escs = await _tick_through_grace(bad_harness, 'D3500A')

        assert len(bad_escs) == 1, bad_escs
        bad = bad_escs[0]
        # The load-bearing signal is UNCHANGED — the diagnosis is additive.
        assert 'DEP_CAPABILITY_NOT_DELIVERED' in bad.summary, bad.summary
        assert _MISAUTHORED_CAP in bad.summary, bad.summary
        assert bad.level == 2 and bad.severity == 'critical'
        # ...and the detail now says WHICH authoring defect fired, plus a
        # remedy. A code without a remedy is a diagnosis a reader cannot act on.
        assert 'AUTHORING DIAGNOSIS' in bad.detail, bad.detail
        assert 'filename_shaped' in bad.detail, bad.detail
        assert _MISAUTHORED_REL_PATH in bad.detail, bad.detail

        bad_status = next(
            (t['status'] for t in bad_session.tasks if str(t.get('id')) == 'D3500A'), None,
        )
        assert bad_status == 'blocked'

        # --- (b) GENUINELY UNDELIVERED: well-formed, simply not landed ---
        good_harness, good_session = _drive_to_l2(
            tmp_path / 'undelivered', tmp_path / 'esc-good',
            marker_rel_path=_UNDELIVERED_REL_PATH,
            check=_grep_check(
                _UNDELIVERED_CAP, _UNDELIVERED_TOKEN, [_UNDELIVERED_REL_PATH],
            ),
            producer_id='P3500B', dependent_id='D3500B',
        )
        good_escs = await _tick_through_grace(good_harness, 'D3500B')

        assert len(good_escs) == 1, good_escs
        good = good_escs[0]
        assert 'DEP_CAPABILITY_NOT_DELIVERED' in good.summary, good.summary
        # NO diagnosis: nothing is wrong with this descriptor. Emitting one
        # here would be the mirror-image failure — an operator "repairing" a
        # correct check instead of chasing the work that never landed.
        assert 'AUTHORING DIAGNOSIS' not in good.detail, good.detail

        good_status = next(
            (t['status'] for t in good_session.tasks if str(t.get('id')) == 'D3500B'), None,
        )
        assert good_status == 'blocked'

        # --- the property itself ---
        assert bad.detail != good.detail, (
            'a mis-authored check and a genuinely undelivered capability must '
            'not produce identical escalation details — that indistinguishability '
            'IS the defect task 3500 exists to close'
        )

    @pytest.mark.asyncio
    async def test_a_raising_lint_never_costs_the_escalation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """(c) THE DIAGNOSIS IS AN ENHANCEMENT, THE ESCALATION IS THE SIGNAL.

        If classification can abort the file, this change has made the system
        strictly worse than the indistinguishable-escalations state it set out
        to fix: an operator would get NO notification at all instead of an
        ambiguous one. So the lint is driven to raise and every load-bearing
        property is re-asserted.
        """
        import shared.delivered_check_polarity as polarity

        def _boom(*args, **kwargs):
            raise RuntimeError('lint exploded')

        monkeypatch.setattr(polarity, 'lint_delivered_checks', _boom)

        harness, session = _drive_to_l2(
            tmp_path / 'raising', tmp_path / 'esc',
            marker_rel_path=_MISAUTHORED_REL_PATH,
            check=_grep_check(
                _MISAUTHORED_CAP, _MISAUTHORED_PATTERN, [_MISAUTHORED_REL_PATH],
            ),
            producer_id='P3500C', dependent_id='D3500C',
        )
        escs = await _tick_through_grace(harness, 'D3500C')

        assert len(escs) == 1, escs
        esc = escs[0]
        assert 'DEP_CAPABILITY_NOT_DELIVERED' in esc.summary, esc.summary
        assert esc.level == 2 and esc.severity == 'critical'
        assert 'AUTHORING DIAGNOSIS' not in esc.detail, esc.detail
        status = next(
            (t['status'] for t in session.tasks if str(t.get('id')) == 'D3500C'), None,
        )
        assert status == 'blocked'

    @pytest.mark.asyncio
    async def test_dedupe_is_unchanged_by_the_diagnosis(self, tmp_path: Path) -> None:
        """(d) The pending-scoped dedupe still holds. The diagnosis is built
        AFTER the dedupe read, so it can neither trigger a second file nor
        change what the existing one matches on."""
        harness, session = _drive_to_l2(
            tmp_path / 'dedupe', tmp_path / 'esc',
            marker_rel_path=_MISAUTHORED_REL_PATH,
            check=_grep_check(
                _MISAUTHORED_CAP, _MISAUTHORED_PATTERN, [_MISAUTHORED_REL_PATH],
            ),
            producer_id='P3500D', dependent_id='D3500D',
        )
        assert len(await _tick_through_grace(harness, 'D3500D')) == 1

        # Re-pend and drive a second full grace arc: still exactly one open L2.
        await _flip_status(session, 'D3500D', 'pending')
        for _ in range(_GRACE_CYCLES_3500):
            await _run_tick(harness)

        assert len(_l2_for(harness, 'D3500D')) == 1
