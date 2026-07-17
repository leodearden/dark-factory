"""Cross-package end-to-end regression: reopen-freshness gate under the
SHIPPED (enforce) default (task 2680, PRD task gamma,
plans/found-on-main-provenance-integrity-prd.md).

Drives PRD boundary-tests #1, #2, and #7 against the REAL fused-memory
``SqliteTaskBackend`` + ``TaskInterceptor`` and the REAL orchestrator
``Scheduler`` + ``ProvenanceConflictSink`` — the only mocked seam is the MCP
wire transport (``Scheduler.dispatch_tool``), bridged directly to the real
interceptor in-process and wrapped in the same envelope
(``{result: {structuredContent: <payload>}}``) that
``extract_structured_rejection``/``extract_rejection`` parse in production.
The full ``Harness`` dispatch loop is intentionally NOT wired: the
load-bearing gate is ``Scheduler.set_task_status``'s rejection
classification (task 2677 / PRD task beta) plus ``ProvenanceConflictSink``'s
dedupe + terminal-this-tick memo — the MCP wire itself is the only seam that
cannot run in-process, so bridging just that keeps both sides' gate logic
real (see plan.json design_decisions).

Critically, ``TaskInterceptor`` is constructed with a bare DEFAULT
``FusedMemoryConfig()`` — no explicit ``reject_stale_done_evidence`` — which
is the entire point of task gamma: this proves 'enforce' is the SHIPPED
default, not merely that the gate works when an operator explicitly turns it
on (task alpha's existing real-backend interceptor tests already cover the
latter with an explicit ``enforce`` config).

Boundary-test #1 (task-1175 shape, reopen sticks / no clobber): done(stale
evidence) -> reopen(pending) -> re-done citing the SAME stale evidence is
rejected; the reopen sticks (no clobber), exactly one born-at-L2
``provenance_conflict`` escalation is filed, and a repeat attempt with
identical evidence folds into ``dedupe_count`` rather than filing a second
escalation.

Boundary-test #2 (legit crash-recovery re-complete): from a reopened task, a
re-done citing FRESH evidence (committer date after ``reopen_at``) is
accepted — no rejection, no escalation.

Boundary-test #7 (commitless kinds exempt): from a reopened task, a
commitless ``operational-verified`` done-write (DeterministicRunner-resume
class) is accepted under the enforce default — the freshness gate only
examines commit-bearing kinds (``merged``/``found_on_main``).
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

pytest.importorskip('fused_memory')

from escalation.queue import EscalationQueue
from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import FusedMemoryConfig, TaskmasterConfig
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.reconciliation.event_buffer import EventBuffer

from orchestrator.config import OrchestratorConfig
from orchestrator.provenance_conflict import ProvenanceConflictSink, stale_evidence_fingerprint
from orchestrator.scheduler import Scheduler, StaleEvidenceRejection

pytestmark = pytest.mark.asyncio

# Bridge-synthesised agent_id for every dispatch_tool('set_task_status', ...)
# call. The real Scheduler.set_task_status never forwards an agent_id over
# the wire (every internal orchestrator caller omits it — see scheduler.py),
# so this in-process bridge supplies a fixed, KNOWN-SAFE value: the exact
# agent_id used throughout fused-memory's own real-backend enforce tests
# (test_task_interceptor.py's test_reopen_freshness_enforce_rejects_stale_
# evidence_real_backend et al.), so this e2e cannot be tripped up by an
# actor-classification gate unrelated to reopen-freshness.
_BRIDGE_AGENT_ID = 'claude-interactive'


def _init_git_repo(path: Path) -> str:
    """Create a minimal git repo at *path* with one commit; return full SHA.

    Mirrors fused-memory/tests/_fm_helpers.py's ``_init_git_repo`` exactly,
    kept local to this file — cross-package test-file import is avoided (see
    plan.json design_decisions/reuse). ``git init`` creates *path* itself.
    """
    subprocess.run(['git', 'init', '-q', '-b', 'main', str(path)], check=True)
    subprocess.run(['git', '-C', str(path), 'config', 'user.email', 't@e.example'], check=True)
    subprocess.run(['git', '-C', str(path), 'config', 'user.name', 'T'], check=True)
    (path / 'seed.txt').write_text('seed\n')
    subprocess.run(['git', '-C', str(path), 'add', '-A'], check=True)
    subprocess.run(['git', '-C', str(path), 'commit', '-q', '-m', 'seed'], check=True)
    return subprocess.run(
        ['git', '-C', str(path), 'rev-parse', 'HEAD'],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _commit_on_main(path: Path, filename: str, date_iso: str) -> str:
    """Create a commit with a controlled committer/author date; return full SHA.

    Mirrors fused-memory/tests/test_task_interceptor.py's ``_commit_on_main``
    exactly (kept local — see ``_init_git_repo`` above). ``'2020-01-01...'``
    == stale, ``'2099-01-01...'`` == fresh, relative to the
    interceptor-stamped ``reopen_at``.
    """
    (path / filename).write_text(f'{filename}\n')
    subprocess.run(['git', '-C', str(path), 'add', '-A'], check=True)
    env = {**os.environ, 'GIT_COMMITTER_DATE': date_iso, 'GIT_AUTHOR_DATE': date_iso}
    subprocess.run(
        ['git', '-C', str(path), 'commit', '-q', '-m', f'evidence: {filename}'],
        check=True, env=env,
    )
    return subprocess.run(
        ['git', '-C', str(path), 'rev-parse', 'HEAD'],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


@pytest_asyncio.fixture
async def event_buffer(tmp_path):
    """Mirrors fused-memory's own ``event_buffer`` fixture (test_task_
    interceptor.py) — constructed directly since that fixture itself is not
    importable cross-package (see plan.json reuse).
    """
    buf = EventBuffer(db_path=tmp_path / 'events.db', buffer_size_threshold=100)
    await buf.initialize()
    yield buf
    await buf.close()


@dataclass
class _Rig:
    """Bundles the real cross-package wiring for one test."""

    repo: Path
    project_root: str
    backend: SqliteTaskBackend
    scheduler: Scheduler
    sink: ProvenanceConflictSink
    queue: EscalationQueue


async def _build_rig(tmp_path: Path, event_buffer: EventBuffer) -> _Rig:
    """Wire a REAL SqliteTaskBackend + REAL TaskInterceptor(DEFAULT config)
    behind a REAL orchestrator Scheduler + ProvenanceConflictSink, bridging
    ONLY the MCP transport (``dispatch_tool``) directly to the interceptor.

    See the module docstring for why the DEFAULT config and the
    dispatch_tool-only bridge are both load-bearing choices, not
    conveniences.
    """
    repo = tmp_path / 'repo'
    _init_git_repo(repo)

    orch_config = OrchestratorConfig(project_root=repo)
    project_root = str(orch_config.project_root)

    backend = SqliteTaskBackend(TaskmasterConfig(project_root=project_root))
    await backend.start()

    # DEFAULT config — no explicit reject_stale_done_evidence — is the whole
    # point of task gamma: proves 'enforce' is the SHIPPED default.
    interceptor = TaskInterceptor(backend, None, event_buffer, config=FusedMemoryConfig())

    async def _bridge_dispatch_tool(name: str, arguments: dict, *, timeout: float = 15) -> dict:
        """Stand-in MCP transport for 'set_task_status' only.

        Calls the real interceptor and wraps its payload in the MCP envelope
        shape ``extract_structured_rejection``/``extract_rejection`` expect,
        so the real ``Scheduler.set_task_status`` classification logic runs
        unmodified against a REAL rejection payload.
        """
        assert name == 'set_task_status', f'unexpected dispatch_tool call: {name!r}'
        payload = await interceptor.set_task_status(
            arguments['id'],
            arguments['status'],
            arguments['project_root'],
            done_provenance=arguments.get('done_provenance'),
            reopen_reason=arguments.get('reopen_reason'),
            agent_id=_BRIDGE_AGENT_ID,
        )
        return {'result': {'structuredContent': payload}}

    scheduler = Scheduler(orch_config)
    scheduler.dispatch_tool = _bridge_dispatch_tool  # type: ignore[method-assign]

    queue = EscalationQueue(tmp_path / 'escalations')
    sink = ProvenanceConflictSink(escalation_queue=queue)

    rig = _Rig(
        repo=repo,
        project_root=project_root,
        backend=backend,
        scheduler=scheduler,
        sink=sink,
        queue=queue,
    )
    # Pin the assumption _read_fresh's fresh-backend-connection check depends
    # on: OrchestratorConfig.project_root must resolve to the SAME tree as
    # the git repo used for evidence commits. If config path resolution ever
    # diverges (symlink, trailing slash, normalization change), git
    # committer-date lookups and backend reads would silently target
    # different trees instead of failing loudly.
    assert Path(rig.project_root).resolve() == repo.resolve(), (
        'OrchestratorConfig(project_root=repo).project_root must resolve '
        'identically to the git repo used for evidence commits/reads'
    )
    return rig


async def _gate_tick(
    rig: _Rig, task_id: str, status: str, **kwargs: Any,
) -> StaleEvidenceRejection | None:
    """Model one dispatch-gate write attempt against the REAL Scheduler.

    Mirrors harness.py's dispatch-gate catch/record pattern (see plan.json
    design_decisions): on a ``done_evidence_stale`` rejection, feeds the real
    ``StaleEvidenceRejection`` to the real ``ProvenanceConflictSink`` and
    returns it so callers can assert its structured fields. Returns ``None``
    on a successful (accepted) write. Any OTHER ``SetTaskStatusRejected``
    subtype is deliberately left to propagate — it would indicate a bug in
    this test's provenance construction, not a case this e2e models.
    """
    try:
        await rig.scheduler.set_task_status(task_id, status, **kwargs)
    except StaleEvidenceRejection as exc:
        rig.sink.record_from_rejection(exc, gate_source='dispatch_gate')
        return exc
    return None


async def _read_fresh(rig: _Rig, task_id: str) -> dict:
    """Read *task_id* via an INDEPENDENT fresh backend connection.

    Mirrors the fused-memory real-backend tests' fresh-backend pattern: this
    confirms a write (or its absence) actually persisted to disk, rather
    than merely reflecting ``rig.backend``'s own connection/cache state.
    """
    fresh = SqliteTaskBackend(TaskmasterConfig(project_root=rig.project_root))
    await fresh.start()
    try:
        return await fresh.get_task(task_id, project_root=rig.project_root)
    finally:
        await fresh.close()


class TestReopenSticksE2E:
    """PRD boundary-tests #1/#2/#7, real backend + real orchestrator gate."""

    async def test_boundary_1_stale_redone_rejected_and_reopen_sticks(
        self, tmp_path, event_buffer,
    ) -> None:
        """task-1175 shape: a reopened task's re-done citing the SAME stale
        (pre-reopen) evidence is rejected by the REAL gate under the
        SHIPPED enforce default — the reopen sticks (no clobber), exactly
        one born-at-L2 provenance_conflict escalation is filed, and a
        repeat attempt with identical evidence folds into dedupe_count
        rather than filing a second escalation or re-clobbering the status.
        """
        rig = await _build_rig(tmp_path, event_buffer)
        try:
            stale_sha = _commit_on_main(
                rig.repo, 'stale_evidence.txt', '2020-01-01T00:00:00+00:00',
            )
            add_result = await rig.backend.add_task(project_root=rig.project_root, title='T')
            task_id = str(add_result['id'])
            stale_provenance = {'kind': 'found_on_main', 'commit': stale_sha, 'note': 'sibling'}

            # (1) First done-write, before any reopen: reopen_at is not yet
            # set, so the freshness gate is inert — accepted.
            rejection = await _gate_tick(rig, task_id, 'done', done_provenance=stale_provenance)
            assert rejection is None
            assert (await _read_fresh(rig, task_id))['status'] == 'done'

            # (2) Reopen: done -> pending, stamps metadata.reopen_at.
            rejection = await _gate_tick(
                rig, task_id, 'pending', reopen_reason='reverting: regression found',
            )
            assert rejection is None
            reopened = await _read_fresh(rig, task_id)
            assert reopened['status'] == 'pending'
            reopen_at = reopened['metadata']['reopen_at']
            assert reopen_at

            # (3) Gate attempt #1: re-done citing the SAME stale evidence —
            # rejected with the typed, structured StaleEvidenceRejection.
            rejection = await _gate_tick(rig, task_id, 'done', done_provenance=stale_provenance)
            assert rejection is not None
            assert rejection.error_code == 'done_evidence_stale'
            assert rejection.task_id == task_id
            assert rejection.evidence_commit == stale_sha
            assert rejection.evidence_committed_at
            assert rejection.reopen_at == reopen_at
            assert rejection.agent_id == _BRIDGE_AGENT_ID

            # The reopen sticks — no clobber of the pending status.
            assert (await _read_fresh(rig, task_id))['status'] == 'pending'

            pending = rig.queue.get_by_task(task_id, status='pending')
            assert len(pending) == 1, f'expected exactly one pending escalation, got {len(pending)}'
            esc = pending[0]
            assert esc.level == 2
            assert esc.severity == 'urgent'
            assert esc.category == 'provenance_conflict'
            assert esc.dedupe_fingerprint == stale_evidence_fingerprint(task_id, stale_sha)
            assert esc.dedupe_count == 0

            assert rig.sink.should_skip(task_id, reopen_at=reopen_at) is True

            # (4) Gate attempt #2: identical stale evidence — folds into the
            # SAME escalation's dedupe_count; no second escalation filed;
            # status still stuck at 'pending' (still no clobber).
            rejection2 = await _gate_tick(rig, task_id, 'done', done_provenance=stale_provenance)
            assert rejection2 is not None
            assert rejection2.error_code == 'done_evidence_stale'

            pending_after_2 = rig.queue.get_by_task(task_id, status='pending')
            assert len(pending_after_2) == 1, (
                f'expected exactly one pending escalation after a repeat rejection, '
                f'got {len(pending_after_2)}'
            )
            assert pending_after_2[0].id == esc.id
            assert pending_after_2[0].dedupe_count == 1

            assert (await _read_fresh(rig, task_id))['status'] == 'pending'
        finally:
            await rig.backend.close()

    async def test_boundary_2_fresh_redone_accepted_no_escalation(
        self, tmp_path, event_buffer,
    ) -> None:
        """Legit crash-recovery re-complete: from a reopened task, a re-done
        citing FRESH evidence (committer date AFTER reopen_at) passes the
        REAL gate under the enforce default — no rejection, the write
        lands, and no provenance_conflict escalation is filed.
        """
        rig = await _build_rig(tmp_path, event_buffer)
        try:
            stale_sha = _commit_on_main(
                rig.repo, 'stale_evidence.txt', '2020-01-01T00:00:00+00:00',
            )
            add_result = await rig.backend.add_task(project_root=rig.project_root, title='T')
            task_id = str(add_result['id'])
            stale_provenance = {'kind': 'found_on_main', 'commit': stale_sha, 'note': 'sibling'}

            rejection = await _gate_tick(rig, task_id, 'done', done_provenance=stale_provenance)
            assert rejection is None

            rejection = await _gate_tick(
                rig, task_id, 'pending', reopen_reason='reverting: regression found',
            )
            assert rejection is None

            fresh_sha = _commit_on_main(
                rig.repo, 'fresh_evidence.txt', '2099-01-01T00:00:00+00:00',
            )
            fresh_provenance = {
                'kind': 'found_on_main', 'commit': fresh_sha, 'note': 'fixed and reverified',
            }

            rejection = await _gate_tick(rig, task_id, 'done', done_provenance=fresh_provenance)
            assert rejection is None

            assert (await _read_fresh(rig, task_id))['status'] == 'done'
            assert rig.queue.get_by_task(task_id, status='pending') == []
        finally:
            await rig.backend.close()

    async def test_boundary_7_commitless_operational_verified_exempt(
        self, tmp_path, event_buffer,
    ) -> None:
        """Commitless kinds exempt: from a reopened task, a commitless
        ``operational-verified`` done-write (DeterministicRunner-resume
        class) is accepted under the enforce default — the freshness gate
        only examines commit-bearing kinds ('merged'/'found_on_main').
        """
        rig = await _build_rig(tmp_path, event_buffer)
        try:
            stale_sha = _commit_on_main(
                rig.repo, 'stale_evidence.txt', '2020-01-01T00:00:00+00:00',
            )
            add_result = await rig.backend.add_task(project_root=rig.project_root, title='T')
            task_id = str(add_result['id'])
            stale_provenance = {'kind': 'found_on_main', 'commit': stale_sha, 'note': 'sibling'}

            rejection = await _gate_tick(rig, task_id, 'done', done_provenance=stale_provenance)
            assert rejection is None

            rejection = await _gate_tick(
                rig, task_id, 'pending', reopen_reason='reverting: regression found',
            )
            assert rejection is None

            operational_provenance = {
                'kind': 'operational-verified',
                'escalation_id': 'esc-prov-conflict-77',
                'note': 'confirmed the service was already healthy; no code change needed',
            }
            rejection = await _gate_tick(
                rig, task_id, 'done', done_provenance=operational_provenance,
            )
            assert rejection is None

            assert (await _read_fresh(rig, task_id))['status'] == 'done'
            assert rig.queue.get_by_task(task_id, status='pending') == []
        finally:
            await rig.backend.close()
