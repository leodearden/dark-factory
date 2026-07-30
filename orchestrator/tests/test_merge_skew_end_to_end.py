"""End-to-end tests for task 2383 β (steps 15-18): merge-skew classification
wired through the REAL production chain on a genuinely FIRST merge attempt.

Unlike test_merge_queue_disposition_wiring.py (unit-level, one link at a
time, with neighboring layers faked) and test_workflow_skew_surfacing.py
(routing-level, driven by a hand-built MergeOutcome), these tests compose
the individually-tested links — dispatch-time base-fact computation (step
10), classification + I4 surfacing (steps 6/8), the workflow_verify producer
emit (step 12), INTEGRATION_SKEW routing to L1 (step 14), and the runs.db
merge_attempt disposition emit (step 18) — together for the first time, so
that every sha/file assertion below flows end-to-end from real git history
rather than from a hand-typed fixture.

"First attempt" / G6 trap: classify_merge_failure_disposition's I5 green fact
is sourced EXCLUSIVELY from EventType.workflow_verify (the branch's own
pre-merge verdict, emitted at the workflow VERIFY→REVIEW edge by step 12) —
never from EventType.merge_verify (the merge worker's post-rebase verify,
which would not exist yet on a first attempt; see merge_disposition.py's
_branch_pre_merge_verify_green docstring).  These tests never emit a
merge_verify row, proving INTEGRATION_SKEW fires on exactly this case.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from escalation.queue import EscalationQueue

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps, MergeResult
from orchestrator.merge_disposition import MergeFailureDisposition
from orchestrator.merge_queue import (
    InflightEntry,
    MergeOutcome,
    MergeRequest,
    QueuedBranch,
    RealMergeItem,
    SpeculativeMergeWorker,
    _MAX_EVENT_EVIDENCE_ITEMS,
)
from orchestrator.scheduler import TaskAssignment
from orchestrator.verify import VerifyResult
from orchestrator.verify_runner import HostLease
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

TASK_ID = '9001'
BRANCH = f'task/{TASK_ID}'

# A failing test whose pytest id + cause-hint both resolve to src/x.py (mirrors
# test_merge_queue_disposition_wiring.py's _XPY_FAILURE fixture).
_XPY_FAILURE = VerifyResult(
    passed=False,
    test_output='FAILED src/x.py::test_bar - AssertionError',
    lint_output='',
    type_output='',
    summary='1 failed',
    cause_hint='src/x.py::test_bar',
    category='test_failure',
)

# Reify 5566's LIVE shell-guard failure shape, captured 2026-07-29 (task 3178).
# reify's verify summary renders a tests/infra/*.sh guard trip as
# ``FAILED <guard>.sh`` — there is no pytest node id anywhere in it. Pre-3178
# the unconstrained ``^FAILED\s+(\S+)`` captured the guard's own FILENAME as a
# "failing test id", which satisfied I7's non-empty-failing_tests requirement
# vacuously and produced a fabricated INTEGRATION_SKEW whenever any main landing
# happened to touch that file. Note the guard NAME is deliberately still a
# candidate FILE (the file-token scan is untouched), so landings are still
# cited — it is the I7 gate, not a starved extraction, that flips the verdict.
_SHELL_GUARD_NAME = 'test_reify_audit_ptodo.sh'
_SHELL_GUARD_FAILURE = VerifyResult(
    passed=False,
    test_output='',
    lint_output='',
    type_output='',
    summary='1 failed',
    cause_hint=f'FAILED {_SHELL_GUARD_NAME}',
    category='test_failure',
)

# Reify 5321's guard, a second live string from the same 07-24..07-28 census.
_SWEEP_GUARD_NAME = 'test_deterministic_gate_closure_staleness_sweep.sh'
_SWEEP_GUARD_FAILURE = VerifyResult(
    passed=False,
    test_output='',
    lint_output='',
    type_output='',
    summary='1 failed',
    cause_hint=f'FAILED {_SWEEP_GUARD_NAME}',
    category='test_failure',
)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _init_git_repo(root: Path) -> None:
    for cmd in [
        ['git', 'init', '-q', '-b', 'main'],
        ['git', 'config', 'user.email', 'test@test.com'],
        ['git', 'config', 'user.name', 'Test'],
    ]:
        subprocess.run(cmd, cwd=root, check=True, capture_output=True)


def _commit_file(root: Path, rel_path: str, content: str, message: str) -> str:
    file_path = root / rel_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)
    subprocess.run(['git', 'add', rel_path], cwd=root, check=True, capture_output=True)
    subprocess.run(
        ['git', 'commit', '-q', '-m', message], cwd=root, check=True, capture_output=True,
    )
    return subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True, check=True,
    ).stdout.strip()


def _make_config(project_root: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=project_root,
        max_concurrent_tasks=1,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    )


def _make_git_ops(project_root: Path) -> GitOps:
    git_ops = MagicMock(spec=GitOps)
    git_ops.project_root = project_root
    git_ops.cleanup_merge_worktree = AsyncMock(return_value=None)
    git_ops.get_main_sha = AsyncMock(return_value='should-not-be-used-sha')
    return git_ops


def _build_skew_topology(repo: Path) -> dict[str, str]:
    """A real git repo with a first-attempt integration-skew shape::

        merge_base (creates src/x.py)
          |-- branch_tip (task/9001; touches an UNRELATED file, so it
          |                merges cleanly against any main descendant)
          `-- landing_sha (main; edits src/x.py — the file the branch's
                            failing test implicates)
    """
    _init_git_repo(repo)
    merge_base_sha = _commit_file(repo, 'src/x.py', 'v1', 'init x (merge-base)')
    subprocess.run(
        ['git', 'checkout', '-q', '-b', BRANCH], cwd=repo, check=True, capture_output=True,
    )
    branch_tip_sha = _commit_file(
        repo, 'src/branch_only.py', 'v1', 'branch work (unrelated file)',
    )
    subprocess.run(['git', 'checkout', '-q', 'main'], cwd=repo, check=True, capture_output=True)
    landing_sha = _commit_file(repo, 'src/x.py', 'v2', 'edit x on main (landing, implicated)')
    return {
        'merge_base_sha': merge_base_sha,
        'branch_tip_sha': branch_tip_sha,
        'landing_sha': landing_sha,
    }


def _build_shell_guard_topology(repo: Path, guard_name: str) -> dict[str, str]:
    """The reify-5566 shape (task 3178): identical to
    :func:`_build_skew_topology` except that the file main lands on is a
    ROOT-LEVEL shell guard rather than a source file::

        merge_base (creates <guard_name> at the repo ROOT)
          |-- branch_tip (task/9001; unrelated file, merges cleanly)
          `-- landing_sha (main; edits <guard_name> — a GENUINE main landing
                            that _implicated_landings really does cite)

    Root-level on purpose: ``_implicated_landings`` passes the candidate token
    to git as a pathspec from the repo root, so a bare ``foo.sh`` token only
    matches a root-level ``foo.sh``. Nesting it under ``tests/infra/`` would
    make the citation vanish and the test would then pass for the WRONG reason
    (no landings implicated -> BRANCH_BUG), hiding rather than pinning the I7
    gate.
    """
    _init_git_repo(repo)
    merge_base_sha = _commit_file(repo, guard_name, '#!/bin/sh\nexit 0\n', 'init guard (merge-base)')
    subprocess.run(
        ['git', 'checkout', '-q', '-b', BRANCH], cwd=repo, check=True, capture_output=True,
    )
    branch_tip_sha = _commit_file(
        repo, 'src/branch_only.py', 'v1', 'branch work (unrelated file)',
    )
    subprocess.run(['git', 'checkout', '-q', 'main'], cwd=repo, check=True, capture_output=True)
    landing_sha = _commit_file(
        repo, guard_name, '#!/bin/sh\nexit 1\n', 'tighten guard on main (landing, implicated)',
    )
    return {
        'merge_base_sha': merge_base_sha,
        'branch_tip_sha': branch_tip_sha,
        'landing_sha': landing_sha,
    }


def _seed_branch_green(store: EventStore, base_sha: str) -> None:
    """Seed the I5 green fact (the branch's OWN pre-merge verdict — the
    ONLY event this harness ever emits; no EventType.merge_verify row is
    ever written, proving classification works on a genuinely first
    attempt)."""
    store.emit(
        EventType.workflow_verify, task_id=TASK_ID,
        data={'passed': True, 'base_sha': base_sha, 'branch': BRANCH},
    )


def _setup_skew_scenario(tmp_path: Path) -> tuple[Path, dict[str, str], EventStore]:
    repo = tmp_path / 'repo'
    repo.mkdir()
    topology = _build_skew_topology(repo)
    store = EventStore(tmp_path / 'runs.db', run_id='run-test')
    _seed_branch_green(store, topology['merge_base_sha'])
    return repo, topology, store


def _setup_shell_guard_scenario(
    tmp_path: Path, guard_name: str,
) -> tuple[Path, dict[str, str], EventStore]:
    """The reify-5566 scenario (task 3178): a real main landing on the guard
    file PLUS a positively-satisfied I5 green fact, so the ONLY thing standing
    between this and an INTEGRATION_SKEW verdict is I7's node-id shape floor."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    topology = _build_shell_guard_topology(repo, guard_name)
    store = EventStore(tmp_path / 'runs.db', run_id='run-test')
    _seed_branch_green(store, topology['merge_base_sha'])
    return repo, topology, store


def _make_item(
    tmp_path: Path,
    repo: Path,
    topology: dict[str, str],
    future: asyncio.Future,
    *,
    merged_branch_tip: str | None,
) -> tuple[MergeRequest, RealMergeItem]:
    config = _make_config(repo)
    merge_wt = tmp_path / 'merge-wt'
    merge_wt.mkdir(exist_ok=True)
    req = MergeRequest(
        task_id=TASK_ID, branch=QueuedBranch.parse(BRANCH, config.git.branch_prefix), worktree=repo, pre_rebased=False,
        task_files=None, module_configs=[], config=config, result=future, lane='normal',
    )
    item = RealMergeItem(
        request=req,
        merge_result=MergeResult(success=True, merge_commit='deadbeef', merge_worktree=merge_wt),
        merge_wt=merge_wt,
        base_sha=topology['landing_sha'],
        speculative=False,
        merged_branch_tip=merged_branch_tip,
    )
    return req, item


def _patched_verify_seams(verify_result: VerifyResult = _XPY_FAILURE):
    """The only two genuine subprocess/probe seams mocked in this harness —
    everything else (merge-base computation, disposition classification, I4
    surfacing, escalation filing) runs against real production code.

    *verify_result* is the post-merge verify verdict the mocked runner returns;
    it defaults to ``_XPY_FAILURE`` so every pre-3178 caller is unchanged.
    """
    return (
        patch(
            'orchestrator.merge_queue.run_scoped_verification',
            new=AsyncMock(return_value=verify_result),
        ),
        patch(
            'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
            new=AsyncMock(return_value=(False, '')),
        ),
    )


async def _finalize_via_real_worker(
    tmp_path: Path, repo: Path, topology: dict[str, str], store: EventStore,
    *, merged_branch_tip: str | None,
    verify_result: VerifyResult = _XPY_FAILURE,
) -> MergeOutcome | None:
    """Drive the REAL ``SpeculativeMergeWorker._finalize_inflight`` (step 18)
    and hand back the MergeOutcome it delivered on ``req.result``.

    Returning the outcome (task 3178) lets a caller assert on
    ``MergeOutcome.skew_evidence`` — the seam the widened step-18 emit guard
    keys on — from the same run that produced the runs.db row, rather than
    inferring the outcome's shape from the row alone.
    """
    git_ops = _make_git_ops(repo)
    worker = SpeculativeMergeWorker(
        git_ops=git_ops, queue=asyncio.Queue(), event_store=store,
    )
    alloc = MagicMock()
    alloc.release = AsyncMock()
    alloc.cancel_and_release = AsyncMock()
    worker._host_allocator = alloc  # type: ignore[attr-defined]

    lease = HostLease(name='local', runner=MagicMock(), is_local=True)
    future: asyncio.Future = asyncio.get_running_loop().create_future()
    _req, item = _make_item(
        tmp_path, repo, topology, future,
        merged_branch_tip=merged_branch_tip,
    )
    entry = InflightEntry(
        item=item,
        lease=lease,
        verify_task=asyncio.ensure_future(worker._run_inflight_verify(item, lease)),
        merge_wt=None,
        was_speculative=False,
    )
    p1, p2 = _patched_verify_seams(verify_result)
    with p1, p2:
        await worker._finalize_inflight(entry)
    return future.result() if future.done() and not future.cancelled() else None


def _assert_evidence_bounded(data: dict) -> None:
    """Every evidence list on a merge_attempt row is capped, and its TRUE
    length is recorded alongside so the truncation is never silent (task
    3178). A silent cap would reproduce this task's own failure mode — a
    reader inferring "3 commits were cited" from a row that was trimmed, just
    as "guards parse zero test ids" was inferred from a row that never carried
    evidence at all.

    (The truncating case itself is pinned at the unit level in
    test_merge_queue_disposition_wiring.py with a 22-SHA bundle; these
    end-to-end topologies are deliberately small, so here the assertion is
    that the totals AGREE with the stored slices.)
    """
    for key in ('failing_tests', 'implicated_commits', 'overlap_files'):
        assert key in data, f'missing evidence key {key!r} in {data!r}'
        stored = data[key]
        assert isinstance(stored, list), f'{key} must be a json list; got {stored!r}'
        assert len(stored) <= _MAX_EVENT_EVIDENCE_ITEMS, (
            f'{key} exceeds the cap: {len(stored)} > {_MAX_EVENT_EVIDENCE_ITEMS}'
        )
        total_key = f'{key}_total'
        assert total_key in data, f'missing {total_key!r} — bounding must not be silent'
        assert data[total_key] == len(stored), (
            f'{total_key}=={data[total_key]!r} but {key} holds {len(stored)} '
            f'items and this topology is below the cap — the two must agree'
        )


# ---------------------------------------------------------------------------
# Step-15 [user-observable signal / G6 first-attempt no-op trap closed]
# ---------------------------------------------------------------------------


class TestFirstAttemptSkewFailureDiagnostic:
    """A first-attempt integration-skew failure — driven through the REAL
    SpeculativeMergeWorker._run_inflight_verify — must surface disposition
    and failure_diagnostic on the delivered MergeOutcome verbatim.  This is
    exactly what the escalation server's merge_request response builder
    (server.py ~1169-1170: ``if outcome.failure_diagnostic is not None:
    result['failure_diagnostic'] = outcome.failure_diagnostic``) forwards to
    an MCP caller unchanged, so asserting on the outcome IS asserting on
    that surface."""

    def test_first_attempt_skew_surfaces_failure_diagnostic_verbatim(
        self, tmp_path: Path,
    ) -> None:
        repo, topology, store = _setup_skew_scenario(tmp_path)
        git_ops = _make_git_ops(repo)
        worker = SpeculativeMergeWorker(
            git_ops=git_ops, queue=asyncio.Queue(), event_store=store,
        )
        lease = HostLease(name='local', runner=MagicMock(), is_local=True)

        async def _run() -> MergeOutcome | None:
            future: asyncio.Future = asyncio.get_running_loop().create_future()
            _req, item = _make_item(
                tmp_path, repo, topology, future,
                merged_branch_tip=topology['branch_tip_sha'],
            )
            p1, p2 = _patched_verify_seams()
            with p1, p2:
                result = await worker._run_inflight_verify(item, lease)
            return result.outcome

        outcome = asyncio.run(_run())

        assert outcome is not None
        assert outcome.disposition == MergeFailureDisposition.INTEGRATION_SKEW, (
            f'Expected INTEGRATION_SKEW on a first attempt with no prior '
            f'merge_verify row; got {outcome.disposition!r}'
        )
        assert outcome.failure_diagnostic is not None
        diag_joined = ' '.join(outcome.failure_diagnostic.values())
        assert 'integration_skew' in diag_joined, diag_joined
        assert topology['landing_sha'] in diag_joined, diag_joined
        assert 'src/x.py' in diag_joined, diag_joined

        assert 'port landed commit' in outcome.reason, outcome.reason
        assert topology['landing_sha'] in outcome.reason, outcome.reason
        assert 'src/x.py' in outcome.reason, outcome.reason


class TestFirstAttemptSkewL1Escalation:
    """The SAME real-worker-produced outcome, routed through the REAL
    (unmocked) TaskWorkflow._submit_to_merge_queue + _mark_blocked +
    escalation_queue.submit, must file an L1 escalation carrying
    category='integration_skew' and a body with the implicated sha + overlap
    files verbatim."""

    def _make_routing_workflow(
        self, config: OrchestratorConfig, worktree: Path,
    ) -> TaskWorkflow:
        assignment = TaskAssignment(
            task_id=TASK_ID,
            task={
                'id': TASK_ID, 'title': 'X', 'description': '',
                'status': 'pending', 'metadata': {'files': ['lib']},
                'dependencies': [],
            },
            modules=['lib'],
        )
        git_ops = MagicMock(spec=GitOps)
        git_ops.get_main_sha = AsyncMock(return_value='deadbeef1234567890')
        git_ops.config = MagicMock(branch_prefix='task/', main_branch='main')
        git_ops.rebind_branch_to_head = AsyncMock(return_value=True)
        scheduler = MagicMock()
        scheduler.set_task_status = AsyncMock()
        workflow = TaskWorkflow(
            assignment=assignment,
            config=config,
            git_ops=git_ops,
            scheduler=scheduler,
            briefing=MagicMock(),
            mcp=MagicMock(),
        )
        workflow.worktree = worktree
        artifacts = TaskArtifacts(worktree)
        artifacts.init(TASK_ID, 'X', 'desc', base_commit='base-sha')
        workflow.artifacts = artifacts
        workflow.plan = {'task_id': TASK_ID, 'steps': []}
        return workflow

    def test_first_attempt_skew_routes_to_l1_escalation_with_diagnostic_body(
        self, tmp_path: Path,
    ) -> None:
        repo, topology, store = _setup_skew_scenario(tmp_path)
        git_ops = _make_git_ops(repo)
        worker = SpeculativeMergeWorker(
            git_ops=git_ops, queue=asyncio.Queue(), event_store=store,
        )
        lease = HostLease(name='local', runner=MagicMock(), is_local=True)

        routing_config = _make_config(tmp_path / 'routing-root')
        task_wt = tmp_path / 'task-wt'
        task_wt.mkdir()
        eq = EscalationQueue(tmp_path / 'escalations')

        async def _run() -> WorkflowOutcome:
            future: asyncio.Future = asyncio.get_running_loop().create_future()
            _req, item = _make_item(
                tmp_path, repo, topology, future,
                merged_branch_tip=topology['branch_tip_sha'],
            )
            p1, p2 = _patched_verify_seams()
            with p1, p2:
                vr = await worker._run_inflight_verify(item, lease)
            outcome = vr.outcome
            assert outcome is not None
            assert outcome.disposition == MergeFailureDisposition.INTEGRATION_SKEW

            workflow = self._make_routing_workflow(routing_config, task_wt)
            workflow.escalation_queue = eq

            async def _fake_enqueue(_queue, request, _event_store, **_kwargs):
                request.result.set_result(outcome)

            workflow.merge_queue = asyncio.Queue()
            with patch(
                'orchestrator.merge_queue.enqueue_merge_request', _fake_enqueue,
            ):
                return await workflow._submit_to_merge_queue(
                    BRANCH, pre_rebased=False, merge_phase=True,
                )

        result = asyncio.run(_run())

        assert result == WorkflowOutcome.BLOCKED
        filed = eq.get_by_task(TASK_ID, level=1)
        assert len(filed) >= 1, f'Expected an L1 escalation; got {filed}'
        esc = filed[0]
        assert esc.category == 'integration_skew', (
            f'Expected category=integration_skew; got {esc.category!r}'
        )
        assert 'integration_skew' in esc.detail, esc.detail
        assert topology['landing_sha'] in esc.detail, esc.detail
        assert 'src/x.py' in esc.detail, esc.detail


# ---------------------------------------------------------------------------
# Step-17 [I4 runs.db surface — merge_attempt carries disposition;
# anti-orphan for γ (task 2384)]
# ---------------------------------------------------------------------------


class TestFirstAttemptSkewMergeAttemptDisposition:
    """After a first-attempt skew failure is finalized through the REAL
    SpeculativeMergeWorker._finalize_inflight, a merge_attempt event for the
    branch's task_id must carry data['disposition']=='integration_skew' in
    the shared EventStore — proving γ's runs.db disposition field (task
    2384) is non-sentinel on the production skew path.  Control: on the
    INDETERMINATE path (absent base facts, boundary row 4) no merge_attempt
    row may carry a 'disposition' key at all — byte-identical to today."""

    def test_first_attempt_skew_emits_merge_attempt_with_disposition(
        self, tmp_path: Path,
    ) -> None:
        repo, topology, store = _setup_skew_scenario(tmp_path)

        asyncio.run(
            _finalize_via_real_worker(
                tmp_path, repo, topology, store,
                merged_branch_tip=topology['branch_tip_sha'],
            )
        )

        rows = [
            r for r in store.fetch_events_by_type(EventType.merge_attempt)
            if r['task_id'] == TASK_ID
        ]
        assert len(rows) == 1, f'Expected exactly one merge_attempt row; got {rows}'
        assert rows[0]['data'].get('disposition') == 'integration_skew', (
            f"Expected data['disposition']=='integration_skew'; got {rows[0]['data']!r}"
        )

    def test_first_attempt_skew_row_carries_bounded_evidence(
        self, tmp_path: Path,
    ) -> None:
        """Task 3178: the skew row no longer persists only
        ``{disposition, outcome}`` — it spells out the evidence the verdict was
        built from, so a census can read the attribution back out of runs.db
        instead of hand-reconstructing it from escalation prose."""
        repo, topology, store = _setup_skew_scenario(tmp_path)

        asyncio.run(
            _finalize_via_real_worker(
                tmp_path, repo, topology, store,
                merged_branch_tip=topology['branch_tip_sha'],
            )
        )

        rows = [
            r for r in store.fetch_events_by_type(EventType.merge_attempt)
            if r['task_id'] == TASK_ID
        ]
        assert len(rows) == 1, f'Expected exactly one merge_attempt row; got {rows}'
        data = rows[0]['data']
        assert data.get('disposition') == 'integration_skew', data
        assert data.get('failing_tests') == ['src/x.py::test_bar'], data
        assert topology['landing_sha'] in data.get('implicated_commits', []), data
        assert 'src/x.py' in data.get('overlap_files', []), data
        _assert_evidence_bounded(data)

    def test_indeterminate_first_attempt_emits_no_disposition_key(
        self, tmp_path: Path,
    ) -> None:
        """merged_branch_tip=None -> base facts absent -> classification is
        skipped inside _run_post_merge_verify (I3) -> disposition stays the
        MergeOutcome default INDETERMINATE -> the guarded step-18 emit must
        NOT fire (byte-identical to pre-β behaviour on this branch).

        Task 3178 made the distinction this pins LOAD-BEARING rather than
        incidental. The step-18 guard no longer keys on the disposition enum
        alone; it also fires whenever a bundle was GATHERED. That splits two
        INDETERMINATEs the old blanket exclusion conflated:

        * CLASSIFICATION-SKIPPED (this test) — no base facts, so the classifier
          never ran, ``MergeOutcome.skew_evidence`` keeps its ``None`` default,
          and NO row is emitted. Byte-identical to pre-3178, which is why the
          assertion below is unchanged.
        * ADJUDICATED (``TestAdjudicatedIndeterminateEmitsEvidenceRow``) — the
          classifier ran, cited real landings, and the I7 shape floor refused to
          promote. Evidence exists, so a row IS emitted with it.

        The companion ``skew_evidence is None`` assertion makes that seam
        explicit: if a future change starts populating it here, the guard would
        silently begin emitting on the skipped path and I3's fail-open
        byte-identity would be lost.
        """
        repo, topology, store = _setup_skew_scenario(tmp_path)

        outcome = asyncio.run(
            _finalize_via_real_worker(
                tmp_path, repo, topology, store,
                merged_branch_tip=None,
            )
        )

        rows_with_disposition = [
            r for r in store.fetch_events_by_type(EventType.merge_attempt)
            if r['task_id'] == TASK_ID and 'disposition' in (r['data'] or {})
        ]
        assert rows_with_disposition == [], (
            f"Expected no merge_attempt row carrying a 'disposition' key on "
            f"the INDETERMINATE path; got {rows_with_disposition}"
        )
        assert outcome is not None
        assert outcome.disposition == MergeFailureDisposition.INDETERMINATE
        assert outcome.skew_evidence is None, (
            f'Classification was SKIPPED (no base facts), so nothing was '
            f'gathered and skew_evidence must stay None — this is the seam the '
            f'step-18 emit guard keys on; got {outcome.skew_evidence!r}'
        )

    def test_classifier_fault_emits_no_disposition_key(
        self, tmp_path: Path,
    ) -> None:
        """I3 fail-open control (task 3178): force the classifier to RAISE on an
        otherwise-skew-shaped scenario. The wrapper's belt-and-suspenders except
        branch degrades to INDETERMINATE with no gathered bundle, so the widened
        guard must still emit nothing — a fault must never fabricate evidence.
        """
        repo, topology, store = _setup_skew_scenario(tmp_path)

        with patch(
            'orchestrator.merge_queue.classify_merge_failure_disposition',
            new=AsyncMock(side_effect=RuntimeError('classifier boom')),
        ):
            outcome = asyncio.run(
                _finalize_via_real_worker(
                    tmp_path, repo, topology, store,
                    merged_branch_tip=topology['branch_tip_sha'],
                )
            )

        rows_with_disposition = [
            r for r in store.fetch_events_by_type(EventType.merge_attempt)
            if r['task_id'] == TASK_ID and 'disposition' in (r['data'] or {})
        ]
        assert rows_with_disposition == [], (
            f'A classifier fault must stay byte-identical (I3): no '
            f'disposition-carrying merge_attempt row; got {rows_with_disposition}'
        )
        assert outcome is not None
        assert outcome.disposition == MergeFailureDisposition.INDETERMINATE
        assert outcome.skew_evidence is None, outcome.skew_evidence


class TestAdjudicatedIndeterminateEmitsEvidenceRow:
    """Task 3178, the headline case: reify 5566's LIVE shell-guard failure
    (``FAILED test_reify_audit_ptodo.sh``) driven through the REAL production
    chain, over a topology where the guard file IS a genuine main landing and
    the I5 green fact IS positively satisfied.

    Pre-3178 this exact shape produced a fabricated INTEGRATION_SKEW — the
    unconstrained pytest regex captured the guard's own filename as a failing
    test id, satisfying I7 vacuously (8 such dispositions between 07-24 and
    07-28). Two halves are asserted here, and they are independent:

    * CORRECTNESS (already green after steps 2/4): the verdict degrades to the
      honest INDETERMINATE, ``failure_diagnostic`` is ``None``, and the blocked
      reason carries NO "port landed commit" directive — nobody is steered off
      their own diff any more.
    * OBSERVABILITY (new here): the degrade is now MEASURABLE. A merge_attempt
      row is emitted carrying ``disposition == 'indeterminate'`` together with
      the gathered evidence, so a census can read straight out of runs.db that
      the gate bit, on which commits, and why (zero node-shaped test ids
      despite cited landings).
    """

    def _run_guard_scenario(
        self, tmp_path: Path, guard_name: str, verify_result: VerifyResult,
    ) -> tuple[MergeOutcome | None, list, dict[str, str]]:
        """Deliberately asserts NOTHING — the correctness assertions must be
        reachable independently of the observability ones, so that the two
        halves of this task fail (and pass) separately."""
        repo, topology, store = _setup_shell_guard_scenario(tmp_path, guard_name)

        outcome = asyncio.run(
            _finalize_via_real_worker(
                tmp_path, repo, topology, store,
                merged_branch_tip=topology['branch_tip_sha'],
                verify_result=verify_result,
            )
        )

        rows = [
            r for r in store.fetch_events_by_type(EventType.merge_attempt)
            if r['task_id'] == TASK_ID
        ]
        return outcome, rows, topology

    @staticmethod
    def _only_row_data(rows: list) -> dict:
        assert len(rows) == 1, f'Expected exactly one merge_attempt row; got {rows}'
        return rows[0]['data']

    def test_reify_5566_shell_guard_degrades_to_indeterminate(
        self, tmp_path: Path,
    ) -> None:
        """CORRECTNESS half, in isolation: green as of steps 2/4, before any of
        the observability wiring exists. This is the assertion that proves the
        8 false INTEGRATION_SKEW dispositions can no longer be produced."""
        outcome, _rows, _topology = self._run_guard_scenario(
            tmp_path, _SHELL_GUARD_NAME, _SHELL_GUARD_FAILURE,
        )

        assert outcome is not None
        assert outcome.disposition == MergeFailureDisposition.INDETERMINATE, (
            f'A shell-guard trip whose only FAILED token is a bare filename '
            f'must degrade to INDETERMINATE, not a fabricated INTEGRATION_SKEW; '
            f'got {outcome.disposition!r}'
        )
        assert outcome.failure_diagnostic is None, outcome.failure_diagnostic
        assert 'port landed commit' not in outcome.reason, outcome.reason
        assert 'integration_skew' not in outcome.reason, outcome.reason

    def test_reify_5566_shell_guard_degrades_and_is_measurable(
        self, tmp_path: Path,
    ) -> None:
        outcome, rows, topology = self._run_guard_scenario(
            tmp_path, _SHELL_GUARD_NAME, _SHELL_GUARD_FAILURE,
        )

        # --- correctness half (steps 2/4) ---------------------------------
        assert outcome is not None
        assert outcome.disposition == MergeFailureDisposition.INDETERMINATE, (
            f'A shell-guard trip whose only FAILED token is a bare filename '
            f'must degrade to INDETERMINATE, not a fabricated INTEGRATION_SKEW; '
            f'got {outcome.disposition!r}'
        )
        assert outcome.failure_diagnostic is None, outcome.failure_diagnostic
        assert 'port landed commit' not in outcome.reason, outcome.reason
        assert 'integration_skew' not in outcome.reason, outcome.reason

        # --- observability half (steps 9/10) ------------------------------
        # The classifier really did cite a landing; only the I7 gate refused to
        # promote it. Both facts are now on the row.
        assert outcome.skew_evidence is not None, (
            'The classifier gathered a bundle (real landing cited) — it must be '
            'carried on the outcome so the step-18 emit can persist it'
        )
        assert outcome.skew_evidence.failing_tests == ()
        assert topology['landing_sha'] in outcome.skew_evidence.implicated_commits

        data = self._only_row_data(rows)
        assert data.get('disposition') == 'indeterminate', data
        assert data.get('failing_tests') == [], (
            f'Zero node-shaped test ids is the REASON the gate bit — the row '
            f'must record it, not omit it: {data!r}'
        )
        assert topology['landing_sha'] in data.get('implicated_commits', []), data
        assert data.get('implicated_commits'), data
        assert _SHELL_GUARD_NAME in data.get('overlap_files', []), data
        _assert_evidence_bounded(data)

    def test_reify_5321_sweep_guard_degrades_and_is_measurable(
        self, tmp_path: Path,
    ) -> None:
        """A second verbatim string from the same census, so the end-to-end
        proof is not resting on a single captured shape."""
        outcome, rows, topology = self._run_guard_scenario(
            tmp_path, _SWEEP_GUARD_NAME, _SWEEP_GUARD_FAILURE,
        )

        assert outcome is not None
        assert outcome.disposition == MergeFailureDisposition.INDETERMINATE
        assert outcome.failure_diagnostic is None
        assert 'port landed commit' not in outcome.reason, outcome.reason
        data = self._only_row_data(rows)
        assert data.get('disposition') == 'indeterminate', data
        assert data.get('failing_tests') == [], data
        assert topology['landing_sha'] in data.get('implicated_commits', []), data
        assert _SWEEP_GUARD_NAME in data.get('overlap_files', []), data
        _assert_evidence_bounded(data)


# ---------------------------------------------------------------------------
# Amendment round 2 [reviewer_comprehensive, workflow.py:1757]: pins the
# documented I5 "any-prior-green" tradeoff across a review-bounce. This is
# NOT a new plan step — it's a regression test confirming TODAY's intended
# (accepted-tradeoff) disposition, so that a future change to the green-fact
# keying (e.g. keying on (task_id, base_sha) instead of task_id alone,
# inside merge_disposition.py — out of β's module scope) is a deliberate
# decision that updates this test, not a silent behavior change.
# ---------------------------------------------------------------------------


def _build_review_bounce_topology(repo: Path) -> dict[str, str]:
    """A real git repo modeling a review-bounce shape::

        merge_base (creates src/x.py)
          |-- pre_review_sha (task/9001; unrelated file — the commit state
          |                    workflow_verify(passed=True) was ACTUALLY
          |                    emitted for, before the review bounce)
          |-- post_bounce_sha (task/9001; edits src/x.py — the branch's OWN
          |                     new bug, introduced AFTER that green record)
          `-- landing_sha (main; also edits src/x.py — unrelated to the
                            branch's bug, but overlaps the same file)

    The branch's failing test (``_XPY_FAILURE``) implicates ``src/x.py``,
    which BOTH the branch's own post-bounce commit and main's unrelated
    landing touch. ``classify_merge_failure_disposition`` never diffs the
    branch's own commits against ``merge_base`` — it only (a) reads the I5
    green fact keyed by task_id (any-prior-green) and (b) greps
    ``merge_base_sha..main_sha`` (a main-only range) for landings touching a
    candidate file — so it cannot distinguish "main moved" from "the branch
    broke itself" here.
    """
    _init_git_repo(repo)
    merge_base_sha = _commit_file(repo, 'src/x.py', 'v1', 'init x (merge-base)')
    subprocess.run(
        ['git', 'checkout', '-q', '-b', BRANCH], cwd=repo, check=True, capture_output=True,
    )
    pre_review_sha = _commit_file(
        repo, 'src/branch_only.py', 'v1', 'pre-review work (unrelated file)',
    )
    post_bounce_sha = _commit_file(
        repo, 'src/x.py', 'v2-branch-bug', "post-review-bounce: branch's own bug in x.py",
    )
    subprocess.run(['git', 'checkout', '-q', 'main'], cwd=repo, check=True, capture_output=True)
    landing_sha = _commit_file(
        repo, 'src/x.py', 'v2-main-landing', 'unrelated main landing, also edits x.py',
    )
    return {
        'merge_base_sha': merge_base_sha,
        'pre_review_sha': pre_review_sha,
        'post_bounce_sha': post_bounce_sha,
        'landing_sha': landing_sha,
    }


class TestReviewBounceStaleGreenTradeoff:
    """Reviewer finding (amendment round 2): I5's any-prior-green semantics
    keep a stale ``workflow_verify(passed=True)`` row alive across a review
    bounce, so a genuine BRANCH_BUG introduced by post-review branch work can
    be misclassified as INTEGRATION_SKEW when an unrelated main landing
    happens to overlap the same file. This is a documented I5 tradeoff
    (``merge_disposition.py``'s any-prior-green keying — out of β's module
    scope to change); this test pins TODAY's intended behaviour so a future
    semantics change there is deliberate, not accidental."""

    def test_stale_green_survives_review_bounce_and_yields_skew(
        self, tmp_path: Path,
    ) -> None:
        repo = tmp_path / 'repo'
        repo.mkdir()
        topology = _build_review_bounce_topology(repo)
        store = EventStore(tmp_path / 'runs.db', run_id='run-test')
        # The stale green: emitted for the PRE-review-bounce commit, exactly
        # as the real VERIFY->REVIEW edge would (step 12) — keyed ONLY on
        # task_id per merge_disposition._branch_pre_merge_verify_green, so it
        # survives the branch's later (buggy) post-bounce commit untouched.
        _seed_branch_green(store, topology['pre_review_sha'])

        git_ops = _make_git_ops(repo)
        worker = SpeculativeMergeWorker(
            git_ops=git_ops, queue=asyncio.Queue(), event_store=store,
        )
        lease = HostLease(name='local', runner=MagicMock(), is_local=True)

        async def _run() -> MergeOutcome | None:
            future: asyncio.Future = asyncio.get_running_loop().create_future()
            _req, item = _make_item(
                tmp_path, repo, topology, future,
                merged_branch_tip=topology['post_bounce_sha'],
            )
            p1, p2 = _patched_verify_seams()
            with p1, p2:
                result = await worker._run_inflight_verify(item, lease)
            return result.outcome

        outcome = asyncio.run(_run())

        assert outcome is not None
        # Documented tradeoff: even though the failure genuinely originates
        # in the branch's OWN post-bounce commit, the stale any-prior-green
        # fact plus the overlapping main landing yield INTEGRATION_SKEW, not
        # BRANCH_BUG. If this assertion starts failing because the
        # classifier now keys the green fact on (task_id, base_sha) or
        # otherwise distinguishes this case, update it to assert BRANCH_BUG
        # instead — that's the deliberate fix the reviewer's suggested_fix
        # named, not a regression.
        assert outcome.disposition == MergeFailureDisposition.INTEGRATION_SKEW, (
            f'Expected the documented stale-green/review-bounce tradeoff to '
            f'yield INTEGRATION_SKEW; got {outcome.disposition!r}.'
        )
        assert outcome.failure_diagnostic is not None
        diag_joined = ' '.join(outcome.failure_diagnostic.values())
        assert topology['landing_sha'] in diag_joined, diag_joined
        assert 'src/x.py' in diag_joined, diag_joined
