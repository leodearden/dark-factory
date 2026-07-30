"""Merge-skew β (task 2383, M2 of plans/merge-skew-attribution-prd.md): wire
alpha's classify_merge_failure_disposition into the production merge-gate
failure path + every I4 surface.

Boundary rows as tests:
  row1 (steps 1-2):   MAIN_RED — _classify_main_health_red stamps disposition.
  I4 surfacing (steps 3-4): _render_skew_surfaces pure helper.
  rows 2/3/4 (steps 5-6):   _classify_disposition_for_outcome async wrapper.
  I4 on the real outcome (steps 7-8): _run_post_merge_verify wiring.

This file stays at the merge_queue.py unit layer and does not touch
TaskWorkflow. The workflow-layer INTEGRATION_SKEW routing contract
(_submit_to_merge_queue -> _mark_blocked(category='integration_skew',
suggested_action='port_landed_change', escalate_to_human=True), steps 13-14)
is asserted directly — on the exact _mark_blocked kwargs — in
test_workflow_skew_surfacing.py::TestSubmitToMergeQueueIntegrationSkewRouting,
and end-to-end through the real merge worker in
test_merge_skew_end_to_end.py::TestFirstAttemptSkewL1Escalation.
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import make_placeholder_future

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps
from orchestrator.merge_disposition import MergeFailureDisposition, SkewEvidence
from orchestrator.merge_queue import (
    MAIN_HEALTH_RED_REASON_PREFIX,
    MergeOutcome,
    MergeRequest,
    _classify_main_health_red,
    _run_post_merge_verify,
)
from orchestrator.merge_types import OutcomeKind, QueuedBranch
from orchestrator.verify import VerifyResult

MAIN_SHA = 'cafecafe1234567890deadbeef'

COMPILE_ERROR_RESULT = VerifyResult(
    passed=False,
    test_output='',
    lint_output='',
    type_output='error TS2322: StatusBar.tsx:12',
    summary='tsc failed',
    cause_hint='error TS2322: StatusBar.tsx',
    category='compile_error',
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_probe_cache():
    """Clear the process-wide _PROBE_CACHE between tests (mirrors
    test_merge_queue_main_health.py's fixture)."""
    from orchestrator.verify import _PROBE_CACHE
    _PROBE_CACHE.clear()
    yield
    _PROBE_CACHE.clear()


def _make_config(tmp_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=tmp_path,
        max_concurrent_tasks=1,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    )


def _make_git_ops(tmp_path: Path) -> GitOps:
    git_ops = MagicMock(spec=GitOps)
    git_ops.project_root = tmp_path
    git_ops.cleanup_merge_worktree = AsyncMock(return_value=None)
    git_ops.get_main_sha = AsyncMock(return_value=MAIN_SHA)
    return git_ops


def _make_req(
    task_id: str,
    worktree: Path,
    config: OrchestratorConfig,
) -> MergeRequest:
    future = make_placeholder_future()
    return MergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(f'task/{task_id}', config.git.branch_prefix),
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=future,
        lane='normal',
    )


# ---------------------------------------------------------------------------
# Step-1/2 [boundary row 1 — MAIN_RED]
# ---------------------------------------------------------------------------


class TestClassifyMainHealthRedSetsDisposition:
    """_classify_main_health_red must stamp disposition=MAIN_RED on the
    outcome it returns when the preexisting probe confirms True (I1: probe
    order preserved; the classifier is never invoked for this bucket), while
    the fix-main reason prefix and dedupe_fingerprint stay unchanged."""

    def test_disposition_is_main_red(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        async def _run() -> MergeOutcome | None:
            with patch(
                'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                new=AsyncMock(return_value=(True, MAIN_SHA)),
            ):
                return await _classify_main_health_red(git_ops, req, COMPILE_ERROR_RESULT)

        outcome = asyncio.run(_run())
        assert outcome is not None
        assert outcome.disposition == MergeFailureDisposition.MAIN_RED, (
            f'Expected disposition=MAIN_RED; got {outcome.disposition!r}'
        )
        assert outcome.reason.startswith(MAIN_HEALTH_RED_REASON_PREFIX), (
            f'fix-main reason prefix must stay unchanged; got {outcome.reason!r}'
        )
        from orchestrator.workflow import compute_preexisting_main_break_fingerprint
        expected_fp = compute_preexisting_main_break_fingerprint(
            'compile_error', 'error TS2322: StatusBar.tsx', MAIN_SHA,
        )
        assert outcome.dedupe_fingerprint == expected_fp, (
            f'dedupe_fingerprint must stay unchanged; '
            f'expected={expected_fp!r} got={outcome.dedupe_fingerprint!r}'
        )

    def test_negative_classification_stays_indeterminate(self, tmp_path: Path) -> None:
        """When probe returns (False, ''), _classify_main_health_red returns
        None (falls through) — disposition is not this function's concern."""
        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        async def _run() -> MergeOutcome | None:
            with patch(
                'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                new=AsyncMock(return_value=(False, '')),
            ):
                return await _classify_main_health_red(git_ops, req, COMPILE_ERROR_RESULT)

        outcome = asyncio.run(_run())
        assert outcome is None


# ---------------------------------------------------------------------------
# Step-3/4 [I4 surfacing content] — _render_skew_surfaces pure helper
# ---------------------------------------------------------------------------


class TestRenderSkewSurfaces:
    """_render_skew_surfaces(disposition, evidence) is a pure helper: only
    INTEGRATION_SKEW (with non-None evidence) yields a non-empty reason_suffix
    + failure_diagnostic dict; every other disposition/evidence combination
    returns ('', None)."""

    def test_integration_skew_with_evidence_renders_surfaces(self) -> None:
        from orchestrator.merge_queue import _render_skew_surfaces

        evidence = SkewEvidence(
            implicated_commits=('abc123deadbeef',),
            failing_tests=('tests/test_foo.py::test_bar',),
            overlap_files=('a/b.py',),
        )
        reason_suffix, failure_diagnostic = _render_skew_surfaces(
            MergeFailureDisposition.INTEGRATION_SKEW, evidence,
        )

        assert 'integration_skew' in reason_suffix, reason_suffix
        assert 'abc123deadbeef' in reason_suffix, reason_suffix
        assert 'a/b.py' in reason_suffix, reason_suffix
        assert 'port landed commit' in reason_suffix, reason_suffix
        assert 'do not hunt your own diff' in reason_suffix, reason_suffix

        assert failure_diagnostic is not None
        assert all(isinstance(v, str) for v in failure_diagnostic.values()), (
            f'failure_diagnostic must be dict[str,str]; got {failure_diagnostic!r}'
        )
        joined = ' '.join(failure_diagnostic.values())
        assert 'integration_skew' in joined, joined
        assert 'abc123deadbeef' in joined, joined
        assert 'a/b.py' in joined, joined
        assert 'tests/test_foo.py::test_bar' in joined, joined

    @pytest.mark.parametrize('disposition', [
        MergeFailureDisposition.MAIN_RED,
        MergeFailureDisposition.BRANCH_BUG,
        MergeFailureDisposition.INDETERMINATE,
    ])
    def test_non_skew_dispositions_render_nothing(
        self, disposition: MergeFailureDisposition,
    ) -> None:
        from orchestrator.merge_queue import _render_skew_surfaces

        evidence = SkewEvidence(
            implicated_commits=('abc123',),
            failing_tests=('t',),
            overlap_files=('f.py',),
        )
        assert _render_skew_surfaces(disposition, evidence) == ('', None)

    def test_integration_skew_with_none_evidence_renders_nothing(self) -> None:
        from orchestrator.merge_queue import _render_skew_surfaces

        assert _render_skew_surfaces(
            MergeFailureDisposition.INTEGRATION_SKEW, None,
        ) == ('', None)


# ---------------------------------------------------------------------------
# Step-5/6 [boundary rows 2/3/4 at the classify-wrapper level]
# ---------------------------------------------------------------------------


def _init_git_repo(root: Path) -> None:
    """Init a real git repo at *root* (mirrors test_merge_disposition.py's
    fixture convention)."""
    for cmd in [
        ['git', 'init', '-q', '-b', 'main'],
        ['git', 'config', 'user.email', 'test@test.com'],
        ['git', 'config', 'user.name', 'Test'],
    ]:
        subprocess.run(cmd, cwd=root, check=True, capture_output=True)


def _commit_file(root: Path, rel_path: str, content: str, message: str) -> str:
    """Write+commit *rel_path* at *root*; return the new commit's full SHA."""
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


# A verify_result whose failing test maps to src/x.py (pytest id path segment +
# cause-hint file token both resolve to src/x.py) — mirrors
# test_merge_disposition.py's _XPY_FAILURE constant.
_XPY_FAILURE = VerifyResult(
    passed=False,
    test_output='FAILED src/x.py::test_bar - AssertionError',
    lint_output='',
    type_output='',
    summary='1 failed',
    cause_hint='src/x.py::test_bar',
    category='test_failure',
)

# A timed-out verify: no test/lint/type output and no cause_hint (mirrors how
# a wall-clock timeout short-circuits before any command produces file-shaped
# output). _extract_failing_tests_and_candidate_files degrades to an empty
# candidate-file set on this input, so classify_merge_failure_disposition
# returns INDETERMINATE before ever issuing a git log call (Amendment,
# reviewer_comprehensive round 3: pins this cheap degrade so a future
# refactor of the extraction ordering can't silently start classifying
# timeout noise).
_TIMED_OUT_FAILURE = VerifyResult(
    passed=False,
    test_output='',
    lint_output='',
    type_output='',
    summary='verify timed out',
    cause_hint='',
    category='',
    timed_out=True,
)


class TestClassifyDispositionForOutcome:
    """_classify_disposition_for_outcome(verify, *, req, merge_base_sha,
    main_sha, event_store) thinly wraps classify_merge_failure_disposition +
    _render_skew_surfaces, with a belt-and-suspenders fail-open (I3) atop the
    classifier's own fail-open."""

    def test_row2_integration_skew(self, tmp_path: Path) -> None:
        from orchestrator.merge_queue import _classify_disposition_for_outcome

        repo = tmp_path / 'repo'
        repo.mkdir()
        _init_git_repo(repo)
        merge_base_sha = _commit_file(repo, 'src/x.py', 'v1', 'init x (merge-base)')
        landing_sha = _commit_file(repo, 'src/x.py', 'v2', 'edit x on main (landing)')

        config = _make_config(repo)
        req = _make_req('2381', repo, config)
        store = EventStore(tmp_path / 'runs.db', run_id='run-test')
        store.emit(
            EventType.workflow_verify, task_id='2381',
            data={'passed': True, 'base_sha': merge_base_sha, 'branch': 'task/2381'},
        )

        async def _run() -> tuple[
            MergeFailureDisposition, dict[str, str] | None, str, SkewEvidence | None,
        ]:
            return await _classify_disposition_for_outcome(
                _XPY_FAILURE, req=req, merge_base_sha=merge_base_sha,
                main_sha=landing_sha, event_store=store,
            )

        disposition, diag, reason_suffix, observed = asyncio.run(_run())
        assert disposition == MergeFailureDisposition.INTEGRATION_SKEW
        assert diag is not None
        assert landing_sha in diag.get('implicated_commits', ''), diag
        assert reason_suffix, 'reason_suffix must be non-empty for INTEGRATION_SKEW'
        assert landing_sha in reason_suffix
        # Task 3178: the 4th element is the raw typed SkewEvidence the
        # classifier gathered — NOT the comma-joined failure_diagnostic strings,
        # which would have to be split back into a list to bound them.
        assert isinstance(observed, SkewEvidence)
        assert landing_sha in observed.implicated_commits

    def test_row3_branch_bug(self, tmp_path: Path) -> None:
        from orchestrator.merge_queue import _classify_disposition_for_outcome

        repo = tmp_path / 'repo'
        repo.mkdir()
        _init_git_repo(repo)
        merge_base_sha = _commit_file(repo, 'src/x.py', 'v1', 'init x (merge-base)')
        # main moves, but only touches an UNRELATED file -> no implicated landing.
        main_sha = _commit_file(repo, 'src/y.py', 'v1', 'add unrelated y on main')

        config = _make_config(repo)
        req = _make_req('2381', repo, config)
        store = EventStore(tmp_path / 'runs.db', run_id='run-test')
        store.emit(
            EventType.workflow_verify, task_id='2381',
            data={'passed': True, 'base_sha': merge_base_sha, 'branch': 'task/2381'},
        )

        async def _run() -> tuple[
            MergeFailureDisposition, dict[str, str] | None, str, SkewEvidence | None,
        ]:
            return await _classify_disposition_for_outcome(
                _XPY_FAILURE, req=req, merge_base_sha=merge_base_sha,
                main_sha=main_sha, event_store=store,
            )

        disposition, diag, reason_suffix, observed = asyncio.run(_run())
        assert disposition == MergeFailureDisposition.BRANCH_BUG
        assert diag is None
        assert reason_suffix == ''
        # Task 3178: no landings were implicated, so there is no bundle to
        # persist — which is what keeps the BRANCH_BUG merge_attempt row's
        # payload byte-identical under step-10's widened emit guard.
        assert observed is None

    def test_row4_classifier_fault_fails_open(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        from orchestrator.merge_queue import _classify_disposition_for_outcome

        repo = tmp_path / 'repo'
        repo.mkdir()
        _init_git_repo(repo)
        merge_base_sha = _commit_file(repo, 'src/x.py', 'v1', 'init x (merge-base)')
        landing_sha = _commit_file(repo, 'src/x.py', 'v2', 'edit x on main (landing)')

        config = _make_config(repo)
        req = _make_req('2381', repo, config)

        async def _run() -> tuple[
            MergeFailureDisposition, dict[str, str] | None, str, SkewEvidence | None,
        ]:
            with (
                patch(
                    'orchestrator.merge_queue.classify_merge_failure_disposition',
                    new=AsyncMock(side_effect=RuntimeError('injected fault')),
                ),
                caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'),
            ):
                return await _classify_disposition_for_outcome(
                    _XPY_FAILURE, req=req, merge_base_sha=merge_base_sha,
                    main_sha=landing_sha, event_store=None,
                )

        disposition, diag, reason_suffix, observed = asyncio.run(_run())
        assert disposition == MergeFailureDisposition.INDETERMINATE
        assert diag is None
        assert reason_suffix == ''
        # Task 3178, the I3 guarantee step-10's widened guard relies on: a
        # classifier fault gathered NOTHING and must not fabricate a bundle, so
        # no merge_attempt row is emitted for the fail-open path.
        assert observed is None
        warning_texts = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert warning_texts, 'Expected a WARNING to be logged on classifier fault'


# ---------------------------------------------------------------------------
# Step-7/8 [I4 on the real outcome + I3 absent-base-facts]
# ---------------------------------------------------------------------------


async def _drive_verify_with_base_facts(
    req: MergeRequest,
    merge_wt: Path,
    git_ops: GitOps,
    *,
    event_store: EventStore | None = None,
    merge_base_sha: str | None = None,
    main_sha: str | None = None,
) -> MergeOutcome | None:
    """Call _run_post_merge_verify with standard test parameters, threading
    the optional merge_base_sha/main_sha kw-params under test."""
    return await _run_post_merge_verify(
        git_ops, req, merge_wt,
        timeouts={},
        enospc_retries={},
        max_timeouts=3,
        max_enospc=1,
        event_store=event_store,
        merge_base_sha=merge_base_sha,
        main_sha=main_sha,
    )


class TestRunPostMergeVerifyDispositionWiring:
    """_run_post_merge_verify threads merge_base_sha/main_sha through to
    _classify_disposition_for_outcome on the generic (not-preexisting)
    task-fault path, attaching disposition/failure_diagnostic/reason_suffix
    to the returned MergeOutcome.  Absent base facts (default None) skips
    classification entirely — byte-identical to today (I3)."""

    def test_base_facts_supplied_with_implicated_landing_yields_integration_skew(
        self, tmp_path: Path,
    ) -> None:
        repo = tmp_path / 'repo'
        repo.mkdir()
        _init_git_repo(repo)
        merge_base_sha = _commit_file(repo, 'src/x.py', 'v1', 'init x (merge-base)')
        landing_sha = _commit_file(repo, 'src/x.py', 'v2', 'edit x on main (landing)')

        config = _make_config(repo)
        git_ops = _make_git_ops(repo)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        task_wt = tmp_path / 'task-wt'
        task_wt.mkdir()
        req = _make_req('2381', task_wt, config)

        store = EventStore(tmp_path / 'runs.db', run_id='run-test')
        store.emit(
            EventType.workflow_verify, task_id='2381',
            data={'passed': True, 'base_sha': merge_base_sha, 'branch': 'task/2381'},
        )

        async def _run() -> MergeOutcome | None:
            with (
                patch(
                    'orchestrator.merge_queue.run_scoped_verification',
                    new=AsyncMock(return_value=_XPY_FAILURE),
                ),
                patch(
                    'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                    new=AsyncMock(return_value=(False, '')),
                ),
            ):
                return await _drive_verify_with_base_facts(
                    req, merge_wt, git_ops,
                    event_store=store,
                    merge_base_sha=merge_base_sha,
                    main_sha=landing_sha,
                )

        outcome = asyncio.run(_run())
        assert outcome is not None
        assert outcome.disposition == MergeFailureDisposition.INTEGRATION_SKEW, (
            f'Expected INTEGRATION_SKEW; got {outcome.disposition!r}'
        )
        assert outcome.failure_diagnostic is not None
        assert landing_sha in outcome.failure_diagnostic.get('implicated_commits', ''), (
            outcome.failure_diagnostic
        )
        assert 'src/x.py' in outcome.failure_diagnostic.get('overlap_files', ''), (
            outcome.failure_diagnostic
        )
        assert 'port landed commit' in outcome.reason, outcome.reason
        assert landing_sha in outcome.reason

    def test_base_facts_supplied_no_implicated_landing_yields_branch_bug(
        self, tmp_path: Path,
    ) -> None:
        repo = tmp_path / 'repo'
        repo.mkdir()
        _init_git_repo(repo)
        merge_base_sha = _commit_file(repo, 'src/x.py', 'v1', 'init x (merge-base)')
        # main moves, but only touches an UNRELATED file -> no implicated landing.
        main_sha = _commit_file(repo, 'src/y.py', 'v1', 'add unrelated y on main')

        config = _make_config(repo)
        git_ops = _make_git_ops(repo)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        task_wt = tmp_path / 'task-wt'
        task_wt.mkdir()
        req = _make_req('2381', task_wt, config)

        store = EventStore(tmp_path / 'runs.db', run_id='run-test')
        store.emit(
            EventType.workflow_verify, task_id='2381',
            data={'passed': True, 'base_sha': merge_base_sha, 'branch': 'task/2381'},
        )

        async def _run() -> MergeOutcome | None:
            with (
                patch(
                    'orchestrator.merge_queue.run_scoped_verification',
                    new=AsyncMock(return_value=_XPY_FAILURE),
                ),
                patch(
                    'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                    new=AsyncMock(return_value=(False, '')),
                ),
            ):
                return await _drive_verify_with_base_facts(
                    req, merge_wt, git_ops,
                    event_store=store,
                    merge_base_sha=merge_base_sha,
                    main_sha=main_sha,
                )

        outcome = asyncio.run(_run())
        assert outcome is not None
        assert outcome.disposition == MergeFailureDisposition.BRANCH_BUG, (
            f'Expected BRANCH_BUG; got {outcome.disposition!r}'
        )
        assert outcome.failure_diagnostic is None
        assert 'port landed commit' not in outcome.reason
        assert outcome.reason.startswith('Post-merge verification failed'), outcome.reason

    def test_base_facts_supplied_timed_out_verify_stays_indeterminate(
        self, tmp_path: Path,
    ) -> None:
        """A timed-out verify (empty test_output/cause_hint) yields no
        candidate files, so classify_merge_failure_disposition degrades to
        INDETERMINATE before ever issuing a git log call — even though base
        facts ARE supplied and classification genuinely runs (unlike the
        merge_base_sha/main_sha=None skip-classification-entirely case
        below). Pins the cheap fail-open degrade so a future refactor of the
        extraction ordering can't silently start classifying timeout noise
        (Amendment, reviewer_comprehensive round 3)."""
        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        task_wt = tmp_path / 'task-wt'
        task_wt.mkdir()
        req = _make_req('2381', task_wt, config)

        async def _run() -> MergeOutcome | None:
            with (
                patch(
                    'orchestrator.merge_queue.run_scoped_verification',
                    new=AsyncMock(return_value=_TIMED_OUT_FAILURE),
                ),
                patch(
                    'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                    new=AsyncMock(return_value=(False, '')),
                ),
            ):
                return await _drive_verify_with_base_facts(
                    req, merge_wt, git_ops,
                    merge_base_sha='deadbeef',
                    main_sha='beadfeed',
                )

        outcome = asyncio.run(_run())
        assert outcome is not None
        assert outcome.disposition == MergeFailureDisposition.INDETERMINATE, (
            f'Expected INDETERMINATE for a timed-out verify with no '
            f'candidate files; got {outcome.disposition!r}'
        )
        assert outcome.failure_diagnostic is None
        assert 'port landed commit' not in outcome.reason
        assert outcome.reason.startswith('Post-merge verification failed'), outcome.reason

    def test_base_facts_absent_skips_classification_byte_identical(
        self, tmp_path: Path,
    ) -> None:
        """merge_base_sha/main_sha=None (default) -> classification skipped
        entirely; disposition stays INDETERMINATE, failure_diagnostic None,
        reason unchanged from today's shape (I3, byte-identical)."""
        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        task_wt = tmp_path / 'task-wt'
        task_wt.mkdir()
        req = _make_req('2381', task_wt, config)

        async def _run() -> MergeOutcome | None:
            with (
                patch(
                    'orchestrator.merge_queue.run_scoped_verification',
                    new=AsyncMock(return_value=_XPY_FAILURE),
                ),
                patch(
                    'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                    new=AsyncMock(return_value=(False, '')),
                ),
            ):
                return await _drive_verify_with_base_facts(req, merge_wt, git_ops)

        outcome = asyncio.run(_run())
        assert outcome is not None
        assert outcome.disposition == MergeFailureDisposition.INDETERMINATE, (
            f'Expected INDETERMINATE (default, classification skipped); '
            f'got {outcome.disposition!r}'
        )
        assert outcome.failure_diagnostic is None
        assert outcome.reason.startswith('Post-merge verification failed'), outcome.reason
        assert 'port landed commit' not in outcome.reason


# ---------------------------------------------------------------------------
# Step-9/10 [2357 dispatch-time base facts on the production path]
# ---------------------------------------------------------------------------


class TestRunInflightVerifyBaseFactsWiring:
    """_run_inflight_verify computes dispatch-time base facts — main_sha from
    item.base_sha (the frozen value captured at merge-dispatch time, NEVER a
    fresh get_main_sha() re-read) and merge_base_sha via git merge-base of
    that base and the item's frozen merged_branch_tip — and threads both
    into _run_post_merge_verify (task 2383 β, 2357)."""

    def test_base_facts_forwarded_from_item_not_fresh_main_read(
        self, tmp_path: Path,
    ) -> None:
        from orchestrator.git_ops import MergeResult
        from orchestrator.merge_queue import RealMergeItem, SpeculativeMergeWorker
        from orchestrator.verify_runner import HostLease

        repo = tmp_path / 'repo'
        repo.mkdir()
        _init_git_repo(repo)
        merge_base_sha = _commit_file(repo, 'src/x.py', 'v1', 'init x (merge-base)')
        subprocess.run(
            ['git', 'checkout', '-q', '-b', 'task/2381'], cwd=repo, check=True,
        )
        branch_tip = _commit_file(repo, 'src/branch_only.py', 'v1', 'task work')
        subprocess.run(['git', 'checkout', '-q', 'main'], cwd=repo, check=True)
        # item.base_sha: the FROZEN dispatch-time main SHA — deliberately
        # DIFFERENT from whatever a fresh get_main_sha() call would return
        # (git_ops.get_main_sha is stubbed to a distinguishing sentinel below
        # so any accidental fresh re-read would leak into the assertion).
        main_sha = _commit_file(repo, 'src/x.py', 'v2', 'edit x on main (landing)')

        config = _make_config(repo)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('2381', repo, config)

        git_ops = MagicMock()
        git_ops.get_main_sha = AsyncMock(return_value='should-not-be-used-sha')

        item = RealMergeItem(
            request=req,
            merge_result=MergeResult(
                success=True, merge_commit='deadbeef', merge_worktree=merge_wt,
            ),
            merge_wt=merge_wt,
            base_sha=main_sha,
            speculative=False,
            merged_branch_tip=branch_tip,
        )
        lease = HostLease(name='laptop', runner=MagicMock(), is_local=False)

        captured: dict = {}

        async def _fake_run_post_merge_verify(*_args, **kwargs):
            captured.update(kwargs)
            return MergeOutcome(
                'blocked', reason='Post-merge verification failed: boom',
                disposition=MergeFailureDisposition.INTEGRATION_SKEW,
            )

        async def _run():
            worker = SpeculativeMergeWorker(git_ops=git_ops, queue=asyncio.Queue())
            with patch(
                'orchestrator.merge_queue._run_post_merge_verify',
                _fake_run_post_merge_verify,
            ):
                return await worker._run_inflight_verify(item, lease)

        result = asyncio.run(_run())

        git_ops.get_main_sha.assert_not_called()
        assert captured.get('main_sha') == main_sha, captured
        assert captured.get('merge_base_sha') == merge_base_sha, captured
        assert result.outcome is not None
        assert result.outcome.disposition == MergeFailureDisposition.INTEGRATION_SKEW

    def test_missing_branch_tip_degrades_merge_base_to_none(
        self, tmp_path: Path,
    ) -> None:
        """merged_branch_tip=None (best-effort unavailable) -> merge_base_sha
        is passed as None rather than raising; classification degrades to
        INDETERMINATE inside _run_post_merge_verify (I3, fail-open)."""
        from orchestrator.git_ops import MergeResult
        from orchestrator.merge_queue import RealMergeItem, SpeculativeMergeWorker
        from orchestrator.verify_runner import HostLease

        repo = tmp_path / 'repo'
        repo.mkdir()
        _init_git_repo(repo)
        main_sha = _commit_file(repo, 'src/x.py', 'v1', 'init x')

        config = _make_config(repo)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('2381', repo, config)
        git_ops = MagicMock()
        git_ops.get_main_sha = AsyncMock(return_value='should-not-be-used-sha')

        item = RealMergeItem(
            request=req,
            merge_result=MergeResult(
                success=True, merge_commit='deadbeef', merge_worktree=merge_wt,
            ),
            merge_wt=merge_wt,
            base_sha=main_sha,
            speculative=False,
            merged_branch_tip=None,
        )
        lease = HostLease(name='laptop', runner=MagicMock(), is_local=False)

        captured: dict = {}

        async def _fake_run_post_merge_verify(*_args, **kwargs):
            captured.update(kwargs)
            return MergeOutcome(
                'blocked', reason='Post-merge verification failed: boom',
            )

        async def _run():
            worker = SpeculativeMergeWorker(git_ops=git_ops, queue=asyncio.Queue())
            with patch(
                'orchestrator.merge_queue._run_post_merge_verify',
                _fake_run_post_merge_verify,
            ):
                return await worker._run_inflight_verify(item, lease)

        asyncio.run(_run())

        assert captured.get('main_sha') == main_sha, captured
        assert captured.get('merge_base_sha') is None, captured


# ---------------------------------------------------------------------------
# Step-5 (task 2869) — _run_post_merge_verify resolves real main HEAD and
# threads it into classification so an orphaned speculative base_sha stops
# fabricating a false INTEGRATION_SKEW (reify esc-5260-8).
# ---------------------------------------------------------------------------


class TestRunPostMergeVerifyRealMainHeadFilter:
    """_run_post_merge_verify resolves the CURRENT real main HEAD via a
    fail-safe ``git_ops.get_main_sha()`` at classification time and threads it
    into ``_classify_disposition_for_outcome``. main_sha stays frozen =
    item.base_sha (a possibly ORPHANED speculative train tip); the ancestor
    filter then prunes the orphan's dangling commits, flipping a would-be
    false INTEGRATION_SKEW to the honest BRANCH_BUG."""

    def test_orphan_speculative_base_yields_branch_bug_not_skew(
        self, tmp_path: Path,
    ) -> None:
        repo = tmp_path / 'repo'
        repo.mkdir()
        _init_git_repo(repo)
        merge_base_sha = _commit_file(
            repo, 'src/x.py', 'v1', 'init x (merge-base / real fork)',
        )
        real_main_head = _commit_file(
            repo, 'src/z.py', 'zzz', 'unrelated real-main advance',
        )
        # Orphan speculative tip off merge_base touching src/x.py, never merged
        # onto real main (the 20-commit train tip in reify esc-5260-8).
        subprocess.run(
            ['git', 'checkout', '-q', '-b', 'orphan', merge_base_sha],
            cwd=repo, check=True, capture_output=True,
        )
        orphan_tip = _commit_file(repo, 'src/x.py', 'v2-orphan', 'orphan speculative edit x')
        subprocess.run(
            ['git', 'checkout', '-q', 'main'], cwd=repo, check=True, capture_output=True,
        )

        config = _make_config(repo)
        git_ops = _make_git_ops(repo)
        # get_main_sha resolves the REAL main tip — which does NOT have the
        # orphan tip as an ancestor.
        git_ops.get_main_sha = AsyncMock(return_value=real_main_head)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        task_wt = tmp_path / 'task-wt'
        task_wt.mkdir()
        req = _make_req('2381', task_wt, config)

        store = EventStore(tmp_path / 'runs.db', run_id='run-test')
        store.emit(
            EventType.workflow_verify, task_id='2381',
            data={'passed': True, 'base_sha': merge_base_sha, 'branch': 'task/2381'},
        )

        async def _run() -> MergeOutcome | None:
            with (
                patch(
                    'orchestrator.merge_queue.run_scoped_verification',
                    new=AsyncMock(return_value=_XPY_FAILURE),
                ),
                patch(
                    'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                    new=AsyncMock(return_value=(False, '')),
                ),
            ):
                return await _drive_verify_with_base_facts(
                    req, merge_wt, git_ops,
                    event_store=store,
                    merge_base_sha=merge_base_sha,
                    main_sha=orphan_tip,  # the frozen orphaned speculative base
                )

        outcome = asyncio.run(_run())
        assert outcome is not None
        assert outcome.disposition == MergeFailureDisposition.BRANCH_BUG, (
            f'Expected BRANCH_BUG (orphan pruned by real-main ancestor filter); '
            f'got {outcome.disposition!r}'
        )
        assert outcome.failure_diagnostic is None
        assert 'port landed commit' not in outcome.reason, outcome.reason
        assert 'do not hunt your own diff' not in outcome.reason, outcome.reason
        assert outcome.reason.startswith('Post-merge verification failed'), outcome.reason

    def test_classify_disposition_for_outcome_forwards_real_main_head_sha(
        self, tmp_path: Path,
    ) -> None:
        """_classify_disposition_for_outcome forwards a supplied
        real_main_head_sha into classify_merge_failure_disposition (RED today:
        the wrapper does not accept the kwarg)."""
        from orchestrator.merge_queue import _classify_disposition_for_outcome

        repo = tmp_path / 'repo'
        repo.mkdir()
        _init_git_repo(repo)
        merge_base_sha = _commit_file(repo, 'src/x.py', 'v1', 'init x')
        config = _make_config(repo)
        req = _make_req('2381', repo, config)

        captured: dict = {}

        async def _fake_classify(**kwargs):
            captured.update(kwargs)
            # 3-tuple since task 3178: (disposition, evidence, observed_evidence).
            return MergeFailureDisposition.BRANCH_BUG, None, None

        async def _run() -> tuple[
            MergeFailureDisposition, dict[str, str] | None, str, SkewEvidence | None,
        ]:
            with patch(
                'orchestrator.merge_queue.classify_merge_failure_disposition',
                new=_fake_classify,
            ):
                return await _classify_disposition_for_outcome(
                    _XPY_FAILURE, req=req, merge_base_sha=merge_base_sha,
                    main_sha='beadfeed', event_store=None,
                    real_main_head_sha='realmainheadsha',
                )

        disposition, _diag, _reason_suffix, _observed = asyncio.run(_run())
        assert captured.get('real_main_head_sha') == 'realmainheadsha', captured
        assert disposition == MergeFailureDisposition.BRANCH_BUG


# ---------------------------------------------------------------------------
# step-5 (task 3178) — bounded SkewEvidence on the merge_attempt payload
# ---------------------------------------------------------------------------


class TestEmitMergeAttemptSkewEvidence:
    """``_emit_merge_attempt`` gains a keyword-only ``skew_evidence`` so a
    merge_attempt row can persist the evidence bundle, not just
    ``{disposition, outcome}``. That observability gap is why the false "shell
    guards parse zero test ids" premise underpinning 2871/2918 survived two
    task cycles: nobody could read the evidence back out of runs.db to check it.

    The helper is deliberately DISPOSITION-AGNOSTIC about evidence — it writes
    whatever bundle it is handed. Deciding *when* to emit at all is the caller's
    guard (step-10), and that separation is what lets one code path serve both
    the INTEGRATION_SKEW row and the adjudicated-INDETERMINATE row."""

    @staticmethod
    def _emit_and_read(tmp_path: Path, **kwargs) -> dict:
        """Emit one merge_attempt against a REAL EventStore and read its data
        back (the direct-emit unit-test convention named in
        ``_emit_merge_attempt``'s docstring)."""
        from orchestrator.merge_queue import _emit_merge_attempt

        store = EventStore(tmp_path / 'runs.db', run_id='run-test')
        _emit_merge_attempt(store, '3178', OutcomeKind.verify_failed, **kwargs)
        rows = list(store.fetch_events_by_type(EventType.merge_attempt))
        assert len(rows) == 1, f'expected exactly one merge_attempt row; got {rows}'
        return rows[0]['data']

    def test_evidence_keys_written_on_integration_skew_row(self, tmp_path: Path):
        data = self._emit_and_read(
            tmp_path,
            disposition=MergeFailureDisposition.INTEGRATION_SKEW,
            skew_evidence=SkewEvidence(
                implicated_commits=('a' * 40,),
                failing_tests=('src/x.py::test_bar',),
                overlap_files=('src/x.py',),
            ),
        )
        assert data['disposition'] == 'integration_skew'
        # Stored as LISTS so the payload stays json-serialisable.
        assert data['failing_tests'] == ['src/x.py::test_bar']
        assert data['implicated_commits'] == ['a' * 40]
        assert data['overlap_files'] == ['src/x.py']

    def test_evidence_keys_written_on_adjudicated_indeterminate_row(
        self, tmp_path: Path,
    ):
        """The payload half of task 3178's acceptance criterion (2): an I7
        degrade persists the evidence the gate REFUSED to promote, and the row
        is self-identifying as indeterminate so a census can compute a real
        denominator for the adjudicated bucket."""
        data = self._emit_and_read(
            tmp_path,
            disposition=MergeFailureDisposition.INDETERMINATE,
            skew_evidence=SkewEvidence(
                implicated_commits=('b' * 40,),
                # The whole point: zero node-shaped ids DESPITE cited landings.
                failing_tests=(),
                overlap_files=('test_reify_audit_ptodo.sh',),
            ),
        )
        assert data['disposition'] == 'indeterminate'
        assert data['failing_tests'] == []
        assert data['implicated_commits'] == ['b' * 40]
        assert data['overlap_files'] == ['test_reify_audit_ptodo.sh']

    def test_lists_are_bounded_and_truncation_records_the_true_total(
        self, tmp_path: Path,
    ):
        """reify 5566 attempt-2 cited 22 SHAs touching 7 files. Bounding keeps a
        row from bloating runs.db, but a SILENT cap would reproduce this task's
        own failure mode — a reader inferring "3 commits were cited" from a
        truncated row, exactly as "guards parse zero test ids" was inferred from
        a row that never carried evidence. ``<key>_total`` makes the truncation
        self-describing."""
        from orchestrator.merge_queue import _MAX_EVENT_EVIDENCE_ITEMS

        shas = tuple(f'{i:040x}' for i in range(22))
        files = tuple(f'src/f{i}.py' for i in range(22))
        tests = tuple(f'src/f{i}.py::test_x' for i in range(22))
        data = self._emit_and_read(
            tmp_path,
            disposition=MergeFailureDisposition.INTEGRATION_SKEW,
            skew_evidence=SkewEvidence(
                implicated_commits=shas, failing_tests=tests, overlap_files=files,
            ),
        )
        cap = _MAX_EVENT_EVIDENCE_ITEMS
        assert cap < 22, 'fixture must exceed the cap for this test to mean anything'
        assert data['implicated_commits'] == list(shas[:cap])
        assert data['failing_tests'] == list(tests[:cap])
        assert data['overlap_files'] == list(files[:cap])
        # Truncation is never silent.
        assert data['implicated_commits_total'] == 22
        assert data['failing_tests_total'] == 22
        assert data['overlap_files_total'] == 22

    def test_omitting_skew_evidence_adds_no_keys(self, tmp_path: Path):
        """The module's established "omitted optional kwarg adds no key"
        convention (as for ``disposition``/``origin_host``/``probe_host``), so
        none of the ~20 existing call sites' payloads shift and none need
        auditing."""
        data = self._emit_and_read(tmp_path)
        assert data == {'outcome': OutcomeKind.verify_failed}

    def test_explicit_none_behaves_identically_to_omitting(self, tmp_path: Path):
        data = self._emit_and_read(tmp_path, skew_evidence=None)
        assert data == {'outcome': OutcomeKind.verify_failed}


# ---------------------------------------------------------------------------
# step-7 (task 3178) — thread the GATHERED bundle from classifier to outcome
# ---------------------------------------------------------------------------


_GUARD_FAILURE = VerifyResult(
    passed=False,
    # reify 5566, verbatim. The unconstrained ^FAILED\s+(\S+) used to return
    # 'test_reify_audit_ptodo.sh' as a "failing test id", satisfying I7 vacuously.
    cause_hint='FAILED test_reify_audit_ptodo.sh',
    test_output='',
    lint_output='',
    type_output='',
    summary='1 failed',
    category='test_failure',
)


def _guard_landing_repo(repo: Path) -> tuple[str, str]:
    """merge-base -> a genuine main landing that touches the guard's own file.
    Committed at the repo ROOT so the bare candidate-file token matches the
    ``git log -- <pathspec>``, which resolves from the repo root."""
    _init_git_repo(repo)
    merge_base_sha = _commit_file(
        repo, 'test_reify_audit_ptodo.sh', 'v1', 'init guard (merge-base)',
    )
    landing_sha = _commit_file(
        repo, 'test_reify_audit_ptodo.sh', 'v2', 'genuine landing on main',
    )
    return merge_base_sha, landing_sha


class TestClassifyDispositionForOutcomeObservedEvidence:
    """The wrapper returns the GATHERED bundle as its 4th element — the one the
    emit needs on BOTH paths — while still feeding the ADJUDICATED bundle to
    ``_render_skew_surfaces`` (so only INTEGRATION_SKEW renders the directive)."""

    def test_i7_degraded_indeterminate_still_returns_the_gathered_bundle(
        self, tmp_path: Path,
    ) -> None:
        """The new capability, and the reason the wrapper hands back
        ``observed_evidence`` rather than the adjudicated ``evidence``: an I7
        degrade cited real landings, and that is exactly what the census needs
        to be able to read."""
        from orchestrator.merge_queue import _classify_disposition_for_outcome

        repo = tmp_path / 'repo'
        repo.mkdir()
        merge_base_sha, landing_sha = _guard_landing_repo(repo)
        config = _make_config(repo)
        req = _make_req('2381', repo, config)
        store = EventStore(tmp_path / 'runs.db', run_id='run-test')
        store.emit(
            EventType.workflow_verify, task_id='2381',
            data={'passed': True, 'base_sha': merge_base_sha, 'branch': 'task/2381'},
        )

        async def _run() -> tuple[
            MergeFailureDisposition, dict[str, str] | None, str, SkewEvidence | None,
        ]:
            return await _classify_disposition_for_outcome(
                _GUARD_FAILURE, req=req, merge_base_sha=merge_base_sha,
                main_sha=landing_sha, event_store=store,
            )

        disposition, diag, reason_suffix, observed = asyncio.run(_run())
        assert disposition == MergeFailureDisposition.INDETERMINATE
        # No directive and no diagnostic — _render_skew_surfaces saw the
        # adjudicated (None) bundle, so its contract is untouched.
        assert diag is None
        assert reason_suffix == ''
        # ...but the gathered bundle IS handed back, with the honest signature
        # of an I7 degrade: landings cited, zero node-shaped test ids.
        assert isinstance(observed, SkewEvidence)
        assert landing_sha in observed.implicated_commits
        assert observed.failing_tests == ()
        assert 'test_reify_audit_ptodo.sh' in observed.overlap_files


class TestMergeOutcomeSkewEvidenceField:
    """``MergeOutcome.skew_evidence`` carries the gathered bundle to the step-18
    emit site. Additive with a None default, so every existing construction site
    is untouched."""

    def test_defaults_to_none(self) -> None:
        outcome = MergeOutcome('blocked', reason='x')
        assert outcome.skew_evidence is None

    def test_accepts_a_bundle(self) -> None:
        evidence = SkewEvidence(
            implicated_commits=('a' * 40,), failing_tests=(), overlap_files=('g.sh',),
        )
        assert MergeOutcome('blocked', reason='x', skew_evidence=evidence).skew_evidence \
            is evidence


class TestRunPostMergeVerifySkewEvidenceWiring:
    """The blocked-outcome construction path in ``_run_post_merge_verify``
    carries the classifier's gathered bundle onto the MergeOutcome — on the
    INTEGRATION_SKEW path AND on the I7-degraded INDETERMINATE path — while the
    classification-SKIPPED path (base facts absent) leaves it None. That last
    case is the seam step-10's emit guard keys on, so it is pinned here."""

    @staticmethod
    def _drive(
        tmp_path: Path, repo: Path, *, verify: VerifyResult,
        merge_base_sha: str | None, main_sha: str | None, task_id: str = '2381',
    ) -> MergeOutcome | None:
        config = _make_config(repo)
        git_ops = _make_git_ops(repo)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        task_wt = tmp_path / 'task-wt'
        task_wt.mkdir()
        req = _make_req(task_id, task_wt, config)
        store = EventStore(tmp_path / 'runs.db', run_id='run-test')
        store.emit(
            EventType.workflow_verify, task_id=task_id,
            data={'passed': True, 'base_sha': merge_base_sha, 'branch': f'task/{task_id}'},
        )

        async def _run() -> MergeOutcome | None:
            with (
                patch(
                    'orchestrator.merge_queue.run_scoped_verification',
                    new=AsyncMock(return_value=verify),
                ),
                patch(
                    'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                    new=AsyncMock(return_value=(False, '')),
                ),
            ):
                return await _drive_verify_with_base_facts(
                    req, merge_wt, git_ops, event_store=store,
                    merge_base_sha=merge_base_sha, main_sha=main_sha,
                )

        return asyncio.run(_run())

    def test_integration_skew_outcome_carries_the_bundle(self, tmp_path: Path) -> None:
        repo = tmp_path / 'repo'
        repo.mkdir()
        _init_git_repo(repo)
        merge_base_sha = _commit_file(repo, 'src/x.py', 'v1', 'init x (merge-base)')
        landing_sha = _commit_file(repo, 'src/x.py', 'v2', 'edit x on main (landing)')

        outcome = self._drive(
            tmp_path, repo, verify=_XPY_FAILURE,
            merge_base_sha=merge_base_sha, main_sha=landing_sha,
        )
        assert outcome is not None
        assert outcome.disposition == MergeFailureDisposition.INTEGRATION_SKEW
        assert isinstance(outcome.skew_evidence, SkewEvidence)
        assert landing_sha in outcome.skew_evidence.implicated_commits

    def test_i7_degraded_indeterminate_outcome_carries_the_bundle(
        self, tmp_path: Path,
    ) -> None:
        """The ADJUDICATED INDETERMINATE: the classifier ran, cited a genuine
        landing, and the I7 gate refused to promote. This is the class task 3178
        makes measurable."""
        repo = tmp_path / 'repo'
        repo.mkdir()
        merge_base_sha, landing_sha = _guard_landing_repo(repo)

        outcome = self._drive(
            tmp_path, repo, verify=_GUARD_FAILURE,
            merge_base_sha=merge_base_sha, main_sha=landing_sha,
        )
        assert outcome is not None
        assert outcome.disposition == MergeFailureDisposition.INDETERMINATE
        assert outcome.failure_diagnostic is None
        assert isinstance(outcome.skew_evidence, SkewEvidence)
        assert landing_sha in outcome.skew_evidence.implicated_commits
        assert outcome.skew_evidence.failing_tests == ()

    def test_classification_skipped_leaves_skew_evidence_none(
        self, tmp_path: Path,
    ) -> None:
        """The SKIPPED INDETERMINATE: base facts absent, so classification never
        runs and nothing is gathered. This is the seam step-10's guard keys on —
        it emits nothing here, byte-identical to pre-3178 behaviour (I3)."""
        repo = tmp_path / 'repo'
        repo.mkdir()
        _init_git_repo(repo)
        _commit_file(repo, 'src/x.py', 'v1', 'init x')

        outcome = self._drive(
            tmp_path, repo, verify=_XPY_FAILURE,
            merge_base_sha=None, main_sha=None,
        )
        assert outcome is not None
        assert outcome.disposition == MergeFailureDisposition.INDETERMINATE
        assert outcome.skew_evidence is None
