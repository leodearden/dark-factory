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
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest
from _merge_lane_fakes import FakeVerifier, VerifyScript
from _merge_queue_harness import drive_verify_and_advance
from _orch_helpers import make_placeholder_future

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps
from orchestrator.merge_disposition import (
    MergeFailureDisposition,
    SkewEvidence,
)
from orchestrator.merge_queue import (
    MAIN_HEALTH_RED_REASON_PREFIX,
    MergeOutcome,
    MergeRequest,
    _classify_main_health_red,
    _resolve_dispatch_time_merge_base,
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


def _make_git_ops(tmp_path: Path) -> MagicMock:
    git_ops = MagicMock(spec=GitOps)
    git_ops.project_root = tmp_path
    git_ops.cleanup_merge_worktree = AsyncMock(return_value=None)
    git_ops.get_main_sha = AsyncMock(return_value=MAIN_SHA)
    return git_ops


def _seed_main_health_probe(
    verify: VerifyResult, main_sha: str, *, preexisting: bool,
) -> None:
    """Pick the main-health probe verdict for *verify* without stubbing the probe.

    The REAL ``verify_failure_is_preexisting_on_main`` consults verify.py's
    process-wide probe cache before it pays the probe-worktree cost, and
    production writes exactly these entries itself — so seeding one is how a
    test chooses the verdict while still running the real function. The key is
    composed with the same normaliser production uses, so the two cannot drift
    apart. ``reset_probe_cache`` clears the cache around every test.

    A seed that MISSED would fall through to a real probe, which a MagicMock
    git_ops degrades to the fail-safe ``(False, '')`` — silently right for the
    negative case. Every caller therefore also asserts
    ``git_ops.ephemeral_worktree`` was never reached, which is what keeps the
    seeding honest.
    """
    from orchestrator.verify import _PROBE_CACHE
    from orchestrator.workflow import _normalize_cause_hint

    key = (main_sha, verify.category or '', _normalize_cause_hint(verify.cause_hint))
    _PROBE_CACHE[key] = (time.monotonic(), preexisting)


def _exploding_branch() -> QueuedBranch:
    """A QueuedBranch stand-in whose bare_id raises — the I3 fail-open's fault seam.

    Delivered through ``MergeRequest(branch=...)``, a public constructor
    argument, so no lane module name has to be patched to reach the guard.
    """
    branch = MagicMock(spec=QueuedBranch)
    type(branch).bare_id = PropertyMock(side_effect=RuntimeError('injected fault'))
    return branch


def _make_req(
    task_id: str,
    worktree: Path,
    config: OrchestratorConfig,
    *,
    branch: QueuedBranch | None = None,
    result: asyncio.Future | None = None,
) -> MergeRequest:
    """Build a MergeRequest for the tests here.

    *result* defaults to a placeholder future, which is what the sync bodies
    below want: they call one function directly and never resolve the request.
    A body that drives a path which DOES resolve it (the worker's own lease
    path) passes a future created on its running loop instead.
    """
    return MergeRequest(
        task_id=task_id,
        branch=branch or QueuedBranch.parse(f'task/{task_id}', config.git.branch_prefix),
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=result if result is not None else make_placeholder_future(),
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

        _seed_main_health_probe(COMPILE_ERROR_RESULT, MAIN_SHA, preexisting=True)
        outcome = asyncio.run(
            _classify_main_health_red(git_ops, req, COMPILE_ERROR_RESULT)
        )

        git_ops.ephemeral_worktree.assert_not_called()
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

        _seed_main_health_probe(COMPILE_ERROR_RESULT, MAIN_SHA, preexisting=False)
        outcome = asyncio.run(
            _classify_main_health_red(git_ops, req, COMPILE_ERROR_RESULT)
        )

        git_ops.ephemeral_worktree.assert_not_called()
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

    def test_row4_fault_while_classifying_fails_open(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """I3 belt-and-suspenders: ANY exception raised while classifying
        degrades to INDETERMINATE.

        The fault is injected through the *request* — the wrapper reads
        ``req.branch.bare_id`` to build the classifier call — rather than by
        replacing ``classify_merge_failure_disposition``, because the
        classifier wraps its whole body in its own fail-open and therefore
        cannot itself raise. A fault in the surrounding marshalling is exactly
        the class this outer guard exists for, and a MergeRequest carrying an
        exploding branch is that fault delivered through a public constructor
        argument.
        """
        from orchestrator.merge_queue import _classify_disposition_for_outcome

        repo = tmp_path / 'repo'
        repo.mkdir()
        _init_git_repo(repo)
        merge_base_sha = _commit_file(repo, 'src/x.py', 'v1', 'init x (merge-base)')
        landing_sha = _commit_file(repo, 'src/x.py', 'v2', 'edit x on main (landing)')

        config = _make_config(repo)
        req = _make_req('2381', repo, config, branch=_exploding_branch())

        async def _run() -> tuple[
            MergeFailureDisposition, dict[str, str] | None, str, SkewEvidence | None,
        ]:
            with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
                return await _classify_disposition_for_outcome(
                    _XPY_FAILURE, req=req, merge_base_sha=merge_base_sha,
                    main_sha=landing_sha, event_store=None,
                )

        disposition, diag, reason_suffix, observed = asyncio.run(_run())
        assert disposition == MergeFailureDisposition.INDETERMINATE
        assert diag is None
        assert reason_suffix == ''
        # Task 3178, the I3 guarantee step-10's widened guard relies on: a
        # fault gathered NOTHING and must not fabricate a bundle, so no
        # merge_attempt row is emitted for the fail-open path.
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
    verify: VerifyResult,
    event_store: EventStore | None = None,
    merge_base_sha: str | None = None,
    main_sha: str | None = None,
) -> MergeOutcome | None:
    """Call _run_post_merge_verify with standard test parameters, threading
    the optional merge_base_sha/main_sha kw-params under test.

    *verify* is the scoped-verify verdict, supplied through the injected
    VerifyPort rather than by stubbing the module-level verify functions, and
    is also what the seeded main-health probe verdict is keyed on: every test
    here is about the NON-preexisting (task-fault) bucket, so the probe is
    seeded False and the generic classification path runs.
    """
    _seed_main_health_probe(verify, await git_ops.get_main_sha(), preexisting=False)
    outcome = await _run_post_merge_verify(
        git_ops, req, merge_wt,
        timeouts={},
        enospc_retries={},
        max_timeouts=3,
        max_enospc=1,
        event_store=event_store,
        merge_base_sha=merge_base_sha,
        main_sha=main_sha,
        verifier=FakeVerifier(default=VerifyScript(result=verify)),
    )
    git_ops.ephemeral_worktree.assert_not_called()  # type: ignore[attr-defined]
    return outcome


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

        outcome = asyncio.run(_drive_verify_with_base_facts(
            req, merge_wt, git_ops, verify=_XPY_FAILURE,
            event_store=store,
            merge_base_sha=merge_base_sha,
            main_sha=landing_sha,
        ))
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

        outcome = asyncio.run(_drive_verify_with_base_facts(
            req, merge_wt, git_ops, verify=_XPY_FAILURE,
            event_store=store,
            merge_base_sha=merge_base_sha,
            main_sha=main_sha,
        ))
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

        outcome = asyncio.run(_drive_verify_with_base_facts(
            req, merge_wt, git_ops, verify=_TIMED_OUT_FAILURE,
            merge_base_sha='deadbeef',
            main_sha='beadfeed',
        ))
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

        outcome = asyncio.run(
            _drive_verify_with_base_facts(req, merge_wt, git_ops, verify=_XPY_FAILURE)
        )
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


class TestDispatchTimeMergeBaseResolution:
    """_resolve_dispatch_time_merge_base is what turns an item's two FROZEN
    dispatch-time facts — base_sha and merged_branch_tip — into the
    merge_base_sha threaded into _run_post_merge_verify (task 2383 beta, 2357).

    Asserted against git itself rather than against a kwarg captured from a
    stubbed _run_post_merge_verify, so what is pinned is the computed value
    and its fail-safe degradation, not the call shape."""

    def test_merge_base_of_the_frozen_base_and_branch_tip(
        self, tmp_path: Path,
    ) -> None:
        repo = tmp_path / 'repo'
        repo.mkdir()
        _init_git_repo(repo)
        fork_point = _commit_file(repo, 'src/x.py', 'v1', 'init x (merge-base)')
        subprocess.run(
            ['git', 'checkout', '-q', '-b', 'task/2381'], cwd=repo, check=True,
        )
        branch_tip = _commit_file(repo, 'src/branch_only.py', 'v1', 'task work')
        subprocess.run(['git', 'checkout', '-q', 'main'], cwd=repo, check=True)
        # main advances past the fork point, so the merge base is deliberately
        # NOT the frozen base itself — a naive "return base_sha" would pass an
        # assertion made against a repo where they coincide.
        base_sha = _commit_file(repo, 'src/x.py', 'v2', 'edit x on main (landing)')

        resolved = asyncio.run(
            _resolve_dispatch_time_merge_base(repo, base_sha, branch_tip)
        )

        assert resolved == fork_point, (
            f'Expected git merge-base({base_sha}, {branch_tip}) == {fork_point}; '
            f'got {resolved!r}'
        )

    def test_missing_branch_tip_degrades_merge_base_to_none(
        self, tmp_path: Path,
    ) -> None:
        """merged_branch_tip=None (best-effort unavailable) -> None rather than
        raising. _run_post_merge_verify then skips classification entirely,
        which is pinned by TestRunPostMergeVerifyDispositionWiring::
        test_base_facts_absent_skips_classification_byte_identical (I3)."""
        repo = tmp_path / 'repo'
        repo.mkdir()
        _init_git_repo(repo)
        base_sha = _commit_file(repo, 'src/x.py', 'v1', 'init x')

        assert asyncio.run(
            _resolve_dispatch_time_merge_base(repo, base_sha, None)
        ) is None


@pytest.mark.asyncio
class TestRunInflightVerifyFreezesDispatchTimeMainSha:
    """The PRODUCTION CALLER half of the two classes either side of this one:
    ``_run_inflight_verify`` feeds ``_run_post_merge_verify`` the FROZEN
    dispatch-time ``item.base_sha`` as ``main_sha`` — never a fresh
    ``git_ops.get_main_sha()`` re-read (task 2383 β, 2357).

    Read off the DISPOSITION the whole path produces, with the real
    ``_run_post_merge_verify`` running underneath, rather than off a kwarg
    captured from a stub of it. What makes the two readings tell apart is the
    orphan fixture of :class:`TestRunPostMergeVerifyRealMainHeadFilter` below,
    lifted one level up:

      FROZEN — ``item.base_sha`` is an ORPHANED speculative tip, so the
          real-main ancestor filter prunes the commit it implicates and the
          verdict is the honest BRANCH_BUG.
      FRESH  — a re-read would name the real main head, whose landing commit
          touches the very file the verify failed on, survives the filter, and
          fabricates INTEGRATION_SKEW.

    Both readings resolve to the SAME ``merge_base_sha`` (the task branch and
    both tips fork at one commit), so the disposition swings on the frozen-vs-
    fresh choice alone. That is what keeps
    :class:`TestRunPostMergeVerifyRealMainHeadFilter`'s premise honest: its
    orphan can only reach the filter if this caller declines to re-read main.
    """

    async def test_frozen_orphan_base_yields_branch_bug_not_skew(
        self, tmp_path: Path,
    ) -> None:
        from orchestrator.git_ops import MergeResult
        from orchestrator.merge_queue import RealMergeItem, SpeculativeMergeWorker

        repo = tmp_path / 'repo'
        repo.mkdir()
        _init_git_repo(repo)
        fork_point = _commit_file(repo, 'src/x.py', 'v1', 'init x (merge-base / real fork)')
        subprocess.run(
            ['git', 'checkout', '-q', '-b', 'task/2381', fork_point],
            cwd=repo, check=True, capture_output=True,
        )
        branch_tip = _commit_file(repo, 'src/branch_only.py', 'v1', 'task work')
        subprocess.run(
            ['git', 'checkout', '-q', 'main'], cwd=repo, check=True, capture_output=True,
        )
        # Real main advances with a LANDING that touches src/x.py — the file
        # _XPY_FAILURE's failing test maps to. A fresh read would find it and
        # call the failure INTEGRATION_SKEW.
        real_main_head = _commit_file(repo, 'src/x.py', 'v2', 'edit x on main (landing)')
        # The frozen dispatch-time base: an orphan speculative tip forked at
        # the same point, touching the same file, never merged onto real main.
        subprocess.run(
            ['git', 'checkout', '-q', '-b', 'orphan', fork_point],
            cwd=repo, check=True, capture_output=True,
        )
        orphan_tip = _commit_file(repo, 'src/x.py', 'v2-orphan', 'orphan speculative edit x')
        subprocess.run(
            ['git', 'checkout', '-q', 'main'], cwd=repo, check=True, capture_output=True,
        )

        config = _make_config(repo)
        git_ops = _make_git_ops(repo)
        git_ops.get_main_sha = AsyncMock(return_value=real_main_head)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        task_wt = tmp_path / 'task-wt'
        task_wt.mkdir()
        req = _make_req(
            '2381', task_wt, config,
            result=asyncio.get_running_loop().create_future(),
        )
        _seed_main_health_probe(_XPY_FAILURE, real_main_head, preexisting=False)

        store = EventStore(tmp_path / 'runs.db', run_id='run-test')
        store.emit(
            EventType.workflow_verify, task_id='2381',
            data={'passed': True, 'base_sha': fork_point, 'branch': 'task/2381'},
        )

        item = RealMergeItem(
            request=req,
            merge_result=MergeResult(
                success=True, merge_commit='deadbeef', merge_worktree=merge_wt,
            ),
            merge_wt=merge_wt,
            base_sha=orphan_tip,
            speculative=True,
            merged_branch_tip=branch_tip,
        )
        worker = SpeculativeMergeWorker(
            git_ops=git_ops,
            queue=asyncio.Queue(),
            event_store=store,
            verifier=FakeVerifier(default=VerifyScript(result=_XPY_FAILURE)),
        )

        advanced = await drive_verify_and_advance(worker, item)

        assert advanced is False, 'a failing verify must not advance main'
        git_ops.ephemeral_worktree.assert_not_called()
        outcome = req.result.result()
        assert outcome.disposition == MergeFailureDisposition.BRANCH_BUG, (
            f'Expected BRANCH_BUG — main_sha must be the FROZEN orphaned '
            f'item.base_sha, whose implicated commit the real-main ancestor '
            f'filter prunes. INTEGRATION_SKEW here means the caller re-read '
            f'main; got {outcome.disposition!r}'
        )


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

        outcome = asyncio.run(_drive_verify_with_base_facts(
            req, merge_wt, git_ops, verify=_XPY_FAILURE,
            event_store=store,
            merge_base_sha=merge_base_sha,
            main_sha=orphan_tip,  # the frozen orphaned speculative base
        ))
        assert outcome is not None
        assert outcome.disposition == MergeFailureDisposition.BRANCH_BUG, (
            f'Expected BRANCH_BUG (orphan pruned by real-main ancestor filter); '
            f'got {outcome.disposition!r}'
        )
        assert outcome.failure_diagnostic is None
        assert 'port landed commit' not in outcome.reason, outcome.reason
        assert 'do not hunt your own diff' not in outcome.reason, outcome.reason
        assert outcome.reason.startswith('Post-merge verification failed'), outcome.reason


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

        return asyncio.run(_drive_verify_with_base_facts(
            req, merge_wt, git_ops, verify=verify, event_store=store,
            merge_base_sha=merge_base_sha, main_sha=main_sha,
        ))

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
