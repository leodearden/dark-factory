"""Tests for orchestrator.merge_disposition — merge-skew attribution mechanism M1
(PRD: plans/merge-skew-attribution-prd.md, task 2381 α).

Git-only, read-only, fail-open classifier that refines a merge-verify failure
into a MergeFailureDisposition (MAIN_RED / INTEGRATION_SKEW / BRANCH_BUG /
INDETERMINATE), given the caller-computed ``preexisting`` bucket. Covers:

  step-1:  surface/contract — MergeFailureDisposition member set, SkewEvidence
           frozen dataclass shape, MergeOutcome.disposition default + dual
           import (merge_types / merge_queue shim), classify_* importability
           and coroutine-ness.
  step-3:  MAIN_RED short-circuit + I1 no-reprobe.
  step-5:  failing-test/candidate-file extraction helper
           (_extract_failing_tests_and_candidate_files).
  step-7:  _implicated_landings against a real git repo (I2 read-only,
           2357 authoritative-SHA plumbing).
  step-9:  _branch_pre_merge_verify_green against a real EventStore (G6).
  step-11: end-to-end INTEGRATION_SKEW + I2 read-only assertion.
  step-13: BRANCH_BUG (no implicated landings).
  step-15: INDETERMINATE degrade (implicated landings but green not confirmed).
  step-17: fail-open fault injection (I3).
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import subprocess
from pathlib import Path

import pytest

from orchestrator.event_store import EventStore, EventType
from orchestrator.verify import VerifyResult

# ---------------------------------------------------------------------------
# Helpers / constants
# ---------------------------------------------------------------------------


def _init_git_repo(root: Path) -> None:
    """Init a real git repo at *root* (subprocess, mirrors
    test_verify_preexisting_main_break.py:383's fixture convention)."""
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


ANY_VERIFY_RESULT = VerifyResult(
    passed=False,
    test_output='FAILED tests/test_foo.py::test_bar',
    lint_output='',
    type_output='',
    summary='1 failed',
    cause_hint='tests/test_foo.py::test_bar',
    category='test_failure',
)

# ---------------------------------------------------------------------------
# step-1 — surface/contract
# ---------------------------------------------------------------------------


class TestMergeFailureDispositionMemberSet:
    """MergeFailureDisposition is a StrEnum with exactly the 4 named members."""

    def test_member_values_match_expected_set_exactly(self):
        from orchestrator.merge_disposition import MergeFailureDisposition
        assert {d.value for d in MergeFailureDisposition} == {
            'main_red', 'integration_skew', 'branch_bug', 'indeterminate',
        }

    def test_member_count_is_four(self):
        from orchestrator.merge_disposition import MergeFailureDisposition
        assert len(list(MergeFailureDisposition)) == 4

    def test_is_strenum_subclass(self):
        from enum import StrEnum

        from orchestrator.merge_disposition import MergeFailureDisposition
        assert issubclass(MergeFailureDisposition, StrEnum)

    def test_named_members_have_expected_values(self):
        from orchestrator.merge_disposition import MergeFailureDisposition
        assert MergeFailureDisposition.MAIN_RED == 'main_red'
        assert MergeFailureDisposition.INTEGRATION_SKEW == 'integration_skew'
        assert MergeFailureDisposition.BRANCH_BUG == 'branch_bug'
        assert MergeFailureDisposition.INDETERMINATE == 'indeterminate'


class TestSkewEvidenceFrozenDataclass:
    """SkewEvidence is a frozen dataclass carrying three tuple[str, ...] fields."""

    def test_construct_and_read_fields(self):
        from orchestrator.merge_disposition import SkewEvidence
        evidence = SkewEvidence(
            implicated_commits=('abc123',),
            failing_tests=('test_foo',),
            overlap_files=('src/foo.py',),
        )
        assert evidence.implicated_commits == ('abc123',)
        assert evidence.failing_tests == ('test_foo',)
        assert evidence.overlap_files == ('src/foo.py',)

    def test_is_frozen(self):
        from orchestrator.merge_disposition import SkewEvidence
        evidence = SkewEvidence(
            implicated_commits=('abc123',),
            failing_tests=('test_foo',),
            overlap_files=('src/foo.py',),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            evidence.implicated_commits = ('def456',)  # type: ignore[misc]


class TestMergeOutcomeDispositionField:
    """MergeOutcome.disposition defaults to INDETERMINATE and is dual-importable.

    Note: only ``MergeOutcome`` itself needs to flow through the merge_queue
    shim (it already does, pre-existing re-export) — the design explicitly
    keeps merge_queue.py unedited, so ``MergeFailureDisposition`` is imported
    from merge_types/merge_disposition, never from merge_queue.
    """

    def test_default_is_indeterminate(self):
        from orchestrator.merge_types import MergeFailureDisposition, MergeOutcome
        outcome = MergeOutcome(status='done')
        assert outcome.disposition == MergeFailureDisposition.INDETERMINATE

    def test_importable_from_merge_types(self):
        from orchestrator.merge_types import MergeFailureDisposition  # noqa: F401

    def test_merge_outcome_importable_from_merge_queue_shim_carries_disposition(self):
        """merge_queue re-exports MergeOutcome (pre-existing shim); the new
        disposition field must flow through with zero merge_queue.py edits."""
        from orchestrator.merge_disposition import MergeFailureDisposition
        from orchestrator.merge_queue import MergeOutcome
        outcome = MergeOutcome(status='done')
        assert outcome.disposition == MergeFailureDisposition.INDETERMINATE

    def test_merge_types_and_merge_queue_export_the_same_mergeoutcome_class(self):
        """The shim must re-export the identical MergeOutcome class, not a copy."""
        from orchestrator.merge_queue import MergeOutcome as ShimMergeOutcome
        from orchestrator.merge_types import MergeOutcome as TypesMergeOutcome
        assert ShimMergeOutcome is TypesMergeOutcome


class TestClassifyMergeFailureDispositionImportable:
    """classify_merge_failure_disposition is importable and a coroutine function."""

    def test_importable_from_merge_disposition_module(self):
        from orchestrator.merge_disposition import (  # noqa: F401
            classify_merge_failure_disposition,
        )

    def test_is_coroutine_function(self):
        from orchestrator.merge_disposition import classify_merge_failure_disposition
        assert inspect.iscoroutinefunction(classify_merge_failure_disposition), (
            'classify_merge_failure_disposition must be an async def (coroutine function)'
        )


# ---------------------------------------------------------------------------
# step-3 — MAIN_RED short-circuit + I1 no-reprobe
# ---------------------------------------------------------------------------


class TestMainRedShortCircuit:
    """preexisting=True short-circuits to MAIN_RED [boundary row 1] without
    ever re-probing verify_failure_is_preexisting_on_main (I1)."""

    def test_preexisting_true_returns_main_red(self):
        from orchestrator.merge_disposition import (
            MergeFailureDisposition,
            classify_merge_failure_disposition,
        )

        result = asyncio.run(classify_merge_failure_disposition(
            verify_result=ANY_VERIFY_RESULT,
            branch='task/999',
            merge_base_sha='base000',
            main_sha='main111',
            preexisting=True,
        ))
        assert result == (MergeFailureDisposition.MAIN_RED, None)

    def test_preexisting_true_never_reprobes_verify_failure_is_preexisting_on_main(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        """I1: classify must never call verify_failure_is_preexisting_on_main
        itself — it only refines the caller-computed ``preexisting`` bucket."""
        from orchestrator import verify as verify_module
        from orchestrator.merge_disposition import (
            MergeFailureDisposition,
            classify_merge_failure_disposition,
        )

        def _raise_if_called(*args, **kwargs):
            raise AssertionError(
                'classify_merge_failure_disposition must never call '
                'verify_failure_is_preexisting_on_main (I1: no re-probing)'
            )

        monkeypatch.setattr(
            verify_module, 'verify_failure_is_preexisting_on_main', _raise_if_called,
        )

        result = asyncio.run(classify_merge_failure_disposition(
            verify_result=ANY_VERIFY_RESULT,
            branch='task/999',
            merge_base_sha='base000',
            main_sha='main111',
            preexisting=True,
        ))
        assert result == (MergeFailureDisposition.MAIN_RED, None)


# ---------------------------------------------------------------------------
# step-5 — failing-test/candidate-file extraction helper (Open Q1)
# ---------------------------------------------------------------------------


class TestExtractFailingTestsAndCandidateFiles:
    """Best-effort failing-test-id/candidate-file extraction from a
    VerifyResult's free-text test_output/cause_hint (Open Q1). Biased so
    ambiguous/uninformative output yields an empty candidate set rather than
    a guess (a false skew is the costly error to avoid)."""

    def test_pytest_style_failed_line_and_cause_hint_path(self):
        from orchestrator.merge_disposition import (
            _extract_failing_tests_and_candidate_files,
        )

        verify_result = VerifyResult(
            passed=False,
            test_output=(
                'FAILED tests/test_foo.py::test_bar - AssertionError\n'
                '=== 1 failed in 0.12s ==='
            ),
            lint_output='',
            type_output='',
            summary='1 failed',
            cause_hint='tests/test_foo.py::test_bar',
            category='test_failure',
        )

        failing_tests, candidate_files = _extract_failing_tests_and_candidate_files(
            verify_result,
        )
        assert 'tests/test_foo.py::test_bar' in failing_tests
        assert 'tests/test_foo.py' in candidate_files

    def test_rust_nextest_style_failed_line_and_file_path_in_cause_hint(self):
        from orchestrator.merge_disposition import (
            _extract_failing_tests_and_candidate_files,
        )

        verify_result = VerifyResult(
            passed=False,
            test_output='test crate::mod::test_x ... FAILED',
            lint_output='',
            type_output='',
            summary='1 failed',
            cause_hint='crates/foo/src/bar.rs: assertion `left == right` failed',
            category='test_failure',
        )

        failing_tests, candidate_files = _extract_failing_tests_and_candidate_files(
            verify_result,
        )
        assert 'crate::mod::test_x' in failing_tests
        assert 'crates/foo/src/bar.rs' in candidate_files

    def test_empty_output_yields_empty_candidate_set(self):
        from orchestrator.merge_disposition import (
            _extract_failing_tests_and_candidate_files,
        )

        verify_result = VerifyResult(
            passed=False,
            test_output='',
            lint_output='',
            type_output='',
            summary='',
            cause_hint='',
            category='',
        )

        failing_tests, candidate_files = _extract_failing_tests_and_candidate_files(
            verify_result,
        )
        assert failing_tests == ()
        assert candidate_files == ()

    def test_uninformative_output_yields_empty_candidate_set(self):
        """No FAILED lines and no file-shaped tokens anywhere -> bias to
        empty rather than guessing (ambiguity -> no implicated landings)."""
        from orchestrator.merge_disposition import (
            _extract_failing_tests_and_candidate_files,
        )

        verify_result = VerifyResult(
            passed=False,
            test_output='Something went wrong. No further details available.',
            lint_output='',
            type_output='',
            summary='error',
            cause_hint='',
            category='unknown',
        )

        failing_tests, candidate_files = _extract_failing_tests_and_candidate_files(
            verify_result,
        )
        assert failing_tests == ()
        assert candidate_files == ()


# ---------------------------------------------------------------------------
# step-7 — _implicated_landings against a real git repo
# ---------------------------------------------------------------------------


class TestImplicatedLandings:
    """_implicated_landings(repo_root, merge_base_sha, main_sha, candidate_files)
    is a read-only ``git log --name-only`` over merge_base_sha..main_sha (I2,
    2357: both SHAs are authoritative caller-supplied inputs)."""

    def test_landing_touches_candidate_file(self, tmp_path: Path):
        from orchestrator.merge_disposition import _implicated_landings

        repo = tmp_path / 'repo'
        repo.mkdir()
        _init_git_repo(repo)
        merge_base_sha = _commit_file(repo, 'src/x.py', 'v1', 'init x')
        landing_sha = _commit_file(repo, 'src/x.py', 'v2', 'edit x on main')

        implicated_commits, overlap_files = asyncio.run(
            _implicated_landings(repo, merge_base_sha, landing_sha, {'src/x.py'}),
        )
        assert landing_sha in implicated_commits
        assert 'src/x.py' in overlap_files

    def test_landing_touches_unrelated_file_no_overlap(self, tmp_path: Path):
        from orchestrator.merge_disposition import _implicated_landings

        repo = tmp_path / 'repo'
        repo.mkdir()
        _init_git_repo(repo)
        merge_base_sha = _commit_file(repo, 'src/x.py', 'v1', 'init x')
        main_sha = _commit_file(repo, 'src/y.py', 'v1', 'add y on main')

        implicated_commits, overlap_files = asyncio.run(
            _implicated_landings(repo, merge_base_sha, main_sha, {'src/x.py'}),
        )
        assert implicated_commits == ()
        assert overlap_files == ()

    def test_non_git_dir_returns_empty_fail_safe(self, tmp_path: Path):
        """repo_root exists on disk but is not a git repo -> git log fails
        (rc != 0); the helper must degrade to empty, never raise (I3-style
        fail-safe for the git-plumbing helper)."""
        from orchestrator.merge_disposition import _implicated_landings

        non_git_dir = tmp_path / 'not_a_repo'
        non_git_dir.mkdir()

        implicated_commits, overlap_files = asyncio.run(
            _implicated_landings(non_git_dir, 'deadbeef', 'cafef00d', {'src/x.py'}),
        )
        assert implicated_commits == ()
        assert overlap_files == ()


# ---------------------------------------------------------------------------
# step-9 — _branch_pre_merge_verify_green against a real EventStore
# ---------------------------------------------------------------------------


def _make_event_store(tmp_path: Path) -> EventStore:
    """A real on-disk EventStore (mirrors event_store's own test convention:
    a runs.db under tmp_path with a fixed run_id)."""
    return EventStore(tmp_path / 'runs.db', run_id='run-test')


def _emit_workflow_verify(
    store: EventStore, task_id: str, *, passed: bool, branch: str = 'task/1',
) -> None:
    store.emit(
        EventType.workflow_verify,
        task_id=task_id,
        data={'passed': passed, 'base_sha': 'basesha', 'branch': branch},
    )


class TestBranchPreMergeVerifyGreen:
    """_branch_pre_merge_verify_green reads EventType.workflow_verify rows for
    the task_id (I5 amendment): True if any passing row, False if rows exist
    but none passed, None if no rows / None inputs / read error. The
    load-bearing amendment proof is that it reads workflow_verify, NOT
    merge_verify."""

    def test_any_prior_green_wins_over_later_failure(self, tmp_path: Path):
        from orchestrator.merge_disposition import _branch_pre_merge_verify_green

        store = _make_event_store(tmp_path)
        _emit_workflow_verify(store, '2381', passed=True)
        _emit_workflow_verify(store, '2381', passed=False)
        # any-prior-green, not most-recent-wins
        assert _branch_pre_merge_verify_green(store, '2381') is True

    def test_only_failing_rows_returns_false(self, tmp_path: Path):
        from orchestrator.merge_disposition import _branch_pre_merge_verify_green

        store = _make_event_store(tmp_path)
        _emit_workflow_verify(store, '2381', passed=False)
        _emit_workflow_verify(store, '2381', passed=False)
        assert _branch_pre_merge_verify_green(store, '2381') is False

    def test_no_rows_for_task_returns_none(self, tmp_path: Path):
        from orchestrator.merge_disposition import _branch_pre_merge_verify_green

        store = _make_event_store(tmp_path)
        # rows exist, but for a DIFFERENT task -> task_id filtering yields None
        _emit_workflow_verify(store, '9999', passed=True)
        assert _branch_pre_merge_verify_green(store, '2381') is None

    def test_empty_store_returns_none(self, tmp_path: Path):
        from orchestrator.merge_disposition import _branch_pre_merge_verify_green

        store = _make_event_store(tmp_path)
        assert _branch_pre_merge_verify_green(store, '2381') is None

    def test_none_event_store_returns_none(self):
        from orchestrator.merge_disposition import _branch_pre_merge_verify_green

        assert _branch_pre_merge_verify_green(None, '2381') is None

    def test_none_task_id_returns_none(self, tmp_path: Path):
        from orchestrator.merge_disposition import _branch_pre_merge_verify_green

        store = _make_event_store(tmp_path)
        _emit_workflow_verify(store, '2381', passed=True)
        assert _branch_pre_merge_verify_green(store, None) is None

    def test_reads_workflow_verify_not_merge_verify(self, tmp_path: Path):
        """The amendment's load-bearing distinction: a passing merge_verify
        row (the merge worker's POST-rebase verdict) is NOT the branch-green
        fact and must be ignored — only workflow_verify counts."""
        from orchestrator.merge_disposition import _branch_pre_merge_verify_green

        store = _make_event_store(tmp_path)
        # A passing merge_verify for the task, but NO workflow_verify.
        store.emit(
            EventType.merge_verify,
            task_id='2381',
            data={'passed': True, 'branch': 'task/2381'},
        )
        # Evidence for I5 is ABSENT (no workflow_verify) -> None, not True.
        assert _branch_pre_merge_verify_green(store, '2381') is None


# ---------------------------------------------------------------------------
# step-11/13/15/17 — end-to-end classify_merge_failure_disposition
# ---------------------------------------------------------------------------


def _head_sha(root: Path) -> str:
    return subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True, check=True,
    ).stdout.strip()


# A verify_result whose failing test maps to src/x.py (pytest id path segment +
# cause-hint file token both resolve to src/x.py).
_XPY_FAILURE = VerifyResult(
    passed=False,
    test_output='FAILED src/x.py::test_bar - AssertionError',
    lint_output='',
    type_output='',
    summary='1 failed',
    cause_hint='src/x.py::test_bar',
    category='test_failure',
)


class TestClassifyIntegrationSkew:
    """End-to-end INTEGRATION_SKEW: a landing on main between the branch's
    merge-base and main's tip touches a file implicated by the branch's
    failing test, AND a workflow_verify(passed=True) row is present for the
    task -> INTEGRATION_SKEW + SkewEvidence. Also asserts I2 (read-only)."""

    def test_implicated_landing_plus_green_yields_integration_skew(self, tmp_path: Path):
        from orchestrator.merge_disposition import (
            MergeFailureDisposition,
            SkewEvidence,
            classify_merge_failure_disposition,
        )

        repo = tmp_path / 'repo'
        repo.mkdir()
        _init_git_repo(repo)
        merge_base_sha = _commit_file(repo, 'src/x.py', 'v1', 'init x (merge-base)')
        landing_sha = _commit_file(repo, 'src/x.py', 'v2', 'edit x on main (landing)')

        store = _make_event_store(tmp_path)
        _emit_workflow_verify(store, '2381', passed=True, branch='task/2381')

        head_before = _head_sha(repo)
        disposition, evidence = asyncio.run(classify_merge_failure_disposition(
            verify_result=_XPY_FAILURE,
            branch='task/2381',
            merge_base_sha=merge_base_sha,
            main_sha=landing_sha,
            preexisting=False,
            task_id='2381',
            repo_root=repo,
            event_store=store,
        ))

        assert disposition == MergeFailureDisposition.INTEGRATION_SKEW
        assert isinstance(evidence, SkewEvidence)
        assert landing_sha in evidence.implicated_commits
        assert 'src/x.py' in evidence.overlap_files
        assert 'src/x.py::test_bar' in evidence.failing_tests
        # I2 read-only: the classifier never mutated the repo.
        assert _head_sha(repo) == head_before
        assert subprocess.run(
            ['git', 'status', '--porcelain'], cwd=repo, capture_output=True, text=True,
        ).stdout == ''


class TestClassifyBranchBug:
    """BRANCH_BUG: candidate files identified and searched, but no landing on
    main touches them -> the failure is the branch's own. Green presence is
    irrelevant when nothing on main is implicated."""

    def test_no_implicated_landing_yields_branch_bug_even_with_green(self, tmp_path: Path):
        from orchestrator.merge_disposition import (
            MergeFailureDisposition,
            classify_merge_failure_disposition,
        )

        repo = tmp_path / 'repo'
        repo.mkdir()
        _init_git_repo(repo)
        merge_base_sha = _commit_file(repo, 'src/x.py', 'v1', 'init x (merge-base)')
        # main moves, but only touches an UNRELATED file.
        main_sha = _commit_file(repo, 'src/y.py', 'v1', 'add unrelated y on main')

        store = _make_event_store(tmp_path)
        _emit_workflow_verify(store, '2381', passed=True, branch='task/2381')

        disposition, evidence = asyncio.run(classify_merge_failure_disposition(
            verify_result=_XPY_FAILURE,
            branch='task/2381',
            merge_base_sha=merge_base_sha,
            main_sha=main_sha,
            preexisting=False,
            task_id='2381',
            repo_root=repo,
            event_store=store,
        ))

        assert disposition == MergeFailureDisposition.BRANCH_BUG
        assert evidence is None


class TestClassifyIndeterminate:
    """INDETERMINATE degrade paths (I5 + fail-open honesty)."""

    def _repo_with_implicated_landing(self, tmp_path: Path) -> tuple[Path, str, str]:
        repo = tmp_path / 'repo'
        repo.mkdir()
        _init_git_repo(repo)
        merge_base_sha = _commit_file(repo, 'src/x.py', 'v1', 'init x (merge-base)')
        landing_sha = _commit_file(repo, 'src/x.py', 'v2', 'edit x on main (landing)')
        return repo, merge_base_sha, landing_sha

    def test_implicated_landing_but_workflow_verify_absent_is_indeterminate(
        self, tmp_path: Path,
    ):
        """The keystone amendment case: a real first-attempt skew (implicated
        landing present) with NO workflow_verify row degrades honestly to
        INDETERMINATE, never a fabricated INTEGRATION_SKEW."""
        from orchestrator.merge_disposition import (
            MergeFailureDisposition,
            classify_merge_failure_disposition,
        )

        repo, merge_base_sha, landing_sha = self._repo_with_implicated_landing(tmp_path)
        store = _make_event_store(tmp_path)  # empty: no workflow_verify rows

        disposition, evidence = asyncio.run(classify_merge_failure_disposition(
            verify_result=_XPY_FAILURE,
            branch='task/2381',
            merge_base_sha=merge_base_sha,
            main_sha=landing_sha,
            preexisting=False,
            task_id='2381',
            repo_root=repo,
            event_store=store,
        ))
        assert disposition == MergeFailureDisposition.INDETERMINATE
        assert evidence is None

    def test_implicated_landing_but_workflow_verify_failed_is_indeterminate(
        self, tmp_path: Path,
    ):
        from orchestrator.merge_disposition import (
            MergeFailureDisposition,
            classify_merge_failure_disposition,
        )

        repo, merge_base_sha, landing_sha = self._repo_with_implicated_landing(tmp_path)
        store = _make_event_store(tmp_path)
        _emit_workflow_verify(store, '2381', passed=False, branch='task/2381')

        disposition, evidence = asyncio.run(classify_merge_failure_disposition(
            verify_result=_XPY_FAILURE,
            branch='task/2381',
            merge_base_sha=merge_base_sha,
            main_sha=landing_sha,
            preexisting=False,
            task_id='2381',
            repo_root=repo,
            event_store=store,
        ))
        assert disposition == MergeFailureDisposition.INDETERMINATE
        assert evidence is None

    def test_repo_root_none_is_indeterminate(self, tmp_path: Path):
        """No repo to search -> evidence unavailable -> INDETERMINATE (never
        BRANCH_BUG guessed from an unsearchable state)."""
        from orchestrator.merge_disposition import (
            MergeFailureDisposition,
            classify_merge_failure_disposition,
        )

        store = _make_event_store(tmp_path)
        _emit_workflow_verify(store, '2381', passed=True)

        disposition, evidence = asyncio.run(classify_merge_failure_disposition(
            verify_result=_XPY_FAILURE,
            branch='task/2381',
            merge_base_sha='base',
            main_sha='main',
            preexisting=False,
            task_id='2381',
            repo_root=None,
            event_store=store,
        ))
        assert disposition == MergeFailureDisposition.INDETERMINATE
        assert evidence is None

    def test_unextractable_failure_is_indeterminate(self, tmp_path: Path):
        """No candidate file could be identified from the failure (ambiguity)
        -> INDETERMINATE, not BRANCH_BUG."""
        from orchestrator.merge_disposition import (
            MergeFailureDisposition,
            classify_merge_failure_disposition,
        )

        repo, merge_base_sha, landing_sha = self._repo_with_implicated_landing(tmp_path)
        store = _make_event_store(tmp_path)
        _emit_workflow_verify(store, '2381', passed=True)

        opaque = VerifyResult(
            passed=False,
            test_output='Something went wrong. No further details available.',
            lint_output='',
            type_output='',
            summary='error',
            cause_hint='',
            category='unknown',
        )
        disposition, evidence = asyncio.run(classify_merge_failure_disposition(
            verify_result=opaque,
            branch='task/2381',
            merge_base_sha=merge_base_sha,
            main_sha=landing_sha,
            preexisting=False,
            task_id='2381',
            repo_root=repo,
            event_store=store,
        ))
        assert disposition == MergeFailureDisposition.INDETERMINATE
        assert evidence is None


class TestClassifyFailOpen:
    """I3: any internal error degrades to (INDETERMINATE, None), never raises."""

    def test_helper_exception_degrades_to_indeterminate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        import orchestrator.merge_disposition as md
        from orchestrator.merge_disposition import (
            MergeFailureDisposition,
            classify_merge_failure_disposition,
        )

        def _boom(*args, **kwargs):
            raise RuntimeError('injected fault')

        monkeypatch.setattr(md, '_extract_failing_tests_and_candidate_files', _boom)

        disposition, evidence = asyncio.run(classify_merge_failure_disposition(
            verify_result=_XPY_FAILURE,
            branch='task/2381',
            merge_base_sha='base',
            main_sha='main',
            preexisting=False,
            task_id='2381',
            repo_root=tmp_path,
            event_store=None,
        ))
        assert disposition == MergeFailureDisposition.INDETERMINATE
        assert evidence is None
