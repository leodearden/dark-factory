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
# step-1 (task 3178) — a bare filename / prose word is NOT a pytest node id
# ---------------------------------------------------------------------------


def _guard_failure(cause_hint: str) -> VerifyResult:
    """The VerifyResult shape reify's verify summary actually renders for a
    shell-guard trip: the guard's own name arrives in ``cause_hint``, and
    ``test_output`` is empty (the guard is not a pytest/nextest runner)."""
    return VerifyResult(
        passed=False,
        cause_hint=cause_hint,
        test_output='',
        lint_output='',
        type_output='',
        summary='1 failed',
        category='test_failure',
    )


# Verbatim ``cause_hint`` strings captured live from reify on 2026-07-29 —
# every one of the 8 INTEGRATION_SKEW dispositions between 07-24 and 07-28
# was a shell/infra guard emitting one of these shapes. Deliberately NOT
# invented fixtures: task 2871's I7 shipped green against a synthetic
# ``HARNESS_KLOC_CAP FAIL …`` string (no ``^FAILED <token>``, so it genuinely
# parsed zero ids) while production emits ``FAILED test_harness_kloc_cap.sh``,
# which parses one. That gap is why the false premise survived two cycles.
_LIVE_SHELL_GUARD_CAUSE_HINTS = [
    ('FAILED test_reify_audit_ptodo.sh', 'test_reify_audit_ptodo.sh', '5566'),
    ('FAILED test_verify_scope.sh', 'test_verify_scope.sh', '5300'),
    (
        'FAILED test_deterministic_gate_closure_staleness_sweep.sh',
        'test_deterministic_gate_closure_staleness_sweep.sh',
        '5321',
    ),
    ('FAILED test_harness_kloc_cap.sh', 'test_harness_kloc_cap.sh', '5316/5373/5302'),
]


class TestShellGuardFilenameIsNotATestId:
    """``_PYTEST_FAILED_ID_RE`` is an unconstrained ``^FAILED\\s+(\\S+)``, so a
    shell-guard trip rendered as ``FAILED <guard>.sh`` yields the guard's own
    FILENAME as a "failing test id" — vacuously satisfying task 2871's I7
    ``failing_tests`` non-empty requirement and producing a false
    INTEGRATION_SKEW. Task 3178 adds a node-id SHAPE floor (``'::' in tid``),
    mirroring the invariant ``_RUST_FAILED_ID_RE`` has always had by
    construction.

    Each case additionally pins that the filename REMAINS in
    ``candidate_files``: the fix must empty ``failing_tests`` only, so
    ``_implicated_landings`` still walks main and the classifier degrades for
    the honest reason (the I7 gate bites) rather than because its evidence was
    starved."""

    @pytest.mark.parametrize(
        ('cause_hint', 'guard_filename', 'reify_task'),
        _LIVE_SHELL_GUARD_CAUSE_HINTS,
        ids=[c[1] for c in _LIVE_SHELL_GUARD_CAUSE_HINTS],
    )
    def test_live_shell_guard_filename_is_not_a_failing_test_id(
        self, cause_hint: str, guard_filename: str, reify_task: str,
    ):
        from orchestrator.merge_disposition import (
            _extract_failing_tests_and_candidate_files,
        )

        failing_tests, candidate_files = _extract_failing_tests_and_candidate_files(
            _guard_failure(cause_hint),
        )
        # RED before the fix: today this is ``(guard_filename,)``.
        assert failing_tests == (), (
            f'reify {reify_task}: guard filename parsed as a test id'
        )
        # GREEN both before and after: candidate-file extraction is untouched,
        # so the landings walk still has something to search for.
        assert guard_filename in candidate_files

    def test_prose_word_after_failed_is_not_a_failing_test_id(self):
        """The real shape at test_verify_classify.py:1117 — the unconstrained
        regex parses the English word "to" as a test id. RED before the fix;
        the shape floor closes this hole too."""
        from orchestrator.merge_disposition import (
            _extract_failing_tests_and_candidate_files,
        )

        failing_tests, _candidate_files = _extract_failing_tests_and_candidate_files(
            _guard_failure('FAILED to release semaphore slot before it timed out'),
        )
        assert failing_tests == ()

    def test_mid_line_failed_token_never_anchored(self):
        """reify 5370's shape: ``FAILED`` appears mid-line, so ``^FAILED`` does
        not anchor and zero ids parse. GREEN today — pinned so a future
        regex-relaxation (dropping the ``^`` anchor or adding ``re.search``)
        cannot silently reopen this shape."""
        from orchestrator.merge_disposition import (
            _extract_failing_tests_and_candidate_files,
        )

        failing_tests, candidate_files = _extract_failing_tests_and_candidate_files(
            _guard_failure(
                'verify.sh: FAILED (exit 1): timeout --kill-after=60 60m nice -n 5 '
                'cargo nextest run --workspace',
            ),
        )
        assert failing_tests == ()
        assert 'verify.sh' in candidate_files

    def test_positive_control_pytest_node_id_survives(self):
        """A genuine pytest node id carries ``::`` and must be unaffected —
        including the test-id -> path heuristic (merge_disposition.py:162-165)
        that feeds its path segment into candidate_files."""
        from orchestrator.merge_disposition import (
            _extract_failing_tests_and_candidate_files,
        )

        verify_result = VerifyResult(
            passed=False,
            test_output='FAILED tests/test_foo.py::test_bar - AssertionError',
            lint_output='',
            type_output='',
            summary='1 failed',
            cause_hint='',
            category='test_failure',
        )

        failing_tests, candidate_files = _extract_failing_tests_and_candidate_files(
            verify_result,
        )
        assert 'tests/test_foo.py::test_bar' in failing_tests
        assert 'tests/test_foo.py' in candidate_files

    def test_positive_control_rust_node_id_survives(self):
        """reify 5187's shape — the ONE genuine skew in the 07-24..07-28
        population. ``_RUST_FAILED_ID_RE`` already requires ``::`` by
        construction, so the Rust branch needs no guard and this true skew is
        provably unaffected by the fix."""
        from orchestrator.merge_disposition import (
            _extract_failing_tests_and_candidate_files,
        )

        verify_result = VerifyResult(
            passed=False,
            test_output='test objective_inheritance_e2e::inherits_from_parent ... FAILED',
            lint_output='',
            type_output='',
            summary='1 failed',
            cause_hint='',
            category='test_failure',
        )

        failing_tests, _candidate_files = _extract_failing_tests_and_candidate_files(
            verify_result,
        )
        assert 'objective_inheritance_e2e::inherits_from_parent' in failing_tests


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
# step-1 (task 2869) — _implicated_landings real_main_head_sha ancestor filter
# ---------------------------------------------------------------------------


def _build_orphan_repo(repo: Path) -> tuple[str, str, str]:
    """Build a repo modelling reify esc-5260-8's orphaned speculative train:

      * ``merge_base`` — the real fork point, touches ``src/x.py`` (on main).
      * ``real_main_head`` — real main advanced by an UNRELATED commit
        (``src/z.py``); does NOT have the orphan tip as an ancestor.
      * ``orphan_tip`` — a commit off ``merge_base`` on a side branch touching
        ``src/x.py``, NEVER merged onto real main (the orphaned speculative
        train tip).

    Returns ``(merge_base, real_main_head, orphan_tip)``; leaves HEAD on main.
    """
    _init_git_repo(repo)
    merge_base = _commit_file(repo, 'src/x.py', 'v1', 'init x (merge-base / real fork)')
    real_main_head = _commit_file(repo, 'src/z.py', 'zzz', 'unrelated real-main advance')
    # Orphan side branch off merge_base, touching the same candidate file,
    # never merged onto real main.
    subprocess.run(
        ['git', 'checkout', '-q', '-b', 'orphan', merge_base],
        cwd=repo, check=True, capture_output=True,
    )
    orphan_tip = _commit_file(repo, 'src/x.py', 'v2-orphan', 'orphan speculative edit x')
    # Restore HEAD to real main so the repo's default state is clean/on-main.
    subprocess.run(
        ['git', 'checkout', '-q', 'main'], cwd=repo, check=True, capture_output=True,
    )
    return merge_base, real_main_head, orphan_tip


class TestImplicatedLandingsAncestorFilter:
    """_implicated_landings gains a ``real_main_head_sha`` kwarg (task 2869,
    reify esc-5260-8): when truthy AND != main_sha, cited commits are filtered
    to ancestors of the CURRENT real main HEAD, so orphaned speculative-train
    commits (never reachable from real main) are pruned while genuine landings
    survive. Prunes ONLY on a definitive not-an-ancestor (git merge-base
    --is-ancestor rc 1); fail-open on None / an unresolvable ref (keeps today's
    citations, never suppresses a genuine skew)."""

    def test_orphan_speculative_commit_is_pruned(self, tmp_path: Path):
        from orchestrator.merge_disposition import _implicated_landings

        repo = tmp_path / 'repo'
        repo.mkdir()
        merge_base, real_main_head, orphan_tip = _build_orphan_repo(repo)

        head_before = _head_sha(repo)
        implicated_commits, overlap_files = asyncio.run(
            _implicated_landings(
                repo, merge_base, orphan_tip, {'src/x.py'},
                real_main_head_sha=real_main_head,
            ),
        )
        # The walk merge_base..orphan_tip cites orphan_tip (it touches src/x.py),
        # but orphan_tip is NOT an ancestor of real main HEAD -> pruned.
        assert implicated_commits == ()
        assert overlap_files == ()
        # I2 read-only: the ancestor probes never mutate the repo.
        assert _head_sha(repo) == head_before
        assert subprocess.run(
            ['git', 'status', '--porcelain'], cwd=repo, capture_output=True, text=True,
        ).stdout == ''

    def test_genuine_landing_kept_even_when_main_advanced_past_dispatch_base(
        self, tmp_path: Path,
    ):
        from orchestrator.merge_disposition import _implicated_landings

        repo = tmp_path / 'repo'
        repo.mkdir()
        _init_git_repo(repo)
        merge_base = _commit_file(repo, 'src/x.py', 'v1', 'init x (merge-base)')
        landing = _commit_file(repo, 'src/x.py', 'v2', 'genuine landing on main')
        # Real main advances PAST the dispatch base with an unrelated commit;
        # the landing remains an ancestor of the new real main HEAD.
        real_main_head = _commit_file(repo, 'src/z.py', 'zzz', 'later unrelated main tip')

        implicated_commits, overlap_files = asyncio.run(
            _implicated_landings(
                repo, merge_base, landing, {'src/x.py'},
                real_main_head_sha=real_main_head,
            ),
        )
        # A genuine real-main ancestor survives even though main advanced past
        # the dispatch base (main_sha=landing, real_main_head is a descendant).
        assert landing in implicated_commits
        assert 'src/x.py' in overlap_files

    def test_fail_open_none_keeps_orphan_citation(self, tmp_path: Path):
        from orchestrator.merge_disposition import _implicated_landings

        repo = tmp_path / 'repo'
        repo.mkdir()
        merge_base, _real_main_head, orphan_tip = _build_orphan_repo(repo)

        implicated_commits, overlap_files = asyncio.run(
            _implicated_landings(
                repo, merge_base, orphan_tip, {'src/x.py'},
                real_main_head_sha=None,
            ),
        )
        # real_main_head_sha=None -> filter skipped -> byte-identical to today.
        assert orphan_tip in implicated_commits
        assert 'src/x.py' in overlap_files

    def test_fail_open_unresolvable_ref_keeps_orphan_citation(self, tmp_path: Path):
        from orchestrator.merge_disposition import _implicated_landings

        repo = tmp_path / 'repo'
        repo.mkdir()
        merge_base, _real_main_head, orphan_tip = _build_orphan_repo(repo)

        implicated_commits, overlap_files = asyncio.run(
            _implicated_landings(
                repo, merge_base, orphan_tip, {'src/x.py'},
                real_main_head_sha='0' * 40,  # a 40-hex SHA absent from the repo
            ),
        )
        # merge-base --is-ancestor errors (rc != 0/1) against an absent ref ->
        # the helper returns None -> prune only on a definitive negative -> keep.
        assert orphan_tip in implicated_commits
        assert 'src/x.py' in overlap_files


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


class TestBranchPreMergeVerifyGreenCrossRun:
    """The durable cross-restart regression the verify checkpoint (task 2752)
    creates: the verify checkpoint SUPPRESSES the current-run workflow_verify
    re-emit at an unchanged branch tip, so on the exact cross-restart fast-path
    this feature targets, the branch's only workflow_verify(passed=True) green
    lives under a PRIOR run_id. I5's branch-green fact MUST still see it — else
    a genuine INTEGRATION_SKEW silently degrades to INDETERMINATE after a fleet
    redeploy. So _branch_pre_merge_verify_green must read the durable cross-run
    history (fetch_events_by_type_all_runs), not the run-scoped current run."""

    def test_prior_run_green_is_visible_cross_run(self, tmp_path: Path):
        from orchestrator.merge_disposition import _branch_pre_merge_verify_green

        db_path = tmp_path / 'runs.db'
        # A green recorded under a PRIOR orchestrator run (before a restart).
        store_prior = EventStore(db_path, run_id='run-prior')
        store_prior.emit(
            EventType.workflow_verify,
            task_id='2381',
            data={
                'passed': True,
                'tip_sha': 'tipA',
                'base_sha': 'b',
                'branch': 'task/2381',
            },
        )

        # A fresh run (post-restart) on the SAME db under a DIFFERENT run_id.
        store_now = EventStore(db_path, run_id='run-now')
        # Precondition: the green is genuinely under a different run_id, so the
        # run-scoped reader (the old backing read) cannot see it at all.
        assert store_now.fetch_events_by_type(EventType.workflow_verify) == []
        # ...but the durable cross-run branch-green fact MUST still see it.
        assert _branch_pre_merge_verify_green(store_now, '2381') is True


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


class TestClassifyIntegrationSkewCrossRun:
    """End-to-end cross-restart INTEGRATION_SKEW: the branch's only
    workflow_verify green lives under a PRIOR run_id (the task-2752 verify
    checkpoint suppressed the current-run re-emit at an unchanged branch tip).
    With an implicated landing on main, classify MUST still yield
    INTEGRATION_SKEW after a fleet redeploy — not degrade to INDETERMINATE for
    want of a current-run green (that would lose the skew-specific handling for
    a branch that really was verified green)."""

    def test_prior_run_green_plus_implicated_landing_yields_integration_skew(
        self, tmp_path: Path,
    ):
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

        db_path = tmp_path / 'runs.db'
        # The green is emitted ONLY under the PRIOR run.
        store_prior = EventStore(db_path, run_id='run-prior')
        store_prior.emit(
            EventType.workflow_verify,
            task_id='2381',
            data={
                'passed': True,
                'tip_sha': 'tipA',
                'base_sha': 'b',
                'branch': 'task/2381',
            },
        )
        # classify runs in a fresh post-restart run with NO green of its own.
        store_now = EventStore(db_path, run_id='run-now')

        disposition, evidence = asyncio.run(classify_merge_failure_disposition(
            verify_result=_XPY_FAILURE,
            branch='task/2381',
            merge_base_sha=merge_base_sha,
            main_sha=landing_sha,
            preexisting=False,
            task_id='2381',
            repo_root=repo,
            event_store=store_now,
        ))

        assert disposition == MergeFailureDisposition.INTEGRATION_SKEW
        assert isinstance(evidence, SkewEvidence)


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


# ---------------------------------------------------------------------------
# step-3 (task 2869) — classify orphan speculative main_sha is NOT a skew
# ---------------------------------------------------------------------------


class TestClassifyOrphanSpeculativeNotSkew:
    """classify_merge_failure_disposition gains a ``real_main_head_sha`` kwarg
    (task 2869, reify esc-5260-8). When ``main_sha`` is an orphaned speculative
    train tip (never fast-forwarded onto real main), the implicated landings
    are filtered to ancestors of the CURRENT real main HEAD, so an orphan-cited
    'skew' collapses to the honest BRANCH_BUG instead of a fabricated
    INTEGRATION_SKEW citing dangling commits. A genuine landing still yields
    INTEGRATION_SKEW; real_main_head_sha=None fails open to today's frame."""

    def test_orphan_speculative_main_sha_plus_green_is_branch_bug_not_skew(
        self, tmp_path: Path,
    ):
        from orchestrator.merge_disposition import (
            MergeFailureDisposition,
            classify_merge_failure_disposition,
        )

        repo = tmp_path / 'repo'
        repo.mkdir()
        merge_base, real_main_head, orphan_tip = _build_orphan_repo(repo)

        store = _make_event_store(tmp_path)
        _emit_workflow_verify(store, '2381', passed=True, branch='task/2381')

        disposition, evidence = asyncio.run(classify_merge_failure_disposition(
            verify_result=_XPY_FAILURE,
            branch='task/2381',
            merge_base_sha=merge_base,
            main_sha=orphan_tip,
            real_main_head_sha=real_main_head,
            preexisting=False,
            task_id='2381',
            repo_root=repo,
            event_store=store,
        ))
        # Orphan cited + green would be a false INTEGRATION_SKEW today; with the
        # ancestor filter the orphan is pruned -> empty implicated set falls
        # through to the honest BRANCH_BUG.
        assert disposition == MergeFailureDisposition.BRANCH_BUG
        assert evidence is None

    def test_genuine_landing_plus_green_still_integration_skew(self, tmp_path: Path):
        from orchestrator.merge_disposition import (
            MergeFailureDisposition,
            SkewEvidence,
            classify_merge_failure_disposition,
        )

        repo = tmp_path / 'repo'
        repo.mkdir()
        _init_git_repo(repo)
        merge_base = _commit_file(repo, 'src/x.py', 'v1', 'init x (merge-base)')
        landing = _commit_file(repo, 'src/x.py', 'v2', 'genuine landing on main')
        # Real main advances past the dispatch base; the landing is an ancestor.
        real_main_head = _commit_file(repo, 'src/z.py', 'zzz', 'later unrelated main tip')

        store = _make_event_store(tmp_path)
        _emit_workflow_verify(store, '2381', passed=True, branch='task/2381')

        disposition, evidence = asyncio.run(classify_merge_failure_disposition(
            verify_result=_XPY_FAILURE,
            branch='task/2381',
            merge_base_sha=merge_base,
            main_sha=landing,
            real_main_head_sha=real_main_head,
            preexisting=False,
            task_id='2381',
            repo_root=repo,
            event_store=store,
        ))
        assert disposition == MergeFailureDisposition.INTEGRATION_SKEW
        assert isinstance(evidence, SkewEvidence)
        assert landing in evidence.implicated_commits

    def test_real_main_head_none_fails_open_to_integration_skew(self, tmp_path: Path):
        """An unresolved real-main (real_main_head_sha=None) degrades to today's
        pre-fix reference frame: the orphan is still cited -> INTEGRATION_SKEW.
        Documents fail-open (never crashes, never suppresses a genuine skew)."""
        from orchestrator.merge_disposition import (
            MergeFailureDisposition,
            classify_merge_failure_disposition,
        )

        repo = tmp_path / 'repo'
        repo.mkdir()
        merge_base, _real_main_head, orphan_tip = _build_orphan_repo(repo)

        store = _make_event_store(tmp_path)
        _emit_workflow_verify(store, '2381', passed=True, branch='task/2381')

        disposition, evidence = asyncio.run(classify_merge_failure_disposition(
            verify_result=_XPY_FAILURE,
            branch='task/2381',
            merge_base_sha=merge_base,
            main_sha=orphan_tip,
            real_main_head_sha=None,
            preexisting=False,
            task_id='2381',
            repo_root=repo,
            event_store=store,
        ))
        assert disposition == MergeFailureDisposition.INTEGRATION_SKEW
        assert evidence is not None


# ---------------------------------------------------------------------------
# step-1 (task 2871) — harness-ratchet / shell-guard failures require a
# parsed failing-test id to be INTEGRATION_SKEW
# ---------------------------------------------------------------------------


class TestClassifyHarnessRatchetNotSkew:
    """Reify esc-5053-13/esc-5056-11: a harness-ratchet / shell-guard
    merge-verify failure (kLOC cap, baseline-manifest grandfathering, other
    ``tests/infra/*.sh`` guards) implicates a genuine real-main-ancestor
    landing (2869's I6 filter keeps it, since it IS a real ancestor) yet
    parses ZERO failing-test ids, because guard output matches neither
    ``_PYTEST_FAILED_ID_RE`` nor ``_RUST_FAILED_ID_RE``. Task 2871 requires
    ``failing_tests`` non-empty for INTEGRATION_SKEW, so these cases degrade
    to the honest INDETERMINATE instead. The minimal-pair control at the end
    pins that ``failing_tests`` is the SOLE discriminant (same merge_base ->
    genuine-landing -> advanced-real-main shape, only the VerifyResult's
    parseability differs)."""

    def test_kloc_cap_guard_with_genuine_landing_is_indeterminate_not_skew(
        self, tmp_path: Path,
    ):
        from orchestrator.merge_disposition import (
            MergeFailureDisposition,
            classify_merge_failure_disposition,
        )

        repo = tmp_path / 'repo'
        repo.mkdir()
        _init_git_repo(repo)
        merge_base = _commit_file(
            repo, 'scripts/run_all.sh', 'v1', 'init run_all.sh (merge-base)',
        )
        landing = _commit_file(
            repo, 'scripts/run_all.sh', 'v2', 'genuine landing on main (kloc cap bump)',
        )
        # Real main advances past the dispatch base; the landing is an ancestor.
        real_main_head = _commit_file(repo, 'src/z.py', 'zzz', 'later unrelated main tip')

        store = _make_event_store(tmp_path)
        _emit_workflow_verify(store, '2381', passed=True, branch='task/2381')

        kloc_cap_failure = VerifyResult(
            passed=False,
            test_output=(
                'HARNESS_KLOC_CAP FAIL crate=foo file=crates/foo/tests/big_migration.rs\n'
                '(cap enforced by scripts/run_all.sh)'
            ),
            lint_output='',
            type_output='',
            summary='1 failed',
            cause_hint='',
            category='test_failure',
        )

        disposition, evidence = asyncio.run(classify_merge_failure_disposition(
            verify_result=kloc_cap_failure,
            branch='task/2381',
            merge_base_sha=merge_base,
            main_sha=landing,
            real_main_head_sha=real_main_head,
            preexisting=False,
            task_id='2381',
            repo_root=repo,
            event_store=store,
        ))
        # The run_all.sh overlap + branch green would be a false
        # INTEGRATION_SKEW today; zero parsed failing-test ids (I7) degrades
        # it to the honest INDETERMINATE instead.
        assert disposition == MergeFailureDisposition.INDETERMINATE
        assert evidence is None

    def test_baseline_manifest_grandfathering_with_genuine_landing_is_indeterminate_not_skew(
        self, tmp_path: Path,
    ):
        from orchestrator.merge_disposition import (
            MergeFailureDisposition,
            classify_merge_failure_disposition,
        )

        repo = tmp_path / 'repo'
        repo.mkdir()
        _init_git_repo(repo)
        merge_base = _commit_file(repo, 'src/foo.py', 'v1', 'init foo (merge-base)')
        landing = _commit_file(
            repo, 'src/foo.py', 'v2', 'genuine landing on main (baseline guard trip)',
        )
        real_main_head = _commit_file(repo, 'src/z.py', 'zzz', 'later unrelated main tip')

        store = _make_event_store(tmp_path)
        _emit_workflow_verify(store, '2381', passed=True, branch='task/2381')

        baseline_failure = VerifyResult(
            passed=False,
            test_output=(
                'HARNESS_LAYOUT_BASELINE FAIL: src/foo.py not in '
                'harness-layout-baseline.manifest'
            ),
            lint_output='',
            type_output='',
            summary='1 failed',
            cause_hint='',
            category='test_failure',
        )

        disposition, evidence = asyncio.run(classify_merge_failure_disposition(
            verify_result=baseline_failure,
            branch='task/2381',
            merge_base_sha=merge_base,
            main_sha=landing,
            real_main_head_sha=real_main_head,
            preexisting=False,
            task_id='2381',
            repo_root=repo,
            event_store=store,
        ))
        # Same shape as the kLOC-cap case, over a SOURCE-file overlap: the
        # fix covers the whole guard/ratchet class, not just one guard.
        assert disposition == MergeFailureDisposition.INDETERMINATE
        assert evidence is None

    def test_minimal_pair_genuine_failing_test_id_still_integration_skew(
        self, tmp_path: Path,
    ):
        """Boundary control (GREEN today and after the fix): same
        merge_base -> genuine-landing -> advanced-real-main shape as the two
        guard cases above, but the VerifyResult carries a genuine parsed
        failing-test id over the overlapping file. Pins that ``failing_tests``
        is the sole discriminant introduced by task 2871 — the fix must not
        over-correct and suppress a true INTEGRATION_SKEW."""
        from orchestrator.merge_disposition import (
            MergeFailureDisposition,
            SkewEvidence,
            classify_merge_failure_disposition,
        )

        repo = tmp_path / 'repo'
        repo.mkdir()
        _init_git_repo(repo)
        merge_base = _commit_file(repo, 'src/x.py', 'v1', 'init x (merge-base)')
        landing = _commit_file(repo, 'src/x.py', 'v2', 'genuine landing on main')
        real_main_head = _commit_file(repo, 'src/z.py', 'zzz', 'later unrelated main tip')

        store = _make_event_store(tmp_path)
        _emit_workflow_verify(store, '2381', passed=True, branch='task/2381')

        disposition, evidence = asyncio.run(classify_merge_failure_disposition(
            verify_result=_XPY_FAILURE,
            branch='task/2381',
            merge_base_sha=merge_base,
            main_sha=landing,
            real_main_head_sha=real_main_head,
            preexisting=False,
            task_id='2381',
            repo_root=repo,
            event_store=store,
        ))
        assert disposition == MergeFailureDisposition.INTEGRATION_SKEW
        assert isinstance(evidence, SkewEvidence)
        assert landing in evidence.implicated_commits
