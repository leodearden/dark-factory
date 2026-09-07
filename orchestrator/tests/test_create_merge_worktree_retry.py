"""Tests for the shared ``git worktree add --detach`` retry driver (task 5140).

``GitOps._create_merge_worktree`` (git_ops.py) issued exactly ONE ``git
worktree add --detach`` and raised ``RuntimeError(f'Failed to create merge
worktree: {err}')`` on any non-zero rc — no retry, no transient/permanent
discrimination, and stdout + rc discarded.  Five archived occurrences under
``data/verify-logs`` (tasks 3692, 3420, 3869, 4215, 4545; 2026-08-10 ->
2026-09-06) show both halves of the cost:

  * 4215 — ``fatal: Invalid path '.../.git/worktrees/_merge-9a6caddb': No
    such file or directory`` on the ADMINISTRATIVE path: a transient
    contention/race shape a bounded retry absorbs.
  * 4545 — ``Failed to create merge worktree: Preparing worktree (detached
    HEAD ae6e7e9)``: stderr carries only git's progress line, the cause line
    is absent, and neither rc nor stdout was captured — undiagnosable.
  * 3692 — ``/repo/.worktrees/cas-b No space left on device``: a full disk,
    which must NOT be blanket-retried (it does not heal in 1.5s of backoff;
    retrying only delays the operator signal).

This module covers the three pieces that close that gap, each shared by BOTH
worktree-minting sites (``_create_merge_worktree`` and ``ephemeral_worktree``)
so exactly one retry loop and one predicate exist in git_ops.py:

  step-1: ``_worktree_add_failure_is_retryable`` — the single shared
          predicate, plus the ``_ENOSPC_MARKERS`` single-vocabulary identity
  step-3: ``GitOps._worktree_add_with_retry`` — the single shared driver
  step-5: ``_create_merge_worktree`` absorbs the transient flake
  step-7: ENOSPC fast-fails, and the final error carries rc + both streams
          + the attempt count

The harness is adopted verbatim in shape from
``orchestrator/tests/test_ephemeral_worktree.py`` (``GitOps(GitConfig(),
tmp_path)`` + a patched ``orchestrator.git_ops._run`` recording argvs +
a patched ``orchestrator.git_ops.asyncio.sleep``) so retry/backoff
assertions read identically across both minting paths and no real backoff
wall-clock enters the suite.
"""
from __future__ import annotations

import orchestrator.git_ops as git_ops_mod
import orchestrator.verify_classify as verify_classify_mod

# ---------------------------------------------------------------------------
# step-1: the single shared retry predicate
# ---------------------------------------------------------------------------

# Verbatim from data/verify-logs (task 3692) — a literal ENOSPC.
ENOSPC_3692_STDERR = '/repo/.worktrees/cas-b No space left on device'

# Verbatim from data/verify-logs (task 4215) — the administrative-path race
# shape a bounded retry is meant to absorb. Note the failing path is
# `.git/worktrees/<name>`, git's ADMINISTRATIVE registration dir, not the
# checkout path.
TRANSIENT_4215_STDERR = (
    'Preparing worktree (detached HEAD 863e3fb)\n'
    "fatal: Invalid path '/tmp/pytest-of-leo/pytest-11406/popen-gw6/"
    "test_k1_sanity_late_arrival_at0/repo/.git/worktrees/_merge-9a6caddb': "
    'No such file or directory'
)

# Verbatim from data/verify-logs (task 4545) — the TRUNCATED shape: git's
# progress line only, with no cause line at all.
TRANSIENT_4545_STDERR = 'Preparing worktree (detached HEAD ae6e7e9)'


class TestWorktreeAddRetryPredicate:
    """step-1: ``git_ops._worktree_add_failure_is_retryable(rc, out, err)``.

    The predicate is deliberately NEGATIVE — retry by default, fail fast only
    on a known non-transient cause — because the two archived TRANSIENT
    samples (4215 and 4545 above) share no token, so a positive
    "contention" allow-list would fail to recognise at least one of the very
    failures this task exists to absorb.

    RED today: ``_worktree_add_failure_is_retryable``,
    ``_WORKTREE_ADD_MAX_ATTEMPTS`` and ``_ENOSPC_MARKERS`` do not exist in
    git_ops.py.
    """

    def test_enospc_on_stderr_is_not_retryable(self) -> None:
        assert git_ops_mod._worktree_add_failure_is_retryable(
            1, '', ENOSPC_3692_STDERR,
        ) is False, (
            'a full disk must fail fast — it does not heal in 1.5s of backoff'
        )

    def test_enospc_on_stdout_is_not_retryable(self) -> None:
        """BOTH streams are inspected, not just stderr."""
        assert git_ops_mod._worktree_add_failure_is_retryable(
            1, ENOSPC_3692_STDERR, '',
        ) is False, 'expected the predicate to inspect stdout as well as stderr'

    def test_os_error_28_is_not_retryable(self) -> None:
        assert git_ops_mod._worktree_add_failure_is_retryable(
            1, '', 'failed to write: os error 28',
        ) is False

    def test_enospc_matching_is_case_insensitive(self) -> None:
        """Mirrors ``merge_queue._verify_hit_enospc``'s whole-output,
        case-insensitive match against the same vocabulary."""
        assert git_ops_mod._worktree_add_failure_is_retryable(
            1, '', 'fatal: write error: ENOSPC',
        ) is False
        assert git_ops_mod._worktree_add_failure_is_retryable(
            1, '', 'No Space Left On Device',
        ) is False

    def test_4215_administrative_race_is_retryable(self) -> None:
        assert git_ops_mod._worktree_add_failure_is_retryable(
            1, '', TRANSIENT_4215_STDERR,
        ) is True

    def test_4545_truncated_shape_is_retryable(self) -> None:
        assert git_ops_mod._worktree_add_failure_is_retryable(
            1, '', TRANSIENT_4545_STDERR,
        ) is True

    def test_empty_output_is_retryable(self) -> None:
        """Retry-by-default: a silent failure carries no evidence that it is
        permanent, and ``test_ephemeral_worktree``'s fixtures drive exactly
        this shape."""
        assert git_ops_mod._worktree_add_failure_is_retryable(1, '', '') is True

    def test_lock_contention_is_retryable(self) -> None:
        """The shape ``test_ephemeral_worktree._make_fake_run`` emits — this
        is what keeps ``TestEphemeralWorktreeRetry`` green byte-for-byte."""
        assert git_ops_mod._worktree_add_failure_is_retryable(
            1, '', 'lock contention',
        ) is True

    def test_max_attempts_is_three(self) -> None:
        """Replaces ``ephemeral_worktree``'s former function-local
        ``_MAX_ADD_RETRIES`` at the identical value, so that method's pinned
        3-attempt / [0.5, 1.0]-backoff tests are unaffected."""
        assert git_ops_mod._WORKTREE_ADD_MAX_ATTEMPTS == 3

    def test_enospc_markers_are_imported_not_recopied(self) -> None:
        """ONE shared vocabulary, imported rather than copied a third time.

        The tuple already exists verbatim twice (verify_classify.py and
        merge_queue.py), each declaring itself a "single grounded
        vocabulary ... do not invent new ENOSPC strings; extend that
        constant instead". Object IDENTITY (not equality) is asserted so a
        future well-meaning "just inline it" refactor turns this test red
        rather than quietly forking the vocabulary into a third copy that
        can drift.
        """
        assert git_ops_mod._ENOSPC_MARKERS is verify_classify_mod._ENOSPC_MARKERS
