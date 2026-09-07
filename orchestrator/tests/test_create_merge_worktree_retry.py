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

import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, call, patch

import pytest

import orchestrator.git_ops as git_ops_mod
import orchestrator.verify_classify as verify_classify_mod
from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps

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


# ---------------------------------------------------------------------------
# step-3: the single shared retry driver
# ---------------------------------------------------------------------------

MERGE_SHA = 'a' * 40


def _make_fake_add(
    results: list[tuple[int, str, str]],
    calls: list[list[str]],
    *,
    mkdir_on_failure: bool = False,
    exists_at_entry: list[bool] | None = None,
):
    """Fake ``orchestrator.git_ops._run`` recording every argv into *calls*.

    Generalises ``test_ephemeral_worktree.py::_make_fake_run`` (which yields
    per-attempt rcs only) to yield a per-attempt ``(rc, stdout, stderr)``
    triple, which the stream-content assertions below need. Results are
    consumed in order; the last entry repeats once exhausted.

    A successful add mkdirs the ``--detach`` target, mirroring real ``git
    worktree add``.  With *mkdir_on_failure* a FAILED add mkdirs it too —
    which is also what real git does, since it creates the target directory
    early, before the add can fail.  When *exists_at_entry* is supplied,
    each add records whether the target already existed on entry, which is
    how the between-attempts residue clearing is observed.
    """
    state = {'add_calls': 0}

    async def _fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if 'worktree' in cmd and 'add' in cmd:
            detach_idx = cmd.index('--detach')
            target = Path(cmd[detach_idx + 1])
            if exists_at_entry is not None:
                exists_at_entry.append(target.exists())
            idx = state['add_calls']
            rc, out, err = results[idx] if idx < len(results) else results[-1]
            state['add_calls'] += 1
            if rc == 0 or mkdir_on_failure:
                target.mkdir(parents=True, exist_ok=True)
            return (rc, out, err)
        return (0, '', '')

    return _fake_run


def _add_argvs(calls: list[list[str]]) -> list[list[str]]:
    return [c for c in calls if 'worktree' in c and 'add' in c]


class TestWorktreeAddWithRetry:
    """step-3: ``GitOps._worktree_add_with_retry(path, ref, *, label)``.

    The single shared driver behind both minting sites. Returns
    ``(rc, stdout, stderr, attempts)`` and NEVER raises on a failed add —
    shaping the error is each caller's job, because the two call sites need
    different exception types (``RuntimeError`` vs the typed
    ``EphemeralWorktreeError`` that verify.py's probes pattern-match on).

    RED today: ``GitOps._worktree_add_with_retry`` does not exist.
    """

    def test_transient_then_success_retries_with_linear_backoff(
        self, tmp_path: Path,
    ) -> None:
        git_ops = GitOps(GitConfig(), tmp_path)
        calls: list[list[str]] = []
        target = tmp_path / 'wt' / '_merge-deadbeef'

        async def _body():
            return await git_ops._worktree_add_with_retry(
                target, MERGE_SHA, label='test',
            )

        with (
            patch(
                'orchestrator.git_ops._run',
                side_effect=_make_fake_add(
                    [(1, '', TRANSIENT_4215_STDERR), (1, '', 'lock contention'), (0, '', '')],
                    calls,
                ),
            ),
            patch('orchestrator.git_ops.asyncio.sleep', new_callable=AsyncMock) as mock_sleep,
        ):
            rc, out, err, attempts = asyncio.run(_body())

        assert rc == 0, f'expected the third attempt to succeed; got rc={rc} err={err!r}'
        assert attempts == 3, f'expected attempts==3; got {attempts}'
        adds = _add_argvs(calls)
        assert len(adds) == 3, f'expected exactly 3 add attempts; got {len(adds)}'
        # Retry reuses the minted path — it does not re-mint per attempt.
        detach_targets = {c[c.index('--detach') + 1] for c in adds}
        assert detach_targets == {str(target)}, (
            f'expected all retries to target the same path; got {detach_targets}'
        )
        assert mock_sleep.await_args_list == [call(0.5), call(1.0)], (
            f'expected backoff sleeps of 0.5s then 1.0s; got {mock_sleep.await_args_list}'
        )

    def test_exhaustion_returns_rather_than_raises(self, tmp_path: Path) -> None:
        """The driver is a pure mechanism: it hands the caller the raw
        quadruple so the caller can build its own message with the streams
        intact, instead of forcing a catch-and-retranslate."""
        git_ops = GitOps(GitConfig(), tmp_path)
        calls: list[list[str]] = []
        target = tmp_path / 'wt' / '_merge-deadbeef'

        async def _body():
            return await git_ops._worktree_add_with_retry(
                target, MERGE_SHA, label='test',
            )

        with (
            patch(
                'orchestrator.git_ops._run',
                side_effect=_make_fake_add([(1, 'OUTMARK', 'ERRMARK')], calls),
            ),
            patch('orchestrator.git_ops.asyncio.sleep', new_callable=AsyncMock) as mock_sleep,
        ):
            rc, out, err, attempts = asyncio.run(_body())

        assert (rc, out, err, attempts) == (1, 'OUTMARK', 'ERRMARK', 3)
        assert len(_add_argvs(calls)) == 3
        # Only BETWEEN attempts — never after the final exhausted one.
        assert mock_sleep.await_args_list == [call(0.5), call(1.0)], (
            f'expected exactly 2 backoff sleeps for 3 attempts; '
            f'got {mock_sleep.await_args_list}'
        )

    def test_enospc_fails_fast_without_retrying(self, tmp_path: Path) -> None:
        """WORK item (2): a full disk is not blanket-retried. One attempt,
        no backoff — 1.5s of sleeping cannot free a byte."""
        git_ops = GitOps(GitConfig(), tmp_path)
        calls: list[list[str]] = []
        target = tmp_path / 'wt' / '_merge-deadbeef'

        async def _body():
            return await git_ops._worktree_add_with_retry(
                target, MERGE_SHA, label='test',
            )

        with (
            patch(
                'orchestrator.git_ops._run',
                side_effect=_make_fake_add([(1, '', ENOSPC_3692_STDERR)], calls),
            ),
            patch('orchestrator.git_ops.asyncio.sleep', new_callable=AsyncMock) as mock_sleep,
        ):
            rc, out, err, attempts = asyncio.run(_body())

        assert rc == 1
        assert attempts == 1, f'expected ENOSPC to fail on the first attempt; got {attempts}'
        assert err == ENOSPC_3692_STDERR, 'expected the ENOSPC stderr to survive to the caller'
        assert len(_add_argvs(calls)) == 1, (
            f'expected exactly ONE add attempt on ENOSPC; got {_add_argvs(calls)}'
        )
        assert mock_sleep.await_args_list == [], (
            f'expected NO backoff sleep on a non-retryable failure; '
            f'got {mock_sleep.await_args_list}'
        )

    def test_success_on_first_try_costs_one_attempt_and_no_sleep(
        self, tmp_path: Path,
    ) -> None:
        git_ops = GitOps(GitConfig(), tmp_path)
        calls: list[list[str]] = []
        target = tmp_path / 'wt' / '_merge-deadbeef'

        async def _body():
            return await git_ops._worktree_add_with_retry(
                target, MERGE_SHA, label='test',
            )

        with (
            patch(
                'orchestrator.git_ops._run',
                side_effect=_make_fake_add([(0, '', '')], calls),
            ),
            patch('orchestrator.git_ops.asyncio.sleep', new_callable=AsyncMock) as mock_sleep,
        ):
            rc, out, err, attempts = asyncio.run(_body())

        assert (rc, attempts) == (0, 1)
        assert len(_add_argvs(calls)) == 1
        assert mock_sleep.await_args_list == []

    def test_partial_directory_residue_is_cleared_between_attempts(
        self, tmp_path: Path,
    ) -> None:
        """Real ``git worktree add`` creates its target directory early,
        before it can fail. Without clearing that residue, attempt N+1
        against the same path would fail deterministically with "'<path>'
        already exists" — converting a bounded retry into a GUARANTEED
        second failure whose operator-visible error names a self-inflicted
        cause instead of the real one.
        """
        git_ops = GitOps(GitConfig(), tmp_path)
        calls: list[list[str]] = []
        exists_at_entry: list[bool] = []
        target = tmp_path / 'wt' / '_merge-deadbeef'

        async def _body():
            return await git_ops._worktree_add_with_retry(
                target, MERGE_SHA, label='test',
            )

        with (
            patch(
                'orchestrator.git_ops._run',
                side_effect=_make_fake_add(
                    [(1, '', 'lock contention'), (1, '', 'lock contention'), (0, '', '')],
                    calls,
                    mkdir_on_failure=True,
                    exists_at_entry=exists_at_entry,
                ),
            ),
            patch('orchestrator.git_ops.asyncio.sleep', new_callable=AsyncMock),
        ):
            rc, out, err, attempts = asyncio.run(_body())

        assert rc == 0 and attempts == 3
        assert exists_at_entry == [False, False, False], (
            'expected the partial directory left by each failed add to be '
            'cleared before the next attempt; observed target.exists() at '
            f'each attempt entry = {exists_at_entry}'
        )

    def test_absorbed_retry_emits_a_greppable_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An absorbed flake must leave a trace an operator can grep, not
        vanish silently — otherwise the retry hides exactly the recurring
        failure rate this task exists to measure."""
        git_ops = GitOps(GitConfig(), tmp_path)
        calls: list[list[str]] = []
        target = tmp_path / 'wt' / '_merge-deadbeef'

        async def _body():
            return await git_ops._worktree_add_with_retry(
                target, MERGE_SHA, label='LABEL-MARKER',
            )

        with (
            caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'),
            patch(
                'orchestrator.git_ops._run',
                side_effect=_make_fake_add(
                    [(1, '', TRANSIENT_4545_STDERR), (0, '', '')], calls,
                ),
            ),
            patch('orchestrator.git_ops.asyncio.sleep', new_callable=AsyncMock),
        ):
            rc, out, err, attempts = asyncio.run(_body())

        assert (rc, attempts) == (0, 2)
        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and r.name == 'orchestrator.git_ops'
        ]
        assert len(warnings) == 1, (
            f'expected exactly one WARNING for the one absorbed retry; '
            f'got {[r.getMessage() for r in warnings]}'
        )
        msg = warnings[0].getMessage()
        assert 'LABEL-MARKER' in msg, (
            f'expected the caller-supplied label in the warning; got {msg!r}'
        )
        assert '1' in msg, f'expected the attempt number in the warning; got {msg!r}'

    def test_worktree_missing_propagates_unretried(self, tmp_path: Path) -> None:
        """A vanished ``project_root`` is not a transient ADD failure — the
        child never ran — so retrying it would burn 1.5s of backoff on a
        condition that cannot heal. ``WorktreeMissing`` is already handled
        as its own typed signal upstream."""
        from orchestrator.git_ops import WorktreeMissing

        git_ops = GitOps(GitConfig(), tmp_path)
        calls: list[list[str]] = []
        target = tmp_path / 'wt' / '_merge-deadbeef'

        async def _fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            raise WorktreeMissing(str(tmp_path))

        async def _body():
            return await git_ops._worktree_add_with_retry(
                target, MERGE_SHA, label='test',
            )

        with (
            patch('orchestrator.git_ops._run', side_effect=_fake_run),
            patch('orchestrator.git_ops.asyncio.sleep', new_callable=AsyncMock) as mock_sleep,
            pytest.raises(WorktreeMissing),
        ):
            asyncio.run(_body())

        assert len(_add_argvs(calls)) == 1, (
            f'expected WorktreeMissing to escape after ONE attempt; got {_add_argvs(calls)}'
        )
        assert mock_sleep.await_args_list == []


# ---------------------------------------------------------------------------
# step-5: _create_merge_worktree absorbs the transient flake
# ---------------------------------------------------------------------------

BASE_SHA = 'b' * 40
MAIN_HEAD_SHA = 'e' * 40


def _make_fake_merge_run(
    add_results: list[tuple[int, str, str]],
    calls: list[list[str]],
    *,
    rev_parse_sha: str = MAIN_HEAD_SHA,
):
    """Fake ``_run`` for ``_create_merge_worktree``.

    Beyond the per-attempt ``git worktree add`` results, this also answers
    the ``git fetch`` / ``git rev-parse`` calls the ``base_sha is None``
    branch makes — ``rev-parse`` returns *rev_parse_sha* on stdout with a
    trailing newline, exactly as real git does, so the caller's ``.strip()``
    is genuinely exercised.
    """
    state = {'add_calls': 0}

    async def _fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if 'worktree' in cmd and 'add' in cmd:
            detach_idx = cmd.index('--detach')
            target = Path(cmd[detach_idx + 1])
            idx = state['add_calls']
            rc, out, err = (
                add_results[idx] if idx < len(add_results) else add_results[-1]
            )
            state['add_calls'] += 1
            if rc == 0:
                target.mkdir(parents=True, exist_ok=True)
            return (rc, out, err)
        if 'rev-parse' in cmd:
            return (0, f'{rev_parse_sha}\n', '')
        return (0, '', '')

    return _fake_run


class TestCreateMergeWorktreeRetry:
    """step-5: the recurring cross-task flake, absorbed.

    Five archived occurrences under ``data/verify-logs`` (2026-08-10 ->
    2026-09-06) blocked a merge outright on a single non-zero
    ``git worktree add`` rc. With the shared driver wired in, the transient
    shapes are retried and the merge proceeds.

    RED on base: ``_create_merge_worktree`` is single-shot, so the first
    non-zero rc raises ``RuntimeError`` and every case here fails.
    """

    def test_transient_4215_shape_is_absorbed_and_the_worktree_is_returned(
        self, tmp_path: Path,
    ) -> None:
        git_ops = GitOps(GitConfig(), tmp_path)
        calls: list[list[str]] = []

        async def _body():
            return await git_ops._create_merge_worktree(base_sha=BASE_SHA)

        with (
            patch(
                'orchestrator.git_ops._run',
                side_effect=_make_fake_merge_run(
                    [(1, '', TRANSIENT_4215_STDERR), (0, '', '')], calls,
                ),
            ),
            patch('orchestrator.git_ops.asyncio.sleep', new_callable=AsyncMock) as mock_sleep,
        ):
            path, sha = asyncio.run(_body())

        assert sha == BASE_SHA
        assert path.parent == git_ops.worktree_base, (
            f'expected the minted path under worktree_base; got {path}'
        )
        assert path.name.startswith('_merge-'), (
            f'expected a _merge-<hex> directory name; got {path.name!r}'
        )
        adds = _add_argvs(calls)
        assert len(adds) == 2, f'expected the flake to be retried once; got {len(adds)} adds'
        assert mock_sleep.await_args_list == [call(0.5)], (
            f'expected exactly one 0.5s backoff; got {mock_sleep.await_args_list}'
        )

    def test_retry_reuses_the_same_minted_path(self, tmp_path: Path) -> None:
        """One uuid is minted per CALL, not per attempt — so nothing else can
        own the path and the between-attempts rmtree is safe."""
        git_ops = GitOps(GitConfig(), tmp_path)
        calls: list[list[str]] = []

        async def _body():
            return await git_ops._create_merge_worktree(base_sha=BASE_SHA)

        with (
            patch(
                'orchestrator.git_ops._run',
                side_effect=_make_fake_merge_run(
                    [(1, '', TRANSIENT_4545_STDERR), (0, '', '')], calls,
                ),
            ),
            patch('orchestrator.git_ops.asyncio.sleep', new_callable=AsyncMock),
        ):
            path, _sha = asyncio.run(_body())

        adds = _add_argvs(calls)
        detach_targets = {c[c.index('--detach') + 1] for c in adds}
        assert detach_targets == {str(path)}, (
            f'expected both attempts to target the single minted path {path}; '
            f'got {detach_targets}'
        )

    def test_base_sha_none_branch_also_retries(self, tmp_path: Path) -> None:
        """The main-HEAD branch (the normal merge path) inherits the retry
        too, and still returns main's rev-parsed SHA, stripped."""
        git_ops = GitOps(GitConfig(), tmp_path)
        calls: list[list[str]] = []

        async def _body():
            return await git_ops._create_merge_worktree()

        with (
            patch(
                'orchestrator.git_ops._run',
                side_effect=_make_fake_merge_run(
                    [(1, '', TRANSIENT_4215_STDERR), (0, '', '')], calls,
                ),
            ),
            patch('orchestrator.git_ops.asyncio.sleep', new_callable=AsyncMock),
        ):
            path, sha = asyncio.run(_body())

        assert sha == MAIN_HEAD_SHA, (
            f"expected main's rev-parsed SHA stripped of its newline; got {sha!r}"
        )
        assert path.name.startswith('_merge-')
        adds = _add_argvs(calls)
        assert len(adds) == 2, f'expected 2 adds on the base_sha=None branch; got {len(adds)}'
        # The add must target main_branch, not a raw sha, on this branch.
        assert adds[0][-1] == git_ops.config.main_branch, (
            f'expected the add to check out {git_ops.config.main_branch!r}; got {adds[0][-1]!r}'
        )
