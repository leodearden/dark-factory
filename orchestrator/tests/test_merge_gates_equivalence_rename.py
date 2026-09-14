"""Gate-level tests for a rename false-positive class of
:func:`orchestrator.merge_gates._check_post_merge_equivalence` — SITE 1.

The defect: the gate's compare set is built by a raw path-string
subtraction, ``[p for p in branch_touched if p not in main_touched]``,
while all three of its set-building diffs pass ``--no-renames``.  A
branch-side rename is therefore SPLIT into two unrelated path strings
(old and new), and main's concurrent edits to the same file land at the
rename SOURCE.  The new path is absent from ``main_touched``, so it
survives the subtraction and the final scoped diff reports it — the gate
blocks a merge that dropped nothing at all.

Measured case: reify task 5694, merge ``d1d857f43545``, escalation
esc-5694-5.  The branch relocated a file main had concurrently edited at
the old path; the merged tree carried BOTH edits, and the gate still
reported "Conflict resolution likely dropped or rewrote work".  The RCA
that followed read the triage diff in the wrong direction and concluded
the opposite of the truth.

The complementary SITE 2 defect — the plan-target drop-guard's
``branch_changed`` / ``dropped_in_merge`` intersection — lives in
``test_merge_gates_drop_guard_rename.py``.  The two share the
``_rename_pairs`` primitive but resolve renames on OPPOSITE ranges, so
they are kept in separate files.

These tests exercise the gate against a REAL git repository, so they live
in this dedicated file rather than in the 24k-line ``test_merge_queue.py``
— the precedent set by the sibling ``test_merge_gates_plan_files_rename.py``
(its docstring, L29-30/36-39: real-git tests plus keeping hot
``merge_queue.py`` out of this task's lock scope).  They import the gate
from ``orchestrator.merge_gates`` DIRECTLY, never through the
``orchestrator.merge_queue`` shim.  The only deviation from pure real-git
is :class:`_RunSpy`, used narrowly for the one property real git will not
produce on demand: a non-zero rc from a specific git subcommand.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, _run

# ---------------------------------------------------------------------------
# Fixtures — the standard real-git fixture triple, copied verbatim from
# test_merge_queue.py:88-122.  Per-file duplication (rather than promotion
# to conftest.py) is the established convention across ~60 sibling test
# files in this suite; promoting it would widen this task's lock scope onto
# a shared conftest for no benefit.
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository with an initial commit."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


async def _setup_repo(repo: Path):
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        # Tests use a tmp repo with no real remote; disabling the push avoids
        # per-test subprocess noise.
        push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(git_config, git_repo)


# ---------------------------------------------------------------------------
# _RunSpy — this file's one, narrowly-scoped deviation from pure real-git
# ---------------------------------------------------------------------------


class _RunSpy:
    """Delegating wrapper around ``merge_gates._run`` with fault injection.

    One gate property is invisible to a pure real-git test: what the gate
    does when a specific git subcommand returns a non-zero rc, which real
    git will not produce on demand.  *fail_when* is a predicate over the
    argv list; a command it matches returns ``(128, '', <fatal>)`` WITHOUT
    being executed.  Everything else delegates to the real ``_run``, so
    the surrounding repository work stays real.

    Modelled on ``test_merge_gates_plan_files_rename.py::_RunSpy`` minus
    its ``calls``/``count`` arm, which this file has no use for.
    """

    def __init__(
        self, fail_when: Callable[[Sequence[str]], bool] | None = None,
    ) -> None:
        self._fail_when = fail_when

    async def __call__(
        self, cmd: list[str], cwd: Path | None = None, **kwargs,
    ) -> tuple[int, str, str]:
        if self._fail_when is not None and self._fail_when(cmd):
            return 128, '', 'fatal: injected failure (test fault injection)\n'
        return await _run(cmd, cwd, **kwargs)
