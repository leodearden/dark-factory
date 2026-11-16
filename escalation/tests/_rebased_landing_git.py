"""Real-git builders for the worker-rebased solo landing, escalation side.

When main moves under a solo merge request's verify,
``orchestrator/src/orchestrator/git_ops.py::GitOps.advance_main`` rebases the
branch onto main and fast-forwards main to the rebased tip.  No merge commit is
written, every branch commit is rewritten, and ``task/<id>`` stays at its
pre-rebase tip: a live ref that is NOT an ancestor of main, no
``Merge task/<id> into main`` marker, and every branch commit patch-equivalent
on main.  These builders produce that shape (and its near misses) in a
throwaway repo for the git-authority tier's patch-id arm.  The shell-side
characterization of the same shape is
``orchestrator/tests/test_rebased_solo_landing_ladder.py`` (task 6936), whose
helpers escalation tests cannot import.  Every builder leaves ``main`` checked
out, because ``GitOps.find_equivalent_commit`` searches ``<base>..HEAD`` in the
project root.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Deterministic identity + no signing, so subjects and shas never depend on
# the developer's git config.
_GIT_ENV_ARGS = (
    '-c', 'user.name=Test',
    '-c', 'user.email=test@test.com',
    '-c', 'commit.gpgsign=false',
)


def git(root: Path, *args: str) -> str:
    """Run git in *root*; return stripped stdout, raising on non-zero.

    Discovery is capped at *root* (``GIT_CEILING_DIRECTORIES``), so a builder
    can never reach a repository enclosing pytest's basetemp.
    """
    env = {**os.environ, 'GIT_CEILING_DIRECTORIES': str(root.resolve().parent)}
    return subprocess.run(
        ['git', *_GIT_ENV_ARGS, *args],
        cwd=root, check=True, capture_output=True, text=True, env=env,
    ).stdout.strip()


def commit(root: Path, subject: str, filename: str) -> str:
    """Write *filename*, commit it with *subject*, and return the new sha.

    Each commit writes its own file so a rebase or cherry-pick never conflicts.
    """
    (root / filename).write_text(f'{subject}\n')
    git(root, 'add', filename)
    git(root, 'commit', '-m', subject)
    return git(root, 'rev-parse', 'HEAD')


def new_repo(root: Path) -> None:
    """Create *root* as a repository on ``main`` holding one base commit."""
    root.mkdir()
    git(root, 'init', '-b', 'main')
    commit(root, 'base', 'base.txt')


@dataclass(frozen=True)
class RebasedSoloLanding:
    """A landed rebased solo branch, and the shas a verdict is judged against.

    ``branch_tip`` is the retained pre-rebase tip of ``task/<task_id>``, which
    is not on main.  ``main_tip`` is main after the fast-forward: the rebased
    twin of ``branch_tip``.  ``citing_sha`` is the rebased commit on main whose
    subject cites the task, or ``None`` when no branch commit cited it.
    """

    repo: Path
    task_id: str
    branch_tip: str
    main_tip: str
    citing_sha: str | None


def _land_rebased(root: Path, branch: str) -> str:
    """Land *branch* the way the worker's rebased path does; return main's new tip.

    A scratch branch carries the rebase so *branch* keeps its pre-rebase tip.
    """
    git(root, 'checkout', '-b', 'scratch', branch)
    git(root, 'rebase', 'main')
    git(root, 'checkout', 'main')
    git(root, 'merge', '--ff-only', 'scratch')
    git(root, 'branch', '-D', 'scratch')
    return git(root, 'rev-parse', 'main')


def build_rebased_solo_landing(root: Path, task_id: str, *, cited: bool) -> RebasedSoloLanding:
    """A two-commit ``task/<task_id>`` landed rebased past one direct-to-main commit.

    The first commit cites the task when *cited*; the tip never does.
    """
    branch = f'task/{task_id}'
    citing_subject = f'impl({task_id}): first'
    new_repo(root)
    git(root, 'checkout', '-b', branch)
    commit(root, citing_subject if cited else 'amend: first', 'first.txt')
    branch_tip = commit(root, 'amend: second', 'second.txt')
    git(root, 'checkout', 'main')
    commit(root, 'docs: direct-to-main', 'direct.md')
    main_tip = _land_rebased(root, branch)
    citing_sha = None
    if cited:
        citing_sha = git(
            root, 'log', 'main', '--fixed-strings', f'--grep={citing_subject}',
            '--max-count=1', '--format=%H',
        )
    return RebasedSoloLanding(
        repo=root, task_id=task_id, branch_tip=branch_tip, main_tip=main_tip,
        citing_sha=citing_sha,
    )


def reapply_tip_on_main(landing: RebasedSoloLanding) -> None:
    """Revert ``landing.main_tip`` on main, then cherry-pick it back.

    Main then holds two commits sharing the branch tip's patch-id and subject,
    so ``git cherry`` still reports every branch commit as contained while a
    patch-id twin lookup has no single answer to give.
    """
    git(landing.repo, 'revert', '--no-edit', landing.main_tip)
    git(landing.repo, 'cherry-pick', landing.main_tip)


def add_unlanded_branch(root: Path, task_id: str) -> str:
    """Add a live ``task/<task_id>`` to the repo at *root*; return its tip.

    The branch carries one commit main lacks, and main then advances by one
    unrelated commit, so the tip is neither main's tip nor an ancestor of it.
    """
    branch = f'task/{task_id}'
    git(root, 'checkout', '-b', branch, 'main')
    tip = commit(root, f'impl({task_id}): unlanded work', f'unlanded-{task_id}.txt')
    git(root, 'checkout', 'main')
    commit(root, 'chore: unrelated work by another task', f'unrelated-{task_id}.txt')
    return tip


def add_net_empty_branch(root: Path, task_id: str) -> str:
    """Add a live ``task/<task_id>`` whose commits cancel out; return its tip.

    The branch commits work and then reverts it, and main advances by one
    unrelated commit.  The tip is neither main's tip nor an ancestor of it,
    and its net diff from the merge base is empty.
    """
    branch = f'task/{task_id}'
    git(root, 'checkout', '-b', branch, 'main')
    work = commit(root, f'impl({task_id}): work', f'work-{task_id}.txt')
    git(root, 'revert', '--no-edit', work)
    tip = git(root, 'rev-parse', 'HEAD')
    git(root, 'checkout', 'main')
    commit(root, 'chore: unrelated work by another task', f'unrelated-{task_id}.txt')
    return tip


def add_main_merged_branch(root: Path, task_id: str, *, earlier_citation: bool) -> str:
    """Add a live ``task/<task_id>`` holding only a ``--no-ff`` merge of main.

    Returns the branch's creation point, which a dispatched task records as
    ``branch_base_sha``.  main advances past that point first, by a commit
    citing the task (an earlier round's) when *earlier_citation*, then by an
    unrelated commit.  The branch has no non-merge commit of its own, so
    ``git cherry main task/<task_id>`` prints nothing.
    """
    branch = f'task/{task_id}'
    base = git(root, 'rev-parse', 'main')
    git(root, 'branch', branch, base)
    if earlier_citation:
        commit(root, f'impl({task_id}): earlier round', f'earlier-{task_id}.txt')
    commit(root, 'chore: unrelated work by another task', f'unrelated-{task_id}.txt')
    git(root, 'checkout', branch)
    git(root, 'merge', '--no-ff', '-m', f'Merge main into {branch}', 'main')
    git(root, 'checkout', 'main')
    return base
