"""The plan-files gate attributes an untouched plan file to a dropped commit (task 4669).

A rebase drops a branch commit whose change the target already carries.  The
plan-files gate reads only ``base..HEAD``, so the file that commit wrote looks
untouched, and the old message blamed the implementation ("no commit on the
branch touched them ... Implementation has not delivered").  The gate now
reads RebaseDropLedger and reports such entries in
``PlanFilesTouchedResult.dropped_by_rebase``, and
``plan_files_not_touched_reason`` words them as what they are.

Part 1 runs against a REAL temp git repository and a real GitOps, whose
rebase_onto_main records the drops.  Part 2 builds PlanFilesTouchedResult
values directly: the reason text is a pure function of them.

Fixture (``dropped`` fixture):
  main: index.html ('v=43'), a.txt, c.txt, old.txt, web/v.txt ('v=43')
  task/7: A edits a.txt; B bumps index.html; B2 bumps web/v.txt; D deletes old.txt
  main then makes B's, B2's and D's changes in one commit, plus m.txt,
  and rebase_onto_main drops B, B2 and D.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import pytest_asyncio

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_lane.gates import (
    PLAN_FILES_NOT_TOUCHED_REASON_PREFIX,
    PlanFilesTouchedResult,
    _check_plan_files_touched_in_branch,
    plan_files_not_touched_reason,
)
from orchestrator.rebase_drops import DroppedCommit

# ---------------------------------------------------------------------------
# Helpers (copied locally from test_git_ops_unstack.py, per-file convention)
# ---------------------------------------------------------------------------

_BUMP = 'chore(7): bump 43 -> 44 (task 7)'
_WEB_BUMP = 'chore(7): bump web 43 -> 44 (task 7)'
_DELETE = 'chore(7): delete old.txt (task 7)'


async def _git(cwd: Path, *args: str) -> str:
    rc, out, err = await _run(['git', *args], cwd=cwd)
    assert rc == 0, f'git {args} failed: {err}'
    return out


async def _init_repo(path: Path) -> None:
    await _git(path, 'init', '-b', 'main')
    await _git(path, 'config', 'user.email', 'test@test.com')
    await _git(path, 'config', 'user.name', 'Test')


async def _commit(
    path: Path, msg: str, writes: dict[str, str], deletes: tuple[str, ...] = (),
) -> str:
    """Write and delete exactly the named files and commit them; never ``add -A``."""
    for name, content in writes.items():
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_text(content)
    if writes:
        await _git(path, 'add', '--', *writes)
    if deletes:
        await _git(path, 'rm', '-q', '--', *deletes)
    await _git(path, 'commit', '-m', msg)
    return (await _git(path, 'rev-parse', '--verify', 'HEAD')).strip()


def _make_git_ops(repo: Path) -> GitOps:
    cfg = GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )
    return GitOps(cfg, repo)


@dataclass(frozen=True)
class Dropped:
    git_ops: GitOps
    base: str
    head: str
    bump: DroppedCommit
    web_bump: DroppedCommit

    async def check(
        self, plan_files: list[str], *, branch: str | None = 'task/7',
    ) -> PlanFilesTouchedResult:
        if branch is None:
            return await _check_plan_files_touched_in_branch(
                plan_files, self.base, self.head, self.git_ops, task_id='7',
            )
        return await _check_plan_files_touched_in_branch(
            plan_files, self.base, self.head, self.git_ops, task_id='7', branch=branch,
        )


@pytest_asyncio.fixture
async def dropped(tmp_path: Path) -> Dropped:
    root = tmp_path / 'repo'
    root.mkdir()
    await _init_repo(root)
    await _commit(root, 'initial', {
        'index.html': 'v=43\n', 'a.txt': 'a\n', 'c.txt': 'c\n',
        'old.txt': 'old\n', 'web/v.txt': 'v=43\n',
    })
    git_ops = _make_git_ops(root)
    wt = git_ops.worktree_base / '7'
    wt.parent.mkdir(parents=True, exist_ok=True)
    await _git(root, 'worktree', 'add', '-b', 'task/7', str(wt), 'main')
    await _commit(wt, 'feat(7): own a (task 7)', {'a.txt': 'a\nown\n'})
    bump = await _commit(wt, _BUMP, {'index.html': 'v=44\n'})
    web_bump = await _commit(wt, _WEB_BUMP, {'web/v.txt': 'v=44\n'})
    await _commit(wt, _DELETE, {}, deletes=('old.txt',))
    base = await _commit(
        root, 'main: the same changes and more',
        {'index.html': 'v=44\n', 'web/v.txt': 'v=44\n', 'm.txt': 'm\n'},
        deletes=('old.txt',),
    )

    assert await git_ops.rebase_onto_main(wt) is True

    head = (await _git(root, 'rev-parse', '--verify', 'task/7')).strip()
    return Dropped(
        git_ops=git_ops, base=base, head=head,
        bump=DroppedCommit(sha=bump, subject=_BUMP, paths=('index.html',)),
        web_bump=DroppedCommit(sha=web_bump, subject=_WEB_BUMP, paths=('web/v.txt',)),
    )


# ---------------------------------------------------------------------------
# Part 1: _check_plan_files_touched_in_branch classifies dropped entries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGateAttributesDroppedCommits:
    async def test_an_untouched_file_a_dropped_commit_wrote_is_attributed(
        self, dropped: Dropped,
    ) -> None:
        check = await dropped.check(['a.txt', 'index.html'])

        assert check.not_touched == ['index.html']
        assert check.dropped_by_rebase == {'index.html': (dropped.bump,)}
        assert check.missing_from_tree == []

    async def test_without_a_branch_nothing_is_attributed(self, dropped: Dropped) -> None:
        with_branch = await dropped.check(['a.txt', 'index.html'])
        without = await dropped.check(['a.txt', 'index.html'], branch=None)

        assert without.dropped_by_rebase == {}
        assert without.not_touched == with_branch.not_touched
        assert without.missing_from_tree == with_branch.missing_from_tree
        assert without.resolved_renames == with_branch.resolved_renames

    async def test_a_directory_entry_is_attributed_through_its_contents(
        self, dropped: Dropped,
    ) -> None:
        check = await dropped.check(['a.txt', 'web/'])

        assert check.not_touched == ['web/']
        assert check.dropped_by_rebase == {'web/': (dropped.web_bump,)}

    async def test_an_entry_missing_from_the_tree_is_never_attributed(
        self, dropped: Dropped,
    ) -> None:
        check = await dropped.check(['a.txt', 'old.txt'])

        assert check.not_touched == ['old.txt']
        assert check.missing_from_tree == ['old.txt']
        assert check.dropped_by_rebase == {}

    async def test_a_dropped_commit_explains_only_the_paths_it_wrote(
        self, dropped: Dropped,
    ) -> None:
        check = await dropped.check(['a.txt', 'c.txt'])

        assert check.not_touched == ['c.txt']
        assert check.dropped_by_rebase == {}


# ---------------------------------------------------------------------------
# Part 2: plan_files_not_touched_reason words each class as what it is
# ---------------------------------------------------------------------------

_SHA = 'c2e8f8c14d' + '0' * 30
_SUBJECT = 'chore(3490): bump the redux cache-buster uniformly, 43 -> 44'
_DROPPED = DroppedCommit(sha=_SHA, subject=_SUBJECT, paths=('index.html',))

_UNWRITTEN_CLAUSE = (
    'The architect declared these files but no commit on the branch touched '
    'them: real.py.  That is what was measured; it does not establish why.  '
    'Two common causes: the declared work is missing, or the file was '
    'correctly declared and legitimately needed no change.  To tell them '
    "apart, check whether each file relates textually to the task's subject, "
    'whether the branch touched sibling files in the same directory, whether '
    'the declared path still exists at branch HEAD, and whether the '
    "branch's own commits are already ancestors of main (git merge-base "
    '--is-ancestor <commit> main), meaning the work already landed.  '
)
_STALE_CLAUSE = (
    'Declared path does not exist in the tree at branch HEAD (renamed or never '
    'created), so the branch-touched set was not consulted for: stale.py — '
    'investigate the declared path, not the branch.  '
)
_RENAME_CLAUSE = 'Rename resolution: old.py -> new.py.'


class TestPlanFilesNotTouchedReason:
    def test_a_dropped_only_reason_does_not_blame_the_implementation(self) -> None:
        reason = plan_files_not_touched_reason(PlanFilesTouchedResult(
            not_touched=['index.html'],
            dropped_by_rebase={'index.html': (_DROPPED,)},
        ))

        assert reason.startswith(f'{PLAN_FILES_NOT_TOUCHED_REASON_PREFIX}: index.html. ')
        assert _DROPPED.label == f'{_SHA[:12]} "{_SUBJECT}"'
        assert _DROPPED.label in reason
        assert 'written on this branch' in reason
        assert 'dropped by a rebase' in reason
        assert 'already carried' in reason
        assert 'not an implementation gap' in reason
        assert 're-derive' in reason.lower()
        assert 'current main' in reason
        assert 'no commit on the branch touched them' not in reason
        assert 'Implementation has not delivered' not in reason

    def test_a_mixed_reason_scopes_each_clause_to_its_own_entries(self) -> None:
        reason = plan_files_not_touched_reason(PlanFilesTouchedResult(
            not_touched=['real.py', 'index.html', 'stale.py'],
            missing_from_tree=['stale.py'],
            resolved_renames={'old.py': 'new.py'},
            dropped_by_rebase={'index.html': (_DROPPED,)},
        ))

        assert reason.startswith(
            f'{PLAN_FILES_NOT_TOUCHED_REASON_PREFIX}: real.py, index.html, stale.py. '
        )
        assert _UNWRITTEN_CLAUSE in reason
        assert _STALE_CLAUSE in reason
        assert reason.endswith(_RENAME_CLAUSE)
        dropped_clause = reason[
            reason.index(_UNWRITTEN_CLAUSE) + len(_UNWRITTEN_CLAUSE):
            reason.index(_STALE_CLAUSE)
        ]
        assert 'index.html' in dropped_clause
        assert _SHA[:12] in dropped_clause
        assert 'real.py' not in dropped_clause
        assert 'stale.py' not in dropped_clause

    def test_a_reason_without_drops_is_worded_as_before(self) -> None:
        reason = plan_files_not_touched_reason(PlanFilesTouchedResult(
            not_touched=['real.py', 'stale.py'],
            missing_from_tree=['stale.py'],
            resolved_renames={'old.py': 'new.py'},
        ))

        assert reason == (
            f'{PLAN_FILES_NOT_TOUCHED_REASON_PREFIX}: real.py, stale.py. '
            f'{_UNWRITTEN_CLAUSE}{_STALE_CLAUSE}{_RENAME_CLAUSE}'
        )


class TestPlanFilesTouchedResultInvariant:
    """``dropped_by_rebase`` is a subset of ``not_touched`` and disjoint from
    ``missing_from_tree``: a value breaking that would render without its
    dropped clause, so construction refuses it, naming the offending keys."""

    def test_a_dropped_entry_the_gate_did_not_fail_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"not in not_touched: \['index\.html'\]"):
            PlanFilesTouchedResult(
                not_touched=['real.py'],
                dropped_by_rebase={'index.html': (_DROPPED,)},
            )

    def test_a_dropped_entry_missing_from_the_tree_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"in missing_from_tree: \['index\.html'\]"):
            PlanFilesTouchedResult(
                not_touched=['index.html'],
                missing_from_tree=['index.html'],
                dropped_by_rebase={'index.html': (_DROPPED,)},
            )
