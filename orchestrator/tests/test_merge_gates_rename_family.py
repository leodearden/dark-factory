"""The audit trail and output parsing of the rename-aware merge gates (task 6412).

Each test drives one of the three gate entry points production calls —
``_check_plan_files_touched_in_branch`` (workflow.py),
``_check_plan_targets_in_tree`` and ``_check_post_merge_equivalence``
(merge_lane/worker.py) — against a real git repository, and pins the exact
log record the gate emits, because for a passing stale-path entry that
record is the gate's only output.  Git failures are injected at the process
boundary by a PATH ``git`` shim (:func:`_fail_git_argv`), never by patching
lane internals.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest
from _git_fixtures import seed_repo

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane.gates import _check_plan_files_touched_in_branch

TASK = 'task-6412'
LANE_LOGGER = 'orchestrator.merge_lane'


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    return seed_repo(tmp_path / 'repo')


@pytest.fixture
def git_ops(git_repo: Path) -> GitOps:
    return GitOps(
        GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
            push_after_advance=False,
        ),
        git_repo,
    )


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ['git', *args], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _commit(repo: Path, files: Mapping[str, str | bytes], msg: str) -> str:
    for rel, content in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content)
    _git(repo, 'add', '-A')
    _git(repo, 'commit', '-q', '-m', msg)
    return _git(repo, 'rev-parse', 'HEAD')


def _mv(repo: Path, old: str, new: str, msg: str) -> str:
    (repo / new).parent.mkdir(parents=True, exist_ok=True)
    _git(repo, 'mv', old, new)
    _git(repo, 'commit', '-q', '-m', msg)
    return _git(repo, 'rev-parse', 'HEAD')


def _module(tag: str) -> str:
    return ''.join(f'{tag} line {i}\n' for i in range(1, 31))


def _record(caplog: pytest.LogCaptureFixture, needle: str) -> str:
    messages = [r.getMessage() for r in caplog.records if needle in r.getMessage()]
    assert len(messages) == 1, (
        f'expected exactly one record containing {needle!r}; got {messages!r}'
    )
    return messages[0]


@pytest.mark.asyncio
class TestPlanFilesAuditTrail:
    """What the plan-files gate logs when it resolves, or refuses to resolve, a stale entry."""

    async def test_invented_path_logs_no_history_not_a_probe_error(
        self, git_repo: Path, git_ops: GitOps, caplog: pytest.LogCaptureFixture,
    ):
        base = _commit(git_repo, {'pkg/real.py': _module('real')}, 'add real')
        head = _commit(git_repo, {'pkg/real.py': _module('edited')}, 'edit real')

        with caplog.at_level(logging.WARNING, logger=LANE_LOGGER):
            result = await _check_plan_files_touched_in_branch(
                ['docs/invented.md'], base, head, git_ops, task_id=TASK,
            )

        assert result.missing_from_tree == ['docs/invented.md']
        assert result.not_touched == ['docs/invented.md']
        assert _record(caplog, 'has NO history').endswith(f'task_id={TASK}')
        assert not [r for r in caplog.records if 'rename-probe' in r.getMessage()]

    async def test_single_hop_resolution_names_the_deleting_commit(
        self, git_repo: Path, git_ops: GitOps, caplog: pytest.LogCaptureFixture,
    ):
        _commit(git_repo, {'src/old_mod.py': _module('mod')}, 'add old_mod')
        sha = _mv(git_repo, 'src/old_mod.py', 'src/new_mod.py', 'rename mod')
        head = _commit(git_repo, {'src/new_mod.py': _module('edited')}, 'edit')

        with caplog.at_level(logging.WARNING, logger=LANE_LOGGER):
            result = await _check_plan_files_touched_in_branch(
                ['src/old_mod.py'], sha, head, git_ops, task_id=TASK,
            )

        assert result.resolved_renames == {'src/old_mod.py': 'src/new_mod.py'}
        assert result.not_touched == []
        assert f'via rename in {sha[:12]};' in _record(caplog, 'resolved to')

    async def test_three_hop_chain_names_the_last_hop(
        self, git_repo: Path, git_ops: GitOps, caplog: pytest.LogCaptureFixture,
    ):
        _commit(git_repo, {'one/mod.py': _module('mod')}, 'add mod')
        _mv(git_repo, 'one/mod.py', 'two/mod.py', 'hop 1')
        _mv(git_repo, 'two/mod.py', 'three/mod.py', 'hop 2')
        last = _mv(git_repo, 'three/mod.py', 'four/mod.py', 'hop 3')
        head = _commit(git_repo, {'four/mod.py': _module('edited')}, 'edit')

        with caplog.at_level(logging.WARNING, logger=LANE_LOGGER):
            result = await _check_plan_files_touched_in_branch(
                ['one/mod.py'], last, head, git_ops, task_id=TASK,
            )

        assert result.resolved_renames == {'one/mod.py': 'four/mod.py'}
        assert result.not_touched == []
        assert (
            f'via 3-hop rename chain ending in {last[:12]};'
            in _record(caplog, 'resolved to')
        )

    async def test_basename_fallback_names_its_mechanism(
        self, git_repo: Path, git_ops: GitOps, caplog: pytest.LogCaptureFixture,
    ):
        _commit(git_repo, {'pkg/old/widget.py': _module('old')}, 'add widget')
        _git(git_repo, 'rm', '-q', 'pkg/old/widget.py')
        _git(git_repo, 'commit', '-q', '-m', 'delete widget')
        base = _commit(git_repo, {'pkg/new/widget.py': _module('new')}, 're-add')
        head = _commit(git_repo, {'pkg/new/widget.py': _module('edited')}, 'edit')

        with caplog.at_level(logging.WARNING, logger=LANE_LOGGER):
            result = await _check_plan_files_touched_in_branch(
                ['pkg/old/widget.py'], base, head, git_ops, task_id=TASK,
            )

        assert result.resolved_renames == {'pkg/old/widget.py': 'pkg/new/widget.py'}
        assert result.not_touched == []
        assert 'via unique basename match;' in _record(caplog, 'resolved to')


@pytest.mark.asyncio
class TestRenameProbeReadsTheDeclaredPathsPair:
    """The rename probe picks the declared path's own R line out of ``git show --name-status -M``."""

    async def test_pair_found_among_other_changes_in_the_deleting_commit(
        self, git_repo: Path, git_ops: GitOps,
    ):
        _commit(
            git_repo,
            {
                'pkg/a0.txt': _module('a0'),
                'pkg/aa_one.py': _module('aa'),
                'pkg/zz_old.py': _module('zz'),
            },
            'add three files',
        )
        (git_repo / 'pkg' / 'a0.txt').write_text(_module('a0 edited'))
        _git(git_repo, 'mv', 'pkg/aa_one.py', 'pkg/aa_two.py')
        _git(git_repo, 'mv', 'pkg/zz_old.py', 'pkg/zz_new.py')
        sha = _commit(git_repo, {}, 'edit one, rename two')
        head = _commit(git_repo, {'pkg/zz_new.py': _module('zz edited')}, 'edit')

        assert _git(
            git_repo, 'show', '--name-status', '-M', '--format=', sha,
        ).splitlines() == [
            'M\tpkg/a0.txt',
            'R100\tpkg/aa_one.py\tpkg/aa_two.py',
            'R100\tpkg/zz_old.py\tpkg/zz_new.py',
        ]

        result = await _check_plan_files_touched_in_branch(
            ['pkg/zz_old.py'], sha, head, git_ops, task_id=TASK,
        )

        assert result.not_touched == []
        assert result.missing_from_tree == []
        assert result.resolved_renames == {'pkg/zz_old.py': 'pkg/zz_new.py'}

    async def test_declared_path_containing_a_space_resolves(
        self, git_repo: Path, git_ops: GitOps,
    ):
        _commit(git_repo, {'pkg/old name.py': _module('spaced')}, 'add spaced')
        sha = _mv(git_repo, 'pkg/old name.py', 'pkg/new_name.py', 'rename')
        head = _commit(git_repo, {'pkg/new_name.py': _module('edited')}, 'edit')

        result = await _check_plan_files_touched_in_branch(
            ['pkg/old name.py'], sha, head, git_ops, task_id=TASK,
        )

        assert result.resolved_renames == {'pkg/old name.py': 'pkg/new_name.py'}
        assert result.not_touched == []

    async def test_rename_resolution_ignores_a_repo_that_disables_diff_renames(
        self, git_repo: Path, git_ops: GitOps,
    ):
        _git(git_repo, 'config', 'diff.renames', 'false')
        _commit(git_repo, {'pkg/before.py': _module('mod')}, 'add before')
        sha = _mv(git_repo, 'pkg/before.py', 'pkg/after.py', 'rename')
        head = _commit(git_repo, {'pkg/after.py': _module('edited')}, 'edit')

        default_show = _git(git_repo, 'show', '--name-status', '--format=', sha)
        assert 'D\tpkg/before.py' in default_show
        assert not [ln for ln in default_show.splitlines() if ln.startswith('R')]

        result = await _check_plan_files_touched_in_branch(
            ['pkg/before.py'], sha, head, git_ops, task_id=TASK,
        )

        assert result.resolved_renames == {'pkg/before.py': 'pkg/after.py'}


_BASE_MODULE = ''.join(f'line{i}\n' for i in range(1, 41))


@pytest.mark.asyncio
class TestRenameCycleThroughMergeHistory:
    """A rename chain that revisits a path (reachable only across a merge) stops at the revisit."""

    async def test_cycle_stops_at_the_revisited_path_and_falls_back_to_the_hop_basename(
        self, git_repo: Path, git_ops: GitOps, caplog: pytest.LogCaptureFixture,
    ):
        _commit(git_repo, {'pkg/x.py': _BASE_MODULE}, 'add x')
        r1 = _mv(git_repo, 'pkg/x.py', 'pkg/a.py', 'x -> a')
        _git(git_repo, 'checkout', '-q', '-b', 'cyc-p', r1)
        p1 = _mv(git_repo, 'pkg/a.py', 'pkg/b.py', 'p: a -> b')
        _git(git_repo, 'checkout', '-q', '-b', 'cyc-q', r1)
        _mv(git_repo, 'pkg/a.py', 'pkg/b.py', 'q: a -> b')
        q2 = _mv(git_repo, 'pkg/b.py', 'pkg/a.py', 'q: b -> a')
        _git(git_repo, 'checkout', '-q', 'cyc-p')
        _git(git_repo, 'merge', '--no-ff', '--no-commit', 'cyc-q')
        _git(git_repo, 'rm', '-q', '-f', '--ignore-unmatch', 'pkg/a.py', 'pkg/b.py')
        merge = _commit(
            git_repo, {'moved/b.py': _BASE_MODULE}, 'merge: drop both, keep moved/b.py',
        )
        head = _commit(
            git_repo, {'moved/b.py': _BASE_MODULE + 'edited\n'}, 'edit moved/b.py',
        )

        def deleting_commit(path: str) -> str:
            return _git(git_repo, 'log', '--diff-filter=D', '-1', '--format=%H', head, '--', path)

        assert _git(git_repo, 'rev-parse', f'{merge}^1', f'{merge}^2').split() == [p1, q2]
        assert deleting_commit('pkg/x.py') == r1
        assert deleting_commit('pkg/a.py') == p1
        assert deleting_commit('pkg/b.py') == q2
        assert _git(
            git_repo, 'show', '--name-status', '-M', '--format=', q2,
        ) == 'R100\tpkg/b.py\tpkg/a.py'

        with caplog.at_level(logging.WARNING, logger=LANE_LOGGER):
            result = await _check_plan_files_touched_in_branch(
                ['pkg/x.py'], merge, head, git_ops, task_id=TASK,
            )

        assert result.not_touched == []
        assert result.missing_from_tree == []
        assert result.resolved_renames == {'pkg/x.py': 'moved/b.py'}
        assert (
            'via unique basename match on pkg/b.py (after 2-hop rename chain);'
            in _record(caplog, 'resolved to')
        )
