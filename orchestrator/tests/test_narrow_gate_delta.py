"""Reading a commit range's tree delta from git and classifying it for the narrow merge gate (task 5411)."""
from __future__ import annotations

from pathlib import Path

import pytest
from _landing_git_scenarios import LandingRepo

from orchestrator.narrow_gate_class import (
    ChangedPath,
    GateClassification,
    GateMiss,
    MergeGateResolution,
    MissReason,
)
from orchestrator.narrow_gate_delta import (
    classify_branch_change,
    classify_tree_delta,
    narrow_member_ids,
    read_tree_delta,
)

DOCS_PATHS = ['*.md', 'docs/**']
_PY = 'pkg/mod.py'
_PY_BASE = '"""Module."""\nx = 1\n'


@pytest.fixture
def repo(tmp_path: Path) -> LandingRepo:
    root = tmp_path / 'repo'
    root.mkdir()
    repo = LandingRepo.init(root)
    repo.commit('base', {'README.md': 'hello\n', 'docs/a.md': 'a\n', _PY: _PY_BASE, 'old.md': 'old\n'})
    return repo


def _full(reason: MissReason, path: str | None) -> GateClassification:
    return GateClassification(MergeGateResolution.FULL, GateMiss(reason, path))


@pytest.mark.asyncio
class TestReadTreeDelta:

    async def test_it_reports_added_modified_and_deleted_paths_with_modes(self, repo: LandingRepo) -> None:
        base = repo.sha('HEAD')
        repo.git('rm', '-q', 'docs/a.md')
        head = repo.commit('change', {'README.md': 'changed\n', 'docs/new.md': 'new\n'})

        delta = await read_tree_delta(repo.root, base, head)

        assert delta == (
            ChangedPath('README.md', 'M', '100644', '100644'),
            ChangedPath('docs/a.md', 'D', '100644', '000000'),
            ChangedPath('docs/new.md', 'A', '000000', '100644'),
        )

    async def test_a_rename_is_a_deletion_plus_an_addition(self, repo: LandingRepo) -> None:
        base = repo.sha('HEAD')
        repo.git('mv', 'old.md', 'docs/moved.md')
        head = repo.commit('rename')

        delta = await read_tree_delta(repo.root, base, head)

        assert delta is not None
        assert {(c.path, c.status) for c in delta} == {('old.md', 'D'), ('docs/moved.md', 'A')}

    async def test_a_chmod_only_change_reports_differing_modes(self, repo: LandingRepo) -> None:
        base = repo.sha('HEAD')
        repo.git('update-index', '--chmod=+x', 'README.md')
        repo.git('commit', '-q', '-m', 'chmod')
        head = repo.sha('HEAD')

        assert await read_tree_delta(repo.root, base, head) == (
            ChangedPath('README.md', 'M', '100644', '100755'),
        )

    async def test_an_identical_range_is_an_empty_delta(self, repo: LandingRepo) -> None:
        head = repo.sha('HEAD')

        assert await read_tree_delta(repo.root, head, head) == ()

    async def test_an_unknown_base_is_unreadable(self, repo: LandingRepo) -> None:
        assert await read_tree_delta(repo.root, '0' * 40, repo.sha('HEAD')) is None

    async def test_a_missing_repo_is_unreadable(self, tmp_path: Path) -> None:
        assert await read_tree_delta(tmp_path / 'absent', 'HEAD~1', 'HEAD') is None


@pytest.mark.asyncio
class TestClassifyTreeDelta:

    async def test_a_docs_only_commit_takes_the_docs_gate(self, repo: LandingRepo) -> None:
        base = repo.sha('HEAD')
        head = repo.commit('docs', {'README.md': 'reworded\n', 'docs/b.md': 'b\n'})

        assert await classify_tree_delta(repo.root, base, head, DOCS_PATHS) == GateClassification(
            MergeGateResolution.DOCS,
        )

    async def test_a_comment_only_py_commit_takes_the_comments_gate(self, repo: LandingRepo) -> None:
        base = repo.sha('HEAD')
        head = repo.commit('comment', {_PY: '"""Reworded module."""\nx = 1  # why\n', 'README.md': 'x\n'})

        assert await classify_tree_delta(repo.root, base, head, DOCS_PATHS) == GateClassification(
            MergeGateResolution.COMMENTS,
        )

    async def test_a_code_commit_takes_the_full_gate(self, repo: LandingRepo) -> None:
        base = repo.sha('HEAD')
        head = repo.commit('code', {_PY: '"""Module."""\nx = 2\n'})

        assert await classify_tree_delta(repo.root, base, head, DOCS_PATHS) == _full(MissReason.CODE_CHANGED, _PY)

    async def test_blobs_are_read_from_git_not_the_working_tree(self, repo: LandingRepo) -> None:
        base = repo.sha('HEAD')
        head = repo.commit('code', {_PY: '"""Module."""\nx = 2\n'})
        repo.write(_PY, _PY_BASE)

        assert await classify_tree_delta(repo.root, base, head, DOCS_PATHS) == _full(MissReason.CODE_CHANGED, _PY)

    async def test_a_leading_blank_line_before_a_shebang_is_not_lost(self, repo: LandingRepo) -> None:
        base = repo.commit('script', {'pkg/tool.py': '\n#!/usr/bin/env python3\nprint(1)\n'})
        head = repo.commit('shebang', {'pkg/tool.py': '#!/usr/bin/env python3\nprint(1)\n'})

        assert await classify_tree_delta(repo.root, base, head, DOCS_PATHS) == _full(
            MissReason.CODE_CHANGED, 'pkg/tool.py',
        )

    async def test_a_non_utf8_py_is_undecodable(self, repo: LandingRepo) -> None:
        base = repo.sha('HEAD')
        (repo.root / _PY).write_bytes(b'"""Module."""\nx = 1  # caf\xe9\n')
        repo.git('add', '-A')
        repo.git('commit', '-q', '-m', 'latin-1 comment')
        head = repo.sha('HEAD')

        assert await classify_tree_delta(repo.root, base, head, DOCS_PATHS) == _full(MissReason.UNDECODABLE, _PY)

    async def test_an_unreadable_range_takes_the_full_gate(self, repo: LandingRepo) -> None:
        classification = await classify_tree_delta(repo.root, '0' * 40, repo.sha('HEAD'), DOCS_PATHS)

        assert classification == _full(MissReason.DELTA_UNREADABLE, None)


@pytest.mark.asyncio
class TestBranchChange:

    async def test_a_docs_branch_stays_docs_after_main_moves_ahead_with_code(self, repo: LandingRepo) -> None:
        repo.git('checkout', '-q', '-b', 'task/docs')
        repo.commit('docs', {'docs/a.md': 'reworded\n'})
        repo.git('checkout', '-q', 'main')
        repo.commit('code on main', {_PY: '"""Module."""\nx = 99\n'})

        classification = await classify_branch_change(repo.root, 'main', 'task/docs', DOCS_PATHS)

        assert classification == GateClassification(MergeGateResolution.DOCS)

    async def test_an_unknown_branch_takes_the_full_gate(self, repo: LandingRepo) -> None:
        classification = await classify_branch_change(repo.root, 'main', 'task/absent', DOCS_PATHS)

        assert classification == _full(MissReason.DELTA_UNREADABLE, None)

    async def test_narrow_member_ids_keeps_only_narrow_branches(self, repo: LandingRepo) -> None:
        for branch, files in (
            ('task/docs', {'docs/a.md': 'reworded\n'}),
            ('task/comments', {_PY: '"""Module."""\nx = 1  # why\n'}),
            ('task/code', {_PY: '"""Module."""\nx = 3\n'}),
        ):
            repo.git('checkout', '-q', '-b', branch, 'main')
            repo.commit(branch, files)
        repo.git('checkout', '-q', 'main')

        members = await narrow_member_ids(repo.root, 'main', {
            '1': 'task/docs', '2': 'task/comments', '3': 'task/code', '4': 'task/absent',
        }, DOCS_PATHS)

        assert members == frozenset({'1', '2'})
