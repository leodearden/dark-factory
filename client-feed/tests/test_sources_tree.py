"""The client-visible tree at a main commit under a policy: kinds, counted lines from git objects, last-change dates."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from _git_repo import Repo, make_repo

from client_feed.contract import (
    FEED_VERSION,
    FileKind,
    RevealBudget,
    TreeFile,
    TreeSnapshot,
    TreeTotals,
)
from client_feed.policy import Policy, matches_any
from client_feed.sources.git import main_head
from client_feed.sources.tree import LineCounts, VisibleTree, visible_tree

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
T1 = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)
T2 = datetime(2026, 1, 3, 12, 0, tzinfo=UTC)
T3 = datetime(2026, 1, 4, 12, 0, tzinfo=UTC)
SENTINEL = datetime(2000, 1, 1, tzinfo=UTC)

A_PY = b'import os\n\n\ndef f():\n    return os.sep\n'


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    return make_repo(tmp_path / 'repo')


def _policy(root: Path, tmp_path: Path, *visible: str, **fields: Any) -> Policy:
    return Policy.model_validate(
        {
            'project_id': 'acme',
            'project_root': str(root),
            'out_dir': str(tmp_path / 'out'),
            'in_dir': str(tmp_path / 'in'),
            'visible_globs': list(visible),
            **fields,
        }
    )


def _snapshot(tree: VisibleTree) -> TreeSnapshot:
    return TreeSnapshot(
        seq=1,
        at=T0,
        project_id='acme',
        feed_version=FEED_VERSION,
        commit=tree.commit,
        totals=tree.totals,
        files=tree.files,
        budget=RevealBudget(
            proportion=0.25, counted_lines=0, budget_lines=0, spent_lines=0, floor_lines=50
        ),
    )


def _blob(repo: Repo, commit: str, path: str) -> str:
    return repo.rev(f'{commit}:{path}')


def test_only_visible_files_are_listed(repo: Repo, tmp_path: Path) -> None:
    repo.write('dark-factory-orchestrator.yaml', 'test_command: pytest\n')
    repo.write('.mcp.json', '{}\n')
    repo.write('plans/review.md', '# secret review\n')
    repo.write('src/a.py', A_PY)
    commit = repo.commit('one', when=T0)
    policy = _policy(repo.root, tmp_path, 'src/**', 'tests/**')

    tree = visible_tree(repo.root, policy, commit, previous=None, line_counts=None)

    assert tree.commit == commit
    assert tree.files == (
        TreeFile(
            path='src/a.py',
            kind=FileKind.PRODUCTION,
            lines=policy.counted_lines(A_PY),
            bytes=len(A_PY),
            last_changed_at=T0,
        ),
    )


def test_kinds_and_totals(repo: Repo, tmp_path: Path) -> None:
    repo.write('.gitattributes', 'src/gen.py linguist-generated\n')
    repo.write('src/a.py', 'a = 1\nb = 2\n')
    repo.write('tests/test_a.py', 't = 1\nu = 2\nv = 3\n')
    repo.write('src/README.md', '# readme\n\ntext\nmore\nlast\n')
    repo.write('src/gen.py', 'g = 1\n' * 7)
    repo.write('src/build/out.py', 'o = 1\n' * 11)
    commit = repo.commit('one', when=T0)
    policy = _policy(repo.root, tmp_path, 'src/**', 'tests/**', generated_globs=['src/build/**'])

    tree = visible_tree(repo.root, policy, commit, previous=None, line_counts=None)

    assert {file.path: (file.kind, file.lines) for file in tree.files} == {
        'src/a.py': (FileKind.PRODUCTION, 2),
        'tests/test_a.py': (FileKind.TEST, 3),
        'src/README.md': (FileKind.DOCS, 4),
        'src/gen.py': (FileKind.GENERATED, 7),
        'src/build/out.py': (FileKind.GENERATED, 11),
    }
    assert [file.path for file in tree.files] == sorted(file.path for file in tree.files)
    assert tree.totals == TreeTotals(files=5, production_lines=2, test_lines=3, docs_lines=4)


def test_symlinks_and_submodules_are_not_listed(repo: Repo, tmp_path: Path) -> None:
    repo.write('src/a.py', A_PY)
    repo.symlink('src/link', '/home/someone/secret')
    repo.gitlink('src/vendored', 'b' * 40)
    commit = repo.commit('one', when=T0)
    policy = _policy(repo.root, tmp_path, 'src/**')

    tree = visible_tree(repo.root, policy, commit, previous=None, line_counts=None)

    assert [file.path for file in tree.files] == ['src/a.py']


def test_lines_come_from_the_commit_not_the_working_tree(repo: Repo, tmp_path: Path) -> None:
    repo.write('src/a.py', 'a = 1\n')
    commit = repo.commit('one', when=T0)
    repo.write('src/a.py', 'a = 1\nb = 2\nc = 3\n')
    policy = _policy(repo.root, tmp_path, 'src/**')

    tree = visible_tree(repo.root, policy, commit, previous=None, line_counts=None)

    assert tree.files[0].lines == 1
    assert tree.files[0].bytes == len(b'a = 1\n')


def test_line_counts_are_cached_by_blob_and_pruned(repo: Repo, tmp_path: Path) -> None:
    repo.write('src/a.py', A_PY)
    repo.write('src/b.py', 'b = 1\n')
    commit = repo.commit('one', when=T0)
    policy = _policy(repo.root, tmp_path, 'src/**')
    a_blob = _blob(repo, commit, 'src/a.py')
    b_blob = _blob(repo, commit, 'src/b.py')
    cached = LineCounts(
        max_line_bytes=policy.max_line_bytes, by_blob={a_blob: 999, 'c' * 40: 5}
    )

    tree = visible_tree(repo.root, policy, commit, previous=None, line_counts=cached)

    assert {file.path: file.lines for file in tree.files} == {'src/a.py': 999, 'src/b.py': 1}
    assert tree.line_counts == LineCounts(
        max_line_bytes=policy.max_line_bytes, by_blob={a_blob: 999, b_blob: 1}
    )


def test_a_cache_for_another_line_width_is_ignored(repo: Repo, tmp_path: Path) -> None:
    repo.write('src/a.py', A_PY)
    commit = repo.commit('one', when=T0)
    policy = _policy(repo.root, tmp_path, 'src/**')
    a_blob = _blob(repo, commit, 'src/a.py')
    stale = LineCounts(max_line_bytes=policy.max_line_bytes + 1, by_blob={a_blob: 999})

    tree = visible_tree(repo.root, policy, commit, previous=None, line_counts=stale)

    assert tree.files[0].lines == policy.counted_lines(A_PY)
    assert tree.line_counts == LineCounts(
        max_line_bytes=policy.max_line_bytes, by_blob={a_blob: policy.counted_lines(A_PY)}
    )


def _history(repo: Repo) -> tuple[str, str]:
    """main: T0 adds a, b and tests/t; a task branch changes c, merged at T2; returns (base, merge)."""
    repo.write('src/a.py', 'a = 1\n')
    repo.write('src/b.py', 'b = 1\n')
    repo.write('tests/t.py', 't = 1\n')
    base = repo.commit('base', when=T0)
    repo.branch('task/7')
    repo.checkout('task/7')
    repo.write('src/c.py', 'c = 1\n')
    repo.commit('c on the branch', when=T1)
    repo.checkout('main')
    merge = repo.merge_no_ff('task/7', 'Merge task/7 into main', when=T2)
    return base, merge


def _dates(tree: VisibleTree) -> dict[str, datetime | None]:
    return {file.path: file.last_changed_at for file in tree.files}


def test_a_first_snapshot_dates_files_by_their_landing_on_main(
    repo: Repo, tmp_path: Path
) -> None:
    _, merge = _history(repo)
    policy = _policy(repo.root, tmp_path, 'src/**')

    tree = visible_tree(repo.root, policy, merge, previous=None, line_counts=None)

    assert _dates(tree) == {'src/a.py': T0, 'src/b.py': T0, 'src/c.py': T2}


def test_a_later_snapshot_redates_only_what_changed(repo: Repo, tmp_path: Path) -> None:
    _, merge = _history(repo)
    policy = _policy(repo.root, tmp_path, 'src/**')
    first = visible_tree(repo.root, policy, merge, previous=None, line_counts=None)
    repo.write('src/b.py', 'b = 2\n')
    later = repo.commit('b changes', when=T3)

    second = visible_tree(
        repo.root, policy, later, previous=_snapshot(first), line_counts=first.line_counts
    )

    assert _dates(second) == {'src/a.py': T0, 'src/b.py': T3, 'src/c.py': T2}


def test_unchanged_files_carry_the_previous_snapshots_date(repo: Repo, tmp_path: Path) -> None:
    _, merge = _history(repo)
    policy = _policy(repo.root, tmp_path, 'src/**')
    first = visible_tree(repo.root, policy, merge, previous=None, line_counts=None)
    carried = first.files[0].model_copy(update={'last_changed_at': SENTINEL})
    previous = _snapshot(first).model_copy(update={'files': (carried, *first.files[1:])})
    repo.write('src/b.py', 'b = 2\n')
    later = repo.commit('b changes', when=T3)

    second = visible_tree(repo.root, policy, later, previous=previous, line_counts=None)

    assert _dates(second) == {'src/a.py': SENTINEL, 'src/b.py': T3, 'src/c.py': T2}


def test_a_file_newly_visible_under_a_wider_policy_is_still_dated(
    repo: Repo, tmp_path: Path
) -> None:
    _, merge = _history(repo)
    narrow = _policy(repo.root, tmp_path, 'src/**')
    first = visible_tree(repo.root, narrow, merge, previous=None, line_counts=None)
    repo.write('src/b.py', 'b = 2\n')
    later = repo.commit('b changes', when=T3)
    wide = _policy(repo.root, tmp_path, 'src/**', 'tests/**')

    second = visible_tree(repo.root, wide, later, previous=_snapshot(first), line_counts=None)

    assert _dates(second) == {'src/a.py': T0, 'src/b.py': T3, 'src/c.py': T2, 'tests/t.py': T0}


def test_a_previous_snapshot_off_the_main_line_is_not_trusted(repo: Repo, tmp_path: Path) -> None:
    _, merge = _history(repo)
    policy = _policy(repo.root, tmp_path, 'src/**')
    first = visible_tree(repo.root, policy, merge, previous=None, line_counts=None)
    repo.branch('task/9')
    repo.checkout('task/9')
    repo.write('src/a.py', 'a = 9\n')
    side = repo.commit('side', when=T3)
    repo.checkout('main')
    stale = [file.model_copy(update={'last_changed_at': SENTINEL}) for file in first.files]
    previous = _snapshot(first).model_copy(update={'commit': side, 'files': tuple(stale)})

    second = visible_tree(repo.root, policy, merge, previous=previous, line_counts=None)

    assert _dates(second) == {'src/a.py': T0, 'src/b.py': T0, 'src/c.py': T2}


def test_this_checkouts_main_lists_only_the_source_and_test_trees(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    commit = main_head(root)
    assert commit is not None
    visible = ('*/src/**', '*/tests/**')
    policy = _policy(root, tmp_path, *visible)

    tree = visible_tree(root, policy, commit, previous=None, line_counts=None)

    files = {file.path: file for file in tree.files}
    assert all(matches_any(visible, path) for path in files)
    assert 'dark-factory-orchestrator.yaml' not in files
    assert '.mcp.json' not in files
    contract = files['client-feed/src/client_feed/contract.py']
    fold_tests = files['client-feed/tests/test_fold.py']
    assert (contract.kind, fold_tests.kind) == (FileKind.PRODUCTION, FileKind.TEST)
    assert contract.lines > 0
    assert fold_tests.lines > 0
