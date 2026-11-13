"""shared.quality_series.git_lines: path classes, module keys and landed lines from real git."""

from __future__ import annotations

from pathlib import Path

import pytest
from quality_series_fixtures import GitRepoBuilder

from shared.quality_series.git_lines import (
    GitObjectMissing,
    GitTimeout,
    GitUnavailable,
    LandedLines,
    ModuleLines,
    PathClass,
    landed_lines,
    module_key,
    path_class,
)
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)


@pytest.mark.parametrize(
    ('path', 'expected'),
    [
        ('uv.lock', PathClass.LOCK),
        ('dashboard/package-lock.json', PathClass.LOCK),
        ('Cargo.lock', PathClass.LOCK),
        ('plans/x.md', PathClass.DOC),
        ('docs/a.txt', PathClass.DOC),
        ('README.md', PathClass.DOC),
        ('data/x.json', PathClass.DATA),
        ('shared/tests/fixtures/a.json', PathClass.DATA),
        ('Makefile', PathClass.DATA),
        ('shared/tests/test_x.py', PathClass.TEST),
        ('orchestrator/conftest.py', PathClass.TEST),
        ('pkg/foo_test.py', PathClass.TEST),
        ('shared/src/shared/stats.py', PathClass.SRC),
        ('scripts/run.sh', PathClass.SRC),
    ],
)
def test_path_class(path: str, expected: PathClass) -> None:
    assert path_class(path) is expected


@pytest.mark.parametrize(
    ('path', 'expected'),
    [
        ('orchestrator/src/orchestrator/merge_lane/gates.py', 'orchestrator/merge_lane'),
        ('orchestrator/src/orchestrator/workflow.py', 'orchestrator/workflow.py'),
        ('fused-memory/src/fused_memory/middleware/x.py', 'fused-memory/middleware'),
        ('crates/reify-core/src/lib.rs', 'crates/reify-core'),
        ('scripts/x.py', 'scripts'),
        ('setup.py', 'setup.py'),
    ],
)
def test_module_key(path: str, expected: str) -> None:
    assert module_key(path) == expected


def test_module_key_rejects_an_empty_path() -> None:
    with pytest.raises(ValueError, match='non-empty path'):
        module_key('')


def _lines(n: int, tag: str) -> str:
    return ''.join(f'{tag} {i}\n' for i in range(n))


@pytest.fixture
def merged(tmp_path: Path) -> tuple[GitRepoBuilder, str, str]:
    repo = GitRepoBuilder(tmp_path / 'repo')
    base = repo.commit({'pkg/src/pkg/mod/b.py': _lines(5, 'old')}, 'root')
    kept = ''.join(f'old {i}\n' for i in range(3, 5))
    merge = repo.branch_merge('101', {
        'pkg/src/pkg/mod/a.py': _lines(10, 'src'),
        'pkg/src/pkg/mod/b.py': kept + _lines(2, 'new'),
        'pkg/tests/test_a.py': _lines(5, 'test'),
        'docs/guide.md': _lines(100, 'doc'),
        'uv.lock': _lines(40, 'lock'),
        'data/x.json': _lines(30, 'data'),
        'pkg/src/pkg/mod/blob.js': bytes(range(256)) * 4,
    })
    return repo, base, merge


class TestLandedLines:
    def test_counts_code_churn_and_module_src_lines(self, merged) -> None:
        repo, base, merge = merged
        src = 10 + 3 + 2
        assert landed_lines(repo.root, base, merge, timeout_s=60) == LandedLines(
            code_lines=src + 5,
            src_lines=src,
            test_lines=5,
            modules=(ModuleLines('pkg/mod', src),),
        )

    def test_modules_sort_by_src_lines_then_name(self, tmp_path: Path) -> None:
        repo = GitRepoBuilder(tmp_path / 'repo')
        base = repo.commit({'README.md': 'x\n'}, 'root')
        merge = repo.branch_merge('7', {
            'p/src/p/zeta/a.py': _lines(4, 'z'),
            'p/src/p/alpha/a.py': _lines(4, 'a'),
            'p/src/p/big.py': _lines(9, 'b'),
        })
        modules = landed_lines(repo.root, base, merge, timeout_s=60).modules
        assert modules == (
            ModuleLines('p/big.py', 9), ModuleLines('p/alpha', 4), ModuleLines('p/zeta', 4),
        )

    def test_a_module_touched_only_by_zero_churn_paths_is_not_a_module(
        self, tmp_path: Path
    ) -> None:
        repo = GitRepoBuilder(tmp_path / 'repo')
        base = repo.commit({'p/src/p/tool/run.sh': 'echo 1\n'}, 'root')
        merge = repo.branch_merge('8', {
            'p/src/p/assets/blob.js': bytes(range(256)) * 4,
            'p/src/p/core/a.py': _lines(3, 'a'),
        })
        (repo.root / 'p/src/p/tool/run.sh').chmod(0o755)
        mode_only = repo.commit({}, 'mode only')
        assert landed_lines(repo.root, base, merge, timeout_s=60).modules == (
            ModuleLines('p/core', 3),
        )
        assert landed_lines(repo.root, merge, mode_only, timeout_s=60) == LandedLines(
            code_lines=0, src_lines=0, test_lines=0, modules=(),
        )

    def test_an_unknown_sha_is_a_missing_object(self, merged) -> None:
        repo, base, _ = merged
        with pytest.raises(GitObjectMissing):
            landed_lines(repo.root, base, 'f' * 40, timeout_s=60)

    def test_a_tiny_timeout_is_a_git_timeout(self, merged) -> None:
        repo, base, merge = merged
        with pytest.raises(GitTimeout):
            landed_lines(repo.root, base, merge, timeout_s=1e-6)

    @pytest.mark.parametrize('timeout_s', [0, -1.0])
    def test_a_non_positive_timeout_is_refused(self, merged, timeout_s: float) -> None:
        repo, base, merge = merged
        with pytest.raises(ValueError):
            landed_lines(repo.root, base, merge, timeout_s=timeout_s)

    def test_a_directory_outside_any_repository_is_git_unavailable(self, tmp_path: Path) -> None:
        plain = tmp_path / 'plain'
        plain.mkdir()
        with pytest.raises(GitUnavailable):
            landed_lines(plain, 'a' * 40, 'b' * 40, timeout_s=60)
