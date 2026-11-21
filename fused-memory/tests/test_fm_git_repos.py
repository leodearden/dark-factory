"""Tests for the real-git repo builders in _fm_git_repos.py."""

import subprocess

import pytest
from _fm_git_repos import (
    _init_git_repo,
    git_stdout,
    init_ab_commit_repo,
    init_merge_history_repo,
)

# ---------------------------------------------------------------------------
# Tests for the shared real-git repo builders (task 4847)
# ---------------------------------------------------------------------------
# Real git repos in tmp_path, read back through git itself, so the commit graph
# test_stages.py's done-provenance tests rely on is pinned by observation.

class TestGitStdout:
    """Unit tests for git_stdout(path, *args)."""

    def test_returns_stripped_stdout(self, tmp_path):
        git_stdout(tmp_path, 'init', '-q')

        assert git_stdout(tmp_path, 'rev-parse', '--is-inside-work-tree') == 'true'

    def test_nonzero_exit_raises_called_process_error(self, tmp_path):
        git_stdout(tmp_path, 'init', '-q')

        with pytest.raises(subprocess.CalledProcessError):
            git_stdout(tmp_path, 'rev-parse', '--verify', 'no-such-ref')


def _hostile_global_git_config(tmp_path, monkeypatch) -> None:
    """Point git's global config at one that forces signing through a failing
    gpg program and routes every hook a builder's commands can trigger to
    ``exit 1``, including the ones ``--no-verify`` does not bypass."""
    hooks = tmp_path / 'hostile-hooks'
    hooks.mkdir()
    for hook in (
        'pre-commit', 'commit-msg', 'pre-merge-commit', 'prepare-commit-msg', 'post-checkout',
    ):
        script = hooks / hook
        script.write_text('#!/bin/sh\nexit 1\n')
        script.chmod(0o755)
    config = tmp_path / 'hostile-gitconfig'
    config.write_text(
        f'[commit]\n\tgpgsign = true\n[gpg]\n\tprogram = false\n[core]\n\thooksPath = {hooks}\n'
    )
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', str(config))


class TestRepoBuildersIgnoreDeveloperGitConfig:
    """Every builder survives a developer's global gpgsign and commit hooks."""

    @pytest.mark.parametrize(
        'builder', [_init_git_repo, init_ab_commit_repo, init_merge_history_repo],
    )
    def test_builder_succeeds_under_global_gpgsign_and_failing_hooks(
        self, tmp_path, monkeypatch, builder,
    ):
        _hostile_global_git_config(tmp_path, monkeypatch)
        repo = tmp_path / 'repo'

        builder(repo)

        assert git_stdout(repo, 'config', '--get', 'commit.gpgsign') == 'false'
        assert git_stdout(repo, 'status', '--porcelain') == ''


class TestProvenanceRepoBuilders:
    """Unit tests for init_ab_commit_repo(path) and init_merge_history_repo(path)."""

    def test_ab_commit_repo_is_one_root_commit_on_main_adding_a_and_b(self, tmp_path):
        from _fm_git_repos import init_ab_commit_repo

        sha = init_ab_commit_repo(tmp_path)

        assert sha == git_stdout(tmp_path, 'rev-parse', 'main')
        assert git_stdout(tmp_path, 'rev-list', '--count', 'main') == '1'
        assert git_stdout(tmp_path, 'ls-tree', '-r', '--name-only', sha).splitlines() == [
            'a.txt',
            'b.txt',
        ]
        assert git_stdout(tmp_path, 'config', '--get', 'commit.gpgsign') == 'false'

    def test_merge_history_single_is_the_ab_commit_prefix(self, tmp_path):
        from _fm_git_repos import init_ab_commit_repo, init_merge_history_repo

        merge_repo = tmp_path / 'm'
        prefix_repo = tmp_path / 'p'
        shas = init_merge_history_repo(merge_repo)
        prefix_sha = init_ab_commit_repo(prefix_repo)

        assert set(shas) == {'single', 'merge', 'conflict_merge'}
        single = shas['single']
        assert len(git_stdout(merge_repo, 'rev-list', '--parents', '-n', '1', single).split()) == 1
        assert git_stdout(merge_repo, 'rev-parse', f'{single}^{{tree}}') == git_stdout(
            prefix_repo, 'rev-parse', f'{prefix_sha}^{{tree}}'
        )

    def test_merge_history_merges_report_full_first_parent_file_lists(self, tmp_path):
        from _fm_git_repos import init_merge_history_repo

        shas = init_merge_history_repo(tmp_path)
        merge, conflict_merge = shas['merge'], shas['conflict_merge']

        for sha in (merge, conflict_merge):
            parents = git_stdout(tmp_path, 'rev-list', '--parents', '-n', '1', sha).split()[1:]
            assert len(parents) == 2, f'{sha} must be a two-parent merge, got {parents}'
        assert git_stdout(tmp_path, 'rev-parse', f'{merge}^1') == shas['single']
        merge_files = set(
            git_stdout(tmp_path, 'diff', '--name-only', f'{merge}^1', merge).splitlines()
        )
        assert merge_files == {'feature.py', *(f'extra_{i}.py' for i in range(1, 5))}
        conflict_files = set(
            git_stdout(
                tmp_path, 'diff', '--name-only', f'{conflict_merge}^1', conflict_merge
            ).splitlines()
        )
        assert {'a.txt', 'conflict_only.py'} <= conflict_files
        assert git_stdout(tmp_path, 'rev-parse', 'main') == conflict_merge
