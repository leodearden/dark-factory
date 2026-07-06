"""Tests for the PROTECTED_PREFIXES band registry and foreign-band refusal
guard (gitops-chokepoints ε, task 2205).

Makes ephemeral-worktree band ownership explicit and machine-checked so a
destructive cleanup sweep can never remove a ``worktree_base`` child band it
does not own.  Covers the shared primitive (``GitOps.protected_prefixes`` /
``GitOps._refuse_foreign_band``) exhaustively, then the two sweep
integrations and the point-site wiring (self-heal rmtrees + harness
substrate-gate pre-clean) that consult it.

Fixtures mirror test_interactive_worktree_reaper.py's real-temp-git-repo
pattern (``_init_repo`` via ``orchestrator.git_ops._run``,
``GitOps(GitConfig(), repo)``).
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import (
    PERSISTENT_MERGE_WORKTREE_NAME,
    PERSISTENT_OFFLINE_DEEP_WORKTREE_NAME,
    PROTECTED_PREFIXES,
    GitOps,
    _run,
)

# ---------------------------------------------------------------------------
# Repo fixture + helpers (mirrors test_interactive_worktree_reaper.py)
# ---------------------------------------------------------------------------


async def _init_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def pp_git_repo(tmp_path: Path) -> Path:
    """Temporary git repo with an initial commit for protected-prefixes tests."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    return repo


# ---------------------------------------------------------------------------
# TestProtectedPrefixesRegistry
# ---------------------------------------------------------------------------


class TestProtectedPrefixesRegistry:
    """RED — module-level PROTECTED_PREFIXES and GitOps.protected_prefixes()."""

    @pytest.mark.parametrize(
        'band',
        ['_lane-', '_spec-', '_merge-', '_solo-', '_substrate-gate-'],
    )
    def test_static_prefix_bands_have_non_empty_owner(self, band: str) -> None:
        assert band in PROTECTED_PREFIXES, (
            f'expected {band!r} to be a key in PROTECTED_PREFIXES; '
            f'got keys={list(PROTECTED_PREFIXES)!r}'
        )
        owner = PROTECTED_PREFIXES[band]
        assert isinstance(owner, str) and owner, (
            f'expected a non-empty owner tag for {band!r}; got {owner!r}'
        )

    @pytest.mark.parametrize(
        'exact_name',
        [PERSISTENT_MERGE_WORKTREE_NAME, PERSISTENT_OFFLINE_DEEP_WORKTREE_NAME],
    )
    def test_exact_persistent_names_have_non_empty_owner(self, exact_name: str) -> None:
        assert exact_name in PROTECTED_PREFIXES, (
            f'expected {exact_name!r} to be a key in PROTECTED_PREFIXES; '
            f'got keys={list(PROTECTED_PREFIXES)!r}'
        )
        owner = PROTECTED_PREFIXES[exact_name]
        assert isinstance(owner, str) and owner, (
            f'expected a non-empty owner tag for {exact_name!r}; got {owner!r}'
        )

    def test_exact_name_keys_match_persistent_name_constants(self) -> None:
        # The registry keys for the two persistent worktrees must be the
        # constants themselves (not independently-typed literals), so the
        # registry cannot silently drift from the canonical names.
        assert '_merge-verify' == PERSISTENT_MERGE_WORKTREE_NAME
        assert '_offline-deep' == PERSISTENT_OFFLINE_DEEP_WORKTREE_NAME
        assert PERSISTENT_MERGE_WORKTREE_NAME in PROTECTED_PREFIXES
        assert PERSISTENT_OFFLINE_DEEP_WORKTREE_NAME in PROTECTED_PREFIXES

    def test_protected_prefixes_method_is_superset_with_default_iact_prefix(
        self, pp_git_repo: Path,
    ) -> None:
        config = GitConfig()
        git_ops = GitOps(config, pp_git_repo)

        registry = git_ops.protected_prefixes()

        for key, owner in PROTECTED_PREFIXES.items():
            assert registry.get(key) == owner, (
                f'expected instance registry to be a superset of the module '
                f'registry for key {key!r}; got {registry.get(key)!r}'
            )
        assert config.iact_prefix == '_iact-', (
            'sanity: default iact_prefix expected to be "_iact-"'
        )
        assert '_iact-' in registry, (
            f'expected the default iact_prefix key in protected_prefixes(); '
            f'got keys={list(registry)!r}'
        )
        assert registry['_iact-'], 'expected a non-empty owner tag for _iact-'

    def test_protected_prefixes_method_respects_custom_iact_prefix(
        self, pp_git_repo: Path,
    ) -> None:
        config = GitConfig(iact_prefix='_custom-')
        git_ops = GitOps(config, pp_git_repo)

        registry = git_ops.protected_prefixes()

        for key, owner in PROTECTED_PREFIXES.items():
            assert registry.get(key) == owner, (
                f'expected instance registry to be a superset of the module '
                f'registry for key {key!r}; got {registry.get(key)!r}'
            )
        assert '_custom-' in registry, (
            f'expected the custom iact_prefix key in protected_prefixes(); '
            f'got keys={list(registry)!r}'
        )
        assert registry['_custom-'], 'expected a non-empty owner tag for _custom-'
        assert '_iact-' not in registry, (
            'the default iact_prefix key must NOT leak in when a custom '
            'prefix is configured'
        )
