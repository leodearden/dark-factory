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
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import (
    PERSISTENT_MERGE_WORKTREE_NAME,
    PERSISTENT_OFFLINE_DEEP_WORKTREE_NAME,
    PROTECTED_PREFIXES,
    GitOps,
    ReapedInteractiveWorktree,
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


# ---------------------------------------------------------------------------
# TestRefuseForeignBand — PREFIX-band matching only (exact-name matching is
# covered separately in TestRefuseForeignBandExactAndConfig once step-6 lands
# the exact-name precedence layer).
# ---------------------------------------------------------------------------


class TestRefuseForeignBand:
    """RED — GitOps._refuse_foreign_band: PREFIX-band matching (step-3/4)."""

    def test_foreign_prefix_band_refused_with_warning(
        self, pp_git_repo: Path, caplog,
    ) -> None:
        """(a-core) A `_iact-<x>` dir, owned={'_merge-'} -> refused (True) +
        WARNING naming the matched band token and its owner."""
        config = GitConfig()
        git_ops = GitOps(config, pp_git_repo)
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
        path = git_ops.worktree_base / '_iact-someone'
        path.mkdir()

        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
            result = git_ops._refuse_foreign_band(
                path, frozenset({'_merge-'}), 'test-context',
            )

        assert result is True, 'expected a foreign band to be refused (True)'
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings, 'expected a WARNING to be logged for the foreign-band refusal'
        joined = '\n'.join(r.getMessage() for r in warnings)
        assert '_iact-' in joined, (
            f'expected the WARNING to name the matched band token (_iact-); got: {joined!r}'
        )
        assert 'interactive' in joined, (
            f'expected the WARNING to name the owner (interactive); got: {joined!r}'
        )

    def test_owned_prefix_band_not_refused_no_warning(
        self, pp_git_repo: Path, caplog,
    ) -> None:
        """(b) A `_merge-<id>` dir, owned={'_merge-'} -> proceeds (False), no WARNING."""
        config = GitConfig()
        git_ops = GitOps(config, pp_git_repo)
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
        path = git_ops.worktree_base / '_merge-abc123'
        path.mkdir()

        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
            result = git_ops._refuse_foreign_band(
                path, frozenset({'_merge-'}), 'test-context',
            )

        assert result is False, 'expected an owned band to proceed (False)'
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not warnings, f'expected NO warning for an owned band; got {warnings!r}'

    def test_non_band_name_not_refused(self, pp_git_repo: Path) -> None:
        """(c) A plain task-id dir (non-band) -> proceeds (False) regardless of owned."""
        config = GitConfig()
        git_ops = GitOps(config, pp_git_repo)
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
        path = git_ops.worktree_base / '123'
        path.mkdir()

        result = git_ops._refuse_foreign_band(path, frozenset(), 'test-context')

        assert result is False, 'expected a non-band name to fail-open (False)'

    def test_non_direct_child_not_refused_even_if_name_matches(
        self, pp_git_repo: Path,
    ) -> None:
        """(d) A nested path whose NAME matches a prefix but whose PARENT is
        NOT worktree_base -> proceeds (False), fail-open."""
        config = GitConfig()
        git_ops = GitOps(config, pp_git_repo)
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
        nested = git_ops.worktree_base / '123' / '_merge-nested'
        nested.mkdir(parents=True)

        result = git_ops._refuse_foreign_band(nested, frozenset(), 'test-context')

        assert result is False, (
            'expected a non-direct-child path to fail-open even though its '
            'name matches a protected prefix'
        )


# ---------------------------------------------------------------------------
# TestRefuseForeignBandExactAndConfig — EXACT-name matching (precedence over
# prefix matching) + config-driven iact_prefix respected at the refusal
# level (step-5/6).
# ---------------------------------------------------------------------------


class TestRefuseForeignBandExactAndConfig:
    """RED (against the step-4 prefix-only impl) — exact-name precedence."""

    def test_exact_persistent_name_with_no_matching_prefix_is_refused(
        self, pp_git_repo: Path, caplog,
    ) -> None:
        """(d) `_offline-deep` matches no prefix key, so a prefix-only impl
        wrongly fails open. Exact-name matching must refuse it + WARNING."""
        config = GitConfig()
        git_ops = GitOps(config, pp_git_repo)
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
        path = git_ops.worktree_base / PERSISTENT_OFFLINE_DEEP_WORKTREE_NAME
        path.mkdir()

        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
            result = git_ops._refuse_foreign_band(
                path, frozenset({'_merge-'}), 'test-context',
            )

        assert result is True, (
            'expected the exact persistent name _offline-deep to be refused '
            '(True) even though it matches no prefix key'
        )
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings, 'expected a WARNING to be logged for the foreign-band refusal'
        joined = '\n'.join(r.getMessage() for r in warnings)
        assert PERSISTENT_OFFLINE_DEEP_WORKTREE_NAME in joined, (
            f'expected the WARNING to name the matched exact token '
            f'({PERSISTENT_OFFLINE_DEEP_WORKTREE_NAME!r}); got: {joined!r}'
        )
        assert 'persistent-offline-deep' in joined, (
            f'expected the WARNING to name the owner (persistent-offline-deep); '
            f'got: {joined!r}'
        )

    def test_exact_name_wins_over_overlapping_prefix_owner(
        self, pp_git_repo: Path, caplog,
    ) -> None:
        """(d) `_merge-verify` matches BOTH the `_merge-` prefix and its own
        exact name. The exact name must win, so a `_merge-` owner still
        refuses the persistent merge-verify worktree."""
        config = GitConfig()
        git_ops = GitOps(config, pp_git_repo)
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
        path = git_ops.worktree_base / PERSISTENT_MERGE_WORKTREE_NAME
        path.mkdir()

        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
            result = git_ops._refuse_foreign_band(
                path, frozenset({'_merge-'}), 'test-context',
            )

        assert result is True, (
            'expected _merge-verify to be refused (True) to a plain '
            '`_merge-` owner — the exact name must win over the prefix'
        )
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings, 'expected a WARNING to be logged for the foreign-band refusal'
        joined = '\n'.join(r.getMessage() for r in warnings)
        assert 'persistent-merge-verify' in joined, (
            f'expected the WARNING to name the owner (persistent-merge-verify); '
            f'got: {joined!r}'
        )

    def test_exact_name_not_refused_when_owned_by_its_own_token(
        self, pp_git_repo: Path, caplog,
    ) -> None:
        """A `_merge-verify` dir with owned={'_merge-verify'} (the persistent
        owner itself) proceeds (False), no WARNING."""
        config = GitConfig()
        git_ops = GitOps(config, pp_git_repo)
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
        path = git_ops.worktree_base / PERSISTENT_MERGE_WORKTREE_NAME
        path.mkdir()

        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
            result = git_ops._refuse_foreign_band(
                path, frozenset({PERSISTENT_MERGE_WORKTREE_NAME}), 'test-context',
            )

        assert result is False, (
            'expected the persistent owner itself to proceed (False)'
        )
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not warnings, f'expected NO warning for an owned exact band; got {warnings!r}'

    def test_custom_iact_prefix_foreign_refused_with_warning(
        self, pp_git_repo: Path, caplog,
    ) -> None:
        """(e) With a custom iact_prefix, a `_custom-<x>` dir is refused to a
        non-owning sweep + WARNING (config-driven band respected)."""
        config = GitConfig(iact_prefix='_custom-')
        git_ops = GitOps(config, pp_git_repo)
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
        path = git_ops.worktree_base / '_custom-someone'
        path.mkdir()

        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
            result = git_ops._refuse_foreign_band(
                path, frozenset({'_merge-'}), 'test-context',
            )

        assert result is True, (
            'expected the custom iact band to be refused (True) to a '
            'non-owning sweep'
        )
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings, 'expected a WARNING to be logged for the foreign-band refusal'
        joined = '\n'.join(r.getMessage() for r in warnings)
        assert '_custom-' in joined, (
            f'expected the WARNING to name the matched band token (_custom-); '
            f'got: {joined!r}'
        )

    def test_custom_iact_prefix_owned_not_refused(
        self, pp_git_repo: Path, caplog,
    ) -> None:
        """(e) With a custom iact_prefix, a `_custom-<x>` dir owned by the
        matching band proceeds (False), no WARNING."""
        config = GitConfig(iact_prefix='_custom-')
        git_ops = GitOps(config, pp_git_repo)
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
        path = git_ops.worktree_base / '_custom-someone'
        path.mkdir()

        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
            result = git_ops._refuse_foreign_band(
                path, frozenset({'_custom-'}), 'test-context',
            )

        assert result is False, (
            'expected the custom iact band to proceed (False) when owned'
        )
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not warnings, f'expected NO warning for an owned custom band; got {warnings!r}'


# ---------------------------------------------------------------------------
# TestPruneStaleMergeForeignBandRefusal — sweep integration (step-7/8).
#
# The foreign candidate must be a REAL registered git worktree, not a bare
# directory: `git worktree remove --force` on a path git does not recognize
# as a worktree fails closed on its own (exit 128, "is not a working tree")
# without touching the directory — so a bare foreign directory would survive
# the sweep even with no guard wired at all, making it a false RED. Using a
# real (but foreign-band-named) registered worktree means the pre-guard
# `git worktree remove --force` call actually succeeds and deletes it,
# reproducing the filter-bug hazard honestly: RED (foreign genuinely removed)
# before step-8, GREEN (guard `continue`s before the removal call) after.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPruneStaleMergeForeignBandRefusal:
    """RED — prune_stale_merge_worktrees consults _refuse_foreign_band (step-7/8)."""

    async def test_foreign_band_survives_filter_bug_owned_still_removed(
        self, pp_git_repo: Path, monkeypatch, caplog,
    ) -> None:
        config = GitConfig()
        git_ops = GitOps(config, pp_git_repo)

        # Owned candidate: a real registered `_merge-<id>` worktree.
        merge_wt, _ = await git_ops._create_merge_worktree()
        assert merge_wt.exists()

        # Foreign candidate: a real registered worktree under a FOREIGN band
        # name (`_iact-`), simulating a filter bug that steers a foreign
        # band's worktree into the merge sweep's removal step.
        foreign_wt = git_ops.worktree_base / '_iact-foreign'
        rc, _, err = await _run(
            ['git', 'worktree', 'add', '--detach', str(foreign_wt), config.main_branch],
            cwd=pp_git_repo,
        )
        assert rc == 0, f'failed to set up foreign worktree fixture: {err}'
        assert foreign_wt.exists()

        async def _fake_iter_merge_worktrees():
            yield merge_wt, merge_wt.resolve()
            yield foreign_wt, foreign_wt.resolve()

        monkeypatch.setattr(git_ops, '_iter_merge_worktrees', _fake_iter_merge_worktrees)

        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
            removed = await git_ops.prune_stale_merge_worktrees()

        # Case (a): foreign band survives, is not reported removed, and a
        # WARNING names its band.
        assert foreign_wt.exists(), (
            'expected the foreign _iact- worktree to survive the sweep '
            '(refused), but it was removed from disk'
        )
        assert str(foreign_wt) not in removed, (
            f'expected the foreign worktree NOT to appear in the removed list: {removed}'
        )
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        joined = '\n'.join(r.getMessage() for r in warnings)
        assert '_iact-' in joined, (
            f'expected a WARNING naming the foreign band token (_iact-); got: {joined!r}'
        )

        # Case (b): owned band still proceeds — removed from disk and
        # reported in the removed list.
        assert str(merge_wt) in removed, (
            f'expected the owned _merge- worktree to be removed; got: {removed}'
        )
        assert not merge_wt.exists(), 'expected the owned worktree to be gone from disk'


# ---------------------------------------------------------------------------
# TestReapInteractiveForeignBandRefusal — sweep integration (step-9/10).
#
# Same false-RED pitfall as the prune sweep applies here: the foreign
# candidate must be a REAL registered git worktree so the pre-guard
# `git worktree remove --force` call actually succeeds and removes it
# (reap_interactive_worktrees additionally falls through its
# missing-`.task/interactive.json`-stamp branch to `reason='stale_no_stamp'`
# for such a foreign worktree, so it reaches the removal call today).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReapInteractiveForeignBandRefusal:
    """RED — reap_interactive_worktrees consults _refuse_foreign_band (step-9/10)."""

    async def test_foreign_band_survives_filter_bug_owned_still_reaped(
        self, pp_git_repo: Path, monkeypatch, caplog,
    ) -> None:
        config = GitConfig()
        git_ops = GitOps(config, pp_git_repo)

        # Owned candidate: a real, reapable `_iact-<slug>` worktree —
        # backdated stamp so it is TTL-idle at `now`.
        info = await git_ops.create_interactive_worktree('someslug')
        now = datetime.now(UTC)
        stamp_path = info.path / '.task' / 'interactive.json'
        stamp = json.loads(stamp_path.read_text())
        stamp['created_at'] = (
            now - timedelta(seconds=config.interactive_worktree_ttl + 3600)
        ).isoformat()
        stamp_path.write_text(json.dumps(stamp))

        # Foreign candidate: a real registered worktree under a FOREIGN band
        # name (`_lane-`), simulating a filter bug that steers a foreign
        # band's worktree into the reap sweep's removal step.
        foreign_wt = git_ops.worktree_base / '_lane-foreign'
        rc, _, err = await _run(
            ['git', 'worktree', 'add', '--detach', str(foreign_wt), config.main_branch],
            cwd=pp_git_repo,
        )
        assert rc == 0, f'failed to set up foreign worktree fixture: {err}'
        assert foreign_wt.exists()

        async def _fake_iter_interactive_worktrees():
            yield info.path, info.path.resolve()
            yield foreign_wt, foreign_wt.resolve()

        monkeypatch.setattr(
            git_ops, '_iter_interactive_worktrees', _fake_iter_interactive_worktrees,
        )

        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
            reaped = await git_ops.reap_interactive_worktrees(now=now)

        # Case (a): foreign band survives, is not reported reaped, and a
        # WARNING from _refuse_foreign_band (not the unrelated "unreadable
        # stamp" warning, which also happens to mention the path) names its
        # band.
        assert foreign_wt.exists(), (
            'expected the foreign _lane- worktree to survive the sweep '
            '(refused), but it was removed from disk'
        )
        assert not any(r.path == foreign_wt for r in reaped), (
            f'expected the foreign worktree NOT to appear in the reaped list: {reaped!r}'
        )
        assert all(isinstance(r, ReapedInteractiveWorktree) for r in reaped)
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            'belongs to protected band' in r.getMessage() and '_lane-' in r.getMessage()
            for r in warnings
        ), (
            f'expected a _refuse_foreign_band WARNING naming the foreign '
            f'band token (_lane-); got: {[r.getMessage() for r in warnings]!r}'
        )

        # Case (b): owned band still proceeds — reaped and removed from disk.
        assert any(r.path == info.path for r in reaped), (
            f'expected the owned _iact- worktree to be reaped; got: {reaped!r}'
        )
        assert not info.path.exists(), 'expected the owned worktree to be gone from disk'
