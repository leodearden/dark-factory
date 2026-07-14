"""Tests for GitOps.ephemeral_worktree() — task θ (verify-plan PRD).

GitOps.ephemeral_worktree(kind, sha) is the single extraction point for the
two main-tip probes' (verify_failure_is_preexisting_on_main, run_main_tip_sweep
in orchestrator/src/orchestrator/verify.py) copy-pasted worktree lifecycle:
mint a path under worktree_base with a kind-specific prefix, retry
`git worktree add --detach` on lock contention, and GUARANTEE scoped cleanup
(`git worktree remove --force` + rmtree) with NO `git worktree prune` ever
issued (DD5 — the comment-only invariant that failed in the 2026-07-04
warm-lane broad-prune registration-wipe incident, df 2097-2100).

Test coverage:
  step-1: WorktreeKind enum + E2 PROTECTED_PREFIXES registration
  step-3: GitOps.ephemeral_worktree CM behavior (naming, retry, cleanup,
          raise-on-add-failure)
  step-5: E1 — both probes route through the CM and NEVER issue
          ['git', 'worktree', 'prune']
"""
from __future__ import annotations

from pathlib import Path

from orchestrator.config import GitConfig
from orchestrator.git_ops import PROTECTED_PREFIXES, GitOps

# ---------------------------------------------------------------------------
# step-1: WorktreeKind enum + E2 registration
# ---------------------------------------------------------------------------


class TestWorktreeKindRegistration:
    """step-1: WorktreeKind enum exists, its values are the probe/sweep
    prefixes, and both are registered in PROTECTED_PREFIXES /
    GitOps.protected_prefixes() (E2) with a non-empty owner.

    RED today: WorktreeKind does not exist yet, so every test here fails on
    the (deliberately in-test, not module-level) import.
    """

    def test_worktree_kind_import_and_values(self) -> None:
        from orchestrator.git_ops import WorktreeKind

        assert WorktreeKind.MAIN_PROBE.value == '_mainprobe-'
        assert WorktreeKind.MAIN_SWEEP.value == '_mainsweep-'

    def test_worktree_kind_values_registered_in_module_prefixes(self) -> None:
        from orchestrator.git_ops import WorktreeKind

        for kind in (WorktreeKind.MAIN_PROBE, WorktreeKind.MAIN_SWEEP):
            assert kind.value in PROTECTED_PREFIXES, (
                f'expected {kind.value!r} to be a key in PROTECTED_PREFIXES; '
                f'got keys={list(PROTECTED_PREFIXES)!r}'
            )
            owner = PROTECTED_PREFIXES[kind.value]
            assert isinstance(owner, str) and owner, (
                f'expected a non-empty owner tag for {kind.value!r}; got {owner!r}'
            )

    def test_worktree_kind_values_registered_in_instance_protected_prefixes(
        self, tmp_path: Path,
    ) -> None:
        from orchestrator.git_ops import WorktreeKind

        git_ops = GitOps(GitConfig(), tmp_path)
        registry = git_ops.protected_prefixes()

        for kind in (WorktreeKind.MAIN_PROBE, WorktreeKind.MAIN_SWEEP):
            assert kind.value in registry, (
                f'expected {kind.value!r} in protected_prefixes(); '
                f'got keys={list(registry)!r}'
            )
            assert registry[kind.value] == PROTECTED_PREFIXES[kind.value], (
                f'expected protected_prefixes() owner for {kind.value!r} to '
                f'match the module registry owner; got '
                f'{registry[kind.value]!r} != {PROTECTED_PREFIXES[kind.value]!r}'
            )
