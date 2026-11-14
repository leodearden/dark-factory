"""The task-mu fork's flake gate (flake-ledger PRD task delta, boundary rows B4/B5).

``verify.py::verify_failure_is_preexisting_on_main`` answers "wholly
preexisting" when every failing id of a merge is also red in main's baseline.
A load flake can starve the same test on both sides, so before that verdict
stands the named failures are re-run in isolation, in a worktree pinned at the
main SHA the verdict is about. These tests drive the real discriminator; only
the baseline, the worktree and the per-module verification are doubled.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator import verify as verify_module
from orchestrator.config import GitConfig, ModuleConfig, OrchestratorConfig
from orchestrator.flake_ledger import FlakeCallSite, FlakeSuppression, FlakeVerdict
from orchestrator.git_ops import EphemeralWorktreeError, WorktreeKind
from orchestrator.verify import VerifyResult

pytestmark = pytest.mark.asyncio

MAIN_SHA = 'a1' * 20
OTHER_MAIN_SHA = 'b2' * 20
MODULE = ModuleConfig(prefix='orchestrator', test_command='pytest tests')
TEST_FILE = 'orchestrator/tests/test_x.py'
NODE_A = f'{TEST_FILE}::test_a'
NODE_B = f'{TEST_FILE}::test_b'
JUNIT_A = 'tests.test_x::test_a'
JUNIT_B = 'tests.test_x::test_b'
JUNIT_C = 'tests.test_x::test_c'


def _failing_result(
    *node_ids: str,
    junit_ids: tuple[str, ...] = (JUNIT_A, JUNIT_B),
    lint_output: str = '',
) -> VerifyResult:
    summary = ''.join(f'FAILED {node_id} - AssertionError\n' for node_id in node_ids)
    return VerifyResult(
        passed=False,
        test_output=f'{summary}{len(node_ids)} failed, 900 passed in 60.00s\n',
        lint_output=lint_output,
        type_output='',
        summary='Failures: tests failed',
        category='test_failure',
        failing_test_ids=list(junit_ids),
        failing_leg_categories=['test_failure'],
    )


def _rerun_result(*, passed: bool) -> VerifyResult:
    return VerifyResult(
        passed=passed,
        test_output='' if passed else f'FAILED {NODE_A} - AssertionError\n1 failed\n',
        lint_output='',
        type_output='',
        summary='isolated re-run',
        category='' if passed else 'test_failure',
    )


class _Harness:
    """A recording MAIN_PROBE worktree, a stubbed main baseline and a stubbed
    per-module re-run that records which tree it ran in."""

    def __init__(self, tmp_path: Path) -> None:
        self.config = OrchestratorConfig(
            project_root=tmp_path,
            max_concurrent_tasks=1,
            git=GitConfig(
                main_branch='main', branch_prefix='task/', remote='origin',
                worktree_dir='.worktrees',
            ),
        )
        self.task_worktree = self._tree(tmp_path / 'task-wt')
        self.probe_worktree = self._tree(tmp_path / 'main-probe')
        self.baseline: frozenset[str] = frozenset({JUNIT_A, JUNIT_B, JUNIT_C})
        self.rerun_passes = True
        self.worktree_raises = False
        self.worktrees: list[tuple[WorktreeKind, str, bool]] = []
        self.rerun_trees: list[Path] = []
        self.observed: list[tuple[FlakeSuppression, str]] = []
        self.main_sha = MAIN_SHA
        self.git_ops = MagicMock()
        self.git_ops.ephemeral_worktree = self._ephemeral_worktree
        self.git_ops.get_main_sha = AsyncMock(side_effect=lambda: self.main_sha)

    @staticmethod
    def _tree(root: Path) -> Path:
        (root / TEST_FILE).parent.mkdir(parents=True)
        (root / TEST_FILE).write_text('def test_a(): pass\n')
        return root

    @contextlib.asynccontextmanager
    async def _ephemeral_worktree(self, kind, sha, *, warm_seed=False):
        self.worktrees.append((kind, sha, warm_seed))
        if self.worktree_raises:
            raise EphemeralWorktreeError('worktree add failed on every attempt')
        yield self.probe_worktree

    async def main_baseline(self, config, module_configs, git_ops, main_sha, **kwargs):
        return self.baseline

    async def run_verification(self, worktree, config, module_config=None, **kwargs):
        self.rerun_trees.append(worktree)
        return _rerun_result(passed=self.rerun_passes)

    async def legacy_probe(self, *args, **kwargs):
        raise AssertionError('the signature-comparison probe must not run on the mu fork')

    async def sink(self, observation: FlakeSuppression, main_sha: str) -> None:
        self.observed.append((observation, main_sha))

    async def classify(self, failing_result: VerifyResult, *, sink=None) -> tuple[bool, str]:
        return await verify_module.verify_failure_is_preexisting_on_main(
            self.task_worktree, self.config, [MODULE], None, failing_result, self.git_ops,
            flake_sink=self.sink if sink is None else sink,
        )


@pytest.fixture
def harness(tmp_path: Path):
    h = _Harness(tmp_path)
    with (
        patch.object(verify_module, 'main_baseline_failing_ids', side_effect=h.main_baseline),
        patch.object(verify_module, 'run_verification', side_effect=h.run_verification),
        patch.object(verify_module, 'run_scoped_verification', side_effect=h.legacy_probe),
    ):
        yield h


class TestTheGateRunsBeforeTheWhollyPreexistingVerdict:
    """B4: a load-flaked wholly-preexisting verdict is re-examined, on main, and downgraded."""

    async def test_b4_discriminator_runs_before_the_preexisting_return_and_downgrades(
        self, harness: _Harness,
    ) -> None:
        verdict = await harness.classify(_failing_result(NODE_A, NODE_B))

        assert verdict == (False, '')
        assert harness.worktrees == [(WorktreeKind.MAIN_PROBE, MAIN_SHA, True)]
        assert harness.rerun_trees and set(harness.rerun_trees) == {harness.probe_worktree}
        [(observation, recorded_sha)] = harness.observed
        assert observation.verdict is FlakeVerdict.passes_in_isolation
        assert observation.call_site == FlakeCallSite.main_probe
        assert set(observation.test_ids) == {NODE_A, NODE_B}
        assert recorded_sha == MAIN_SHA

    async def test_a_raising_sink_never_changes_the_verdict(self, harness: _Harness) -> None:
        async def broken_sink(observation: FlakeSuppression, main_sha: str) -> None:
            raise RuntimeError('ledger unwritable')

        verdict = await harness.classify(_failing_result(NODE_A, NODE_B), sink=broken_sink)

        assert verdict == (False, '')

    async def test_a_passing_rerun_is_never_cached(self, harness: _Harness) -> None:
        await harness.classify(_failing_result(NODE_A, NODE_B))
        await harness.classify(_failing_result(NODE_A, NODE_B))

        assert len(harness.worktrees) == 2
        assert len(harness.observed) == 2


class TestAConfirmedRedKeepsTheVerdict:
    """B5: still red in isolation on main -> today's verdict, byte-identical, no row."""

    async def test_b5_failing_rerun_returns_preexisting_unchanged_and_writes_no_row(
        self, harness: _Harness,
    ) -> None:
        harness.rerun_passes = False

        verdict = await harness.classify(_failing_result(NODE_A, NODE_B))

        assert verdict == (True, MAIN_SHA)
        assert harness.observed == []
        assert len(harness.worktrees) == 1

    async def test_b5_confirmed_red_is_cached_per_main_sha_and_ids(
        self, harness: _Harness,
    ) -> None:
        harness.rerun_passes = False
        same = _failing_result(NODE_A, NODE_B)

        assert await harness.classify(same) == (True, MAIN_SHA)
        assert await harness.classify(same) == (True, MAIN_SHA)
        assert len(harness.worktrees) == 1

        other_ids = _failing_result(NODE_A, NODE_B, f'{TEST_FILE}::test_c',
                                    junit_ids=(JUNIT_A, JUNIT_B, JUNIT_C))
        assert await harness.classify(other_ids) == (True, MAIN_SHA)
        assert len(harness.worktrees) == 2

        harness.main_sha = OTHER_MAIN_SHA
        assert await harness.classify(same) == (True, OTHER_MAIN_SHA)
        assert [sha for _, sha, _ in harness.worktrees] == [MAIN_SHA, MAIN_SHA, OTHER_MAIN_SHA]


class TestEveryOtherOutcomeKeepsTheVerdict:
    """The gate is a one-way ratchet: only a positive isolated pass changes anything."""

    @pytest.mark.parametrize('shape', ['unmapped_node_ids', 'other_leg_failed', 'worktree_add_fails'])
    async def test_unconfirmable_keeps_the_verdict_and_records_nothing(
        self, harness: _Harness, shape: str,
    ) -> None:
        failing = _failing_result(NODE_A, NODE_B)
        if shape == 'unmapped_node_ids':
            failing = _failing_result('elsewhere/tests/test_gone.py::test_a', NODE_B)
        elif shape == 'other_leg_failed':
            failing = _failing_result(NODE_A, NODE_B, lint_output='E501 line too long')
        else:
            harness.worktree_raises = True

        assert await harness.classify(failing) == (True, MAIN_SHA)
        assert await harness.classify(failing) == (True, MAIN_SHA)

        assert harness.observed == []
        assert harness.rerun_trees == []
        assert len(harness.worktrees) == 2, 'an unconfirmable answer must not be cached'

    async def test_no_recoverable_node_ids_skips_the_worktree_entirely(
        self, harness: _Harness,
    ) -> None:
        failing = _failing_result()
        failing.test_output = 'FAILED some test\n'

        assert await harness.classify(failing) == (True, MAIN_SHA)
        assert harness.worktrees == []
        assert harness.observed == []

    async def test_partial_coverage_keeps_the_verdict(self, harness: _Harness) -> None:
        failing = _failing_result(NODE_A, NODE_B, junit_ids=(JUNIT_A, JUNIT_B, JUNIT_C))
        failing.test_output = f'FAILED {NODE_A} - AssertionError\nFAILED {NODE_B}\n'

        assert await harness.classify(failing) == (True, MAIN_SHA)
        assert harness.rerun_trees, 'the re-run happened; only its coverage was short'
        assert harness.observed == []
