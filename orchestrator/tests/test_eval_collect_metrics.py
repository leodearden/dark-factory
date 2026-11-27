"""runner.collect_metrics, the executor's metric collection, measured against a real git worktree."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _eval_diff_repo import repo_with_commit

from orchestrator.config import PriceEntry
from orchestrator.evals.cell_sandbox import CellSandbox
from orchestrator.evals.metrics import EndpointKind, EvalMetrics, RoleUnderTest
from orchestrator.evals.runner import collect_metrics
from orchestrator.implementer_brief_variant import ImplementerBriefVariant
from orchestrator.verify import VerifyResult

_RUNNER_LOGGER = 'orchestrator.evals.runner'


def _workflow() -> MagicMock:
    wf = MagicMock()
    for name in (
        'total_duration_ms', 'total_output_tokens', 'total_input_tokens',
        'total_cache_read_tokens', 'total_cache_create_tokens', 'total_turns',
        'execute_iterations', 'verify_attempts', 'judge_invocations',
        'judge_early_exits',
    ):
        setattr(wf.metrics, name, 0)
    wf.metrics.total_cost_usd = 0.0
    wf.metrics.judge_cost_usd = 0.0
    wf.config.env_overrides = {}
    wf.config.max_execute_iterations = 20
    wf.artifacts.read_plan.return_value = {'steps': []}
    wf.artifacts.aggregate_reviews.return_value = None
    return wf


async def _collect(
    worktree: Path,
    base_commit: str,
    *,
    role: RoleUnderTest = RoleUnderTest.IMPLEMENTER,
    replay_frame: str | None = None,
    workflow: MagicMock | None = None,
    sandbox_backend: CellSandbox = 'landlock',
    endpoint_kind: EndpointKind = EndpointKind.NATIVE_ANTHROPIC,
) -> EvalMetrics:
    verification = VerifyResult(
        passed=True, test_output='', lint_output='', type_output='', summary='',
    )
    with patch('orchestrator.verify.run_verification', AsyncMock(return_value=verification)):
        return await collect_metrics(
            workflow or _workflow(), worktree, {'id': 't', 'pre_task_commit': base_commit},
            role=role, replay_frame=replay_frame, sandbox_backend=sandbox_backend,
            endpoint_kind=endpoint_kind,
        )


def _runner_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage() for r in caplog.records
        if r.name == _RUNNER_LOGGER and r.levelno >= logging.WARNING
    ]


@pytest.mark.asyncio
async def test_collect_metrics_sets_role_and_replay_frame_on_the_dataclass(tmp_path: Path):
    repo, base = repo_with_commit(tmp_path / 'repo', {})

    m = await _collect(repo, base, role=RoleUnderTest.JUDGE, replay_frame='f')

    assert m.role_under_test is RoleUnderTest.JUDGE
    assert m.replay_frame == 'f'


@pytest.mark.asyncio
async def test_collect_metrics_stamps_the_sandbox_backend(tmp_path: Path):
    repo, base = repo_with_commit(tmp_path / 'repo', {})

    m = await _collect(repo, base, sandbox_backend='bwrap')

    assert m.sandbox_backend == 'bwrap'


@pytest.mark.asyncio
async def test_collect_metrics_stamps_the_endpoint_kind(tmp_path: Path):
    repo, base = repo_with_commit(tmp_path / 'repo', {})

    m = await _collect(repo, base, endpoint_kind=EndpointKind.SELF_HOSTED_VLLM)

    assert m.endpoint_kind is EndpointKind.SELF_HOSTED_VLLM


@pytest.mark.asyncio
async def test_the_cost_trust_flag_follows_the_env_not_the_endpoint_kind(tmp_path: Path):
    repo, base = repo_with_commit(tmp_path / 'repo', {})
    proxied = _workflow()
    proxied.config.env_overrides = {
        'ANTHROPIC_BASE_URL': 'https://token-plan-ams.xiaomimimo.com/anthropic',
    }

    proxied_native = await _collect(
        repo, base, workflow=proxied, endpoint_kind=EndpointKind.NATIVE_ANTHROPIC,
    )
    unproxied_vllm = await _collect(repo, base, endpoint_kind=EndpointKind.SELF_HOSTED_VLLM)

    assert proxied_native.cli_cost_untrusted is True
    assert proxied_native.endpoint_kind is EndpointKind.NATIVE_ANTHROPIC
    assert unproxied_vllm.cli_cost_untrusted is False
    assert unproxied_vllm.endpoint_kind is EndpointKind.SELF_HOSTED_VLLM


@pytest.mark.asyncio
class TestDiffStats:
    """lines_changed / files_changed measure the base..HEAD diff.

    A measurement that failed reads (-1, -1) with a WARNING, never a silent
    (0, 0) that would pass for a genuine no-change run (task 1809).
    """

    async def test_committed_change_counts_lines_and_files(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        repo, base = repo_with_commit(
            tmp_path / 'repo',
            {'README.md': '# Changed\n', 'src/b.py': 'x = 1\ny = 2\n'},
        )

        with caplog.at_level(logging.WARNING, logger=_RUNNER_LOGGER):
            m = await _collect(repo, base)

        assert (m.lines_changed, m.files_changed) == (4, 2)
        assert not _runner_warnings(caplog)

    async def test_nothing_committed_since_base_is_zero_zero(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        repo, base = repo_with_commit(tmp_path / 'repo', {})

        with caplog.at_level(logging.WARNING, logger=_RUNNER_LOGGER):
            m = await _collect(repo, base)

        assert (m.lines_changed, m.files_changed) == (0, 0)
        assert not _runner_warnings(caplog)

    async def test_unknown_base_commit_is_the_sentinel_and_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        repo, _base = repo_with_commit(tmp_path / 'repo', {})

        with caplog.at_level(logging.WARNING, logger=_RUNNER_LOGGER):
            m = await _collect(repo, 'deadbeef')

        assert (m.lines_changed, m.files_changed) == (-1, -1)
        assert _runner_warnings(caplog)

    async def test_absent_worktree_is_the_sentinel_and_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=_RUNNER_LOGGER):
            m = await _collect(tmp_path / 'absent', 'deadbeef')

        assert (m.lines_changed, m.files_changed) == (-1, -1)
        assert _runner_warnings(caplog)


def _glm_cell_workflow(price: PriceEntry) -> MagicMock:
    """A proxied glm-5.3 cell carrying the measured endpoint__46074e01 counts."""
    wf = _workflow()
    wf.config.models.implementer = 'glm-5.3'
    wf.config.env_overrides = {'ANTHROPIC_BASE_URL': 'https://x'}
    wf.config.prices = {'glm-5.3': price}
    wf.metrics.total_input_tokens = 1_193
    wf.metrics.total_output_tokens = 12_537
    wf.metrics.total_cache_read_tokens = 1_370_402
    wf.metrics.total_cache_create_tokens = 13_702
    wf.metrics.total_cost_usd = 99.0
    return wf


@pytest.mark.asyncio
class TestCollectMetricsPricesCacheTokens:
    async def test_cache_tokens_are_priced_at_their_own_rates(self, tmp_path: Path):
        repo, base = repo_with_commit(tmp_path / 'repo', {})
        price = PriceEntry(
            input_per_1m=1.40, output_per_1m=4.40,
            cached_input_per_1m=0.26, cache_write_per_1m=1.40,
        )

        m = await _collect(repo, base, workflow=_glm_cell_workflow(price))

        assert m.cost_usd == pytest.approx(0.43232032)
        assert m.cost_source == 'price_table'
        assert m.cache_read_tokens == 1_370_402

    async def test_missing_cache_rates_label_the_cell(self, tmp_path: Path):
        repo, base = repo_with_commit(tmp_path / 'repo', {})
        price = PriceEntry(input_per_1m=1.40, output_per_1m=4.40)

        m = await _collect(repo, base, workflow=_glm_cell_workflow(price))

        assert m.cost_source == 'price_table_no_cache_rate'


def _zero_work_workflow(variant: ImplementerBriefVariant, iterations: int) -> MagicMock:
    wf = _workflow()
    wf.config.implementer_brief_variant = variant
    wf.config.prices = {}
    wf.metrics.execute_iterations = iterations
    wf.artifacts.read_plan.return_value = {
        'prerequisites': [],
        'steps': [{'id': f's{i}', 'status': 'pending'} for i in range(3)],
    }
    return wf


@pytest.mark.asyncio
class TestFalseGreenAtTheEnforcedCap:
    """The false-green guard reads the cap the EXECUTE loop enforced, not max_execute_iterations.

    Under one-step a 3-step plan stops at 3 + slack = 6 iterations, far short
    of the eval default of 20, so a zero-work $0 T/T/T run must still be nulled.
    """

    async def test_one_step_run_stopped_at_its_cap_is_flagged(self, tmp_path: Path):
        repo, base = repo_with_commit(tmp_path / 'repo', {})

        m = await _collect(
            repo, base,
            workflow=_zero_work_workflow(ImplementerBriefVariant.ONE_STEP, 6),
        )

        assert m.tests_pass is None
        assert m.lint_clean is None
        assert m.typecheck_clean is None

    async def test_whole_plan_run_short_of_max_execute_iterations_is_not_flagged(
        self, tmp_path: Path,
    ):
        repo, base = repo_with_commit(tmp_path / 'repo', {})

        m = await _collect(
            repo, base,
            workflow=_zero_work_workflow(ImplementerBriefVariant.WHOLE_PLAN, 6),
        )

        assert m.tests_pass is True
