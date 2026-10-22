"""A resumed Claude CLI process is charged its own cost, not its session's running total.

The recorded sessions in ``fixtures/cumulative_cost/sessions.json`` go through
the real invoke path: ``invoke_claude_agent`` spawns the fake ``claude`` from
``shared.testing_fake_claude`` (selected through ``CLAUDE_BINARY``), and the
real parser reads what it prints.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from shared.cli_invoke import AgentResult, invoke_claude_agent
from shared.process_cost import ProcessCostBasis
from shared.testing_fake_claude import FakeClaudeCli, install_fake_claude_cli

_SESSIONS = json.loads(
    (Path(__file__).parent / 'fixtures' / 'cumulative_cost' / 'sessions.json').read_text()
)
STEWARD_4895 = '59607ecd-9ecd-4f90-9ce6-bc71a7fb33e9'
CROSSED_UPGRADE_POSITIVE = 'f615388b-10b7-4dc9-b86b-4d011dcb0069'
CROSSED_UPGRADE_NEGATIVE = '4bd5a89f-1992-4691-a968-191f2ccc9abf'

InstallFake = Callable[[Sequence[str]], FakeClaudeCli]


def _stdout_lines(session_id: str) -> list[str]:
    return [json.dumps(entry['stdout']) for entry in _SESSIONS[session_id]]


def _reported_totals(session_id: str) -> list[float]:
    return [entry['stdout']['total_cost_usd'] for entry in _SESSIONS[session_id]]


@pytest.fixture
def install_fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> InstallFake:
    def install(responses: Sequence[str]) -> FakeClaudeCli:
        fake = install_fake_claude_cli(tmp_path / 'fake-claude', responses)
        monkeypatch.setenv('CLAUDE_BINARY', str(fake.executable))
        monkeypatch.delenv('DF_AGENT_CPU_GOVERN', raising=False)
        return fake
    return install


async def _invoke(cwd: Path, **kwargs: Any) -> AgentResult:
    return await invoke_claude_agent(prompt='p', system_prompt='s', cwd=cwd, **kwargs)


async def _chain(
    install_fake: InstallFake, cwd: Path, session_id: str,
) -> tuple[FakeClaudeCli, list[AgentResult]]:
    """Run the session's invocations in order, each resuming the previous one."""
    fake = install_fake(_stdout_lines(session_id))
    results = [await _invoke(cwd)]
    for _ in _SESSIONS[session_id][1:]:
        prev = results[-1]
        results.append(await _invoke(
            cwd,
            resume_session_id=prev.session_id,
            resume_prior_total_cost_usd=prev.reported_total_cost_usd,
        ))
    return fake, results


class TestInvokePathDerivesPerProcessCost:
    async def test_steward_session_costs_each_process_its_own_spend(
        self, install_fake: InstallFake, tmp_path: Path,
    ) -> None:
        fake, results = await _chain(install_fake, tmp_path, STEWARD_4895)

        assert [r.cost_usd for r in results] == pytest.approx(
            [2.288609, 1.117410, 1.876402, 2.033117, 1.307432], abs=1e-6,
        )
        assert [r.cost_basis for r in results] == (
            [ProcessCostBasis.REPORTED] + [ProcessCostBasis.SESSION_DELTA] * 4
        )
        assert [r.cost_delta_clamped for r in results] == [False] * 5
        assert [r.reported_total_cost_usd for r in results] == pytest.approx(
            _reported_totals(STEWARD_4895), abs=1e-6,
        )
        assert sum(r.cost_usd for r in results) == pytest.approx(8.622969, abs=1e-6)
        for argv in fake.recorded_argv()[1:5]:
            resume_at = argv.index('--resume')
            assert argv[resume_at + 1] == STEWARD_4895

    async def test_session_crossing_the_upgrade_is_clamped_and_never_negative(
        self, install_fake: InstallFake, tmp_path: Path, caplog,
    ) -> None:
        with caplog.at_level(logging.WARNING, logger='shared.process_cost'):
            _, results = await _chain(install_fake, tmp_path, CROSSED_UPGRADE_POSITIVE)

        assert [r.cost_usd for r in results] == pytest.approx(
            [2.423063, 1.2188795, 3.6470905], abs=1e-6,
        )
        assert [r.cost_basis for r in results] == [
            ProcessCostBasis.REPORTED,
            ProcessCostBasis.USAGE_ESTIMATE,
            ProcessCostBasis.USAGE_ESTIMATE,
        ]
        assert [r.cost_delta_clamped for r in results] == [False, True, True]
        assert all(r.cost_usd >= 0 for r in results)
        assert CROSSED_UPGRADE_POSITIVE in caplog.text

    async def test_negative_session_delta_is_clamped(
        self, install_fake: InstallFake, tmp_path: Path,
    ) -> None:
        _, results = await _chain(install_fake, tmp_path, CROSSED_UPGRADE_NEGATIVE)

        assert [r.cost_usd for r in results] == pytest.approx(
            [1.7857265, 1.5471005], abs=1e-6,
        )
        assert [r.cost_delta_clamped for r in results] == [False, True]
        assert all(r.cost_usd >= 0 for r in results)

    async def test_resume_without_prior_costs_its_priced_usage(
        self, install_fake: InstallFake, tmp_path: Path,
    ) -> None:
        install_fake(_stdout_lines(STEWARD_4895)[3:4])

        result = await _invoke(tmp_path, resume_session_id=STEWARD_4895)

        assert result.cost_usd == pytest.approx(2.0331165, abs=1e-6)
        assert result.cost_basis is ProcessCostBasis.USAGE_ESTIMATE
        assert result.reported_total_cost_usd == pytest.approx(7.3155365, abs=1e-6)

    async def test_fresh_invocation_ignores_a_prior(
        self, install_fake: InstallFake, tmp_path: Path,
    ) -> None:
        install_fake(_stdout_lines(STEWARD_4895)[1:2])

        result = await _invoke(tmp_path, resume_prior_total_cost_usd=2.288609)

        assert result.cost_usd == pytest.approx(3.4060185, abs=1e-6)
        assert result.cost_basis is ProcessCostBasis.REPORTED
        assert result.cost_delta_clamped is False

    def test_agent_result_defaults_to_an_unclamped_reported_cost(self) -> None:
        result = AgentResult(success=True, output='')

        assert result.cost_basis is ProcessCostBasis.REPORTED
        assert result.reported_total_cost_usd is None
        assert result.cost_delta_clamped is False
