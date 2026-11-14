"""A steward resuming one CLI session is charged each process's own spend.

Since Claude CLI 2.1.277 a resumed process reports its session's running
total. The steward resumes on every continuation, so it used to re-count all
earlier spend each time: steward session 59607ecd (task 4895) really cost
$8.62, was counted as $26.92, and starved its own fifth invocation's budget.

Everything here runs through the real chain: ``_handle_escalation`` ->
``_invoke_with_session`` -> ``invoke_with_cap_retry`` -> ``invoke_agent`` ->
``invoke_claude_agent``. The fake ``claude`` (``shared.testing_fake_claude``,
selected through ``CLAUDE_BINARY``) answers the recorded stdouts in
``shared/tests/fixtures/cumulative_cost/sessions.json``.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from escalation.models import Escalation
from shared.testing_fake_claude import FakeClaudeCli, install_fake_claude_cli

from orchestrator.event_store import EventType

_SESSIONS = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / 'shared' / 'tests' / 'fixtures' / 'cumulative_cost' / 'sessions.json'
    ).read_text()
)
STEWARD_4895 = '59607ecd-9ecd-4f90-9ce6-bc71a7fb33e9'
CROSSED_UPGRADE_POSITIVE = 'f615388b-10b7-4dc9-b86b-4d011dcb0069'


def _stdout_lines(session_id: str) -> list[str]:
    return [json.dumps(entry['stdout']) for entry in _SESSIONS[session_id]]


def _make_escalation(escalation_id: str) -> Escalation:
    return Escalation(
        id=escalation_id,
        task_id='42',
        agent_role='orchestrator',
        severity='blocking',
        category='limit_exhausted',
        summary='execute limit exhausted',
    )


def _escalations(count: int) -> list[Escalation]:
    return [_make_escalation(f'esc-42-{n}') for n in range(1, count + 1)]


def _invocation_ends(event_store: MagicMock) -> list[dict[str, Any]]:
    return [
        c.kwargs
        for c in event_store.emit.call_args_list
        if c.args and c.args[0] == EventType.invocation_end
    ]


@pytest.fixture
def install_fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def install(responses: Sequence[str]) -> FakeClaudeCli:
        fake = install_fake_claude_cli(tmp_path / 'fake-claude', responses)
        monkeypatch.setenv('CLAUDE_BINARY', str(fake.executable))
        monkeypatch.delenv('DF_AGENT_CPU_GOVERN', raising=False)
        return fake
    return install


@pytest.fixture
def steward(make_steward):
    """A real steward under the production lifetime cap, whose queue resolves each record."""
    built = make_steward(
        config_overrides={'steward_lifetime_budget': 20.0},
        event_store=MagicMock(),
    )
    built.escalation_queue.get.side_effect = lambda escalation_id: dataclasses.replace(
        _make_escalation(escalation_id), status='resolved',
    )
    return built


async def _handle_all(steward, escalations: Sequence[Escalation]) -> None:
    for escalation in escalations:
        await steward._handle_escalation(escalation)


def _resumed_session(argv: list[str]) -> str | None:
    return argv[argv.index('--resume') + 1] if '--resume' in argv else None


def _max_budget(argv: list[str]) -> str:
    return argv[argv.index('--max-budget-usd') + 1]


@pytest.mark.asyncio
class TestStewardSessionCost:
    async def test_each_continuation_is_charged_its_own_spend(
        self, steward, install_fake,
    ) -> None:
        fake = install_fake(_stdout_lines(STEWARD_4895))

        await _handle_all(steward, _escalations(5))

        ends = _invocation_ends(steward.event_store)
        assert [e['cost_usd'] for e in ends] == pytest.approx(
            [2.288609, 1.117410, 1.876402, 2.033117, 1.307432], abs=1e-6,
        )
        assert [e['data']['cost_basis'] for e in ends] == (
            ['reported'] + ['session_delta'] * 4
        )
        assert [e['data']['cost_delta_clamped'] for e in ends] == [False] * 5
        assert steward.metrics.total_cost_usd == pytest.approx(8.622969, abs=1e-6)
        argv = fake.recorded_argv()
        assert _max_budget(argv[4]) == '5.0'
        assert [_resumed_session(a) for a in argv[1:5]] == [STEWARD_4895] * 4

    async def test_session_crossing_the_cli_upgrade_is_clamped(
        self, steward, install_fake,
    ) -> None:
        install_fake(_stdout_lines(CROSSED_UPGRADE_POSITIVE))

        await _handle_all(steward, _escalations(3))

        ends = _invocation_ends(steward.event_store)
        assert steward.metrics.total_cost_usd == pytest.approx(7.289033, abs=1e-6)
        assert [e['data']['cost_delta_clamped'] for e in ends] == [False, True, True]
        assert all(e['cost_usd'] >= 0 for e in ends)

    async def test_empty_output_drops_the_session_and_its_prior(
        self, steward, install_fake,
    ) -> None:
        fresh, resume = _stdout_lines(STEWARD_4895)[:2]
        fake = install_fake([fresh, '', fresh, resume])
        first, second, third = _escalations(3)

        await _handle_all(steward, [first, second, second, third])

        argv = fake.recorded_argv()
        assert _resumed_session(argv[2]) is None
        ends = _invocation_ends(steward.event_store)
        assert ends[2]['data']['cost_basis'] == 'reported'
        assert ends[3]['cost_usd'] == pytest.approx(1.1174095, abs=1e-6)
        assert ends[3]['data']['cost_basis'] == 'session_delta'
