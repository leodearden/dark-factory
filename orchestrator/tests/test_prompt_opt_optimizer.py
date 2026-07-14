"""Tests for orchestrator.evals.prompt_opt.optimizer — the frontier-optimizer reflection step (T6).

Uses an injected async fake `invoke_fn`, mirroring
orchestrator/tests/test_reviewer_trial_runner.py's SimpleNamespace/AsyncMock
convention, so no real LLM call is made (hermetic).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from orchestrator.evals.prompt_opt.optimizer import propose_heuristics_edit
from orchestrator.evals.prompt_opt.scorer import ScoredItem

# A string that must NEVER reach the optimizer — proves the CONTRACT is not
# passed in (D-3: optimizer.py's signature has no contract parameter at all,
# so this also documents that guarantee for future readers/refactors).
_SENTINEL_CONTRACT = 'SENTINEL-CONTRACT-TEXT-NEVER-SEEN-BY-OPTIMIZER'


def _make_result(heuristics_text: str) -> SimpleNamespace:
    return SimpleNamespace(
        success=True,
        output=heuristics_text,
        structured_output=None,
        cost_usd=0.1,
    )


class TestProposeHeuristicsEdit:
    @pytest.mark.asyncio
    async def test_invoke_fn_called_with_optimizer_model(self) -> None:
        fake_invoke = AsyncMock(return_value=_make_result('new heuristics'))

        await propose_heuristics_edit(
            'current heuristics text',
            [],
            [],
            max_edits=4,
            optimizer_model='frontier-y',
            invoke_fn=fake_invoke,
            cwd=Path('/tmp'),
        )

        assert fake_invoke.await_args.kwargs['model'] == 'frontier-y'

    @pytest.mark.asyncio
    async def test_prompt_contains_heuristics_scores_and_rejected_but_not_contract(self) -> None:
        fake_invoke = AsyncMock(return_value=_make_result('new heuristics'))
        scored_minibatch = [
            ScoredItem(item='item-1', rollout='rollout-1', score=0.42),
            ScoredItem(item='item-2', rollout='rollout-2', score=0.77),
        ]
        rejected_buffer = ['rejected candidate text A']

        await propose_heuristics_edit(
            'THE CURRENT HEURISTICS BLOCK',
            scored_minibatch,
            rejected_buffer,
            max_edits=3,
            optimizer_model='frontier-y',
            invoke_fn=fake_invoke,
            cwd=Path('/tmp'),
        )

        _, kwargs = fake_invoke.call_args
        combined = kwargs['prompt'] + kwargs['system_prompt']

        assert 'THE CURRENT HEURISTICS BLOCK' in combined
        assert '0.42' in combined
        assert '0.77' in combined
        assert 'rejected candidate text A' in combined
        assert _SENTINEL_CONTRACT not in combined

    @pytest.mark.asyncio
    async def test_max_edits_budget_conveyed_in_prompt(self) -> None:
        fake_invoke = AsyncMock(return_value=_make_result('new heuristics'))

        await propose_heuristics_edit(
            'current heuristics',
            [],
            [],
            max_edits=3,
            optimizer_model='frontier-y',
            invoke_fn=fake_invoke,
            cwd=Path('/tmp'),
        )

        _, kwargs = fake_invoke.call_args
        assert '3' in kwargs['prompt']

    @pytest.mark.asyncio
    async def test_returns_output_text(self) -> None:
        fake_invoke = AsyncMock(return_value=_make_result('THE NEW HEURISTICS'))

        result = await propose_heuristics_edit(
            'current',
            [],
            [],
            max_edits=4,
            optimizer_model='frontier-y',
            invoke_fn=fake_invoke,
            cwd=Path('/tmp'),
        )

        assert result == 'THE NEW HEURISTICS'

    @pytest.mark.asyncio
    async def test_returns_structured_output_heuristics_field_when_present(self) -> None:
        fake_invoke = AsyncMock(return_value=SimpleNamespace(
            success=True,
            output='fallback text',
            structured_output={'heuristics': 'STRUCTURED HEURISTICS'},
            cost_usd=0.1,
        ))

        result = await propose_heuristics_edit(
            'current',
            [],
            [],
            max_edits=4,
            optimizer_model='frontier-y',
            invoke_fn=fake_invoke,
            cwd=Path('/tmp'),
        )

        assert result == 'STRUCTURED HEURISTICS'

    @pytest.mark.asyncio
    async def test_default_invoke_fn_is_invoke_agent(self) -> None:
        import inspect

        from orchestrator.agents.invoke import invoke_agent

        sig = inspect.signature(propose_heuristics_edit)
        assert sig.parameters['invoke_fn'].default is invoke_agent
