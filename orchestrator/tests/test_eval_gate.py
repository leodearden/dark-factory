"""The eval usage-gate lifecycle: build, own, borrow, tear down (task 6737).

``orchestrator.evals.eval_gate`` is the one home for how an eval entry point
gets its account-failover gate. These tests drive it through its public seams
only: ``build_eval_usage_gate``'s own tests control the ``UsageGate``
constructor it calls, and ``owned_usage_gate`` / ``campaign_usage_gate`` are
handed their builder as ``factory=``. The rationale behind each behaviour
pinned here lives in docs/plan-scoring-and-judge.md, "Eval usage-gate
lifecycle (task 6737)".
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _campaign_gate_helpers import eval_base_config
from _orch_helpers import build_usage_gate
from shared.config_models import UsageCapConfig
from shared.testing import make_gate_mock

from orchestrator.evals import eval_gate
from orchestrator.evals.eval_gate import (
    GATE_UNSET,
    build_eval_usage_gate,
    campaign_usage_gate,
    owned_usage_gate,
)

_GATE_LOGGER = 'orchestrator.evals.eval_gate'


def _with_usage_cap(tmp_path: Path, *, enabled: bool):
    base = eval_base_config(tmp_path)
    return base.model_copy(
        update={'usage_cap': base.usage_cap.model_copy(update={'enabled': enabled})},
    )


def _zero_account_gate(shutdown: AsyncMock):
    """A real gate that constructed fine but resolved no accounts."""
    gate = build_usage_gate([], [])
    gate.shutdown = shutdown
    return gate


@pytest.mark.asyncio
class TestBuildEvalUsageGate:
    async def test_disabled_usage_cap_builds_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        constructor = MagicMock()
        monkeypatch.setattr(eval_gate, 'UsageGate', constructor)

        gate = await build_eval_usage_gate(_with_usage_cap(tmp_path, enabled=False))

        assert gate is None
        constructor.assert_not_called()

    async def test_a_raising_constructor_degrades_to_ungated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        caplog.set_level(logging.WARNING, logger=_GATE_LOGGER)
        monkeypatch.setattr(
            eval_gate, 'UsageGate', MagicMock(side_effect=RuntimeError('boom')),
        )

        gate = await build_eval_usage_gate(_with_usage_cap(tmp_path, enabled=True))

        assert gate is None
        assert 'without failover' in caplog.text
        assert 'boom' in caplog.text

    async def test_a_zero_account_gate_is_torn_down_and_discarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        caplog.set_level(logging.WARNING, logger=_GATE_LOGGER)
        shutdown = AsyncMock()
        monkeypatch.setattr(
            eval_gate, 'UsageGate', MagicMock(return_value=_zero_account_gate(shutdown)),
        )

        gate = await build_eval_usage_gate(_with_usage_cap(tmp_path, enabled=True))

        assert gate is None
        shutdown.assert_awaited_once()
        assert 'ZERO accounts' in caplog.text

    async def test_a_failing_zero_account_teardown_still_degrades(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        empty = _zero_account_gate(AsyncMock(side_effect=RuntimeError('teardown boom')))
        monkeypatch.setattr(eval_gate, 'UsageGate', MagicMock(return_value=empty))

        gate = await build_eval_usage_gate(_with_usage_cap(tmp_path, enabled=True))

        assert gate is None

    async def test_a_healthy_gate_is_built_once_from_the_usage_cap_leaf(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        healthy = make_gate_mock()
        constructor = MagicMock(return_value=healthy)
        monkeypatch.setattr(eval_gate, 'UsageGate', constructor)
        orch_config = _with_usage_cap(tmp_path, enabled=True)

        gate = await build_eval_usage_gate(orch_config)

        assert gate is healthy
        constructor.assert_called_once()
        passed = constructor.call_args.args[0]
        assert isinstance(passed, UsageCapConfig)
        assert passed.enabled is True
        assert passed == orch_config.usage_cap


@pytest.mark.asyncio
class TestOwnedUsageGate:
    """The executor's half: build-and-own when unset, borrow otherwise."""

    async def test_unset_builds_yields_and_tears_down_its_own_gate(
        self, tmp_path: Path,
    ):
        gate = make_gate_mock()
        factory = AsyncMock(return_value=gate)
        orch_config = eval_base_config(tmp_path)

        async with owned_usage_gate(GATE_UNSET, orch_config, factory=factory) as yielded:
            assert yielded is gate
            gate.shutdown.assert_not_awaited()

        factory.assert_awaited_once_with(orch_config)
        gate.shutdown.assert_awaited_once()

    async def test_an_owned_gate_is_torn_down_when_the_body_raises(
        self, tmp_path: Path,
    ):
        gate = make_gate_mock()
        factory = AsyncMock(return_value=gate)

        with pytest.raises(RuntimeError, match='cell exploded'):
            async with owned_usage_gate(
                GATE_UNSET, eval_base_config(tmp_path), factory=factory,
            ):
                raise RuntimeError('cell exploded')

        gate.shutdown.assert_awaited_once()

    async def test_a_failing_teardown_is_swallowed_and_names_the_label(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ):
        caplog.set_level(logging.WARNING, logger=_GATE_LOGGER)
        gate = make_gate_mock()
        gate.shutdown = AsyncMock(side_effect=RuntimeError('teardown boom'))

        async with owned_usage_gate(
            GATE_UNSET, eval_base_config(tmp_path),
            factory=AsyncMock(return_value=gate), label='df_task_1 × cand',
        ) as yielded:
            assert yielded is gate

        assert 'shutdown failed' in caplog.text
        assert 'df_task_1 × cand' in caplog.text

    async def test_a_degraded_build_yields_none_and_tears_nothing_down(
        self, tmp_path: Path,
    ):
        factory = AsyncMock(return_value=None)

        async with owned_usage_gate(
            GATE_UNSET, eval_base_config(tmp_path), factory=factory,
        ) as yielded:
            assert yielded is None

        factory.assert_awaited_once()

    async def test_an_injected_gate_is_borrowed_never_built_never_torn_down(
        self, tmp_path: Path,
    ):
        borrowed = make_gate_mock()
        factory = AsyncMock()

        with pytest.raises(RuntimeError, match='cell exploded'):
            async with owned_usage_gate(
                borrowed, eval_base_config(tmp_path), factory=factory,
            ) as yielded:
                assert yielded is borrowed
                raise RuntimeError('cell exploded')

        factory.assert_not_awaited()
        borrowed.shutdown.assert_not_awaited()

    async def test_an_injected_none_stays_ungated(self, tmp_path: Path):
        factory = AsyncMock()

        async with owned_usage_gate(
            None, eval_base_config(tmp_path), factory=factory,
        ) as yielded:
            assert yielded is None

        factory.assert_not_awaited()


@pytest.mark.asyncio
class TestCampaignUsageGate:
    """The campaign's half: ONE gate built from the base config, torn down once."""

    async def test_yields_the_built_gate_and_builds_exactly_once(
        self, tmp_path: Path,
    ):
        gate = make_gate_mock()
        factory = AsyncMock(return_value=gate)
        base = eval_base_config(tmp_path)

        async with campaign_usage_gate(base, factory=factory) as yielded:
            assert yielded is gate

        factory.assert_awaited_once()
        assert factory.await_args is not None
        assert factory.await_args.args[0] is base

    async def test_gate_is_shut_down_on_normal_exit(self, tmp_path: Path):
        gate = make_gate_mock()

        async with campaign_usage_gate(
            eval_base_config(tmp_path), factory=AsyncMock(return_value=gate),
        ):
            gate.shutdown.assert_not_awaited()

        gate.shutdown.assert_awaited_once()

    async def test_gate_is_shut_down_when_the_campaign_raises(self, tmp_path: Path):
        """A campaign that dies mid-fan-out must not leak the probe loop."""
        gate = make_gate_mock()

        with pytest.raises(RuntimeError, match='campaign exploded'):
            async with campaign_usage_gate(
                eval_base_config(tmp_path), factory=AsyncMock(return_value=gate),
            ):
                raise RuntimeError('campaign exploded')

        gate.shutdown.assert_awaited_once()

    async def test_a_degraded_pool_yields_none(self, tmp_path: Path):
        async with campaign_usage_gate(
            eval_base_config(tmp_path), factory=AsyncMock(return_value=None),
        ) as gate:
            assert gate is None

    async def test_none_base_config_yields_none_without_building(self):
        factory = AsyncMock()

        async with campaign_usage_gate(None, factory=factory) as gate:
            assert gate is None

        factory.assert_not_awaited()

    async def test_a_failing_shutdown_is_swallowed_and_warned(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ):
        caplog.set_level(logging.WARNING, logger=_GATE_LOGGER)
        gate = make_gate_mock()
        gate.shutdown = AsyncMock(side_effect=RuntimeError('teardown boom'))

        async with campaign_usage_gate(
            eval_base_config(tmp_path), factory=AsyncMock(return_value=gate),
        ) as yielded:
            assert yielded is gate

        assert 'shutdown failed' in caplog.text
