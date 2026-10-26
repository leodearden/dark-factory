"""How an eval entry point builds, borrows and tears down its account-failover gate.

Rationale: docs/plan-scoring-and-judge.md, "Eval usage-gate lifecycle (task 6737)".
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from shared.usage_gate import UsageGate

from orchestrator.config import OrchestratorConfig

logger = logging.getLogger(__name__)


class _UnsetGate:
    """The type of :data:`GATE_UNSET`: the caller passed no ``usage_gate`` at all."""


GATE_UNSET = _UnsetGate()

InjectedGate = UsageGate | None | _UnsetGate
"""A ``usage_gate=`` argument: a borrowed gate, ``None`` (deliberately ungated), or
:data:`GATE_UNSET` (build one and own it)."""

GateFactory = Callable[[OrchestratorConfig], Awaitable[UsageGate | None]]


async def build_eval_usage_gate(orch_config: OrchestratorConfig) -> UsageGate | None:
    """The account-failover gate for an eval, or ``None`` when it degrades to ungated."""
    if not orch_config.usage_cap.enabled:
        return None
    try:
        gate = UsageGate(orch_config.usage_cap)
    except Exception as exc:
        logger.warning(f'Failed to create UsageGate for eval: {exc} — running without failover')
        return None
    if gate.account_count == 0:
        logger.warning(
            'UsageGate resolved ZERO accounts (no CLAUDE_OAUTH_* env and no '
            'default credential?) — running without failover rather than '
            'refusing every invocation'
        )
        await _shut_down_best_effort(gate, 'the discarded zero-account gate')
        return None
    return gate


async def _shut_down_best_effort(gate: UsageGate, label: str) -> None:
    try:
        await gate.shutdown()
    except Exception:
        logger.warning(f'UsageGate shutdown failed for {label} — continuing', exc_info=True)


@asynccontextmanager
async def owned_usage_gate(
    injected: InjectedGate,
    orch_config: OrchestratorConfig,
    *,
    factory: GateFactory = build_eval_usage_gate,
    label: str = 'eval',
) -> AsyncIterator[UsageGate | None]:
    """Yield the gate *injected* names; one built here is torn down on exit."""
    if not isinstance(injected, _UnsetGate):
        yield injected
        return
    gate = await factory(orch_config)
    try:
        yield gate
    finally:
        if gate is not None:
            await _shut_down_best_effort(gate, label)


@asynccontextmanager
async def campaign_usage_gate(
    base_config: OrchestratorConfig | None,
    *,
    factory: GateFactory = build_eval_usage_gate,
) -> AsyncIterator[UsageGate | None]:
    """Own ONE gate for a whole eval campaign, built from its base config."""
    if base_config is None:
        yield None
        return
    async with owned_usage_gate(
        GATE_UNSET, base_config, factory=factory, label='campaign',
    ) as gate:
        yield gate
