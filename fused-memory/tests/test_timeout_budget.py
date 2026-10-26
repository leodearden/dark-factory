"""Tests for fused_memory.backends.timeout_budget — a backend call under a budget."""
from __future__ import annotations

import asyncio

import pytest

from fused_memory.backends.timeout_budget import await_within


async def _resolve(value):
    return value


async def _hang():
    await asyncio.Event().wait()


async def _raise(exc):
    raise exc


@pytest.mark.asyncio
async def test_result_inside_budget_is_returned_unchanged() -> None:
    payload = {'results': [1, 2, 3]}

    result = await await_within(_resolve(payload), store='Graphiti', op='search', budget_s=5.0)

    assert result is payload


@pytest.mark.asyncio
async def test_overrun_raises_timeout_naming_store_op_and_budget() -> None:
    with pytest.raises(TimeoutError) as excinfo:
        await await_within(_hang(), store='Graphiti', op='search', budget_s=0.05)

    assert str(excinfo.value) == 'Graphiti search timed out after 0.05s'
    assert isinstance(excinfo.value.__cause__, TimeoutError), (
        "wait_for's own TimeoutError must be chained as __cause__, not discarded"
    )


@pytest.mark.asyncio
async def test_non_timeout_exception_passes_through_unwrapped() -> None:
    original = ConnectionError('falkordb refused')

    with pytest.raises(ConnectionError) as excinfo:
        await await_within(_raise(original), store='Mem0', op='get', budget_s=5.0)

    assert excinfo.value is original
