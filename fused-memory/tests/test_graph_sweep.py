"""The best-effort per-graph loop every read-only sweep shares (task 6849).

A failed graph is recorded as structure, never folded into a zero, and an
operator's Ctrl-C or a cancellation still aborts the sweep.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import FrozenInstanceError, fields

import pytest

from fused_memory.maintenance.graph_sweep import GraphFailure, read_each_graph


def reader(failing: dict[str, BaseException] | None = None):
    """An async read_one that answers ``'read:<key>'`` and records every key it was asked for."""
    failing = failing or {}
    asked: list[str] = []

    async def read_one(key: str) -> str:
        asked.append(key)
        if key in failing:
            raise failing[key]
        return f'read:{key}'

    return read_one, asked


class TestReadEachGraph:
    @pytest.mark.asyncio
    async def test_results_arrive_in_key_order_one_per_key(self):
        read_one, _ = reader()

        results, failures = await read_each_graph(['b', 'a', 'c'], read_one, what='probe')

        assert results == ('read:b', 'read:a', 'read:c')
        assert failures == ()

    @pytest.mark.asyncio
    async def test_a_failing_graph_is_recorded_and_the_rest_are_still_read(self):
        read_one, asked = reader({'broken': RuntimeError('FalkorDB connection reset')})
        keys = ['home', 'broken', 'clean']

        results, failures = await read_each_graph(keys, read_one, what='probe')

        assert failures == (
            GraphFailure(group_id='broken', error='RuntimeError: FalkorDB connection reset'),
        )
        assert results == ('read:home', 'read:clean')
        assert asked == keys
        assert len(results) + len(failures) == len(keys)

    @pytest.mark.asyncio
    async def test_a_failure_is_logged_at_error_naming_what_and_the_graph(self, caplog):
        read_one, _ = reader({'broken': RuntimeError('FalkorDB connection reset')})

        with caplog.at_level(logging.ERROR, logger='fused_memory.maintenance.graph_sweep'):
            await read_each_graph(['broken'], read_one, what='leak census')

        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert 'leak census' in errors[0].getMessage()
        assert "'broken'" in errors[0].getMessage()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('raised', [asyncio.CancelledError, KeyboardInterrupt])
    async def test_base_exceptions_abort_the_sweep_instead_of_being_recorded(self, raised):
        read_one, asked = reader({'home': raised()})

        with pytest.raises(raised):
            await read_each_graph(['home', 'clean'], read_one, what='probe')

        assert asked == ['home']

    def test_a_graph_failure_is_frozen(self):
        failure = GraphFailure(group_id='broken', error='RuntimeError: boom')

        for field in fields(failure):
            with pytest.raises(FrozenInstanceError):
                setattr(failure, field.name, 'other')

    @pytest.mark.asyncio
    async def test_no_keys_reads_nothing(self):
        read_one, asked = reader()

        results, failures = await read_each_graph([], read_one, what='probe')

        assert (results, failures) == ((), ())
        assert asked == []
