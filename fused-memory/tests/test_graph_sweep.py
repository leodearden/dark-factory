"""The best-effort per-graph loop every read-only sweep shares (task 6849).

A failed graph is recorded as structure, never folded into a zero, and an
operator's Ctrl-C or a cancellation still aborts the sweep. Whether a sweep's
numbers can be trusted as a whole is decided once, by ``sweep_complete``.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import FrozenInstanceError, dataclass, fields

import pytest

from fused_memory.maintenance.graph_sweep import (
    GraphFailure,
    SweepReads,
    read_each_graph,
    sweep_complete,
)


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

        reads = await read_each_graph(['b', 'a', 'c'], read_one, what='probe')

        assert reads.results == ('read:b', 'read:a', 'read:c')
        assert reads.failures == ()

    @pytest.mark.asyncio
    async def test_a_failing_graph_is_recorded_and_the_rest_are_still_read(self):
        read_one, asked = reader({'broken': RuntimeError('FalkorDB connection reset')})
        keys = ['home', 'broken', 'clean']

        reads = await read_each_graph(keys, read_one, what='probe')

        assert reads.failures == (
            GraphFailure(group_id='broken', error='RuntimeError: FalkorDB connection reset'),
        )
        assert reads.results == ('read:home', 'read:clean')
        assert asked == keys
        assert len(reads.results) + len(reads.failures) == len(keys)

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

    @pytest.mark.asyncio
    async def test_the_loop_is_timed_in_milliseconds(self):
        read_one, _ = reader()

        reads = await read_each_graph(['home'], read_one, what='probe')

        assert isinstance(reads.elapsed_ms, float)
        assert reads.elapsed_ms >= 0

    @pytest.mark.asyncio
    async def test_what_it_read_and_each_failure_are_frozen(self):
        read_one, _ = reader({'broken': RuntimeError('boom')})

        reads = await read_each_graph(['home', 'broken'], read_one, what='probe')

        for frozen in (reads, reads.failures[0]):
            for field in fields(frozen):
                with pytest.raises(FrozenInstanceError):
                    setattr(frozen, field.name, 'other')
        assert isinstance(reads, SweepReads)

    @pytest.mark.asyncio
    async def test_no_keys_reads_nothing(self):
        read_one, asked = reader()

        reads = await read_each_graph([], read_one, what='probe')

        assert (reads.results, reads.failures) == ((), ())
        assert asked == []


@dataclass(frozen=True)
class Read:
    """A per-graph result that only says whether its graph was read whole."""

    complete: bool


BROKEN = GraphFailure(group_id='broken', error='RuntimeError: boom')


class TestSweepComplete:
    def test_whole_reads_and_no_failures_are_complete(self):
        assert sweep_complete((Read(True), Read(True)), ()) is True

    def test_one_failed_graph_makes_the_sweep_incomplete_though_every_read_was_whole(self):
        assert sweep_complete((Read(True), Read(True)), (BROKEN,)) is False

    def test_one_partial_read_makes_the_sweep_incomplete(self):
        assert sweep_complete((Read(True), Read(False)), ()) is False

    def test_a_sweep_that_read_no_graph_is_not_complete(self):
        assert sweep_complete((), ()) is False

    def test_a_sweep_whose_every_graph_failed_is_not_complete(self):
        assert sweep_complete((), (BROKEN,)) is False
