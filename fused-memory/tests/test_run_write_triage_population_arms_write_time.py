"""Tests for the π2 write-time mode of run_write_triage_population_arms.py (task 6530).

PRD §12 D19: a write is judged on a slate restricted to the records that
existed when it was written, its band re-decided by the shipped
``decide_band``. No network, no key: the OpenAI SDK is faked at
``openai.AsyncOpenAI`` wherever a judge is called.
"""
from __future__ import annotations

import copy
import dataclasses
import functools
import types
from pathlib import Path
from typing import Any

import pytest
from _fm_helpers import load_script_module

from fused_memory.models.enums import SourceStore
from fused_memory.models.memory import MemoryResult

SCRIPTS = Path(__file__).parent.parent / 'scripts'

#: The snapshot's band edges, rounded.
T_HIGH = 0.887
T_LOW = 0.523

WRITTEN = '2026-10-01T00:00:00+00:00'
BEFORE = '2026-09-30T23:59:59+00:00'
AFTER = '2026-10-02T00:00:00+00:00'


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'run_write_triage_population_arms.py', 'run_write_triage_population_arms',
    )


def _retrieval() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'eval_write_triage_retrieval.py', 'eval_write_triage_retrieval',
    )


def _candidate(memory_id: str, cosine: float, created_at: str | None, **metadata: Any) -> dict:
    """A frozen slate record: ``normalize()`` of a store row plus its created_at."""
    row = MemoryResult(
        id=memory_id, content=f'content of {memory_id}', source_store=SourceStore.mem0,
        metadata={'store_score': cosine, **metadata}, created_at=created_at,
    )
    return {**_retrieval().normalize(row), 'created_at': created_at}


def _write(
    candidates: list[dict],
    *,
    memory_id: str = 'w',
    created_at: str | None = WRITTEN,
    band: str = 'judge',
    band_winner_id: str | None = None,
) -> dict:
    """A write in the frozen-snapshot shape; its frozen winner is the top cosine by default."""
    top = max(candidates, key=lambda c: c['store_score'], default=None)
    return {
        'memory_id': memory_id, 'project_id': 'reify', 'category': 'procedural_knowledge',
        'created_at': created_at, 'content': f'entry {memory_id}',
        'metadata': {}, 'recon_marker': False, 'declares_attach_keys': False,
        'band': band,
        'band_winner_id': band_winner_id or (None if top is None else top['canonical_id']),
        'similarity': None if top is None else top['store_score'],
        'retrieved_count': len(candidates), 'self_retrieved': False, 'own_children_dropped': 0,
        'candidates': candidates,
    }


def _ids(write: dict) -> list[str]:
    return [c['memory_id'] for c in write['candidates']]


def _write_time(write: dict) -> dict:
    return _mod().write_time_slate(write, t_high=T_HIGH, t_low=T_LOW)


class TestTheWriteTimeSlate:
    def test_only_candidates_created_strictly_before_the_write_are_kept(self) -> None:
        write = _write([
            _candidate('early', 0.80, BEFORE),
            _candidate('same', 0.78, WRITTEN),
            _candidate('later', 0.75, '2026-10-01T00:00:01+00:00'),
            _candidate('undated', 0.70, None),
            _candidate('naive', 0.65, '2026-09-30T00:00:00'),
            _candidate('older', 0.60, '2026-09-29T00:00:00+00:00'),
        ])
        assert _ids(_write_time(write)) == ['early', 'older']

    def test_the_band_winner_is_re_decided_over_the_earlier_candidates(self) -> None:
        write = _write([
            _candidate('late', 0.86, AFTER),
            _candidate('a', 0.70, BEFORE),
            _candidate('b', 0.60, BEFORE),
        ])
        assert write['band_winner_id'] == 'late'
        viewed = _write_time(write)
        assert (viewed['band'], viewed['band_winner_id'], viewed['similarity']) == (
            'judge', 'a', 0.70,
        )

    def test_a_kept_child_winner_hoists_to_its_parent(self) -> None:
        write = _write([
            _candidate('late', 0.86, AFTER),
            _candidate('kid', 0.75, BEFORE, kind='sighting', parent_id='P'),
            _candidate('a', 0.70, BEFORE),
        ])
        viewed = _write_time(write)
        assert (viewed['band'], viewed['band_winner_id'], viewed['similarity']) == (
            'judge', 'P', 0.75,
        )

    def test_a_write_whose_earlier_candidates_are_all_below_t_low_is_stored(self) -> None:
        write = _write([
            _candidate('late', 0.80, AFTER),
            _candidate('a', 0.50, BEFORE),
            _candidate('b', 0.40, BEFORE),
        ])
        viewed = _write_time(write)
        assert (viewed['band'], viewed['band_winner_id'], viewed['similarity']) == (
            'stored', None, None,
        )
        assert _ids(viewed) == ['a', 'b']

    def test_a_write_with_no_earlier_candidate_is_stored(self) -> None:
        write = _write([_candidate('late', 0.80, AFTER), _candidate('later', 0.70, AFTER)])
        viewed = _write_time(write)
        assert (viewed['band'], viewed['band_winner_id'], viewed['similarity']) == (
            'stored', None, None,
        )
        assert viewed['candidates'] == []

    def test_the_view_is_stamped_and_carries_every_other_field_unchanged(self) -> None:
        write = _write([_candidate('late', 0.86, AFTER), _candidate('a', 0.70, BEFORE)])
        frozen = copy.deepcopy(write)
        viewed = _write_time(write)
        assert viewed['slates'] == 'write-time'
        decided = {'slates', 'candidates', 'band', 'band_winner_id', 'similarity'}
        assert {k: v for k, v in viewed.items() if k not in decided} == {
            k: v for k, v in frozen.items() if k not in decided
        }
        assert set(viewed) == set(frozen) | {'slates'}

    def test_the_input_write_and_its_slate_are_not_mutated(self) -> None:
        write = _write([_candidate('late', 0.86, AFTER), _candidate('a', 0.70, BEFORE)])
        frozen = copy.deepcopy(write)
        candidates = write['candidates']
        _write_time(write)
        assert write == frozen
        assert write['candidates'] is candidates
        assert candidates == frozen['candidates']

    @pytest.mark.parametrize('created_at', [None, '2026-10-01T00:00:00', 'yesterday'])
    def test_a_write_without_an_aware_instant_is_refused_naming_it(
        self, created_at: str | None,
    ) -> None:
        write = _write([_candidate('a', 0.70, BEFORE)], memory_id='undated-w', created_at=created_at)
        with pytest.raises(ValueError, match='undated-w'):
            _write_time(write)

    def test_instants_with_different_offsets_are_compared_as_instants(self) -> None:
        write = _write([
            # Sorts before the write as text, but is 03:00Z: after the write.
            _candidate('looks-early', 0.80, '2026-09-30T20:00:00-07:00'),
            # Sorts after the write as text, but is 01:00Z: before the write.
            _candidate('looks-late', 0.70, '2026-10-01T03:00:00+02:00'),
        ], created_at='2026-10-01T02:00:00+00:00')
        assert _ids(_write_time(write)) == ['looks-late']


# ---------------------------------------------------------------------------
# The run set
# ---------------------------------------------------------------------------

SNAPSHOT_SHA = 'a' * 64
#: How many writes of the judge-band order a sampled run draws.
SAMPLE = 5


def _judge_band_write(memory_id: str) -> dict:
    return _write(
        [_candidate('a', 0.70, BEFORE), _candidate('b', 0.60, BEFORE)], memory_id=memory_id,
    )


def _leaver(memory_id: str) -> dict:
    """In the judge band only because of a record created after it."""
    return _write(
        [_candidate('late', 0.80, AFTER), _candidate('a', 0.40, BEFORE)], memory_id=memory_id,
    )


def _snapshot(leavers: frozenset[str] = frozenset()) -> dict:
    """Eight judge-band writes (*leavers* among them), one restated and one stored."""
    writes = [
        _leaver(m) if m in leavers else _judge_band_write(m)
        for m in (f'w{i:02d}' for i in range(8))
    ]
    writes += [
        # Restated only because of a later record: the judge band under the filter.
        _write(
            [_candidate('late', 0.95, AFTER), _candidate('a', 0.70, BEFORE)],
            memory_id='det', band='restated',
        ),
        _write([_candidate('a', 0.40, BEFORE)], memory_id='low', band='stored'),
    ]
    return {'t_high': T_HIGH, 't_low': T_LOW, 'candidate_k': 20, 'writes': writes}


def _order(snapshot: dict) -> list[str]:
    return [w['memory_id'] for w in _mod().judge_band_order(snapshot, SNAPSHOT_SHA)]


def _with_leavers() -> tuple[dict, tuple[str, ...]]:
    """The snapshot with the 2nd and 4th writes of the order leaving the band."""
    order = _order(_snapshot())
    leavers = (order[1], order[3])
    return _snapshot(frozenset(leavers)), leavers


def _draw(snapshot: dict, slates: Any, max_writes: int | None = SAMPLE) -> Any:
    return _mod().draw_run_set(snapshot, SNAPSHOT_SHA, max_writes=max_writes, slates=slates)


def _by_id(snapshot: dict) -> dict[str, dict]:
    return {w['memory_id']: w for w in snapshot['writes']}


class TestTheRunSet:
    def test_frozen_is_the_order_prefix_each_write_stamped_frozen(self) -> None:
        snapshot, _ = _with_leavers()
        run_set = _draw(snapshot, _mod().Slates.FROZEN)
        frozen = _by_id(snapshot)
        assert [w['memory_id'] for w in run_set.writes] == _order(snapshot)[:SAMPLE]
        for write in run_set.writes:
            assert write == {**frozen[write['memory_id']], 'slates': 'frozen'}
        assert (run_set.slates, run_set.sample_size, run_set.left_judge_band) == (
            'frozen', SAMPLE, (),
        )

    def test_a_run_set_is_immutable(self) -> None:
        run_set = _draw(_snapshot(), _mod().Slates.FROZEN)
        with pytest.raises(dataclasses.FrozenInstanceError):
            run_set.sample_size = 7  # type: ignore[misc]

    def test_write_time_is_the_same_prefix_less_the_writes_that_left_the_band(self) -> None:
        snapshot, leavers = _with_leavers()
        run_set = _draw(snapshot, _mod().Slates.WRITE_TIME)
        frozen = _by_id(snapshot)
        sample = _order(snapshot)[:SAMPLE]
        assert [w['memory_id'] for w in run_set.writes] == [m for m in sample if m not in leavers]
        for write in run_set.writes:
            assert write == _mod().write_time_slate(
                frozen[write['memory_id']], t_high=T_HIGH, t_low=T_LOW,
            )
        assert run_set.left_judge_band == leavers
        assert (run_set.slates, run_set.sample_size) == ('write-time', SAMPLE)
        assert len(run_set.writes) + len(run_set.left_judge_band) == SAMPLE

    def test_a_write_that_left_the_band_is_never_back_filled(self) -> None:
        snapshot, _ = _with_leavers()
        run_set = _draw(snapshot, _mod().Slates.WRITE_TIME)
        assert _order(snapshot)[SAMPLE] not in {w['memory_id'] for w in run_set.writes}

    @pytest.mark.parametrize('max_writes', [SAMPLE, None])
    def test_a_frozen_restated_write_the_filter_would_judge_is_not_drawn(
        self, max_writes: int | None,
    ) -> None:
        snapshot, _ = _with_leavers()
        assert _mod().write_time_slate(
            _by_id(snapshot)['det'], t_high=T_HIGH, t_low=T_LOW,
        )['band'] == 'judge'
        run_set = _draw(snapshot, _mod().Slates.WRITE_TIME, max_writes=max_writes)
        assert 'det' not in {w['memory_id'] for w in run_set.writes}

    def test_no_limit_draws_the_whole_frozen_judge_band(self) -> None:
        snapshot, leavers = _with_leavers()
        run_set = _draw(snapshot, _mod().Slates.WRITE_TIME, max_writes=None)
        assert run_set.sample_size == 8
        assert len(run_set.writes) == 8 - len(leavers)
