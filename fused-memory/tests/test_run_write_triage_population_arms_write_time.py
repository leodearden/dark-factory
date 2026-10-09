"""Tests for the π2 write-time mode of run_write_triage_population_arms.py (task 6530).

PRD §12 D19: a write is judged on a slate restricted to the records that
existed when it was written, its band re-decided by the shipped
``decide_band``. No network, no key: the OpenAI SDK is faked at
``openai.AsyncOpenAI`` wherever a judge is called.
"""
from __future__ import annotations

import asyncio
import copy
import dataclasses
import functools
import json
import re
import types
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _fm_helpers import load_script_module

from fused_memory.config.schema import FusedMemoryConfig
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


def _freeze() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'freeze_write_triage_population.py', 'freeze_write_triage_population',
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


# ---------------------------------------------------------------------------
# The write-time run
# ---------------------------------------------------------------------------

#: A candidate line of the rendered user prompt, capturing its id.
_ID_LINE = re.compile(r'^- id: (\S+)$', re.MULTILINE)

WRITE_TIME_ARM_NAMES = ['gpt-4o-mini@5', 'gpt-5.6-terra:none@5', 'gpt-6.1-sol:low@5']


def _response() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        output_text='{"verdict": "distinct"}', status='completed', incomplete_details=None,
        usage=types.SimpleNamespace(
            input_tokens=1_000, output_tokens=50,
            output_tokens_details=types.SimpleNamespace(reasoning_tokens=3),
        ),
    )


class _Provider:
    """The fake Responses endpoint: answers ``distinct``, keeping each rendered prompt."""

    def __init__(self) -> None:
        self.prompts: list[tuple[str, str]] = []

    async def _create(self, **kwargs: Any) -> types.SimpleNamespace:
        memory_id = kwargs['input'].split('\n')[1].removeprefix('entry ')
        self.prompts.append((memory_id, kwargs['input']))
        return _response()

    def client(self) -> MagicMock:
        """A fake ``AsyncOpenAI`` that is its own async context manager, as the SDK is."""
        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client.responses.create = AsyncMock(side_effect=self._create)
        return client

    def called(self) -> list[str]:
        return [memory_id for memory_id, _ in self.prompts]

    def shown(self, memory_id: str) -> list[str]:
        [prompt] = [prompt for called, prompt in self.prompts if called == memory_id]
        return _ID_LINE.findall(prompt)


def _late_winner(memory_id: str) -> dict:
    """Frozen band winner ``late`` was created after the write; so was ``same``'s instant."""
    return _write([
        _candidate('late', 0.86, AFTER),
        _candidate('a', 0.70, BEFORE),
        _candidate('same', 0.68, WRITTEN),
        _candidate('b', 0.60, BEFORE),
        _candidate('c', 0.55, BEFORE),
    ], memory_id=memory_id)


def _run_snapshot(tmp_path: Path) -> Path:
    """Six judge-band writes: ``w-gone`` leaves the band, ``w-late``'s winner postdates it."""
    writes = [_judge_band_write(f'w{i:02d}') for i in range(4)]
    writes += [_leaver('w-gone'), _late_winner('w-late')]
    writes += [_write([_candidate('a', 0.40, BEFORE)], memory_id='low', band='stored')]
    return _freeze().write_snapshot(tmp_path / 'data', {
        'schema_version': 2, 'frozen_at': '2026-10-05T07:00:00+00:00',
        'projects': ['dark_factory', 'reify'], 'candidate_k': 20,
        't_high': T_HIGH, 't_low': T_LOW, 'writes': writes, 'targets': {},
        'excluded_writes': {'undated': 0, 'vanished': 0},
        'slate_rows_dropped': {'self': 0, 'own_children': 0},
    })


def _service() -> types.SimpleNamespace:
    return types.SimpleNamespace(config=FusedMemoryConfig())


def _arm(name: str) -> Any:
    [arm] = [a for a in _mod().ARMS if a.name == name]
    return arm


def _run(path: Path, provider: _Provider, arms_dir: Path, **slates: Any) -> dict:
    snapshot, sha = _freeze().load_snapshot(path)
    with patch('openai.AsyncOpenAI', return_value=provider.client()):
        return asyncio.run(_mod().run_arms(
            snapshot, sha, arms_dir, arms=[_arm('gpt-4o-mini@5')], service=_service(),
            max_writes=None, concurrency=2, budget_usd=15.0, sleep=AsyncMock(), **slates,
        ))


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


class TestTheWriteTimeRun:
    @staticmethod
    def _write_time(tmp_path: Path) -> tuple[dict, list[dict], _Provider]:
        path = _run_snapshot(tmp_path)
        provider = _Provider()
        arms_dir = path.parent / 'arms-write-time'
        summary = _run(path, provider, arms_dir, slates=_mod().Slates.WRITE_TIME)
        return summary, _rows(arms_dir / 'gpt-4o-mini@5.jsonl'), provider

    def test_one_row_per_write_still_in_the_band_none_for_the_leaver(
        self, tmp_path: Path,
    ) -> None:
        _, rows, provider = self._write_time(tmp_path)
        judged = {'w00', 'w01', 'w02', 'w03', 'w-late'}
        assert sorted(row['memory_id'] for row in rows) == sorted(judged)
        assert sorted(provider.called()) == sorted(judged)
        assert {row['slates'] for row in rows} == {'write-time'}

    def test_a_later_band_winner_is_replaced_by_the_best_earlier_candidate(
        self, tmp_path: Path,
    ) -> None:
        _, rows, provider = self._write_time(tmp_path)
        [row] = [row for row in rows if row['memory_id'] == 'w-late']
        assert row['band_winner_id'] == 'a'
        assert provider.shown('w-late') == ['a', 'b', 'c']

    def test_the_summary_says_which_slates_and_what_left_the_band(self, tmp_path: Path) -> None:
        summary, _, _ = self._write_time(tmp_path)
        assert (summary['slates'], summary['sample_size'], summary['left_judge_band']) == (
            'write-time', 6, ['w-gone'],
        )
        assert summary['arms']['gpt-4o-mini@5'] == {
            'complete': True, 'rows': 5, 'missing': 0, 'written_now': 5,
        }

    def test_the_default_is_still_the_frozen_slate(self, tmp_path: Path) -> None:
        path = _run_snapshot(tmp_path)
        provider = _Provider()
        summary = _run(path, provider, path.parent / 'arms')
        rows = _rows(path.parent / 'arms' / 'gpt-4o-mini@5.jsonl')
        assert len(rows) == 6
        assert {row['slates'] for row in rows} == {'frozen'}
        assert (summary['slates'], summary['left_judge_band']) == ('frozen', [])
        assert 'late' in provider.shown('w-late')


class TestTheWriteTimeCommand:
    def test_it_runs_the_write_time_arms_into_their_own_directory(self, tmp_path: Path) -> None:
        path = _run_snapshot(tmp_path)
        frozen_arms = path.parent / 'arms'
        frozen_arms.mkdir()
        frozen_row = frozen_arms / 'gpt-4o-mini@5.jsonl'
        body = (json.dumps({'memory_id': 'w00', 'snapshot_sha256': 'f' * 64, 'usd': 0.0}) + '\n')
        frozen_row.write_bytes(body.encode())
        with patch('openai.AsyncOpenAI', return_value=_Provider().client()):
            code = _mod().main([
                'run', '--slates', 'write-time', '--snapshot', str(path), '--budget-usd', '15',
            ])
        assert code == 0
        written = sorted(p.name for p in (path.parent / 'arms-write-time').iterdir())
        assert written == sorted(f'{name}.jsonl' for name in WRITE_TIME_ARM_NAMES)
        assert frozen_row.read_bytes() == body.encode()
        assert list(frozen_arms.iterdir()) == [frozen_row]

    def test_an_arm_outside_the_write_time_table_is_refused_before_any_call(
        self, tmp_path: Path,
    ) -> None:
        path = _run_snapshot(tmp_path)
        provider = _Provider()
        with (
            patch('openai.AsyncOpenAI', return_value=provider.client()),
            pytest.raises(ValueError, match='gpt-6.1-sol:low@20'),
        ):
            _mod().main([
                'run', '--slates', 'write-time', '--snapshot', str(path),
                '--arms', 'gpt-6.1-sol:low@20',
            ])
        assert provider.prompts == []

    def test_the_write_time_arms_are_three_members_of_the_one_arm_table(self) -> None:
        arms = _mod().WRITE_TIME_ARMS
        assert [arm.name for arm in arms] == WRITE_TIME_ARM_NAMES
        assert all(any(arm is member for member in _mod().ARMS) for arm in arms)
