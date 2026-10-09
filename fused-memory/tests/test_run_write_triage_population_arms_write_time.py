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


# ---------------------------------------------------------------------------
# Publish
# ---------------------------------------------------------------------------

_PACKAGE = Path(__file__).parent.parent
_LONG = 'z' * 5_000
#: Of the seven judge-band writes, the six the publish fixture samples.
PUBLISH_SAMPLE = 6
JUDGED = ['w-marker', 'w-late', 'w-long', 'w-kid', 'w-keys']


def _sha(data: bytes | str) -> str:
    import hashlib  # noqa: PLC0415

    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def _calibrate() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'calibrate_write_triage.py', 'calibrate_write_triage')


def _scorer() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'score_write_triage_pairs.py', 'score_write_triage_pairs')


def _wording() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'write_triage_judge_wording.py', 'write_triage_judge_wording',
    )


def _plain(memory_id: str, **fields: Any) -> dict:
    write = _write(
        [_candidate('a', 0.70, BEFORE), _candidate('b', 0.60, BEFORE)], memory_id=memory_id,
    )
    return {**write, **fields}


def _publish_writes() -> list[dict]:
    """Seven judge-band writes, one restated, one stored; slate sizes 2, 3 or 4 at write time."""
    return [
        _leaver('w-gone'),
        _write([
            _candidate('late', 0.86, AFTER), _candidate('a', 0.70, BEFORE),
            _candidate('b', 0.60, BEFORE), _candidate('c', 0.55, BEFORE),
        ], memory_id='w-late'),
        _plain('w-marker', recon_marker=True),
        _plain('w-keys', declares_attach_keys=True),
        _write([
            _candidate('a', 0.70, BEFORE), {**_candidate('b', 0.60, BEFORE), 'content': _LONG},
        ], memory_id='w-long'),
        _write([
            _candidate('kid', 0.75, BEFORE, kind='sighting', parent_id='P'),
            _candidate('a', 0.70, BEFORE), _candidate('b', 0.60, BEFORE),
            _candidate('d', 0.55, BEFORE),
        ], memory_id='w-kid'),
        _plain('w-plain'),
        _write([_candidate('a', 0.95, BEFORE)], memory_id='det', band='restated'),
        _write([_candidate('a', 0.40, BEFORE)], memory_id='low', band='stored'),
    ]


class _PublishedWriteTime:
    """A write-time run over the first six of seven judge-band writes, one of which left the band."""

    def __init__(self, tmp_path: Path) -> None:
        (tmp_path / '.git').mkdir()
        self.path = _freeze().write_snapshot(tmp_path / 'data', {
            'schema_version': 2, 'frozen_at': '2026-10-05T07:00:00+00:00',
            'projects': ['dark_factory', 'reify'], 'candidate_k': 20,
            't_high': T_HIGH, 't_low': T_LOW, 'writes': _publish_writes(),
            'targets': {'P': {'project_id': 'reify', 'content': 'the parent'}},
            # `vanished` is what puts w-plain seventh in this snapshot's hash order.
            'excluded_writes': {'undated': 1, 'vanished': 6},
            'slate_rows_dropped': {'self': 4, 'own_children': 3},
        })
        self.snapshot, self.sha = _freeze().load_snapshot(self.path)
        self.order = [w['memory_id'] for w in _mod().judge_band_order(self.snapshot, self.sha)]
        assert self.order[PUBLISH_SAMPLE] == 'w-plain', self.order
        self.run_set = _mod().draw_run_set(
            self.snapshot, self.sha, max_writes=PUBLISH_SAMPLE, slates=_mod().Slates.WRITE_TIME,
        )
        assert [w['memory_id'] for w in self.run_set.writes] == JUDGED
        self.rows = {arm.name: self._rows(arm) for arm in _mod().WRITE_TIME_ARMS}
        self._answer('gpt-4o-mini@5', {'w-late': ('amended', 'a'), 'w-kid': ('restated', 'P')})
        self._answer('gpt-5.6-terra:none@5', {'w-long': ('contested', 'b')})
        self._answer('gpt-6.1-sol:low@5', {
            'w-marker': ('amended', 'gone'), 'w-late': ('amended', 'a'),
        })
        mini = self.rows['gpt-4o-mini@5']
        for row, seconds in zip(mini, (1.0, 2.0, 3.0, 4.0, 5.0), strict=True):
            row['judge_seconds'] = seconds
        mini[0]['transport_failures'] = ['TimeoutError', 'RateLimitError']
        mini[1]['transport_failures'] = ['TimeoutError']
        mini[2]['usd'] = None
        mini[4]['parse_failure'] = True

    def _rows(self, arm: Any) -> list[dict]:
        return [{
            'arm': arm.name, 'judge_model': arm.model, 'reasoning_effort': arm.reasoning_effort,
            'width': arm.width, 'wording': arm.wording, 'snapshot_sha256': self.sha,
            'field_chars': 4_000, 'timeout_seconds': 15.0,
            'memory_id': write['memory_id'], 'project_id': write['project_id'],
            'category': write['category'], 'recon_marker': write['recon_marker'],
            'declares_attach_keys': write['declares_attach_keys'],
            'band': write['band'], 'band_winner_id': write['band_winner_id'],
            'slates': 'write-time',
            'outcome': 'stored', 'verdict_candidate_id': None, 'judged_candidate_id': None,
            'raw_text': '{"verdict": "distinct"}', 'usage': None, 'usd': 0.001,
            'judge_seconds': 1.0, 'parse_failure': False, 'failure': None,
            'attempts': 1, 'transport_failures': [],
        } for write in self.run_set.writes]

    def _answer(self, arm_name: str, answers: dict[str, tuple[str, str]]) -> None:
        for row in self.rows[arm_name]:
            if row['memory_id'] in answers:
                outcome, target = answers[row['memory_id']]
                row.update(outcome=outcome, verdict_candidate_id=target, judged_candidate_id=target)

    def arms_dir(self) -> Path:
        return self.path.parent / 'arms-write-time'

    def write_arm_files(self) -> None:
        self.arms_dir().mkdir(exist_ok=True)
        for name, rows in self.rows.items():
            _mod().arm_path(self.arms_dir(), name).write_text(
                ''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows),
            )

    def artifact(self, **kwargs: Any) -> dict:
        self.write_arm_files()
        return _mod().build_write_time_population_artifact(
            self.snapshot, self.sha, self.path, self.rows,
            sample_size=PUBLISH_SAMPLE, budget_usd=15.0, **kwargs,
        )

    def pairs(self, already_rated: Any = frozenset()) -> tuple[list[dict], dict]:
        return _mod().build_write_time_pairs_to_rate(
            self.snapshot, self.sha, self.rows,
            sample_size=PUBLISH_SAMPLE, already_rated=already_rated,
        )


@pytest.fixture
def published(tmp_path: Path) -> _PublishedWriteTime:
    return _PublishedWriteTime(tmp_path)


class TestTheWriteTimePopulationBlock:
    def test_what_was_sampled_what_left_and_what_the_filter_changed(
        self, published: _PublishedWriteTime,
    ) -> None:
        assert published.artifact()['population'] == {
            'slates': 'write-time',
            'slates_rule': _mod().WRITE_TIME_RULE,
            'n_writes': 9,
            'n_judge_band_frozen': 7,
            'judge_band_sample': {'order': 'sha256(snapshot_sha256:memory_id)', 'size': 6, 'of': 7},
            'n_judge_band_write_time': 5,
            'excluded_by_write_time_filter': {
                'count': 1, 'memory_ids': ['w-gone'], 'by_band': {'stored': 1},
            },
            'band_winner_changed': 1,
            'writes_with_later_candidates': 2,
            'later_candidates_dropped': 1,
            'write_time_slate_size': {'min': 2, 'median': 2, 'max': 4},
            'recon_marker_run_set': 1,
            'declares_attach_keys_run_set': 1,
            'projects': ['dark_factory', 'reify'],
            'frozen_at': '2026-10-05T07:00:00+00:00',
            'snapshot_sha256': published.sha,
            'snapshot_path': 'data/write-triage-population-2026-10-05/snapshot.json',
            't_high': T_HIGH,
            't_low': T_LOW,
        }

    def test_the_rule_is_stated_in_words(self) -> None:
        assert 'before' in _mod().WRITE_TIME_RULE


class TestTheWriteTimeArmRows:
    def test_one_row_per_write_time_arm_in_order(self, published: _PublishedWriteTime) -> None:
        arms = published.artifact()['arms']
        assert [row['arm'] for row in arms] == WRITE_TIME_ARM_NAMES
        for row, arm in zip(arms, _mod().WRITE_TIME_ARMS, strict=True):
            assert (row['model'], row['reasoning_effort'], row['width'], row['wording']) == (
                arm.model, arm.reasoning_effort, arm.width, arm.wording,
            )
            assert row['calls'] == 5
            assert row['cases_path'] == (
                f'data/write-triage-population-2026-10-05/arms-write-time/{arm.name}.jsonl'
            )

    def test_an_arms_provenance_and_measurements(self, published: _PublishedWriteTime) -> None:
        [row] = [r for r in published.artifact()['arms'] if r['arm'] == 'gpt-4o-mini@5']
        cases = published.arms_dir() / 'gpt-4o-mini@5.jsonl'
        seconds = _calibrate().summarize_distribution([1.0, 2.0, 3.0, 4.0, 5.0])
        assert row == {
            'arm': 'gpt-4o-mini@5', 'model': 'gpt-4o-mini', 'provider': 'openai',
            'reasoning_effort': None, 'width': 5, 'wording': 'shipped',
            'system_prompt_sha256': _sha(_wording().system_prompt('shipped')),
            'field_chars': 4_000, 'timeout_seconds': 15.0,
            'calls': 5, 'parse_failures': 1,
            'transport_failures': 3,
            'transport_failures_by_type': {'RateLimitError': 1, 'TimeoutError': 2},
            'usd': 0.004, 'unpriced_calls': 1,
            'p50_seconds': seconds['median'], 'p95_seconds': seconds['p95'],
            'outcomes': {'amended': 1, 'restated': 1, 'stored': 3},
            'cases_path': 'data/write-triage-population-2026-10-05/arms-write-time/gpt-4o-mini@5.jsonl',
            'cases_sha256': _sha(cases.read_bytes()),
        }

    def test_the_spend_and_no_wording_readout(self, published: _PublishedWriteTime) -> None:
        artifact = published.artifact()
        assert artifact['spend'] == {
            'usd_total': pytest.approx(0.014),
            'budget_usd': 15.0,
            'list_prices_as_of': _scorer().LIST_PRICES_AS_OF,
            'list_prices_source': _scorer().LIST_PRICES_SOURCE,
        }
        assert 'wording_attribution' not in artifact

    def test_the_artifact_carries_the_pairs_block_it_was_given(
        self, published: _PublishedWriteTime,
    ) -> None:
        block = {'n_pairs': 3, 'sha256': 'b' * 64}
        assert published.artifact(pairs_to_rate=block)['pairs_to_rate'] == block


def _drop_arm(p: _PublishedWriteTime) -> str:
    del p.rows['gpt-5.6-terra:none@5']
    return 'gpt-5.6-terra:none@5'


def _foreign_row(p: _PublishedWriteTime) -> str:
    p.rows['gpt-6.1-sol:low@5'][1]['snapshot_sha256'] = 'f' * 64
    return 'gpt-6.1-sol:low@5'


def _frozen_slate_row(p: _PublishedWriteTime) -> str:
    p.rows['gpt-4o-mini@5'][2]['slates'] = 'frozen'
    return 'gpt-4o-mini@5'


def _row_without_slates(p: _PublishedWriteTime) -> str:
    del p.rows['gpt-4o-mini@5'][2]['slates']
    return 'gpt-4o-mini@5'


def _judged_twice(p: _PublishedWriteTime) -> str:
    rows = p.rows['gpt-6.1-sol:low@5']
    rows.append(dict(rows[0]))
    return 'gpt-6.1-sol:low@5'


def _leaver_judged(p: _PublishedWriteTime) -> str:
    rows = p.rows['gpt-5.6-terra:none@5']
    rows.append({**rows[0], 'memory_id': 'w-gone'})
    return 'gpt-5.6-terra:none@5'


def _a_write_missed(p: _PublishedWriteTime) -> str:
    p.rows['gpt-6.1-sol:low@5'].pop()
    return 'gpt-6.1-sol:low@5'


def _outside_the_sample(p: _PublishedWriteTime) -> str:
    p.rows['gpt-4o-mini@5'][-1]['memory_id'] = 'w-plain'
    return 'gpt-4o-mini@5'


_REFUSED = [
    _drop_arm, _foreign_row, _frozen_slate_row, _row_without_slates, _judged_twice,
    _leaver_judged, _a_write_missed, _outside_the_sample,
]


class TestPublishWriteTimeRefuses:
    @pytest.mark.parametrize('spoil', _REFUSED, ids=lambda f: f.__name__.lstrip('_'))
    def test_the_artifact_refuses_naming_the_arm(
        self, published: _PublishedWriteTime, spoil: Any,
    ) -> None:
        arm_name = spoil(published)
        with pytest.raises(ValueError, match=re.escape(arm_name)):
            published.artifact()

    @pytest.mark.parametrize('spoil', _REFUSED, ids=lambda f: f.__name__.lstrip('_'))
    def test_the_pairs_refuse_the_same_runs(
        self, published: _PublishedWriteTime, spoil: Any,
    ) -> None:
        arm_name = spoil(published)
        with pytest.raises(ValueError, match=re.escape(arm_name)):
            published.pairs()


class TestTheWriteTimePairs:
    def test_one_blind_row_per_distinct_judged_pair(self, published: _PublishedWriteTime) -> None:
        rows, stats = published.pairs()
        assert {(row['entry_id'], row['target_id']) for row in rows} == {
            ('w-late', 'a'), ('w-kid', 'P'), ('w-long', 'b'), ('w-marker', 'gone'),
        }
        assert all(
            set(row) == {'entry_id', 'target_id', 'entry_text', 'target_text'} for row in rows
        )
        assert stats['n_pairs'] == 4

    def test_texts_come_from_the_slate_then_the_targets_and_are_cut_at_4000(
        self, published: _PublishedWriteTime,
    ) -> None:
        rows, stats = published.pairs()
        text = {(row['entry_id'], row['target_id']): row for row in rows}
        assert text[('w-late', 'a')]['entry_text'] == 'entry w-late'
        assert text[('w-late', 'a')]['target_text'] == 'content of a'
        assert text[('w-kid', 'P')]['target_text'] == 'the parent'
        assert text[('w-long', 'b')]['target_text'] == (
            _LONG[:4_000] + '…[truncated, 5000 chars total]'
        )
        assert text[('w-marker', 'gone')]['target_text'] is None
        assert stats['missing_target_text'] == 1

    def test_the_order_is_the_snapshot_hash_order_of_the_pair(
        self, published: _PublishedWriteTime,
    ) -> None:
        rows, stats = published.pairs()
        keys = [f'{published.sha}:{row["entry_id"]}:{row["target_id"]}' for row in rows]
        assert keys == sorted(keys, key=_sha)
        assert stats['order'] == 'sha256(snapshot_sha256:entry_id:target_id)'

    def test_pairs_already_rated_are_left_out_and_counted(
        self, published: _PublishedWriteTime,
    ) -> None:
        rows, stats = published.pairs(already_rated={('w-late', 'a'), ('x', 'y')})
        assert ('w-late', 'a') not in {(row['entry_id'], row['target_id']) for row in rows}
        assert (stats['n_pairs'], stats['excluded_already_rated']) == (3, 1)


def _verdicts(path: Path, *pairs: tuple[str, str]) -> Path:
    path.write_text(''.join(
        json.dumps({'entry_id': entry, 'target_id': target, 'verdict': 'duplicate'}) + '\n'
        for entry, target in pairs
    ))
    return path


class TestThePublishWriteTimeCommand:
    @staticmethod
    def _rated(tmp_path: Path) -> tuple[Path, Path]:
        corpus = _verdicts(tmp_path / 'verdicts.jsonl', ('w-late', 'a'), ('x', 'y'))
        seed = _verdicts(tmp_path / 'seed.jsonl', ('w-kid', 'P'))
        return corpus, seed

    @staticmethod
    def _publish(published: _PublishedWriteTime, *args: str) -> int:
        return _mod().main([
            'publish-write-time', '--snapshot', str(published.path),
            '--max-writes', str(PUBLISH_SAMPLE), *args,
        ])

    def test_the_artifact_names_the_pairs_and_the_verdicts_they_were_kept_apart_from(
        self, published: _PublishedWriteTime, tmp_path: Path,
    ) -> None:
        published.write_arm_files()
        corpus, seed = self._rated(tmp_path)
        assert self._publish(
            published,
            '--population-out', str(tmp_path / 'pop.json'),
            '--pairs-out', str(tmp_path / 'pairs.jsonl'),
            '--already-rated', str(corpus), str(seed),
        ) == 0
        artifact = json.loads((tmp_path / 'pop.json').read_text(encoding='utf-8'))
        body = (tmp_path / 'pairs.jsonl').read_bytes()
        pairs = [json.loads(line) for line in body.decode().splitlines()]
        assert {(p['entry_id'], p['target_id']) for p in pairs} == {
            ('w-long', 'b'), ('w-marker', 'gone'),
        }
        assert artifact['pairs_to_rate'] == {
            'n_pairs': 2, 'excluded_already_rated': 2, 'missing_target_text': 1,
            'order': 'sha256(snapshot_sha256:entry_id:target_id)',
            'path': 'pairs.jsonl', 'sha256': _sha(body),
            'already_rated': [
                {'path': 'verdicts.jsonl', 'rows': 2, 'sha256': _sha(corpus.read_bytes())},
                {'path': 'seed.jsonl', 'rows': 1, 'sha256': _sha(seed.read_bytes())},
            ],
        }
        assert artifact['spend']['budget_usd'] == 15.0
        assert artifact['population']['n_judge_band_write_time'] == 5
        assert list(tmp_path.glob('*.tmp')) == []

    def test_a_refused_run_writes_neither_file(
        self, published: _PublishedWriteTime, tmp_path: Path,
    ) -> None:
        _frozen_slate_row(published)
        published.write_arm_files()
        corpus, seed = self._rated(tmp_path)
        with pytest.raises(ValueError, match='gpt-4o-mini@5'):
            self._publish(
                published,
                '--population-out', str(tmp_path / 'pop.json'),
                '--pairs-out', str(tmp_path / 'pairs.jsonl'),
                '--already-rated', str(corpus), str(seed),
            )
        assert not (tmp_path / 'pop.json').exists()
        assert not (tmp_path / 'pairs.jsonl').exists()

    def test_the_defaults_are_the_write_time_files_never_the_pi_ones(self) -> None:
        calibration = _PACKAGE / 'calibration'
        defaults = {
            name: getattr(_mod(), name) for name in (
                'DEFAULT_WRITE_TIME_POPULATION_OUT', 'DEFAULT_WRITE_TIME_PAIRS_OUT',
                'DEFAULT_VERDICT_CORPUS', 'DEFAULT_ALREADY_RATED',
            )
        }
        assert defaults == {
            'DEFAULT_WRITE_TIME_POPULATION_OUT': calibration / 'write_triage_population_write_time.json',
            'DEFAULT_WRITE_TIME_PAIRS_OUT': calibration / 'write_triage_pairs_to_rate_write_time.jsonl',
            'DEFAULT_VERDICT_CORPUS': calibration / 'write_triage_pair_verdicts.jsonl',
            'DEFAULT_ALREADY_RATED': (
                _PACKAGE / 'tests' / 'fixtures' / 'write_triage_pair_verdicts_seed.jsonl'
            ),
        }

    def test_the_parser_writes_to_its_defaults(
        self, published: _PublishedWriteTime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        published.write_arm_files()
        corpus, seed = self._rated(tmp_path)
        monkeypatch.setattr(_mod(), 'DEFAULT_WRITE_TIME_POPULATION_OUT', tmp_path / 'pop.json')
        monkeypatch.setattr(_mod(), 'DEFAULT_WRITE_TIME_PAIRS_OUT', tmp_path / 'pairs.jsonl')
        monkeypatch.setattr(_mod(), 'DEFAULT_VERDICT_CORPUS', corpus)
        monkeypatch.setattr(_mod(), 'DEFAULT_ALREADY_RATED', seed)
        assert self._publish(published) == 0
        artifact = json.loads((tmp_path / 'pop.json').read_text(encoding='utf-8'))
        assert artifact['pairs_to_rate']['path'] == 'pairs.jsonl'
        assert [s['path'] for s in artifact['pairs_to_rate']['already_rated']] == [
            'verdicts.jsonl', 'seed.jsonl',
        ]

    def test_the_sample_size_is_required(self, published: _PublishedWriteTime) -> None:
        with pytest.raises(SystemExit):
            _mod().main(['publish-write-time', '--snapshot', str(published.path)])


# ---------------------------------------------------------------------------
# The committed artifacts
# ---------------------------------------------------------------------------

_REPO = _PACKAGE.parent
_COMMITTED_WRITE_TIME = _PACKAGE / 'calibration' / 'write_triage_population_write_time.json'
_COMMITTED_WRITE_TIME_PAIRS = _PACKAGE / 'calibration' / 'write_triage_pairs_to_rate_write_time.jsonl'
_COMMITTED_PI = _PACKAGE / 'calibration' / 'write_triage_population.json'
_TRUNCATION = re.compile(r'…\[truncated, \d+ chars total\]$')


@functools.cache
def _committed() -> dict:
    return json.loads(_COMMITTED_WRITE_TIME.read_text(encoding='utf-8'))


@functools.cache
def _committed_pairs() -> tuple[bytes, tuple[dict, ...]]:
    body = _COMMITTED_WRITE_TIME_PAIRS.read_bytes()
    return body, tuple(json.loads(line) for line in body.decode('utf-8').splitlines())


class TestCommittedWriteTimePopulationIsTraceable:
    """The committed π2 artifacts say which arms ran, on which slates, over what.

    No numeric floor is asserted (e.g. ``n_judge_band_write_time >= 300``):
    that is Γ3's predicate, and a floor in a test would make the test the decision.
    """

    def test_the_arms_are_the_write_time_table_in_order(self) -> None:
        rows = _committed()['arms']
        assert [row['arm'] for row in rows] == [arm.name for arm in _mod().WRITE_TIME_ARMS]
        for row, arm in zip(rows, _mod().WRITE_TIME_ARMS, strict=True):
            assert {'calls', 'parse_failures', 'usd'} <= set(row), row['arm']
            assert (row['model'], row['reasoning_effort'], row['width'], row['wording']) == (
                arm.model, arm.reasoning_effort, arm.width, arm.wording,
            )
            assert row['system_prompt_sha256'] == _sha(_wording().system_prompt(arm.wording))
            assert row['field_chars'] == 4_000
            assert '/arms-write-time/' in row['cases_path'], row['arm']
            assert '/arms/' not in row['cases_path'], row['arm']

    def test_every_arm_judged_every_write_still_in_the_band(self) -> None:
        population = _committed()['population']
        n_judged = population['n_judge_band_write_time']
        excluded = population['excluded_by_write_time_filter']
        for row in _committed()['arms']:
            assert row['calls'] == n_judged, row['arm']
        assert n_judged + excluded['count'] == population['judge_band_sample']['size']
        assert len(excluded['memory_ids']) == excluded['count']
        assert population['slates'] == 'write-time'

    def test_the_sample_is_pis_sample_of_the_same_snapshot(self) -> None:
        pi = json.loads(_COMMITTED_PI.read_text(encoding='utf-8'))['population']
        population = _committed()['population']
        assert population['snapshot_sha256'] == pi['snapshot_sha256']
        assert population['judge_band_sample']['size'] == pi['n_judge_band']

    def test_the_spend_is_the_sum_of_the_arms(self) -> None:
        import math  # noqa: PLC0415

        artifact = _committed()
        assert artifact['spend']['usd_total'] == round(
            math.fsum(row['usd'] for row in artifact['arms']), 6,
        )
        assert artifact['spend']['budget_usd'] is not None

    def test_the_pairs_file_is_the_one_the_artifact_names(self) -> None:
        body, rows = _committed_pairs()
        block = _committed()['pairs_to_rate']
        assert len(rows) == block['n_pairs']
        assert _sha(body) == block['sha256']

    def test_every_pair_row_is_blind(self) -> None:
        _, rows = _committed_pairs()
        for row in rows:
            assert set(row) == {'entry_id', 'target_id', 'entry_text', 'target_text'}, row

    def test_no_text_exceeds_the_rater_cap(self) -> None:
        _, rows = _committed_pairs()
        for row in rows:
            for key in ('entry_text', 'target_text'):
                text = row[key]
                if text is not None and len(text) > 4_000:
                    assert len(_TRUNCATION.sub('', text)) == 4_000, (row['entry_id'], key)
                    assert _TRUNCATION.search(text), (row['entry_id'], key)

    def test_no_pair_is_one_already_rated_when_it_was_published(self) -> None:
        sources = _committed()['pairs_to_rate']['already_rated']
        assert {
            'fused-memory/calibration/write_triage_pair_verdicts.jsonl',
            'fused-memory/tests/fixtures/write_triage_pair_verdicts_seed.jsonl',
        } <= {source['path'] for source in sources}
        _, rows = _committed_pairs()
        asked = {(row['entry_id'], row['target_id']) for row in rows}
        for source in sources:
            # Verdict files are only appended to, so the published prefix is still there.
            lines = (_REPO / source['path']).read_bytes().splitlines(keepends=True)
            prefix = lines[:source['rows']]
            assert len(prefix) == source['rows'], source['path']
            assert _sha(b''.join(prefix)) == source['sha256'], source['path']
            rated = {(v['entry_id'], v['target_id']) for v in map(json.loads, prefix)}
            assert not asked & rated, source['path']
