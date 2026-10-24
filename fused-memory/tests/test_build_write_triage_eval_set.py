"""Tests for build_write_triage_eval_set.py — the fully labelled rubric v2.1 eval set.

Synthetic only: a tmp frozen snapshot, tmp live slates and a tmp scratch.
An in-test rater writes each shard's ratings file, quoting verbatim from the
texts its rater saw (the pattern of test_build_write_triage_v2_soak_baseline.py::_rate).
"""
from __future__ import annotations

import functools
import json
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from _fm_helpers import load_script_module

from fused_memory.models.enums import SourceStore
from fused_memory.models.memory import MemoryResult

SCRIPTS = Path(__file__).parent.parent / 'scripts'

T_HIGH = 0.887
T_LOW = 0.523
WRITTEN = '2026-10-01T00:00:00+00:00'
BEFORE = '2026-09-30T23:59:59+00:00'
LIVE_AT = '2026-10-12T00:00:00+00:00'
LIVE_BEFORE = '2026-10-11T20:00:00+00:00'


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'build_write_triage_eval_set.py',
                              mod_name='build_write_triage_eval_set')


@functools.cache
def _rubric() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'write_triage_reader_rubric.py',
                              mod_name='write_triage_reader_rubric')


@functools.cache
def _rounds() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'write_triage_rating_rounds.py',
                              mod_name='write_triage_rating_rounds')


@functools.cache
def _score() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'score_write_triage_reader.py',
                              mod_name='score_write_triage_reader')


def _freeze() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'freeze_write_triage_population.py',
                              'freeze_write_triage_population')


def _retrieval() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'eval_write_triage_retrieval.py',
                              'eval_write_triage_retrieval')


def _candidate(memory_id: str, cosine: float, **metadata: Any) -> dict[str, Any]:
    """A frozen slate record created before the offline writes."""
    row = MemoryResult(id=memory_id, content=f'content of {memory_id}',
                       source_store=SourceStore.mem0,
                       metadata={'store_score': cosine, **metadata}, created_at=BEFORE)
    return {**_retrieval().normalize(row), 'created_at': BEFORE}


def _offline(memory_id: str, candidates: list[dict[str, Any]], *, project: str = 'dark_factory',
             band: str = 'judge', **flags: Any) -> dict[str, Any]:
    top = max(candidates, key=lambda c: c['store_score'])
    return {
        'memory_id': memory_id, 'project_id': project, 'category': 'procedural_knowledge',
        'created_at': WRITTEN, 'content': f'entry {memory_id}', 'metadata': {},
        'recon_marker': False, 'declares_attach_keys': False, 'band': band,
        'band_winner_id': top['canonical_id'], 'similarity': top['store_score'],
        'retrieved_count': len(candidates), 'self_retrieved': False, 'own_children_dropped': 0,
        'candidates': candidates, **flags,
    }


def _snapshot(root: Path) -> Path:
    writes = [
        _offline('o1', [_candidate('a', 0.80), _candidate('b', 0.70), _candidate('c', 0.60),
                        _candidate('d', 0.55)]),
        _offline('o2', [_candidate('kid', 0.75, kind='sighting', parent_id='P'),
                        _candidate('e', 0.65)], project='reify'),
        _offline('o-marker', [_candidate('a', 0.70)], recon_marker=True),
        _offline('o-keys', [_candidate('a', 0.70)], declares_attach_keys=True),
        _offline('o-high', [_candidate('a', 0.95)], band='restated'),
    ]
    return _freeze().write_snapshot(root / 'data', {
        'schema_version': 2, 'frozen_at': '2026-10-05T07:00:00+00:00',
        'projects': ['dark_factory', 'reify'], 'candidate_k': 20,
        't_high': T_HIGH, 't_low': T_LOW, 'writes': writes,
        'targets': {'P': {'content': 'the parent P', 'project_id': 'reify'}},
        'excluded_writes': {'undated': 0, 'vanished': 0},
        'slate_rows_dropped': {'self': 0, 'own_children': 0},
    })


def _live_candidate(memory_id: str, cosine: float, *, canonical: str | None = None,
                    edited: bool = False, **metadata: Any) -> dict[str, Any]:
    return {
        'memory_id': memory_id, 'canonical_id': canonical or memory_id,
        'category': 'procedural_knowledge', 'store_score': cosine,
        'metadata': {'category': 'procedural_knowledge', 'store_score': cosine, **metadata},
        'created_at': LIVE_BEFORE, 'content': f'live text of {memory_id}',
        'edited_after_write': edited,
    }


def _live_row(memory_id: str, project: str, candidates: list[dict[str, Any]], *,
              targets: dict[str, Any] | None = None, outcome: str = 'stored',
              target: str | None = None) -> dict[str, Any]:
    return {
        'memory_id': memory_id, 'project_id': project, 'category': 'procedural_knowledge',
        'created_at': LIVE_AT, 'content': f'live entry {memory_id}', 'metadata': {},
        'band': 'judge', 'band_winner_id': candidates[0]['canonical_id'],
        'similarity': candidates[0]['store_score'], 'candidates': candidates,
        'targets': targets or {}, 'production_outcome': outcome, 'production_target': target,
    }


LIVE_HEADER = {'strata': {'solar_challenge': 760, 'solar_challenge_platform': 553},
               'allocation': {'solar_challenge': 1, 'solar_challenge_platform': 1}}


def _live_rows() -> list[dict[str, Any]]:
    return [
        _live_row('l1', 'solar_challenge',
                  [_live_candidate('f', 0.80, edited=True), _live_candidate('g', 0.70)],
                  outcome='amended', target='f'),
        _live_row('l2', 'solar_challenge_platform',
                  [_live_candidate('kid2', 0.78, canonical='Q', kind='sighting', parent_id='Q'),
                   _live_candidate('h', 0.60)],
                  targets={'Q': {'content': 'the parent Q', 'created_at': LIVE_BEFORE,
                                 'edited_after_write': True}}),
    ]


def _live_slates(root: Path) -> Path:
    path = root / 'live_slates.jsonl'
    path.write_text(''.join(json.dumps(line) + '\n' for line in [LIVE_HEADER, *_live_rows()]))
    return path


ALL_PAIRS = {('o1', 'a'), ('o1', 'b'), ('o1', 'c'), ('o1', 'd'), ('o2', 'P'), ('o2', 'e'),
             ('l1', 'f'), ('l1', 'g'), ('l2', 'Q'), ('l2', 'h')}


def _target_rows(write: Any) -> list[tuple[str, int, float, bool | None]]:
    return [(t['target_id'], t['rank'], round(t['cosine'], 6), t['edited_after_write'])
            for t in write.row['targets']]


class TestTheWrites:
    def test_offline_writes_are_the_judged_write_time_draw_with_its_top_targets(
        self, tmp_path: Path,
    ) -> None:
        snapshot, sha = _freeze().load_snapshot(_snapshot(tmp_path))
        writes = {write.row['entry_id']: write for write in _mod().offline_writes(snapshot, sha)}
        assert set(writes) == {'o1', 'o2'}
        assert _target_rows(writes['o1']) == [('a', 1, 0.8, None), ('b', 2, 0.7, None),
                                              ('c', 3, 0.6, None), ('d', 4, 0.55, None)]
        assert _target_rows(writes['o2']) == [('P', 1, 0.75, None), ('e', 2, 0.65, None)]
        row = writes['o2'].row
        assert {key: row[key] for key in ('entry_id', 'source', 'project', 'category', 'created_at',
                                          'stratum', 'production_outcome', 'production_target')} == {
            'entry_id': 'o2', 'source': 'offline', 'project': 'reify',
            'category': 'procedural_knowledge', 'created_at': WRITTEN,
            'stratum': 'offline-write-time', 'production_outcome': 'stored',
            'production_target': None,
        }
        assert writes['o2'].child_text == 'entry o2'
        assert writes['o2'].target_texts == {'P': 'the parent P', 'e': 'content of e'}

    def test_live_writes_read_the_rebuilt_slates(self) -> None:
        writes = {write.row['entry_id']: write
                  for write in _mod().live_writes(LIVE_HEADER, _live_rows())}
        assert _target_rows(writes['l1']) == [('f', 1, 0.8, True), ('g', 2, 0.7, False)]
        assert _target_rows(writes['l2']) == [('Q', 1, 0.78, True), ('h', 2, 0.6, False)]
        row = writes['l1'].row
        assert (row['source'], row['project'], row['stratum'], row['production_outcome'],
                row['production_target']) == (
            'live', 'solar_challenge', 'live-solar_challenge', 'amended', 'f')
        assert writes['l2'].target_texts == {'Q': 'the parent Q', 'h': 'live text of h'}

    def test_each_write_gives_one_current_basis_entry_per_target(self) -> None:
        [l1, _] = _mod().live_writes(LIVE_HEADER, _live_rows())
        assert l1.entries() == [
            _rounds().ShardEntry('l1', 'f', 'current', 'solar_challenge', 'live entry l1',
                                 'live text of f'),
            _rounds().ShardEntry('l1', 'g', 'current', 'solar_challenge', 'live entry l1',
                                 'live text of g'),
        ]

    def test_the_full_label_outcome_is_restated(self) -> None:
        assert _mod().FULL_LABEL_OUTCOME == 'restated'
        assert _mod().EVAL_WIDTH == 5


def _vote(entry: str, target: str, role: str, **answers: Any) -> dict[str, Any]:
    row = {'entry_id': entry, 'target_id': target, 'basis': 'current', 'rater': f'opus-{role}',
           'role': role, 'batch': f'{role}_00', 'reason': 'r', **answers}
    if answers.get('conflict') == 'YES':
        row.setdefault('conflict_quote', {'parent': 'p', 'child': 'c'})
        row.setdefault('scope', 'CENTRAL')
    if answers.get('loss') == 'YES':
        row.setdefault('loss_spans', ['c'])
    if 'UNCLEAR' in answers.values():
        row['unclear_reason'] = 'cut'
    return row


KEPT = {'home': 'YES', 'conflict': 'NO', 'loss': 'NO'}


def _rows(*pairs: tuple[str, str]) -> list[dict[str, Any]]:
    return [{'entry_id': entry, 'targets': [{'target_id': target, 'rank': 1, 'cosine': 0.7,
                                             'edited_after_write': None}]}
            for entry, target in pairs]


class TestPairDerivations:
    def _derive(self, *votes: dict[str, Any]) -> Any:
        return _mod().pair_derivations(_rows(('w', 't')), list(votes))[('w', 't')]

    def test_a_pair_both_raters_answered_fully_is_decided_on_every_binary_it_reads(self) -> None:
        derivation = self._derive(_vote('w', 't', 'primary', **KEPT), _vote('w', 't', 'second', **KEPT))
        assert isinstance(derivation, _rubric().Decided)
        assert {str(binary) for _, binary in derivation.consumed} == {'conflict', 'home', 'loss'}

    def test_home_tied_under_conflict_yes_is_incomplete(self) -> None:
        derivation = self._derive(
            _vote('w', 't', 'primary', home='YES', conflict='YES', action_changing='YES'),
            _vote('w', 't', 'second', home='NO', conflict='YES', action_changing='YES'),
        )
        assert isinstance(derivation, _rubric().Incomplete)
        assert [(gap.target_id, str(gap.binary), str(gap.kind)) for gap in derivation.missing] == [
            ('t', 'home', 'tied')]

    def test_home_yes_and_conflict_no_without_a_loss_answer_is_open(self) -> None:
        derivation = self._derive(
            _vote('w', 't', 'primary', home='YES', conflict='UNCLEAR'),
            _vote('w', 't', 'second', home='UNCLEAR', conflict='NO'),
        )
        assert [(gap.target_id, str(gap.binary), str(gap.kind)) for gap in derivation.missing] == [
            ('t', 'loss', 'open')]

    def test_every_pair_of_every_write_is_derived(self) -> None:
        derivations = _mod().pair_derivations(_rows(('w', 't'), ('w', 'u'), ('x', 't')), [])
        assert set(derivations) == {('w', 't'), ('w', 'u'), ('x', 't')}
        assert all(isinstance(d, _rubric().Incomplete) for d in derivations.values())


Triple = tuple[str, str, str]
Rule = Callable[[dict[str, str]], dict[str, Any]]


def _triple(item: dict[str, str]) -> Triple:
    return (item['entry_id'], item['target_id'], item['basis'])


def _kept(_: dict[str, str]) -> dict[str, Any]:
    return dict(KEPT)


def _conflict_yes(text: dict[str, str]) -> dict[str, Any]:
    return {'home': 'YES', 'conflict': 'YES', 'scope': 'CENTRAL', 'action_changing': 'YES',
            'conflict_quote': {'parent': text['parent_text'], 'child': text['child_text']}}


def _loss_yes(text: dict[str, str]) -> dict[str, Any]:
    return {'home': 'YES', 'conflict': 'NO', 'loss': 'YES', 'loss_spans': [text['child_text']]}


def _conflict_no(_: dict[str, str]) -> dict[str, Any]:
    return {'home': 'YES', 'conflict': 'NO', 'loss': 'NO'}


def _conflict_unclear(_: dict[str, str]) -> dict[str, Any]:
    return {'home': 'YES', 'conflict': 'UNCLEAR', 'unclear_reason': 'cut where the claims meet'}


def _home_unclear(_: dict[str, str]) -> dict[str, Any]:
    return {'home': 'UNCLEAR', 'conflict': 'NO', 'unclear_reason': 'cut at the subject'}


def _home_only(_: dict[str, str]) -> dict[str, Any]:
    return {'home': 'NO'}


def _lacks_loss(_: dict[str, str]) -> dict[str, Any]:
    return {'home': 'YES', 'conflict': 'NO'}


def _unfound_quote(text: dict[str, str]) -> dict[str, Any]:
    return {**_conflict_yes(text), 'conflict_quote': {'parent': text['parent_text'],
                                                      'child': 'not in it'}}


class _EvalSet:
    """The tmp inputs and scratch, the CLI over them, and a fake rater."""

    def __init__(self, root: Path) -> None:
        self.snapshot = _snapshot(root)
        self.live = _live_slates(root)
        self.scratch = root / 'rounds'
        self.out = root / 'out'

    def run(self, command: str, *extra: str) -> int:
        return _mod().main([command, '--snapshot', str(self.snapshot), '--live-slates',
                            str(self.live), '--scratch', str(self.scratch), *extra])

    def assemble(self) -> int:
        return self.run('assemble', '--out-dir', str(self.out))

    def key(self) -> dict[str, Any]:
        return json.loads((self.scratch / 'key.json').read_text())

    def shard_files(self) -> dict[str, bytes]:
        return {path.name: path.read_bytes() for path in (self.scratch / 'shards').iterdir()}

    def holding(self, pair: tuple[str, str]) -> list[str]:
        key = self.key()
        return sorted(name for name, spec in key['shards'].items()
                      if any(_triple(key['items'][item])[:2] == pair for item in spec['items']))

    def rate(self, overrides: dict[tuple[str, tuple[str, str]], Rule] | None = None) -> None:
        """A ratings file for each shard without one; *overrides* maps (role, pair) to a rule."""
        key = self.key()
        texts = {_triple(row): row for row in json.loads((self.scratch / 'texts.json').read_text())}
        ratings = self.scratch / 'ratings'
        ratings.mkdir(exist_ok=True)
        for name, spec in key['shards'].items():
            path = ratings / f'{name}.jsonl'
            if path.exists():
                continue
            lines = []
            for item_id in spec['items']:
                triple = _triple(key['items'][item_id])
                rule = (overrides or {}).get((spec['role'], triple[:2]), _kept)
                lines.append({'id': item_id, **rule(texts[triple]), 'reason': 'names both claims'})
            path.write_text(''.join(json.dumps(line) + '\n' for line in lines))

    def jsonl(self, name: str) -> list[dict[str, Any]]:
        path = self.out / f'write_triage_eval_set_v2_{name}'
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def summary(self) -> dict[str, Any]:
        return json.loads((self.out / 'write_triage_eval_set_v2_summary.json').read_text())


@pytest.fixture
def evalset(tmp_path: Path) -> _EvalSet:
    built = _EvalSet(tmp_path)
    assert built.run('shards') == 0
    return built


class TestTheShardsCommand:
    def test_every_top_five_pair_is_in_one_primary_and_one_second_shard(
        self, evalset: _EvalSet,
    ) -> None:
        key = evalset.key()
        assert {_triple(item)[:2] for item in key['items'].values()} == ALL_PAIRS
        assert {_triple(item)[2] for item in key['items'].values()} == {'current'}
        for pair in ALL_PAIRS:
            assert sorted(key['shards'][name]['role'] for name in evalset.holding(pair)) == [
                'primary', 'second'], pair

    def test_writes_the_texts_the_rater_prompt_and_a_gitignore_for_both(
        self, evalset: _EvalSet,
    ) -> None:
        texts = json.loads((evalset.scratch / 'texts.json').read_text())
        assert {(row['entry_id'], row['target_id']) for row in texts} == ALL_PAIRS
        assert (evalset.scratch / 'rater_prompt.md').read_text() == _rounds().rater_prompt()
        ignored = (evalset.scratch / '.gitignore').read_text().split()
        assert set(ignored) == {'shards/', 'texts.json', 'rater_prompt.md'}

    def test_a_rerun_adds_nothing_and_renders_a_deleted_shard_again(
        self, evalset: _EvalSet,
    ) -> None:
        key_bytes, shard_files = (evalset.scratch / 'key.json').read_bytes(), evalset.shard_files()
        (evalset.scratch / 'shards' / 'primary_00.txt').unlink()
        (evalset.scratch / 'texts.json').unlink()
        assert evalset.run('shards') == 0
        assert (evalset.scratch / 'key.json').read_bytes() == key_bytes
        assert evalset.shard_files() == shard_files
        assert (evalset.scratch / 'texts.json').exists()

    def test_refuses_a_scratch_inside_the_snapshots_directory(self, tmp_path: Path) -> None:
        built = _EvalSet(tmp_path)
        built.scratch = built.snapshot.parent / 'rounds'
        with pytest.raises(SystemExit):
            built.run('shards')
        assert not built.scratch.exists()


class TestTheRounds:
    def test_reask_supersedes_a_contract_breaking_line_and_an_unfound_quote(
        self, evalset: _EvalSet,
    ) -> None:
        evalset.rate({('primary', ('o1', 'a')): _lacks_loss,
                      ('second', ('l1', 'f')): _unfound_quote})
        with pytest.raises(ValueError, match='reask'):
            evalset.assemble()
        assert evalset.run('reask') == 0
        key = evalset.key()
        reasks = {name: spec for name, spec in key['shards'].items() if name.startswith('reask_')}
        assert {(spec['rater'], len(spec['items'])) for spec in reasks.values()} == {
            ('opus-primary', 1), ('opus-second', 1)}
        evalset.rate()
        assert evalset.assemble() == 0
        batches = {(row['entry_id'], row['target_id'], row['rater']): row['batch']
                   for row in evalset.jsonl('verdicts.jsonl')}
        assert batches['o1', 'a', 'opus-primary'].startswith('reask_')
        assert batches['l1', 'f', 'opus-second'].startswith('reask_')
        assert evalset.summary()['reasked'] == 2

    def test_third_routes_only_ties_and_open_follow_ups_with_a_new_rater_each_round(
        self, evalset: _EvalSet,
    ) -> None:
        tied, open_pair = ('o1', 'a'), ('l2', 'Q')
        evalset.rate({('primary', tied): _conflict_yes, ('second', tied): _conflict_no,
                      ('primary', open_pair): _conflict_unclear,
                      ('second', open_pair): _home_unclear})
        assert evalset.run('third') == 0
        key = evalset.key()
        assert {_triple(key['items'][item])[:2] for item in key['shards']['third_00']['items']} == {
            tied, open_pair}
        assert key['shards']['third_00']['rater'] == 'opus-third-00'
        evalset.rate({('third', tied): _conflict_unclear})
        assert evalset.run('third') == 0
        key = evalset.key()
        assert {_triple(key['items'][item])[:2] for item in key['shards']['third_01']['items']} == {
            tied}
        assert key['shards']['third_01']['rater'] == 'opus-third-01'
        evalset.rate({('third', tied): _kept})
        shard_files = evalset.shard_files()
        assert evalset.run('third') == 0
        assert evalset.shard_files() == shard_files
        assert evalset.assemble() == 0

    def test_third_refuses_an_unrated_decisive_binary(self, evalset: _EvalSet) -> None:
        unrated = ('o2', 'e')
        evalset.rate({('primary', unrated): _home_only, ('second', unrated): _home_only})
        with pytest.raises(ValueError, match=r"'o2'.*conflict"):
            evalset.run('third')

    def test_assemble_refuses_an_incomplete_pair_naming_each_gap(self, evalset: _EvalSet) -> None:
        tied = ('o1', 'b')
        evalset.rate({('primary', tied): _conflict_yes, ('second', tied): _conflict_no})
        with pytest.raises(ValueError, match=r'tied o1 b conflict'):
            evalset.assemble()
        assert not evalset.out.exists()


class TestTheAssembledEvalSet:
    @pytest.fixture
    def assembled(self, evalset: _EvalSet) -> _EvalSet:
        evalset.rate({('primary', ('o1', 'c')): _loss_yes, ('second', ('o1', 'c')): _loss_yes,
                      ('primary', ('l1', 'g')): _home_unclear,
                      ('second', ('l1', 'g')): _home_unclear})
        assert evalset.assemble() == 0
        return evalset

    def test_writes_one_row_per_write(self, assembled: _EvalSet) -> None:
        rows = assembled.jsonl('writes.jsonl')
        assert [row['entry_id'] for row in rows] == ['l1', 'l2', 'o1', 'o2']
        assert {(row['entry_id'], target['target_id']) for row in rows
                for target in row['targets']} == ALL_PAIRS

    def test_verdicts_are_every_effective_rater_line_with_digest_and_no_derived_row(
        self, assembled: _EvalSet,
    ) -> None:
        verdicts = assembled.jsonl('verdicts.jsonl')
        _rubric().resolve_judgements(verdicts)
        assert sorted(((row['entry_id'], row['target_id']), row['role']) for row in verdicts) == sorted(
            (pair, role) for pair in ALL_PAIRS for role in ('primary', 'second'))
        order = [(row['entry_id'], row['target_id'], row['basis'], row['rater']) for row in verdicts]
        assert order == sorted(order)
        loss = [row for row in verdicts if row.get('loss') == 'YES']
        assert len(loss) == 2 and all(row['digest'] == 'YES' for row in loss)
        assert [row for row in verdicts if 'digest' in row and row.get('loss') != 'YES'] == []

    def test_the_summary_reports_labels_kappa_volume_strata_and_inputs(
        self, assembled: _EvalSet,
    ) -> None:
        summary = assembled.summary()
        assert summary['labels'] == {'decided': 9, 'unclear': 1}
        assert summary['unclear_pairs'] == [
            {'entry_id': 'l1', 'target_id': 'g', 'binaries': ['home']}]
        assert summary['pairs'] == {'offline': 6, 'live': 4}
        assert summary['full_label_outcome'] == 'restated'
        assert summary['rating_volume'] == {'current': {'primary': 10, 'second': 10}}
        assert summary['reasked'] == 0
        assert set(summary['shards']) == set(assembled.key()['shards'])
        assert summary['strata'] == {'offline-write-time': 2, 'live-solar_challenge': 760,
                                     'live-solar_challenge_platform': 553}
        _, snapshot_sha = _freeze().load_snapshot(assembled.snapshot)
        assert summary['inputs']['snapshot_sha256'] == snapshot_sha
        assert summary['inputs']['live_slates_sha256'] == _sha256(assembled.live)

    def test_kappa_is_over_each_decided_pairs_consumed_answers(self, assembled: _EvalSet) -> None:
        verdicts = assembled.jsonl('verdicts.jsonl')
        rubric = _rubric()
        derivations = _mod().pair_derivations(assembled.jsonl('writes.jsonl'), verdicts)
        consumed = [(rubric.PairKey(entry, target, rubric.Basis.CURRENT), binary)
                    for (entry, target), derivation in derivations.items()
                    if isinstance(derivation, rubric.Decided)
                    for _, binary in derivation.consumed]
        expected = _score().kappa_over_consumed(consumed, rubric.resolve_judgements(verdicts))
        assert assembled.summary()['kappa'] == json.loads(json.dumps(expected))
        assert assembled.summary()['kappa']['by_binary']['home']['n_consumed'] == 9


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split_row(entry: str, source: str, project: str, created_at: str) -> dict[str, Any]:
    return {'entry_id': entry, 'source': source, 'project': project,
            'category': 'procedural_knowledge', 'created_at': created_at,
            'stratum': f'{source}-{project}', 'production_outcome': 'stored',
            'production_target': None,
            'targets': [{'target_id': f'{entry}-t', 'rank': 1, 'cosine': 0.7,
                         'edited_after_write': None}]}


def _split_verdicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_vote(row['entry_id'], f'{row["entry_id"]}-t', role, **KEPT)
            for row in rows for role in ('primary', 'second')]


def _day(day: int, hour: int) -> str:
    return f'2026-10-{day:02d}T{hour:02d}:30:00+00:00'


def _split_writes() -> list[dict[str, Any]]:
    """40 offline writes, 50 held-out live writes, and 60 other live writes a day from 10-12 to 10-18."""
    rows = [_split_row(f'off-{index:03d}', 'offline', 'dark_factory', _day(1 + index % 4, 3))
            for index in range(40)]
    rows += [_split_row(f'scp-{index:03d}', 'live', 'solar_challenge_platform',
                        _day(12 + index % 7, 5)) for index in range(50)]
    rows += [_split_row(f'live-{day}-{index:02d}', 'live', ('reify', 'solar_challenge')[index % 2],
                        _day(day, index % 24)) for day in range(12, 19) for index in range(60)]
    return sorted(rows, key=lambda row: row['entry_id'])


def _ids(split: dict[str, Any], name: str) -> set[str]:
    return set(split['splits'][name])


class TestTheSplit:
    def _frozen(self, rows: list[dict[str, Any]] | None = None) -> tuple[Any, list[Any], list[Any]]:
        writes = rows or _split_writes()
        verdicts = _split_verdicts(writes)
        return _mod().freeze_split(writes, verdicts), writes, verdicts

    def test_the_constants(self) -> None:
        assert _mod().HELD_OUT_PROJECT == 'solar_challenge_platform'
        assert _mod().TEST_TARGET_WRITES == 300

    def test_test_holds_the_held_out_project_and_every_live_write_after_the_cutoff(self) -> None:
        split, writes, _ = self._frozen()
        by_id = {row['entry_id']: row for row in writes}
        assert split['held_out_project'] == 'solar_challenge_platform'
        assert split['cutoff'] == '2026-10-15T00:00:00+00:00'
        expected = {row['entry_id'] for row in writes if row['source'] == 'live' and (
            row['project'] == 'solar_challenge_platform' or row['created_at'] >= '2026-10-15')}
        assert _ids(split, 'test') == expected
        assert len(expected) == 50 + 4 * 60
        assert all(by_id[entry]['source'] == 'live' for entry in _ids(split, 'test'))

    def test_a_tie_between_two_midnights_goes_to_the_later_one(self) -> None:
        rows = [_split_row('first', 'live', 'reify', _day(15, 23))]
        rows += [_split_row(f'a{index:03d}', 'live', 'reify', _day(16, 6)) for index in range(40)]
        rows += [_split_row(f'b{index:03d}', 'live', 'reify', _day(17, 6)) for index in range(280)]
        split, _, _ = self._frozen(rows)
        assert split['cutoff'] == '2026-10-17T00:00:00+00:00'
        assert len(_ids(split, 'test')) == 280

    def test_tuning_is_dev_and_train_halves_in_seeded_hash_order(self) -> None:
        import hashlib

        split, writes, _ = self._frozen()
        tuning = sorted((row['entry_id'] for row in writes if row['entry_id'] not in _ids(split, 'test')),
                        key=lambda entry: hashlib.sha256(f'7160:{entry}'.encode()).hexdigest())
        half = len(tuning) // 2
        assert _ids(split, 'dev') == set(tuning[:half])
        assert _ids(split, 'train') == set(tuning[half:])

    def test_the_splits_are_disjoint_and_cover_every_write(self) -> None:
        split, writes, _ = self._frozen()
        names = ('test', 'dev', 'train')
        assert sum(len(split['splits'][name]) for name in names) == len(writes)
        assert set().union(*(_ids(split, name) for name in names)) == {
            row['entry_id'] for row in writes}
        assert all(split['splits'][name] == sorted(split['splits'][name]) for name in names)

    def test_counts_each_split_by_source_and_project(self) -> None:
        split, _, _ = self._frozen()
        assert split['counts']['test'] == {'live': {'reify': 120, 'solar_challenge': 120,
                                                    'solar_challenge_platform': 50}}
        assert sum(sum(by_project.values()) for name in ('dev', 'train')
                   for by_project in split['counts'][name].values()) == 40 + 3 * 60

    def test_is_deterministic_and_label_blind(self) -> None:
        split, writes, verdicts = self._frozen()
        assert _mod().freeze_split(writes, verdicts) == split
        relabelled = [
            {**{key: value for key, value in row.items() if key != 'loss'},
             'home': 'NO', 'conflict': 'NO'} if row['role'] == 'primary' else row
            for row in verdicts
        ]
        again = _mod().freeze_split(writes, relabelled)
        assert (again['splits'], again['cutoff'], again['counts']) == (
            split['splits'], split['cutoff'], split['counts'])

    def test_carries_its_rule_and_the_digests_it_froze(self) -> None:
        import hashlib

        split, writes, verdicts = self._frozen()
        assert 'solar_challenge_platform' in split['rule'] and '300' in split['rule']

        def body(rows: list[dict[str, Any]]) -> bytes:
            return ''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows).encode()

        test = _ids(split, 'test')
        assert split['writes_sha256'] == hashlib.sha256(body(writes)).hexdigest()
        assert split['verdicts_sha256'] == hashlib.sha256(body(verdicts)).hexdigest()
        assert split['test_sha256'] == hashlib.sha256(
            body([row for row in writes if row['entry_id'] in test])
            + body([row for row in verdicts if row['entry_id'] in test])).hexdigest()

    def test_verify_passes_the_frozen_inputs_and_tolerates_a_tuning_relabel(self) -> None:
        split, writes, verdicts = self._frozen()
        _mod().verify_split(split, writes, verdicts)
        tuning = next(iter(_ids(split, 'train')))
        edited = [{**row, 'reason': 'refined'} if row['entry_id'] == tuning else row
                  for row in verdicts]
        _mod().verify_split(split, writes, edited)

    def test_verify_refuses_a_verdict_edit_on_a_test_pair(self) -> None:
        split, writes, verdicts = self._frozen()
        test = next(iter(_ids(split, 'test')))
        edited = [{**row, 'reason': 'refined'} if row['entry_id'] == test else row
                  for row in verdicts]
        with pytest.raises(ValueError, match='test_sha256'):
            _mod().verify_split(split, writes, edited)

    def test_verify_refuses_moved_ids(self) -> None:
        split, writes, verdicts = self._frozen()
        moved = [{**row, 'created_at': _day(18, 1)} if row['entry_id'] == 'live-12-00' else row
                 for row in writes]
        with pytest.raises(ValueError, match='live-12-00'):
            _mod().verify_split(split, moved, verdicts)

    def test_tuning_writes_never_yield_a_test_write(self) -> None:
        split, writes, _ = self._frozen()
        tuning = [row['entry_id'] for row in _mod().tuning_writes(writes, split)]
        assert set(tuning) == _ids(split, 'dev') | _ids(split, 'train')
        assert not set(tuning) & _ids(split, 'test')


class TestTheSplitCommand:
    def _committed(self, root: Path) -> Path:
        out = root / 'calibration'
        out.mkdir()
        writes = _split_writes()
        for name, rows in (('writes', writes), ('verdicts', _split_verdicts(writes))):
            (out / f'write_triage_eval_set_v2_{name}.jsonl').write_text(
                ''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
        return out

    def test_writes_the_split_and_refuses_to_overwrite_it_without_refreeze(
        self, tmp_path: Path,
    ) -> None:
        out = self._committed(tmp_path)
        path = out / 'write_triage_eval_set_v2_split.json'
        assert _mod().main(['split', '--eval-dir', str(out)]) == 0
        frozen = json.loads(path.read_text())
        assert frozen['cutoff'] == '2026-10-15T00:00:00+00:00'
        path.write_text('{}')
        with pytest.raises(SystemExit):
            _mod().main(['split', '--eval-dir', str(out)])
        assert path.read_text() == '{}'
        assert _mod().main(['split', '--eval-dir', str(out), '--refreeze']) == 0
        assert json.loads(path.read_text()) == frozen


class TestTheOpusCorpusIsOpusOnly:
    def test_assemble_refuses_a_cross_family_row_naming_it(self, evalset: _EvalSet) -> None:
        evalset.rate()
        key = evalset.key()
        [item] = key['shards']['primary_00']['items'][:1]
        key['shards']['cross_00'] = {'role': 'cross_family', 'rater': 'gpt-6.1-sol:high',
                                     'round': 0, 'items': [item]}
        (evalset.scratch / 'key.json').write_text(json.dumps(key))
        (evalset.scratch / 'ratings' / 'cross_00.jsonl').write_text(
            json.dumps({'id': item, **KEPT, 'reason': 'r'}) + '\n')
        with pytest.raises(ValueError, match=r'cross_family.*cross_00'):
            evalset.assemble()
        assert not evalset.out.exists()
