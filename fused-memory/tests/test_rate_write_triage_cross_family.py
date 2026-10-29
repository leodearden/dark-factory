"""Tests for rate_write_triage_cross_family.py — the cross-family re-rate of the eval set.

No network: every item goes to a fake async ``ask(instructions, item_text)``
port that answers from a script per pair.
"""
from __future__ import annotations

import asyncio
import functools
import json
import types
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from _fm_helpers import load_script_module

SCRIPTS = Path(__file__).parent.parent / 'scripts'
BATCH = 'cross-family-2026-10-24'


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'rate_write_triage_cross_family.py',
                              mod_name='rate_write_triage_cross_family')


@functools.cache
def _rounds() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'write_triage_rating_rounds.py',
                              mod_name='write_triage_rating_rounds')


@functools.cache
def _stats() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'write_triage_stats.py', mod_name='write_triage_stats')


KEPT = {'home': 'YES', 'conflict': 'NO', 'loss': 'NO'}
DISPUTED = {'home': 'YES', 'conflict': 'YES', 'scope': 'CENTRAL', 'action_changing': 'YES'}
CONTROLS = [f'c{index}' for index in range(1, 6)]


def _vote(entry: str, target: str, role: str, **answers: Any) -> dict[str, Any]:
    row = {'entry_id': entry, 'target_id': target, 'basis': 'current', 'rater': f'opus-{role}',
           'role': role, 'batch': f'{role}_00', 'reason': 'r', **answers}
    if answers.get('conflict') == 'YES':
        row['conflict_quote'] = {'parent': f'parent of {entry}.', 'child': f'child of {entry}.'}
    if 'UNCLEAR' in answers.values():
        row['unclear_reason'] = 'cut'
    return row


def _writes() -> list[dict[str, Any]]:
    pairs = [('w1', 't1'), ('w2', 't2'), *((entry, 't') for entry in CONTROLS), ('u1', 't')]
    return [{'entry_id': entry, 'source': 'live', 'project': 'df',
             'targets': [{'target_id': target, 'rank': 1, 'cosine': 0.7,
                          'edited_after_write': None}]}
            for entry, target in pairs]


def _verdicts() -> list[dict[str, Any]]:
    """w1 splits on HOME and w2 on CONFLICT, each settled by a third; the controls agree; u1 is unclear."""
    rows = [
        _vote('w1', 't1', 'primary', **KEPT), _vote('w1', 't1', 'second', home='NO', conflict='NO'),
        _vote('w1', 't1', 'third', **KEPT),
        _vote('w2', 't2', 'primary', **DISPUTED), _vote('w2', 't2', 'second', **KEPT),
        _vote('w2', 't2', 'third', **DISPUTED),
        _vote('u1', 't', 'primary', home='UNCLEAR', conflict='NO'),
        _vote('u1', 't', 'second', home='UNCLEAR', conflict='NO'),
    ]
    rows += [_vote(entry, 't', role, **KEPT) for entry in CONTROLS for role in ('primary', 'second')]
    return rows


def _entries() -> dict[tuple[str, str], Any]:
    rounds = _rounds()
    return {(row['entry_id'], target['target_id']): rounds.ShardEntry(
        row['entry_id'], target['target_id'], 'current', 'df', f'child of {row["entry_id"]}.',
        f'parent of {row["entry_id"]}.') for row in _writes() for target in row['targets']}


class TestTheSelection:
    def test_discordant_pairs_are_those_whose_consumed_answers_split(self) -> None:
        assert _mod().discordant_pairs(_writes(), _verdicts()) == [('w1', 't1'), ('w2', 't2')]

    def test_control_pairs_are_a_seeded_sample_of_the_concordant_pairs(self) -> None:
        control = _mod().control_pairs(_writes(), _verdicts(), seed=7, n=3)
        assert len(control) == 3 and {entry for entry, _ in control} <= set(CONTROLS)
        assert control == _mod().control_pairs(_writes(), _verdicts(), seed=7, n=3)
        assert {entry for entry, _ in _mod().control_pairs(_writes(), _verdicts(), seed=7, n=10)} == set(
            CONTROLS)

    def test_the_defaults(self) -> None:
        cross = _mod()
        assert (cross.DEFAULT_MODEL, cross.DEFAULT_EFFORT, cross.DEFAULT_BUDGET_USD) == (
            'gpt-6.1-sol', 'high', 15.0)
        assert (cross.CONTROL_N, cross.MAX_REASKS, cross.ITEM_NAME) == (100, 2, 'cross_family')


ITEM_ID = 'cross_family-000'
"""The one item id of a single-pair shard named ITEM_NAME."""


def _reply(**answers: Any) -> str:
    return json.dumps({'id': ITEM_ID, **answers, 'reason': 'names both claims'})


HONEST = _reply(**KEPT)
DISPUTE = _reply(**DISPUTED, conflict_quote={'parent': 'parent of w2.', 'child': 'child of w2.'})
UNFOUND = _reply(**DISPUTED, conflict_quote={'parent': 'parent of w2.', 'child': 'not in it'})


class _Ask:
    """The fake port: each pair answers from its script, in order, then honestly."""

    def __init__(self, scripts: dict[str, list[str]] | None = None,
                 usage: dict[str, int] | None = None) -> None:
        self.scripts = {entry: list(texts) for entry, texts in (scripts or {}).items()}
        self.usage = usage or {'input_tokens': 1_000, 'output_tokens': 100}
        self.calls: list[tuple[str, str]] = []

    def entry_of(self, item_text: str) -> str:
        [entry] = [entry for entry, _ in _entries() if f'child of {entry}.' in item_text]
        return entry

    async def __call__(self, instructions: str, item_text: str) -> Any:
        self.calls.append((instructions, item_text))
        script = self.scripts.get(self.entry_of(item_text)) or []
        return _mod().Reply(text=script.pop(0) if script else HONEST, usage=dict(self.usage))

    def asked(self) -> list[str]:
        return [self.entry_of(item) for _, item in self.calls]


SELECTION = [(('w1', 't1'), 'discordant'), (('w2', 't2'), 'discordant'), (('c1', 't'), 'control')]


def _rate(out: Path, ask: _Ask, *, budget: float = 15.0,
          selection: Sequence[Any] = SELECTION) -> dict[str, Any]:
    return asyncio.run(_mod().rate_pairs(
        selection, _entries(), ask, out=out, model='gpt-6.1-sol', effort='high', batch=BATCH,
        budget_usd=budget,
    ))


def _records(out: Path) -> dict[tuple[str, str], dict[str, Any]]:
    return {(record['entry_id'], record['target_id']): record
            for record in _mod().read_records(out)}


class TestRating:
    def test_each_item_is_the_answer_contract_and_one_pair_rendered_as_the_raters_saw_it(
        self, tmp_path: Path,
    ) -> None:
        ask = _Ask()
        _rate(tmp_path / 'cf.jsonl', ask, selection=SELECTION[:1])
        [(instructions, item)] = ask.calls
        assert instructions == _rounds().answer_contract()
        assert item == _rounds().shard([_entries()['w1', 't1']], name=_mod().ITEM_NAME).text

    def test_a_reply_becomes_a_cross_family_row(self, tmp_path: Path) -> None:
        out = tmp_path / 'cf.jsonl'
        _rate(out, _Ask({'w2': [DISPUTE]}))
        record = _records(out)['w2', 't2']
        assert (record['selection'], record['attempts'], record['failure']) == ('discordant', 1, None)
        row = record['row']
        assert {key: row[key] for key in ('entry_id', 'target_id', 'basis', 'rater', 'role',
                                          'batch')} == {
            'entry_id': 'w2', 'target_id': 't2', 'basis': 'current',
            'rater': 'gpt-6.1-sol:high', 'role': 'cross_family', 'batch': BATCH,
        }
        assert (row['conflict'], row['scope'], 'id' in row) == ('YES', 'CENTRAL', False)

    def test_a_broken_reply_is_asked_again_up_to_twice_then_recorded_as_a_failure(
        self, tmp_path: Path,
    ) -> None:
        out = tmp_path / 'cf.jsonl'
        ask = _Ask({'w1': ['not json', _reply(home='YES', conflict='NO'), HONEST],
                    'w2': [UNFOUND, UNFOUND, UNFOUND]})
        _rate(out, ask)
        records = _records(out)
        assert (records['w1', 't1']['attempts'], records['w1', 't1']['row']['loss']) == (3, 'NO')
        failed = records['w2', 't2']
        assert (failed['attempts'], failed['row']) == (3, None)
        assert 'conflict_quote.child' in failed['failure']
        assert ask.asked().count('w2') == 3

    def test_a_reply_naming_its_own_pair_is_asked_again(self, tmp_path: Path) -> None:
        out = tmp_path / 'cf.jsonl'
        ask = _Ask({'w1': [_reply(**KEPT, entry_id='w2'), HONEST]})
        _rate(out, ask, selection=SELECTION[:1])
        record = _records(out)['w1', 't1']
        assert (record['attempts'], record['row']['entry_id']) == (2, 'w1')

    def test_a_rerun_skips_every_pair_already_recorded(self, tmp_path: Path) -> None:
        out = tmp_path / 'cf.jsonl'
        _rate(out, _Ask(), selection=SELECTION[:1])
        ask = _Ask()
        _rate(out, ask)
        assert sorted(ask.asked()) == ['c1', 'w2']
        assert set(_records(out)) == {pair for pair, _ in SELECTION}

    def test_the_run_stops_before_a_call_that_would_pass_the_budget(self, tmp_path: Path) -> None:
        out = tmp_path / 'cf.jsonl'
        costly = {'input_tokens': 1_000_000, 'output_tokens': 0}
        summary = _rate(out, _Ask(usage=costly), budget=5.0)
        assert summary['stopped'] is True
        assert summary['spent_usd'] == pytest.approx(4.0)
        assert summary['remaining'] == 1
        assert len(_records(out)) == 2
        assert all(record['usd'] == pytest.approx(2.0) for record in _records(out).values())
        ask = _Ask(usage=costly)
        assert _rate(out, ask, budget=5.0)['stopped'] is True
        assert ask.calls == []
        resumed = _rate(out, _Ask(usage=costly), budget=100.0)
        assert (resumed['stopped'], resumed['remaining']) == (False, 0)
        assert resumed['spent_usd'] == pytest.approx(6.0)


def _cross_records() -> list[dict[str, Any]]:
    def record(entry: str, target: str, selection: str, row: dict[str, Any] | None,
               failure: str | None = None) -> dict[str, Any]:
        return {'entry_id': entry, 'target_id': target, 'selection': selection,
                'row': None if row is None else {**_vote(entry, target, 'third', **row),
                                                 'rater': 'gpt-6.1-sol:high',
                                                 'role': 'cross_family', 'batch': BATCH},
                'failure': failure, 'attempts': 1 if failure is None else 3, 'usd': 0.01}

    return [
        record('w1', 't1', 'discordant', KEPT),
        record('w2', 't2', 'discordant', DISPUTED),
        record('c1', 't', 'control', KEPT),
        record('c2', 't', 'control', None, failure='not JSON'),
    ]


V1_ROWS = [{'entry_id': 'w1', 'target_id': 't1', 'verdict': 'RELATED'},
           {'entry_id': 'c1', 'target_id': 't', 'verdict': 'SAME'},
           {'entry_id': 'zz', 'target_id': 't', 'verdict': 'EXTENDS'}]


def _report() -> dict[str, Any]:
    return _mod().report(_cross_records(), _writes(), _verdicts(), V1_ROWS)


def _agreement(agree: int, n: int) -> dict[str, Any]:
    stats = _stats()
    return {'n': n, 'agree': agree, 'rate': stats.rate(agree, n), 'wilson95': stats.wilson95(agree, n)}


class TestTheReport:
    def test_agreement_per_binary_against_the_resolved_label_and_each_opus_rater(self) -> None:
        home = _report()['agreement']['discordant']['home']
        assert home['vs_resolved'] == {**_agreement(2, 2), 'kappa': _stats().cohen_kappa(
            [('YES', 'YES'), ('YES', 'YES')])}
        assert home['vs_primary'] == _agreement(2, 2)
        assert home['vs_second'] == _agreement(1, 2)
        conflict = _report()['agreement']['discordant']['conflict']
        assert conflict['vs_resolved'] == {**_agreement(2, 2), 'kappa': _stats().cohen_kappa(
            [('NO', 'NO'), ('YES', 'YES')])}
        assert conflict['vs_second'] == _agreement(1, 2)
        assert _report()['agreement']['control']['home']['vs_resolved']['n'] == 1

    def test_counts_coverage_failures_and_spend(self) -> None:
        report = _report()
        assert report['coverage'] == {
            'discordant': {'selected': 2, 'rated': 2, 'failed': 0},
            'control': {'rated': 1, 'failed': 1},
        }
        assert report['failures'] == [{'entry_id': 'c2', 'target_id': 't', 'failure': 'not JSON'}]
        assert report['raters'] == ['gpt-6.1-sol:high']
        assert report['spend_usd'] == pytest.approx(0.04)

    def test_the_related_boundary_block(self) -> None:
        boundary = _report()['related_boundary']
        assert boundary['home_split'] == {'n': 1, 'cross_family_yes': 1, 'agree_resolved': 1}
        assert boundary['by_v1_word'] == {
            'RELATED/UNRELATED': {'YES': 1, 'NO': 0, 'UNCLEAR': 0},
            'SAME/SUBSUMED': {'YES': 1, 'NO': 0, 'UNCLEAR': 0},
            'EXTENDS': {'YES': 0, 'NO': 0, 'UNCLEAR': 0},
            'CORRECTS': {'YES': 0, 'NO': 0, 'UNCLEAR': 0},
        }

    def test_renders_deterministically_with_one_row_per_selection_binary_and_rater(self) -> None:
        report = _report()
        markdown = _mod().render_markdown(report)
        assert markdown == _mod().render_markdown(_report())
        block = report['agreement']['discordant']['home']['vs_second']
        assert f'| discordant | home | second | {block["n"]} | {block["agree"]} |' in markdown


class TestTheReportCommand:
    def test_writes_the_json_and_its_rendering(self, tmp_path: Path) -> None:
        records = tmp_path / 'cf.jsonl'
        records.write_text(''.join(json.dumps(record) + '\n' for record in _cross_records()))
        writes, verdicts, v1 = (tmp_path / name for name in ('w.jsonl', 'v.jsonl', 'v1.jsonl'))
        for path, rows in ((writes, _writes()), (verdicts, _verdicts()), (v1, V1_ROWS)):
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        out = tmp_path / 'report'
        assert _mod().main(['report', '--records', str(records), '--writes', str(writes),
                            '--verdicts', str(verdicts), '--v1-verdicts', str(v1),
                            '--out', str(out)]) == 0
        written = json.loads(out.with_suffix('.json').read_text())
        assert written == json.loads(json.dumps(_report()))
        assert out.with_suffix('.md').read_text() == _mod().render_markdown(written)


class _FakeResponses:
    def __init__(self, calls: list[dict[str, Any]]) -> None:
        self.calls = calls

    async def create(self, **request: Any) -> Any:
        self.calls.append(request)
        usage = types.SimpleNamespace(input_tokens=10, output_tokens=2)
        return types.SimpleNamespace(output_text='{}', usage=usage)


class _FakeClient:
    calls: list[dict[str, Any]] = []

    def __init__(self) -> None:
        self.responses = _FakeResponses(_FakeClient.calls)

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None


class TestTheOpenAIPort:
    def test_a_json_object_request_names_json_in_its_input_messages(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The Responses API refuses a json_object format unless an input message says 'json'."""
        import openai
        _FakeClient.calls = []
        monkeypatch.setattr(openai, 'AsyncOpenAI', _FakeClient)
        ask = _mod().openai_ask('gpt-6.1-sol', 'high')
        reply = asyncio.run(ask(_rounds().answer_contract(), 'ITEM WITHOUT THE WORD'))
        [request] = _FakeClient.calls
        assert reply.usage == {'input_tokens': 10, 'output_tokens': 2}
        assert request['text'] == {'format': {'type': 'json_object'}}
        assert request['reasoning'] == {'effort': 'high'}
        messages = request['input']
        assert [message['role'] for message in messages] == ['developer', 'user']
        assert messages[0]['content'] == _rounds().answer_contract()
        assert messages[1]['content'] == 'ITEM WITHOUT THE WORD'
        assert any('json' in message['content'].lower() for message in messages)
