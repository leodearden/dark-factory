"""Tests for scripts/sitting/preparation.py — the prepared-judgement store bridging the nightly run and the sitting (task 5376)."""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from sitting import preparation as mod
from sitting.gates import UNKNOWN, Fact
from sitting.inventory import decision_key, escalation_key, key_str

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
KEY = escalation_key('/src/dark-factory/data/escalations', 'esc-3881-3')
OPTIONS = (
    mod.Option('A', 'close as ruled', 'task 3881 keeps its retargeted scope; the L2 leaves the queue'),
    mod.Option('B', 'hold for a re-ruling', 'task 3881 stays blocked until Leo rules again'),
)


def _prep(**overrides) -> mod.Preparation:
    fields = {
        'item_key': KEY,
        'question': 'Close esc-3881-3 against the option-C ruling Leo already made?',
        'options': OPTIONS,
        'recommendation': mod.Recommended('A', 'task 3881 description opens RETARGETED 2026-09-01'),
        'on_apply': 'esc-3881-3 resolves close_only; task 3881 x_ruling stamped',
        'prepared_at': '2026-09-26T05:40:00+00:00',
        'prepared_by': 'nightly-fable',
    }
    return mod.Preparation(**{**fields, **overrides})


class TestOption:
    @pytest.mark.parametrize('ramification', ['', '   '])
    def test_an_option_is_never_named_without_its_ramification(self, ramification):
        with pytest.raises(ValueError):
            mod.Option('A', 'close', ramification)


class TestPreparationInvariants:
    def test_a_well_formed_preparation_constructs(self):
        prep = _prep(
            cites=mod.Cites(escalation_ids=('esc-3881-2',), task_ids=('3881',)),
            gate_facts=mod.GateFacts(executed=Fact(held=True, evidence='description rewritten'), pins_recovery=()),
            standing=mod.Standing('hold', 'Leo', mod.TaskStatusIs('4811', ('done',)), 'Leo HOLD 2026-08-27'),
        )

        assert prep.recommendation == mod.Recommended('A', 'task 3881 description opens RETARGETED 2026-09-01')
        assert prep.gate_facts.ruling is UNKNOWN

    def test_no_lean_is_an_explicit_recommendation(self):
        prep = _prep(recommendation=mod.NoLean('both options are reversible and cost the same'))

        assert isinstance(prep.recommendation, mod.NoLean)

    @pytest.mark.parametrize('recommendation', [
        mod.Recommended('C', 'evidence'),
    ])
    def test_the_recommended_label_must_be_one_of_the_options(self, recommendation):
        with pytest.raises(ValueError):
            _prep(recommendation=recommendation)

    @pytest.mark.parametrize('build', [
        lambda: mod.Recommended('A', ''),
        lambda: mod.Recommended('', 'evidence'),
        lambda: mod.NoLean(''),
        lambda: mod.NoLean('  '),
    ])
    def test_a_recommendation_needs_its_evidence_or_reason(self, build):
        with pytest.raises(ValueError):
            build()

    def test_the_recommendation_must_be_one_of_the_two_kinds(self):
        with pytest.raises(ValueError):
            _prep(recommendation='A')  # type: ignore[arg-type]

    @pytest.mark.parametrize('question', [
        'Close it. Then retarget 3881?',
        'Close it? Or hold it?',
        '',
    ])
    def test_the_question_is_a_single_sentence(self, question):
        with pytest.raises(ValueError):
            _prep(question=question)

    @pytest.mark.parametrize('question', [
        'Adopt v1.2 of the gate as "done. finished." says?',
        'Ship the fix described in `a.b. c.`?',
        'Close esc-3881-3?',
    ])
    def test_terminal_marks_inside_quotes_or_mid_token_do_not_count(self, question):
        assert _prep(question=question).question == question

    def test_duplicate_option_labels_are_refused(self):
        with pytest.raises(ValueError):
            _prep(options=(OPTIONS[0], OPTIONS[0]))

    @pytest.mark.parametrize('overrides', [
        {'prepared_at': 'yesterday'},
        {'prepared_by': ''},
        {'on_apply': ''},
        {'item_key': ('esc', 'esc-1-1')},
    ])
    def test_provenance_and_identity_are_required(self, overrides):
        with pytest.raises(ValueError):
            _prep(**overrides)

    def test_a_standing_item_needs_no_apply_line(self):
        standing = mod.Standing('owned', 'task 5255', mod.EscalationClosed('esc-6798-1'), 'x_coalesced_into=5255')

        assert _prep(on_apply='', standing=standing).on_apply == ''

    def test_cites_are_structured_ids(self):
        with pytest.raises(ValueError):
            mod.Cites(escalation_ids=('see esc-1-1',))
        with pytest.raises(ValueError):
            mod.Cites(task_ids=('task 12',))


class TestStanding:
    @pytest.mark.parametrize('kind', ['pin', 'hold', 'leo_owned', 'owned'])
    def test_known_kinds(self, kind):
        assert mod.Standing(kind, 'Leo', mod.Manual('Leo says when'), 'handover §1').kind == kind

    def test_unknown_kind_is_refused(self):
        with pytest.raises(ValueError):
            mod.Standing('parked', 'Leo', mod.Manual('x'), 'evidence')

    def test_release_predicate_is_the_closed_union(self):
        with pytest.raises(ValueError):
            mod.Standing('hold', 'Leo', 'when 4811 lands', 'evidence')  # type: ignore[arg-type]

    def test_empty_evidence_is_refused(self):
        with pytest.raises(ValueError):
            mod.Standing('hold', 'Leo', mod.Manual('x'), '')


class TestJsonPayload:
    def test_round_trip_through_the_documented_shape(self):
        prep = _prep(
            cites=mod.Cites(escalation_ids=('esc-3881-2',), task_ids=('3881',)),
            gate_facts=mod.GateFacts(
                ruling=Fact(held=True, evidence='Leo: option C', source_kind='task_description'),
                pins_recovery=(),
            ),
            standing=mod.Standing('hold', 'Leo', mod.TaskStatusIs('4811', ('done', 'cancelled')), 'handover §1'),
        )

        payload = mod.to_json_payload(prep)

        assert json.loads(json.dumps(payload)) == payload
        assert mod.from_json_payload(payload) == prep

    @pytest.mark.parametrize('release', [
        mod.TaskStatusIs('4811', ('done',)), mod.EscalationClosed('esc-1-1'), mod.Manual('Leo says so'),
    ])
    def test_every_release_predicate_round_trips(self, release):
        prep = _prep(standing=mod.Standing('pin', 'esc-3105-5', release, 'veto-pin-do-not-close:3105'))

        assert mod.from_json_payload(mod.to_json_payload(prep)) == prep

    def test_no_lean_and_decision_keys_round_trip(self):
        prep = _prep(item_key=decision_key('df-esc-1-1'), recommendation=mod.NoLean('evenly balanced'))

        assert mod.from_json_payload(mod.to_json_payload(prep)) == prep

    def test_unknown_keys_are_named_in_the_refusal(self):
        payload = {**mod.to_json_payload(_prep()), 'confidence': 0.9, 'mood': 'good'}

        with pytest.raises(ValueError, match='confidence.*mood|mood.*confidence'):
            mod.from_json_payload(payload)

    def test_unknown_nested_keys_are_named_too(self):
        payload = mod.to_json_payload(_prep())
        payload['options'][0]['risk'] = 'high'

        with pytest.raises(ValueError, match='risk'):
            mod.from_json_payload(payload)

    def test_missing_required_keys_are_named(self):
        payload = mod.to_json_payload(_prep())
        del payload['question']

        with pytest.raises(ValueError, match='question'):
            mod.from_json_payload(payload)


class TestLoad:
    def test_missing_file_is_an_empty_store(self, tmp_path):
        store = mod.load(tmp_path / 'preparation.json')

        assert store.entries == {}
        assert store.get(KEY) is None

    @pytest.mark.parametrize('body', ['{"preparations": [', '["not", "a", "store"]', '{"preparations": [{"x": 1}]}'])
    def test_corrupt_file_is_loud_and_names_the_path(self, tmp_path, body):
        path = tmp_path / 'preparation.json'
        path.write_text(body)

        with pytest.raises(mod.PreparationStoreCorrupt) as caught:
            mod.load(path)

        assert caught.value.path == path
        assert str(path) in str(caught.value)


class TestRecord:
    def test_record_then_load(self, tmp_path):
        path = tmp_path / 'sitting' / 'preparation.json'

        mod.record(path, [mod.to_json_payload(_prep())])

        assert mod.load(path).get(KEY) == _prep()

    def test_newer_prepared_at_wins_per_key(self, tmp_path):
        path = tmp_path / 'preparation.json'
        newer = _prep(prepared_at='2026-09-26T09:00:00+00:00', recommendation=mod.NoLean('re-read'))
        older = _prep(prepared_at='2026-09-25T09:00:00+00:00')

        mod.record(path, [mod.to_json_payload(newer)])
        store = mod.record(path, [mod.to_json_payload(older)])

        assert store.get(KEY) == newer
        assert mod.load(path).get(KEY) == newer

    def test_other_keys_are_kept_on_merge(self, tmp_path):
        path = tmp_path / 'preparation.json'
        other = _prep(item_key=decision_key('lone'))

        mod.record(path, [mod.to_json_payload(_prep())])
        mod.record(path, [mod.to_json_payload(other)])

        assert set(mod.load(path).entries) == {key_str(KEY), key_str(decision_key('lone'))}

    def test_one_invalid_entry_writes_nothing(self, tmp_path):
        path = tmp_path / 'preparation.json'
        mod.record(path, [mod.to_json_payload(_prep())])
        before = path.read_bytes()
        good = mod.to_json_payload(_prep(item_key=decision_key('fresh')))
        bad = {**mod.to_json_payload(_prep(item_key=decision_key('bad'))), 'question': 'Two? Questions?'}

        with pytest.raises(ValueError):
            mod.record(path, [good, bad])

        assert path.read_bytes() == before

    def test_the_write_is_atomic_and_leaves_no_temp_files(self, tmp_path):
        path = tmp_path / 'preparation.json'

        mod.record(path, [mod.to_json_payload(_prep())])

        assert sorted(p.name for p in tmp_path.iterdir()) == ['preparation.json', 'preparation.json.lock']


_RECORD_EACH = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from sitting import preparation
for payload in json.load(sys.stdin):
    preparation.record(Path(sys.argv[1]), [payload])
"""


class TestConcurrentRecord:
    def test_two_processes_lose_no_entry(self, tmp_path):
        # Real interpreters rather than fork(): an xdist worker is multi-threaded, and forking it can deadlock.
        path = tmp_path / 'preparation.json'
        batches = [
            [mod.to_json_payload(_prep(item_key=decision_key(f'w{worker}-{n}'))) for n in range(15)]
            for worker in (1, 2)
        ]
        workers = [
            subprocess.Popen(
                [sys.executable, '-c', _RECORD_EACH, str(path), str(SCRIPTS_DIR)],
                stdin=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            for _ in batches
        ]

        for worker, batch in zip(workers, batches, strict=True):
            assert worker.stdin is not None
            worker.stdin.write(json.dumps(batch))
            worker.stdin.close()
        results = [(worker.wait(timeout=120), worker.stderr.read() if worker.stderr else '') for worker in workers]

        assert [code for code, _ in results] == [0, 0], results
        assert len(mod.load(path).entries) == 30


class TestPrune:
    def test_drops_items_no_longer_open(self):
        closed = decision_key('closed')
        store = mod.PreparationStore({
            key_str(KEY): _prep(),
            key_str(closed): _prep(item_key=closed),
        })

        pruned, dropped = mod.prune(store, [KEY])

        assert set(pruned.entries) == {key_str(KEY)}
        assert dropped == (closed,)


def test_default_path_is_under_the_preparers_own_state_dir():
    assert Path('data', 'sitting', 'preparation.json') == mod.DEFAULT_PREPARATION_PATH


def test_replace_keeps_invariants():
    with pytest.raises(ValueError):
        replace(_prep(), options=())
