"""The link adjudicator's eval against blind majority verdicts (task 6184, PRD H3).

The pure core is tested through the script's public functions, loaded by path.
The end-to-end cases drive ``run(argv, env)`` against the in-process server of
``_link_heal_harness``, run the script as a subprocess over the committed λ
fixtures, and feed its report to the readiness gate Γ_A runs.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import pytest_asyncio
from _fm_helpers import load_script_module
from _link_heal_harness import (
    ALL_LINK_HEAL_PREFIXES,
    DF,
    LinkHealHarness,
    build_harness,
)

from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.maintenance.link_adjudicator import AdjudicationFailure, LinkVerdict
from fused_memory.maintenance.link_heal import Verdict
from fused_memory.maintenance.link_heal_store import text_sha256
from fused_memory.server.grouped_read import SIGHTING_KIND

FUSED_MEMORY = Path(__file__).resolve().parent.parent
SCRIPT_PATH = FUSED_MEMORY / 'scripts' / 'eval_link_adjudicator.py'
GATE_SCRIPT = FUSED_MEMORY.parent / 'scripts' / 'check_write_triage_readiness_gate.py'
TRIAGE_VERDICTS = FUSED_MEMORY / 'tests' / 'fixtures' / 'link_adjudicator_triage_verdicts.jsonl'
TRIAGE_PAIRS = FUSED_MEMORY / 'tests' / 'fixtures' / 'link_adjudicator_triage_pairs.jsonl'

#: Gate Γ_A's six bounds, copied verbatim from task 6186's metadata.before_done.args,
#: which is their home.
GAMMA_A_REQUIRES = (
    ('r', 'selection.misfile_recall', '>=', '0.60'),
    ('r', 'selection.false_detach_rate', '<=', '0.02'),
    ('r', 'selection.corrects_recall', '>=', '0.60'),
    ('r', 'selection.false_corrects_rate', '<=', '0.10'),
    ('r', 'selection.parse_failure_rate', '<=', '0.05'),
    ('r', 'population.n_pairs', '>=', '300'),
)

ev = load_script_module(SCRIPT_PATH, mod_name='eval_link_adjudicator')

FIGURES = (
    'misfile_recall',
    'false_detach_rate',
    'corrects_recall',
    'false_corrects_rate',
    'parse_failure_rate',
)


def _item(key: str, truth, *, kind: str | None = None, majority: Verdict | None = None):
    return ev.ScoredItem(
        key=key,
        child_text=f'child {key}',
        parent_text=f'parent {key}',
        truth=truth,
        kind_at_rating=kind,
        majority=majority,
    )


def _said(key: str, verdict: Verdict) -> LinkVerdict:
    return LinkVerdict(key, 'arm', verdict=verdict, reason='r')


def _failed(key: str) -> LinkVerdict:
    return LinkVerdict(key, 'arm', failure=AdjudicationFailure.PARSE_FAILURE, detail='d')


T = ev.TruthClass


class TestTruthClass:
    @pytest.mark.parametrize(
        ('verdict', 'truth'),
        [
            (Verdict.RELATED, T.MISFILE),
            (Verdict.UNRELATED, T.MISFILE),
            (Verdict.CORRECTS, T.CORRECTS),
            (Verdict.SAME, T.AGREEING),
            (Verdict.EXTENDS, T.AGREEING),
            (Verdict.SUBSUMED, T.AGREEING),
            (Verdict.UNCLEAR, T.UNCLEAR),
        ],
    )
    def test_every_verdict_word_has_one_truth_class(self, verdict, truth):
        assert ev.truth_class(verdict) is truth


class TestScoreArm:
    def _scored(self) -> dict:
        items = [
            _item('m1', T.MISFILE), _item('m2', T.MISFILE), _item('m3', T.MISFILE),
            _item('c1', T.CORRECTS), _item('c2', T.CORRECTS),
            _item('a1', T.AGREEING), _item('a2', T.AGREEING), _item('a3', T.AGREEING),
            _item('u1', T.UNCLEAR),
        ]
        verdicts = [
            _said('m1', Verdict.RELATED), _said('m2', Verdict.UNRELATED), _failed('m3'),
            _said('c1', Verdict.CORRECTS), _failed('c2'),
            _said('a1', Verdict.RELATED), _said('a2', Verdict.CORRECTS), _said('a3', Verdict.SAME),
            _said('u1', Verdict.SAME),
        ]
        return ev.score_arm(items, verdicts)

    def test_the_five_figures_are_confusion_counts_against_the_majority(self):
        scored = self._scored()

        assert (scored['misfile_recall_num'], scored['misfile_recall_den']) == (2, 3)
        assert (scored['false_detach_rate_num'], scored['false_detach_rate_den']) == (1, 5)
        assert (scored['corrects_recall_num'], scored['corrects_recall_den']) == (1, 2)
        assert (scored['false_corrects_rate_num'], scored['false_corrects_rate_den']) == (1, 3)
        assert (scored['parse_failure_rate_num'], scored['parse_failure_rate_den']) == (2, 9)
        assert scored['misfile_recall'] == pytest.approx(2 / 3)
        assert scored['parse_failure_rate'] == pytest.approx(2 / 9)

    def test_each_figure_carries_its_counts_and_a_wilson_interval(self):
        scored = self._scored()

        for name in FIGURES:
            assert isinstance(scored[name], float)
            assert type(scored[f'{name}_num']) is int
            assert type(scored[f'{name}_den']) is int
            low, high = scored[f'{name}_ci95']
            assert 0.0 <= low <= scored[name] <= high <= 1.0

    def test_a_zero_denominator_gives_none_not_zero(self):
        scored = ev.score_arm([_item('a1', T.AGREEING)], [_said('a1', Verdict.SAME)])

        assert scored['misfile_recall'] is None
        assert scored['misfile_recall_ci95'] is None
        assert scored['misfile_recall_den'] == 0
        assert scored['corrects_recall'] is None
        assert scored['false_detach_rate'] == 0.0


class TestWilson95:
    def test_nine_of_fifteen(self):
        low, high = ev.wilson95(9, 15)

        assert low == pytest.approx(0.357, abs=1e-3)
        assert high == pytest.approx(0.802, abs=1e-3)

    def test_no_trials_is_none(self):
        assert ev.wilson95(0, 0) is None


class TestKindAgreement:
    def test_an_arm_agrees_when_it_heals_the_kind_the_majority_heals(self):
        items = [
            _item('s1', T.AGREEING, kind='sighting', majority=Verdict.EXTENDS),
            _item('s2', T.AGREEING, kind='sighting', majority=Verdict.SAME),
            _item('h1', T.MISFILE, kind=None, majority=Verdict.RELATED),
            _item('h2', T.AGREEING, kind='correction', majority=Verdict.EXTENDS),
            _item('am', T.AGREEING, kind='amendment', majority=Verdict.EXTENDS),
            _item('pe', T.AGREEING, kind='peer', majority=Verdict.SAME),
        ]
        verdicts = [
            _said('s1', Verdict.EXTENDS),
            _said('s2', Verdict.EXTENDS),
            _said('h1', Verdict.UNRELATED),
            _failed('h2'),
            _said('am', Verdict.RELATED),
            _said('pe', Verdict.RELATED),
        ]

        agreement = ev.kind_agreement(items, verdicts)

        assert (agreement['kind_agreement_num'], agreement['kind_agreement_den']) == (2, 4)
        assert agreement['kind_agreement'] == pytest.approx(0.5)
        baseline = (
            agreement['kind_agreement_always_amendment_baseline_num'],
            agreement['kind_agreement_always_amendment_baseline_den'],
        )
        assert baseline == (2, 4)


class TestSelectArm:
    def _row(self, arm: str, recall, detach, corrects, cost: float) -> dict:
        return {
            'arm': arm,
            'misfile_recall': recall,
            'false_detach_rate': detach,
            'false_corrects_rate': corrects,
            'cost_usd': cost,
            'nested': {'arm': arm},
        }

    def test_the_highest_misfile_recall_wins(self):
        rows = [self._row('a', 0.5, 0.0, 0.0, 1.0), self._row('b', 0.8, 0.1, 0.1, 9.0)]

        assert ev.select_arm(rows)['arm'] == 'b'

    @pytest.mark.parametrize(
        ('rows', 'winner'),
        [
            ([('a', 0.8, 0.02, 0.0, 1.0), ('b', 0.8, 0.01, 0.5, 9.0)], 'b'),
            ([('a', 0.8, 0.01, 0.10, 1.0), ('b', 0.8, 0.01, 0.05, 9.0)], 'b'),
            ([('a', 0.8, 0.01, 0.05, 2.0), ('b', 0.8, 0.01, 0.05, 1.0)], 'b'),
            ([('a', None, 0.0, 0.0, 0.0), ('b', 0.1, 0.9, 0.9, 9.0)], 'b'),
            ([('a', 0.8, None, 0.0, 0.0), ('b', 0.8, 0.9, 0.9, 9.0)], 'b'),
        ],
    )
    def test_ties_break_on_detach_then_flag_then_cost_and_none_ranks_worst(self, rows, winner):
        assert ev.select_arm([self._row(*row) for row in rows])['arm'] == winner

    def test_the_selection_is_a_deep_copy_of_the_chosen_row(self):
        rows = [self._row('a', 0.5, 0.0, 0.0, 1.0)]

        selection = ev.select_arm(rows)

        assert selection == rows[0]
        selection['nested']['arm'] = 'changed'
        assert rows[0]['nested']['arm'] == 'a'


def _vote(entry: str, target: str, verdict: str, rater: str = 'r0') -> dict:
    return {'entry_id': entry, 'target_id': target, 'verdict': verdict, 'rater': rater, 'batch': 'b'}


def _pair(entry: str, target: str, entry_text: str | None = 'child', target_text: str | None = 'parent') -> dict:
    return {'entry_id': entry, 'entry_text': entry_text, 'target_id': target, 'target_text': target_text}


class TestTriageItems:
    def test_votes_resolve_to_truth_and_join_their_texts(self):
        votes = [
            _vote('e1', 't1', 'RELATED'),
            _vote('e2', 't2', 'RELATED'), _vote('e2', 't2', 'EXTENDS', rater='r1'),
            _vote('e3', 't3', 'SAME'),
            _vote('e4', 't4', 'EXTENDS'),
            _vote('e5', 't5', 'CORRECTS'),
            _vote('e6', 't6', 'UNCLEAR'),
        ]
        pairs = [
            _pair('e1', 't1', 'child one', 'parent one'),
            _pair('e2', 't2'),
            _pair('e4', 't4', target_text=None),
            _pair('e5', 't5'),
            _pair('e6', 't6'),
            _pair('e7', 't7'),
        ]

        population = ev.triage_items(votes, pairs)

        truths = {item.key: item.truth for item in population.items}
        assert truths == {'e1:t1': T.MISFILE, 'e5:t5': T.CORRECTS, 'e6:t6': T.UNCLEAR}
        first = next(item for item in population.items if item.key == 'e1:t1')
        assert (first.child_text, first.parent_text) == ('child one', 'parent one')
        assert dict(population.excluded) == {
            ev.Exclusion.TIED: 1,
            ev.Exclusion.NO_TEXTS: 2,
            ev.Exclusion.UNRATED: 1,
        }

    @pytest.mark.parametrize('missing', ['entry_id', 'target_id', 'entry_text', 'target_text'])
    def test_a_pairs_row_without_a_key_is_refused(self, missing):
        row = _pair('e1', 't1')
        del row[missing]

        with pytest.raises(ValueError, match=missing):
            ev.triage_items([_vote('e1', 't1', 'SAME')], [row])

    def test_a_pair_listed_twice_is_refused(self):
        pairs = [_pair('e1', 't1', 'child one'), _pair('e1', 't1', 'child again')]

        with pytest.raises(ValueError, match='twice'):
            ev.triage_items([_vote('e1', 't1', 'SAME')], pairs)


class TestBuildReport:
    def test_the_report_states_its_population_arms_selection_and_provenance(self):
        population = ev.Population(
            items=(
                _item('m', T.MISFILE), _item('c', T.CORRECTS),
                _item('a', T.AGREEING), _item('u', T.UNCLEAR),
            ),
            excluded={ev.Exclusion.TIED: 2, ev.Exclusion.UNRATED: 1},
        )
        arm = {'arm': 'fake', 'misfile_recall': 1.0, 'false_detach_rate': 0.0,
               'false_corrects_rate': 0.0, 'cost_usd': 0.0}

        report = ev.build_report(
            population, [arm],
            mode=ev.Mode.TRIAGE,
            corpus_sha256='c' * 64,
            pairs_sha256='p' * 64,
            brief_sha256='b' * 64,
            field_chars=4000,
            shard_size=40,
        )

        assert report['population'] == {
            'n_pairs': 4,
            'n_misfile': 1,
            'n_corrects': 1,
            'n_belongs': 2,
            'n_unclear': 1,
            'excluded': 3,
            'excluded_by_reason': {'tied': 2, 'unrated': 1},
        }
        assert report['arms'] == [arm]
        assert report['selection'] == arm
        provenance = report['provenance']
        assert provenance['mode'] == 'triage'
        assert provenance['corpus_sha256'] == 'c' * 64
        assert provenance['pairs_sha256'] == 'p' * 64
        assert provenance['brief_sha256'] == 'b' * 64
        assert (provenance['field_chars'], provenance['shard_size']) == (4000, 40)
        assert provenance['text_keys'] == {'child': 'entry_text', 'parent': 'target_text'}
        assert 'write_triage_pairs_to_rate.jsonl' in provenance['text_key_basis']


@pytest_asyncio.fixture
async def harness(mock_config, tmp_path):
    built = await build_harness(
        mock_config, tmp_path, metadata_patch_prefixes=ALL_LINK_HEAL_PREFIXES,
    )
    yield built
    await built.journal.close()


def eval_env(harness: LinkHealHarness, monkeypatch, tmp_path: Path):
    monkeypatch.setenv('CONFIG_PATH', str(tmp_path / 'missing.yaml'))
    caller = harness.tool_caller()
    return ev.EvalEnv(
        config_loader=lambda _path: FusedMemoryConfig(),
        tool_caller_for=lambda _url: contextlib.nullcontext(caller),
    )


def _hand_link_ids(index: int) -> tuple[str, str]:
    return (f'{index:08x}-e1e1-4e1e-8e1e-{index:012x}', f'{index:08x}-f1f1-4f1f-8f1f-{index:012x}')


def seed_hand_links(harness: LinkHealHarness, count: int) -> list[dict]:
    """*count* rated sightings, live, and the corpus rows rating them at their live hashes."""
    rows = []
    for index in range(count):
        child, parent = _hand_link_ids(index)
        child_text, parent_text = f'eval child {index}', f'eval parent {index}'
        harness.seed_link(
            kind=SIGHTING_KIND, child=child, parent=parent,
            child_text=child_text, parent_text=parent_text,
        )
        rows.append({
            'item_id': f'H{index:03d}',
            'project': DF,
            'entry_id': child,
            'target_id': parent,
            'verdict': 'EXTENDS',
            'kind_at_rating': SIGHTING_KIND,
            'child_sha256': text_sha256(child_text),
            'parent_sha256': text_sha256(parent_text),
            'rated_text_matches_live': True,
        })
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return path


class TestHandLinkMode:
    @pytest.mark.asyncio
    async def test_pairs_whose_live_texts_moved_on_are_excluded_by_reason(
        self, harness, monkeypatch, tmp_path, capsys,
    ):
        rows = seed_hand_links(harness, 4)
        rows[1]['rated_text_matches_live'] = False
        corpus = write_jsonl(tmp_path / 'corpus.jsonl', rows)
        edited_child, _parent = _hand_link_ids(0)
        harness.mem0.payload(DF, edited_child)['data'] = 'eval child 0, edited after export'
        _child, deleted_parent = _hand_link_ids(2)
        del harness.mem0.points[(DF, deleted_parent)]
        out = tmp_path / 'report.json'

        code = await ev.run(
            ['--corpus', str(corpus), '--arms', 'fake', '--out', str(out)],
            eval_env(harness, monkeypatch, tmp_path),
        )

        assert code == 0
        report = json.loads(out.read_text())
        assert report == json.loads(capsys.readouterr().out)
        population = report['population']
        assert population['n_pairs'] == 1
        assert population['excluded'] == 3
        assert population['excluded_by_reason'] == {
            'child_text_changed': 1, 'rated_text_mismatch': 1, 'parent_missing': 1,
        }
        (arm,) = report['arms']
        assert arm['arm'] == 'fake'
        assert arm['kind_agreement_den'] == 1
        assert report['provenance']['mode'] == 'hand_link'

    @pytest.mark.asyncio
    async def test_a_malformed_corpus_is_refused_with_exit_2(
        self, harness, monkeypatch, tmp_path, capsys,
    ):
        corpus = tmp_path / 'corpus.jsonl'
        corpus.write_text('not json\n')
        out = tmp_path / 'report.json'

        code = await ev.run(
            ['--corpus', str(corpus), '--arms', 'fake', '--out', str(out)],
            eval_env(harness, monkeypatch, tmp_path),
        )

        assert code == 2
        assert 'line 1' in capsys.readouterr().err
        assert not out.exists()


def _jsonl_text(*rows: object) -> str:
    return ''.join(json.dumps(row) + '\n' for row in rows)


class TestTriageModeRefusesMalformedInput:
    @pytest.mark.parametrize(
        ('verdicts', 'pairs', 'named'),
        [
            pytest.param(
                _jsonl_text(_vote('e1', 't1', 'SAME')),
                _jsonl_text({'entry_text': 'c', 'target_id': 't1', 'target_text': 'p'}),
                'entry_id',
                id='pairs-row-without-entry-id',
            ),
            pytest.param(
                _jsonl_text(_vote('e1', 't1', 'SAME')),
                _jsonl_text(['e1', 't1']),
                'line 1: not a JSON object',
                id='pairs-line-a-list',
            ),
            pytest.param(
                _jsonl_text(_vote('e1', 't1', 'SAME')),
                _jsonl_text(_pair('e1', 't1'), 'e1'),
                'line 2: not a JSON object',
                id='pairs-line-a-string',
            ),
            pytest.param(
                _jsonl_text(_vote('e1', 't1', 'SAME'), 7),
                _jsonl_text(_pair('e1', 't1')),
                'line 2: not a JSON object',
                id='verdicts-line-a-number',
            ),
            pytest.param(
                _jsonl_text(_vote('e1', 't1', 'SAME')),
                _jsonl_text(_pair('e1', 't1'), _pair('e1', 't1')),
                'twice',
                id='pair-listed-twice',
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_it_exits_2_naming_the_fault(
        self, verdicts, pairs, named, monkeypatch, tmp_path, capsys,
    ):
        monkeypatch.setenv('CONFIG_PATH', str(tmp_path / 'missing.yaml'))
        verdicts_path = tmp_path / 'verdicts.jsonl'
        verdicts_path.write_text(verdicts)
        pairs_path = tmp_path / 'pairs.jsonl'
        pairs_path.write_text(pairs)
        out = tmp_path / 'report.json'

        code = await ev.run(
            [
                '--corpus', str(verdicts_path), '--pairs', str(pairs_path),
                '--arms', 'fake', '--out', str(out),
            ],
            ev.EvalEnv(config_loader=lambda _path: FusedMemoryConfig()),
        )

        assert code == 2
        assert named in capsys.readouterr().err
        assert not out.exists()


class TestArmsArgument:
    @pytest.mark.parametrize('arms', ['', ' , '])
    def test_an_empty_arm_list_is_refused(self, arms):
        with pytest.raises(SystemExit):
            ev.build_parser().parse_args(['--corpus', 'c.jsonl', '--arms', arms])

    def test_arms_are_split_and_de_duplicated_in_order(self):
        args = ev.build_parser().parse_args(['--corpus', 'c.jsonl', '--arms', 'opus, fake,opus'])

        assert args.arms == ['opus', 'fake']


def run_triage_eval(out: Path, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable, 'scripts/eval_link_adjudicator.py',
            '--corpus', str(TRIAGE_VERDICTS.relative_to(FUSED_MEMORY)),
            '--pairs', str(TRIAGE_PAIRS.relative_to(FUSED_MEMORY)),
            '--arms', 'fake',
            '--out', str(out),
        ],
        cwd=FUSED_MEMORY,
        env={**os.environ, 'CONFIG_PATH': str(tmp_path / 'missing.yaml')},
        capture_output=True,
        text=True,
        timeout=240,
    )


@pytest.fixture
def triage_report(tmp_path) -> Path:
    out = tmp_path / 'triage-report.json'
    completed = run_triage_eval(out, tmp_path)
    assert completed.returncode == 0, completed.stderr
    return out


class TestTriageModeAsAUserRunsIt:
    @pytest.mark.timeout(600)
    def test_the_fake_arm_scores_the_committed_fixtures(self, triage_report):
        report = json.loads(triage_report.read_text())

        population = report['population']
        assert type(population['n_pairs']) is int
        assert population['n_pairs'] > 0
        assert population['excluded_by_reason'] == {'no_texts': 2, 'tied': 1, 'unrated': 1}
        assert population['excluded'] == 4
        (arm,) = report['arms']
        assert arm['arm'] == 'fake'
        for name in FIGURES:
            assert isinstance(arm[name], float), name
            for suffix in ('_num', '_den', '_ci95'):
                assert f'{name}{suffix}' in arm
        assert report['selection'] == arm
        assert report['provenance']['mode'] == 'triage'

    @pytest.mark.timeout(600)
    def test_the_fake_arm_is_deterministic(self, triage_report, tmp_path):
        again = tmp_path / 'again.json'
        assert run_triage_eval(again, tmp_path).returncode == 0

        first = json.loads(triage_report.read_text())['arms']
        second = json.loads(again.read_text())['arms']
        assert first == second


class TestTheReadinessGateReadsTheReport:
    @pytest.mark.timeout(600)
    def test_every_gamma_a_bound_resolves_to_a_number(self, triage_report):
        argv = [sys.executable, str(GATE_SCRIPT), '--gate', 'Gamma-A', '--report', 'r', str(triage_report)]
        for require in GAMMA_A_REQUIRES:
            argv += ['--require', *require]

        completed = subprocess.run(argv, capture_output=True, text=True, timeout=60)

        verdict = json.loads(completed.stdout.strip().splitlines()[-2])
        assert len(verdict['checks']) == 6
        for check in verdict['checks']:
            assert check['note'] is None, check
            assert isinstance(check['actual'], (int, float)), check
            assert not isinstance(check['actual'], bool), check
