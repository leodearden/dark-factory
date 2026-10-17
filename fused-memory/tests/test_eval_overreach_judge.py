"""Tests for scripts/eval_overreach_judge.py (task 6137, design.md §6.1 and §8 leaf 2).

Both scripts are loaded by path; same-file reuse in load_script_module makes
the eval script's own ``import audit_ruling_overreach`` and this file's load
resolve to one module object, whichever runs first.
"""
from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from _fm_helpers import load_script_module

from fused_memory.services.overreach_guard import PROMPT_VERSION, EdgeJudgement, JudgeEdge
from fused_memory.services.overreach_shadow import (
    SIDECAR_FILENAME,
    JudgedEpisode,
    OverreachShadowStore,
    ShadowOutcome,
)

SCRIPTS = Path(__file__).parent.parent / 'scripts'
SCRIPT_PATH = SCRIPTS / 'eval_overreach_judge.py'

mod = load_script_module(SCRIPT_PATH, mod_name='eval_overreach_judge')
audit = load_script_module(SCRIPTS / 'audit_ruling_overreach.py', mod_name='audit_ruling_overreach')

DF, REIFY = 'dark_factory', 'reify'
RULING, OTHER = 'ruling_lexeme', 'other_decisions'
WEIGHTS = {DF: {RULING: 2.0}, REIFY: {OTHER: 10.0}}


def _minted(edge_uuid: str) -> dict[str, Any]:
    return {
        'edge_uuid': edge_uuid, 'fact': f'fact of {edge_uuid}', 'source_name': 'S',
        'target_name': 'T', 'served': True, 'live_strict': True,
    }


def _row(graph: str, uuid: str, stratum: str, edges: list[str]) -> dict[str, Any]:
    return {
        'graph': graph, 'uuid': uuid, 'stratum': stratum,
        'classifiers': ['category_decisions'], 'created_at': '2026-09-01T00:00:00+00:00',
        'category': 'decisions_and_rationale', 'content': f'the body of episode {uuid}',
        'minted': [_minted(edge) for edge in edges], 'corroborated_count': 0,
    }


def _verdicts_obj(rows: list[dict[str, Any]], labels: dict[tuple[str, str], str]) -> dict:
    return {
        'sample': audit.SampleDefinition().to_dict(),
        'verdicts': [
            {
                'graph': row['graph'], 'episode_uuid': row['uuid'],
                'edge_uuid': edge['edge_uuid'], 'fact': edge['fact'],
                'label': labels[(row['graph'], edge['edge_uuid'])],
                'rationale': f"adjudicated {edge['edge_uuid']}",
            }
            for row in rows for edge in row['minted']
        ],
    }


def _report_obj(weights: dict[str, dict[str, float]]) -> dict:
    return {'adjudicated': {'population_weighted': {'weights': weights}}}


# --------------------------------------------------------------------------- #
# A hand-built confusion over two cells of unequal weight
# --------------------------------------------------------------------------- #
#   cell (dark_factory, ruling_lexeme), w=2:   a1 a2 TP, a3 FP, a4 FN
#   cell (reify, other_decisions),     w=10:  b1 TP, b2 b3 FP, b4 FN (unjudged), b5 TN

SCORE_ROWS = [
    _row(DF, 'A1', RULING, ['a1', 'a2', 'a3', 'a4']),
    _row(REIFY, 'B1', OTHER, ['b1', 'b2', 'b3', 'b4', 'b5']),
]
SCORE_VERDICTS = {
    (DF, 'a1'): 'overreach', (DF, 'a2'): 'overreach', (DF, 'a3'): 'holding',
    (DF, 'a4'): 'overreach',
    (REIFY, 'b1'): 'overreach', (REIFY, 'b2'): 'holding', (REIFY, 'b3'): 'context',
    (REIFY, 'b4'): 'overreach', (REIFY, 'b5'): 'holding',
}
SCORE_JUDGED = {
    (DF, 'a1'): 'overreach', (DF, 'a2'): 'overreach', (DF, 'a3'): 'overreach',
    (DF, 'a4'): 'holding',
    (REIFY, 'b1'): 'overreach', (REIFY, 'b2'): 'overreach', (REIFY, 'b3'): 'overreach',
    (REIFY, 'b5'): 'holding',
}


def _score() -> dict[str, Any]:
    verdict_set = mod.validate(SCORE_ROWS, _verdicts_obj(SCORE_ROWS, SCORE_VERDICTS))
    return mod.score(SCORE_ROWS, verdict_set, SCORE_JUDGED, WEIGHTS)


class TestOneCopyOfTheEstimator:
    def test_the_estimator_is_the_audit_estimator(self) -> None:
        assert mod.weighted_rate is audit.weighted_rate

    def test_the_script_defines_no_estimator_of_its_own(self) -> None:
        tree = ast.parse(SCRIPT_PATH.read_text())
        defined = {
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        assert not defined & {'weighted_rate', '_wilson'}

    def test_the_estimator_is_imported_from_the_audit(self) -> None:
        imported = {
            alias.name
            for node in ast.walk(ast.parse(SCRIPT_PATH.read_text()))
            if isinstance(node, ast.ImportFrom) and node.module == 'audit_ruling_overreach'
            for alias in node.names
        }
        assert 'weighted_rate' in imported


class TestScore:
    def test_precision_is_population_weighted(self) -> None:
        assert _score()['precision'] == audit.weighted_rate([(2.0, 2, 3), (10.0, 1, 3)])

    def test_recall_is_population_weighted(self) -> None:
        assert _score()['recall'] == audit.weighted_rate([(2.0, 2, 3), (10.0, 1, 2)])

    def test_unweighted_reference_figures(self) -> None:
        unweighted = _score()['unweighted']
        assert unweighted['precision'] == audit.rate(3, 6)
        assert unweighted['recall'] == audit.rate(3, 5)

    def test_the_heavy_cell_dominates(self) -> None:
        score = _score()
        assert score['precision']['rate'] != score['unweighted']['precision']['rate']

    def test_confusion_is_reported_per_cell(self) -> None:
        cells = _score()['confusion_by_cell']
        assert (cells[DF][RULING]['tp'], cells[DF][RULING]['fp'], cells[DF][RULING]['fn']) == (2, 1, 1)
        assert (cells[REIFY][OTHER]['tp'], cells[REIFY][OTHER]['fp'], cells[REIFY][OTHER]['fn']) == (1, 2, 1)

    def test_an_unjudged_overreach_edge_is_a_false_negative_and_listed(self) -> None:
        score = _score()
        assert [(u['graph'], u['edge_uuid']) for u in score['unjudged']] == [(REIFY, 'b4')]
        fn = [d for d in score['disagreements'] if d['edge_uuid'] == 'b4']
        assert len(fn) == 1 and fn[0]['judge'] is None

    def test_disagreements_list_every_false_positive_and_negative(self) -> None:
        disagreements = _score()['disagreements']
        assert {(d['graph'], d['edge_uuid'], d['verdict'], d['judge']) for d in disagreements} == {
            (DF, 'a3', 'holding', 'overreach'), (DF, 'a4', 'overreach', 'holding'),
            (REIFY, 'b2', 'holding', 'overreach'), (REIFY, 'b3', 'context', 'overreach'),
            (REIFY, 'b4', 'overreach', None),
        }
        assert all(d['verdict_rationale'] == f"adjudicated {d['edge_uuid']}" for d in disagreements)


def _score_of(precision: tuple[int, int], recall: tuple[int, int]) -> dict[str, Any]:
    return {
        'precision': audit.weighted_rate([(1.0, *precision)]),
        'recall': audit.weighted_rate([(1.0, *recall)]),
    }


class TestTrigger:
    def test_the_thresholds_are_design_6_1s(self) -> None:
        assert (mod.PRECISION_MIN, mod.PRECISION_LOWER_MIN, mod.RECALL_MIN) == (0.80, 0.65, 0.50)

    @pytest.mark.parametrize(('precision', 'recall', 'passes'), [
        pytest.param((80, 100), (50, 100), True, id='both-on-the-boundary'),
        pytest.param((79, 100), (50, 100), False, id='precision-just-under'),
        pytest.param((32, 40), (50, 100), True, id='lower-bound-just-over'),
        pytest.param((24, 30), (50, 100), False, id='lower-bound-just-under'),
        pytest.param((80, 100), (49, 100), False, id='recall-just-under'),
    ])
    def test_offline(self, precision, recall, passes) -> None:
        assert mod.trigger(_score_of(precision, recall), 'offline')['passes'] is passes

    def test_the_boundary_cases_straddle_the_lower_bound(self) -> None:
        assert audit.weighted_rate([(1.0, 32, 40)])['ci'][0] >= 0.65
        assert audit.weighted_rate([(1.0, 24, 30)])['ci'][0] < 0.65

    def test_online_ignores_recall(self) -> None:
        assert mod.trigger(_score_of((80, 100), (0, 100)), 'online')['passes'] is True

    @pytest.mark.parametrize('mode', ['offline', 'online'])
    def test_an_uncomputed_rate_never_passes(self, mode) -> None:
        score = {'precision': audit.weighted_rate([]), 'recall': audit.weighted_rate([])}
        assert mod.trigger(score, mode)['passes'] is False


# --------------------------------------------------------------------------- #
# The CLI, offline
# --------------------------------------------------------------------------- #
# Two cells of unequal weight, each two episodes of three overreach and one
# holding edge: an oracle judge's precision lower bound clears 0.65.

RUN_ROWS = [
    _row(graph, f'{graph}-ep{i}', stratum, [f'{graph}-ep{i}-e{j}' for j in range(4)])
    for graph, stratum in ((DF, RULING), (REIFY, OTHER))
    for i in range(2)
]
RUN_VERDICTS = {
    (row['graph'], edge['edge_uuid']): ('holding' if edge['edge_uuid'].endswith('e3') else 'overreach')
    for row in RUN_ROWS for edge in row['minted']
}


class _OracleLLM:
    """Answers every episode with its adjudicated labels; raises for *failing* bodies."""

    model = 'fake-judge-1'

    def __init__(self, failing: frozenset[str] = frozenset()) -> None:
        self.failing = failing
        self.calls = 0

    async def generate_response(self, messages, response_model=None, max_tokens=None, **kwargs):
        self.calls += 1
        (row,) = [r for r in RUN_ROWS if r['content'] in messages[1].content]
        if row['uuid'] in self.failing:
            raise RuntimeError('provider down')
        return {'judgements': [
            {'edge': f'E{i}', 'label': RUN_VERDICTS[(row['graph'], edge['edge_uuid'])],
             'rationale': 'oracle'}
            for i, edge in enumerate(row['minted'], start=1)
        ]}


def _write_inputs(tmp_path: Path, rows, verdicts_obj, report_obj) -> dict[str, Path]:
    paths = {
        'worksheet': tmp_path / 'worksheet.jsonl', 'verdicts': tmp_path / 'verdicts.json',
        'report': tmp_path / 'report.json', 'out': tmp_path / 'out' / 'eval.json',
    }
    paths['worksheet'].write_text(''.join(json.dumps(row) + '\n' for row in rows))
    paths['verdicts'].write_text(json.dumps(verdicts_obj))
    paths['report'].write_text(json.dumps(report_obj))
    return paths


def _argv(paths: dict[str, Path], mode: str = 'offline') -> list[str]:
    return [
        '--mode', mode, '--worksheet', str(paths['worksheet']),
        '--verdicts', str(paths['verdicts']), '--report', str(paths['report']),
        '--out', str(paths['out']),
    ]


def _run_inputs(tmp_path: Path) -> dict[str, Path]:
    return _write_inputs(tmp_path, RUN_ROWS, _verdicts_obj(RUN_ROWS, RUN_VERDICTS), _report_obj(WEIGHTS))


class TestRunOffline:
    def test_an_oracle_judge_scores_perfectly_and_passes(self, tmp_path: Path) -> None:
        paths = _run_inputs(tmp_path)
        llm = _OracleLLM()
        assert mod.run(_argv(paths), llm=llm) == 0
        out = json.loads(paths['out'].read_text())
        assert llm.calls == len(RUN_ROWS)
        assert out['mode'] == 'offline'
        assert out['precision']['rate'] == 1.0
        assert out['recall']['rate'] == 1.0
        assert out['trigger']['passes'] is True
        assert out['prompt_versions'] == [PROMPT_VERSION]
        assert out['judge_models'] == ['fake-judge-1']
        assert out['unjudged'] == [] and out['failures'] == []

    def test_a_judge_failure_leaves_that_episode_unjudged(self, tmp_path: Path) -> None:
        paths = _run_inputs(tmp_path)
        failing = RUN_ROWS[0]
        assert mod.run(_argv(paths), llm=_OracleLLM(failing=frozenset({failing['uuid']}))) == 0
        out = json.loads(paths['out'].read_text())
        assert [(f['graph'], f['episode_uuid'], f['error_type']) for f in out['failures']] == [
            (failing['graph'], failing['uuid'], 'RuntimeError'),
        ]
        assert {u['edge_uuid'] for u in out['unjudged']} == {
            edge['edge_uuid'] for edge in failing['minted']
        }
        assert out['recall']['rate'] < 1.0


class TestRunRejectsBadInputs:
    def _assert_rejected(self, paths: dict[str, Path]) -> None:
        llm = _OracleLLM()
        assert mod.run(_argv(paths), llm=llm) == 2
        assert not paths['out'].exists()
        assert llm.calls == 0

    def test_a_verdict_on_an_edge_the_worksheet_lacks(self, tmp_path: Path) -> None:
        verdicts = _verdicts_obj(RUN_ROWS, RUN_VERDICTS)
        verdicts['verdicts'].append({
            'graph': DF, 'episode_uuid': 'not-sampled', 'edge_uuid': 'stray', 'fact': 'f',
            'label': 'holding', 'rationale': 'r',
        })
        self._assert_rejected(_write_inputs(tmp_path, RUN_ROWS, verdicts, _report_obj(WEIGHTS)))

    def test_a_worksheet_edge_with_no_verdict(self, tmp_path: Path) -> None:
        verdicts = _verdicts_obj(RUN_ROWS, RUN_VERDICTS)
        verdicts['verdicts'].pop()
        self._assert_rejected(_write_inputs(tmp_path, RUN_ROWS, verdicts, _report_obj(WEIGHTS)))

    def test_an_unknown_label(self, tmp_path: Path) -> None:
        verdicts = _verdicts_obj(RUN_ROWS, RUN_VERDICTS)
        verdicts['verdicts'][0]['label'] = 'maybe'
        self._assert_rejected(_write_inputs(tmp_path, RUN_ROWS, verdicts, _report_obj(WEIGHTS)))

    def test_a_report_with_no_weight_for_a_worksheet_cell(self, tmp_path: Path) -> None:
        report = _report_obj({DF: {RULING: 2.0}})
        verdicts = _verdicts_obj(RUN_ROWS, RUN_VERDICTS)
        self._assert_rejected(_write_inputs(tmp_path, RUN_ROWS, verdicts, report))

    def test_online_mode_needs_a_sidecar(self, tmp_path: Path) -> None:
        paths = _run_inputs(tmp_path)
        with pytest.raises(SystemExit) as exit_info:
            mod.run(_argv(paths, mode='online'), llm=_OracleLLM())
        assert exit_info.value.code == 2
        assert not paths['out'].exists()


# --------------------------------------------------------------------------- #
# The CLI, online: the shadow's own labels, read back from its sidecar
# --------------------------------------------------------------------------- #
# Each episode: (graph, stratum, uuid, judge model, prompt version, edge groups),
# an edge group being (verdict, sidecar label, count). A None label is an edge
# the shadow never judged.
#   cell (dark_factory, ruling_lexeme), w=2:  10 TP, 1 FP, 11 FN, 1 TN
#   cell (reify, other_decisions),     w=10: 10 TP, 11 FN (one unjudged), 1 TN
# Precision clears both bounds; recall stays under 0.50.

OLDER_PROMPT = 'prompt-v-older'
ONLINE_EPISODES = [
    (DF, RULING, 'df-on0', 'judge-a', PROMPT_VERSION,
     (('overreach', 'overreach', 10), ('holding', 'overreach', 1))),
    (DF, RULING, 'df-on1', 'judge-a', PROMPT_VERSION,
     (('overreach', 'holding', 11), ('holding', 'holding', 1))),
    (REIFY, OTHER, 'rf-on0', 'judge-b', OLDER_PROMPT, (('overreach', 'overreach', 10),)),
    (REIFY, OTHER, 'rf-on1', 'judge-b', OLDER_PROMPT,
     (('overreach', 'holding', 10), ('overreach', None, 1), ('holding', 'holding', 1))),
]


def _online_edges(uuid: str, groups) -> list[tuple[str, str, str | None]]:
    """``(edge_uuid, verdict, sidecar label)`` for each of *uuid*'s edges, in order."""
    flat = [(verdict, label) for verdict, label, count in groups for _ in range(count)]
    return [(f'{uuid}-e{i}', verdict, label) for i, (verdict, label) in enumerate(flat)]


ONLINE_ROWS = [
    _row(graph, uuid, stratum, [edge for edge, _, _ in _online_edges(uuid, groups)])
    for graph, stratum, uuid, _, _, groups in ONLINE_EPISODES
]
ONLINE_VERDICTS = {
    (graph, edge): verdict
    for graph, _, uuid, _, _, groups in ONLINE_EPISODES
    for edge, verdict, _ in _online_edges(uuid, groups)
}
UNJUDGED_EDGE = (REIFY, 'rf-on1-e10')


async def _seed_sidecar(data_dir: Path) -> Path:
    store = await OverreachShadowStore.open(data_dir)
    try:
        for graph, _, uuid, judge_model, prompt_version, groups in ONLINE_EPISODES:
            edges = _online_edges(uuid, groups)
            await store.record(
                episode=JudgedEpisode(
                    graph=graph, episode_uuid=uuid, judge_model=judge_model,
                    prompt_version=prompt_version, classifiers=('category_decisions',),
                    minted_count=len(edges), outcome=ShadowOutcome.judged,
                ),
                judgements=[
                    (JudgeEdge(uuid=edge, fact=f'fact of {edge}', source_name='S', target_name='T'),
                     EdgeJudgement(edge_uuid=edge, label=label, rationale=f'shadow on {edge}'))
                    for edge, _, label in edges if label is not None
                ],
            )
        # A flag outside the worksheet: it must not count as a false positive.
        await store.record(
            episode=JudgedEpisode(
                graph=DF, episode_uuid='unsampled', judge_model='judge-c',
                prompt_version='prompt-v-unsampled', classifiers=('category_decisions',),
                minted_count=1, outcome=ShadowOutcome.judged,
            ),
            judgements=[(
                JudgeEdge(uuid='stray', fact='f', source_name='S', target_name='T'),
                EdgeJudgement(edge_uuid='stray', label='overreach', rationale='r'),
            )],
        )
    finally:
        await store.close()
    return data_dir / SIDECAR_FILENAME


class _NeverCalledLLM:
    model = 'never'

    async def generate_response(self, *args, **kwargs):
        raise AssertionError('this run must never consult an LLM')


def _online_argv(paths: dict[str, Path], sidecar: Path) -> list[str]:
    return [*_argv(paths, mode='online'), '--sidecar', str(sidecar)]


class TestRunOnline:
    @pytest.fixture
    def online(self, tmp_path: Path) -> tuple[dict[str, Path], Path]:
        paths = _write_inputs(
            tmp_path, ONLINE_ROWS, _verdicts_obj(ONLINE_ROWS, ONLINE_VERDICTS), _report_obj(WEIGHTS),
        )
        return paths, asyncio.run(_seed_sidecar(tmp_path / 'sidecar'))

    def _run(self, online) -> dict[str, Any]:
        paths, sidecar = online
        assert mod.run(_online_argv(paths, sidecar), llm=_NeverCalledLLM()) == 0
        return json.loads(paths['out'].read_text())

    def test_precision_and_recall_come_from_the_sidecar_labels(self, online) -> None:
        out = self._run(online)
        assert out['mode'] == 'online'
        assert out['precision'] == audit.weighted_rate([(2.0, 10, 11), (10.0, 10, 10)])
        assert out['recall'] == audit.weighted_rate([(2.0, 10, 21), (10.0, 10, 21)])

    def test_the_unjudged_edge_is_a_listed_false_negative(self, online) -> None:
        out = self._run(online)
        assert [(u['graph'], u['edge_uuid']) for u in out['unjudged']] == [UNJUDGED_EDGE]
        assert out['confusion_by_cell'][REIFY][OTHER]['fn'] == 11
        assert out['coverage'] == {'offered': len(ONLINE_VERDICTS), 'judged': len(ONLINE_VERDICTS) - 1}

    def test_the_recorded_judges_and_prompts_are_reported(self, online) -> None:
        out = self._run(online)
        assert out['judge_models'] == ['judge-a', 'judge-b']
        assert out['prompt_versions'] == sorted([PROMPT_VERSION, OLDER_PROMPT])

    def test_the_trigger_ignores_a_recall_under_the_offline_floor(self, online) -> None:
        out = self._run(online)
        assert out['recall']['rate'] < mod.RECALL_MIN
        assert out['trigger']['passes'] is True
        assert 'recall' not in out['trigger']['checks']

    def test_the_sidecar_is_opened_read_only(self, online) -> None:
        _, sidecar = online
        before = (sidecar.read_bytes(), sidecar.stat().st_mtime_ns)
        self._run(online)
        assert (sidecar.read_bytes(), sidecar.stat().st_mtime_ns) == before

    def test_online_mode_never_builds_an_llm(self, online, monkeypatch) -> None:
        def _refuse(*args, **kwargs):
            raise AssertionError('online mode must never build an LLM')

        monkeypatch.setattr(mod, 'build_llm_client', _refuse)
        paths, sidecar = online
        assert mod.run(_online_argv(paths, sidecar)) == 0

    def test_a_missing_sidecar_is_unavailable_and_writes_nothing(self, online, tmp_path: Path) -> None:
        paths, _ = online
        missing = tmp_path / 'nowhere' / SIDECAR_FILENAME
        assert mod.run(_online_argv(paths, missing), llm=_NeverCalledLLM()) == 1
        assert not paths['out'].exists()
        assert not missing.exists()

    def test_offline_mode_rejects_a_sidecar(self, online) -> None:
        paths, sidecar = online
        with pytest.raises(SystemExit) as exit_info:
            mod.run([*_argv(paths), '--sidecar', str(sidecar)], llm=_NeverCalledLLM())
        assert exit_info.value.code == 2
        assert not paths['out'].exists()
