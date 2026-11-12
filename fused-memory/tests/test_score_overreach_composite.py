"""Tests for scripts/score_overreach_composite.py (task 9040).

The composite is Stage 1's retire/keep decision on the overreach flags the
judge handed it. These tests seed a real sidecar through OverreachShadowStore
and answer the graph reads with a double behind the read-only seam.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from _fm_helpers import load_script_module
from _overreach_shadow_fixtures import judged_edge, judged_episode

from fused_memory.backends.falkor_read_only import ReadOnlyGraphReader
from fused_memory.services.overreach_guard import PROMPT_VERSION
from fused_memory.services.overreach_shadow import SIDECAR_FILENAME, OverreachShadowStore

SCRIPTS = Path(__file__).parent.parent / 'scripts'
mod = load_script_module(SCRIPTS / 'score_overreach_composite.py', mod_name='score_overreach_composite')
audit = load_script_module(SCRIPTS / 'audit_ruling_overreach.py', mod_name='audit_ruling_overreach')

DF, REIFY, THIRD = 'dark_factory', 'reify', 'third'
WINDOW_START, WINDOW_END = '2026-10-20T00:00:00+00:00', '2026-10-30T00:00:00+00:00'
JUDGED = datetime(2026, 10, 19, 6, 0, tzinfo=UTC)
BEFORE = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)
IN_WINDOW = datetime(2026, 10, 25, 6, 0, tzinfo=UTC)
RETIRED_AT = '2026-10-25T06:05:00+00:00'

EDGES: dict[str, dict[str, tuple[str | None, str | None]]] = {
    DF: {'r1': (RETIRED_AT, None), 'r2': (RETIRED_AT, None), 'r3': (RETIRED_AT, None),
         'k1': (None, None), 'k2': (None, RETIRED_AT), 's1': (RETIRED_AT, RETIRED_AT),
         'h1': (None, None), 'o1': (RETIRED_AT, None), 'p1': (None, None)},
    REIFY: {'rr1': (RETIRED_AT, None), 'rk1': (None, None)},
    THIRD: {'x1': (RETIRED_AT, None)},
}
"""``(invalid_at, expired_at)`` per live edge; g1 is gone from dark_factory."""

EPISODES = {
    DF: {'ep-r': 'the body of ep-r', 'ep-k': 'the body of ep-k', 'ep-s': 'the body of ep-s',
         'ep-o': 'the body of ep-o'},
    REIFY: {'ep-rr': 'the body of ep-rr'},
    THIRD: {'ep-x': 'the body of ep-x'},
}


class _SetClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


async def _seed_sidecar(data_dir: Path) -> Path:
    """dark_factory: r1-r3 retired, k1-k2 kept, s1 superseded, g1 gone, all overreach and handed
    in the window; h1 a handed holding; o1 handed before the window; p1 never handed.
    reify: rr1 retired, rk1 kept. third: x1, a graph nobody asks for."""
    clock = _SetClock(JUDGED)
    store = await OverreachShadowStore.open(data_dir, clock=clock)
    try:
        await store.record(episode=judged_episode('ep-r', graph=DF), judgements=[
            judged_edge(uuid, 'overreach') for uuid in ('r1', 'r2', 'r3')
        ])
        await store.record(episode=judged_episode('ep-k', graph=DF), judgements=[
            judged_edge('k1', 'overreach'), judged_edge('k2', 'overreach'), judged_edge('h1', 'holding'),
        ])
        await store.record(episode=judged_episode('ep-s', graph=DF), judgements=[
            judged_edge('s1', 'overreach'), judged_edge('g1', 'overreach'),
        ])
        await store.record(episode=judged_episode('ep-o', graph=DF), judgements=[
            judged_edge('o1', 'overreach'), judged_edge('p1', 'overreach'),
        ])
        await store.record(episode=judged_episode('ep-rr', graph=REIFY), judgements=[
            judged_edge('rr1', 'overreach'), judged_edge('rk1', 'overreach'),
        ])
        await store.record(episode=judged_episode('ep-x', graph=THIRD), judgements=[judged_edge('x1', 'overreach')])
        clock.now = BEFORE
        await store.record_handoffs(DF, ['o1'])
        clock.now = IN_WINDOW
        await store.record_handoffs(DF, ['r1', 'r2', 'r3', 'k1', 'k2', 'h1', 's1', 'g1'])
        await store.record_handoffs(REIFY, ['rr1', 'rk1'])
        await store.record_handoffs(THIRD, ['x1'])
    finally:
        await store.close()
    return data_dir / SIDECAR_FILENAME


@pytest.fixture
def sidecar(tmp_path: Path) -> Path:
    return asyncio.run(_seed_sidecar(tmp_path / 'data'))


class _StateGraph:
    """Answers the edge-state and episode-content reads for one graph; records what it was sent."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.sent: list[tuple[str, dict[str, Any]]] = []

    async def ro_query(self, cypher: str, params: dict[str, Any] | None = None) -> Any:
        assert params is not None
        self.sent.append((cypher, params))
        if cypher == mod.EDGE_STATE_CYPHER:
            edges = EDGES[self.name]
            rows = [[uuid, *edges[uuid], f'source of {uuid}', f'target of {uuid}']
                    for uuid in params['uuids'] if uuid in edges]
        elif cypher == mod.EPISODE_CONTENT_CYPHER:
            episodes = EPISODES[self.name]
            rows = [[uuid, episodes[uuid]] for uuid in params['uuids'] if uuid in episodes]
        else:
            raise AssertionError(f'unexpected read {cypher!r}')
        return SimpleNamespace(result_set=rows)


class _Graphs:
    def __init__(self) -> None:
        self.opened: dict[str, _StateGraph] = {}

    def __call__(self, graph: str) -> ReadOnlyGraphReader:
        self.opened[graph] = _StateGraph(graph)
        return ReadOnlyGraphReader(graph_name=graph, graph=self.opened[graph])


def _paths(tmp_path: Path, tag: str = 'run') -> dict[str, Path]:
    return {'sample': tmp_path / tag / 'composite-sample.json', 'worksheet': tmp_path / tag / 'worksheet.jsonl'}


def _sample_argv(sidecar: Path, paths: dict[str, Path], *, cap: int = 2,
                 window: tuple[str, str] = (WINDOW_START, WINDOW_END)) -> list[str]:
    return ['sample', '--sidecar', str(sidecar), '--graph', DF, '--graph', REIFY,
            '--window-start', window[0], '--window-end', window[1], '--cap', str(cap),
            '--out-sample', str(paths['sample']), '--out-worksheet', str(paths['worksheet'])]


def _sampled(sidecar: Path, tmp_path: Path, graphs: _Graphs | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = _paths(tmp_path)
    assert mod.run(_sample_argv(sidecar, paths), reader_factory=graphs or _Graphs()) == 0
    worksheet = [json.loads(line) for line in paths['worksheet'].read_text().splitlines()]
    return json.loads(paths['sample'].read_text()), worksheet


def _first_by_hash(graph: str, uuids: list[str], cap: int) -> list[str]:
    return sorted(uuids, key=lambda uuid: audit.sample_key(graph, uuid))[:cap]


class TestDecisionOf:
    @pytest.mark.parametrize(('invalid_at', 'expired_at', 'decision'), [
        (RETIRED_AT, None, 'retired'),
        (RETIRED_AT, RETIRED_AT, 'superseded'),
        (None, None, 'kept'),
        (None, RETIRED_AT, 'kept'),
    ])
    def test_a_live_edge_is_classified_by_its_stamps(
        self, invalid_at: str | None, expired_at: str | None, decision: str,
    ) -> None:
        state = mod.EdgeState(invalid_at=invalid_at, expired_at=expired_at, source_name='S', target_name='T')
        assert mod.decision_of(state) == decision

    def test_an_absent_edge_is_gone(self) -> None:
        assert mod.decision_of(None) == 'gone'


class TestTheSample:
    def test_only_handed_overreach_flags_in_the_window_and_the_requested_graphs(
        self, sidecar: Path, tmp_path: Path,
    ) -> None:
        sample, _ = _sampled(sidecar, tmp_path)
        sampled = {(f['graph'], f['edge_uuid']) for f in sample['flags']}
        assert sampled == {
            *((DF, uuid) for uuid in _first_by_hash(DF, ['r1', 'r2', 'r3'], 2)),
            (DF, 'k1'), (DF, 'k2'), (REIFY, 'rr1'), (REIFY, 'rk1'),
        }

    def test_each_cell_takes_the_first_cap_by_the_audits_sample_key(
        self, sidecar: Path, tmp_path: Path,
    ) -> None:
        sample, _ = _sampled(sidecar, tmp_path)
        retired = [f['edge_uuid'] for f in sample['flags'] if (f['graph'], f['decision']) == (DF, 'retired')]
        assert sorted(retired) == sorted(_first_by_hash(DF, ['r1', 'r2', 'r3'], 2))
        assert mod.sample_key is audit.sample_key

    def test_each_sampled_flag_carries_its_decision_and_handoff(self, sidecar: Path, tmp_path: Path) -> None:
        sample, _ = _sampled(sidecar, tmp_path)
        by_edge = {(f['graph'], f['edge_uuid']): f for f in sample['flags']}
        assert by_edge[(DF, 'k2')] == {
            'graph': DF, 'edge_uuid': 'k2', 'episode_uuid': 'ep-k', 'fact': 'fact of k2',
            'decision': 'kept', 'handed_at': IN_WINDOW.isoformat(),
        }
        assert by_edge[(REIFY, 'rr1')]['decision'] == 'retired'

    def test_superseded_and_gone_flags_are_counted_but_never_sampled(
        self, sidecar: Path, tmp_path: Path,
    ) -> None:
        sample, _ = _sampled(sidecar, tmp_path)
        assert {f['decision'] for f in sample['flags']} == {'retired', 'kept'}
        census = sample['census']['by_graph'][DF]
        assert (census['superseded']['flags'], census['gone']['flags']) == (1, 1)

    def test_the_census_counts_every_handed_flag_per_graph_and_decision_per_day(
        self, sidecar: Path, tmp_path: Path,
    ) -> None:
        census = _sampled(sidecar, tmp_path)[0]['census']
        assert census['window_days'] == 10.0
        assert census['by_graph'] == {
            DF: {'handed': 7, 'retired': {'flags': 3, 'per_day': 0.3}, 'kept': {'flags': 2, 'per_day': 0.2},
                 'superseded': {'flags': 1, 'per_day': 0.1}, 'gone': {'flags': 1, 'per_day': 0.1}},
            REIFY: {'handed': 2, 'retired': {'flags': 1, 'per_day': 0.1}, 'kept': {'flags': 1, 'per_day': 0.1},
                    'superseded': {'flags': 0, 'per_day': 0.0}, 'gone': {'flags': 0, 'per_day': 0.0}},
        }
        assert census['prompt_versions'] == [PROMPT_VERSION]

    def test_the_sample_definition_is_frozen_into_the_file(self, sidecar: Path, tmp_path: Path) -> None:
        sample, _ = _sampled(sidecar, tmp_path)
        assert sample['sample'] == audit.SampleDefinition(
            window_start=WINDOW_START, window_end=WINDOW_END, cap=2,
            strata=('retired', 'kept'), hash_rule='sha256(graph:edge_uuid)',
        ).to_dict()
        assert mod.HASH_RULE == 'sha256(graph:edge_uuid)' and mod.SAMPLED_DECISIONS == ('retired', 'kept')
        assert sample['graphs'] == [DF, REIFY]
        assert datetime.fromisoformat(sample['sampled_at']).tzinfo is not None

    def test_the_sample_file_holds_no_episode_body_and_no_judge_rationale(
        self, sidecar: Path, tmp_path: Path,
    ) -> None:
        _sampled(sidecar, tmp_path)
        text = _paths(tmp_path)['sample'].read_text()
        assert 'the body of' not in text and 'span for' not in text

    def test_a_rerun_is_byte_identical_apart_from_sampled_at(self, sidecar: Path, tmp_path: Path) -> None:
        first, second = _paths(tmp_path, 'first'), _paths(tmp_path, 'second')
        assert mod.run(_sample_argv(sidecar, first), reader_factory=_Graphs()) == 0
        assert mod.run(_sample_argv(sidecar, second), reader_factory=_Graphs()) == 0
        assert first['worksheet'].read_bytes() == second['worksheet'].read_bytes()
        one, other = (json.loads(p['sample'].read_text()) for p in (first, second))
        for document in (one, other):
            document.pop('sampled_at')
        assert one == other

    def test_small_read_batches_give_the_same_sample(
        self, sidecar: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        whole, _ = _sampled(sidecar, tmp_path / 'whole')
        monkeypatch.setattr(mod, 'UUID_BATCH', 2)
        graphs = _Graphs()
        batched, _ = _sampled(sidecar, tmp_path / 'batched', graphs)
        assert batched['flags'] == whole['flags'] and batched['census'] == whole['census']
        assert all(len(params['uuids']) <= 2 for g in graphs.opened.values() for _, params in g.sent)

    def test_only_the_requested_graphs_are_read_and_only_by_the_two_reads(
        self, sidecar: Path, tmp_path: Path,
    ) -> None:
        graphs = _Graphs()
        _sampled(sidecar, tmp_path, graphs)
        assert set(graphs.opened) == {DF, REIFY}
        assert {cypher for g in graphs.opened.values() for cypher, _ in g.sent} == {
            mod.EDGE_STATE_CYPHER, mod.EPISODE_CONTENT_CYPHER,
        }


class TestTheWorksheet:
    def test_one_row_per_sampled_episode_with_its_body_and_flagged_facts(
        self, sidecar: Path, tmp_path: Path,
    ) -> None:
        sample, worksheet = _sampled(sidecar, tmp_path)
        assert {(r['graph'], e['edge_uuid']) for r in worksheet for e in r['edges']} == {
            (f['graph'], f['edge_uuid']) for f in sample['flags']
        }
        rows = {(r['graph'], r['uuid']): r for r in worksheet}
        assert rows[(DF, 'ep-k')]['content'] == 'the body of ep-k'
        assert {e['fact'] for e in rows[(DF, 'ep-k')]['edges']} == {'fact of k1', 'fact of k2'}
        assert rows[(DF, 'ep-k')]['edges'][0]['source_name'].startswith('source of ')

    def test_it_names_no_decision_and_no_judge_rationale(self, sidecar: Path, tmp_path: Path) -> None:
        _, worksheet = _sampled(sidecar, tmp_path)
        text = _paths(tmp_path)['worksheet'].read_text()
        assert 'span for' not in text and 'retired' not in text and 'kept' not in text
        assert all('decision' not in row and all('decision' not in e for e in row['edges']) for row in worksheet)

    def test_rows_and_edges_are_ordered_by_hash_not_by_cell(self, sidecar: Path, tmp_path: Path) -> None:
        _, worksheet = _sampled(sidecar, tmp_path)
        assert [(r['graph'], r['uuid']) for r in worksheet] == sorted(
            ((r['graph'], r['uuid']) for r in worksheet), key=lambda key: audit.sample_key(*key),
        )
        for row in worksheet:
            hashes = [hashlib.sha256(f"{row['graph']}:{e['edge_uuid']}".encode()).hexdigest() for e in row['edges']]
            assert hashes == sorted(hashes)


class TestSampleFailsClosed:
    def _assert_nothing_written(self, paths: dict[str, Path]) -> None:
        assert not paths['sample'].exists() and not paths['worksheet'].exists()

    def test_a_missing_sidecar_exits_1(self, tmp_path: Path) -> None:
        paths = _paths(tmp_path)
        assert mod.run(_sample_argv(tmp_path / 'absent.db', paths), reader_factory=_Graphs()) == 1
        self._assert_nothing_written(paths)

    def test_an_unreadable_graph_exits_1(self, sidecar: Path, tmp_path: Path) -> None:
        class _Down:
            name = DF

            async def ro_query(self, cypher: str, params: dict[str, Any] | None = None) -> Any:
                raise ConnectionError('falkordb down')

        paths = _paths(tmp_path)
        code = mod.run(_sample_argv(sidecar, paths),
                       reader_factory=lambda graph: ReadOnlyGraphReader(graph_name=graph, graph=_Down()))
        assert code == 1
        self._assert_nothing_written(paths)

    def test_a_defect_in_a_read_is_raised_not_reported_as_unavailability(
        self, sidecar: Path, tmp_path: Path,
    ) -> None:
        class _Broken:
            name = DF

            async def ro_query(self, cypher: str, params: dict[str, Any] | None = None) -> Any:
                raise KeyError('a defect, not an outage')

        paths = _paths(tmp_path)
        with pytest.raises(KeyError):
            mod.run(_sample_argv(sidecar, paths),
                    reader_factory=lambda graph: ReadOnlyGraphReader(graph_name=graph, graph=_Broken()))
        self._assert_nothing_written(paths)

    def test_every_reader_built_is_closed_when_a_later_one_cannot_be(
        self, sidecar: Path, tmp_path: Path,
    ) -> None:
        closed: list[str] = []

        class _Recording(ReadOnlyGraphReader):
            async def aclose(self) -> None:
                closed.append(self.graph_name)

        def factory(graph: str) -> ReadOnlyGraphReader:
            if graph == REIFY:
                raise ConnectionError('falkordb down')
            return _Recording(graph_name=graph, graph=_StateGraph(graph))

        paths = _paths(tmp_path)
        assert mod.run(_sample_argv(sidecar, paths), reader_factory=factory) == 1
        assert closed == [DF]
        self._assert_nothing_written(paths)

    @pytest.mark.parametrize('window', [
        (WINDOW_END, WINDOW_START),
        (WINDOW_START, WINDOW_START),
        ('2026-10-20T00:00:00', WINDOW_END),
        ('not a time', WINDOW_END),
    ])
    def test_an_invalid_window_exits_2(self, sidecar: Path, tmp_path: Path, window: tuple[str, str]) -> None:
        paths = _paths(tmp_path)
        assert mod.run(_sample_argv(sidecar, paths, window=window), reader_factory=_Graphs()) == 2
        self._assert_nothing_written(paths)

    def test_a_cap_below_one_exits_2(self, sidecar: Path, tmp_path: Path) -> None:
        paths = _paths(tmp_path)
        assert mod.run(_sample_argv(sidecar, paths, cap=0), reader_factory=_Graphs()) == 2
        self._assert_nothing_written(paths)


# --------------------------------------------------------------------------- #
# The score: Stage 1's decision against the adjudicated sample
# --------------------------------------------------------------------------- #

judge_eval = load_script_module(SCRIPTS / 'eval_overreach_judge.py', mod_name='eval_overreach_judge')

COMPOSITE_DEFINITION = audit.SampleDefinition(
    window_start=WINDOW_START, window_end=WINDOW_END, cap=20,
    strata=('retired', 'kept'), hash_rule='sha256(graph:edge_uuid)',
)

Cells = dict[tuple[str, str], tuple[int, tuple[str, ...]]]
"""(graph, decision) -> (census population, the sampled flags' labels)."""

FAILING: Cells = {
    (DF, 'retired'): (30, ('overreach', 'overreach', 'holding')),
    (DF, 'kept'): (50, ('overreach', 'holding', 'holding', 'context', 'context')),
    (REIFY, 'retired'): (4, ('overreach', 'context')),
    (REIFY, 'kept'): (10, ('overreach', 'overreach')),
}
"""Cell weights 10, 10, 2 and 5 over a 10-day window."""

PASSING: Cells = {
    (DF, 'retired'): (30, ('overreach',) * 10),
    (DF, 'kept'): (50, ('holding',) * 5),
    (REIFY, 'retired'): (20, ('overreach',) * 10),
    (REIFY, 'kept'): (10, ('context',) * 2),
}

JUDGE_EVAL = {'mode': 'offline', 'recall': {'rate': 0.1467, 'ci': [0.0583, 0.3231], 'effective_n': 27.0}}


def _composite(cells: Cells) -> tuple[dict[str, Any], dict[str, Any]]:
    """A composite-sample.json and a verdict file labelling every sampled flag."""
    by_graph: dict[str, dict[str, Any]] = {
        graph: {'handed': 0} | {d: {'flags': 0, 'per_day': 0.0} for d in ('retired', 'kept', 'superseded', 'gone')}
        for graph in (DF, REIFY)
    }
    by_graph[DF]['superseded'] = {'flags': 7, 'per_day': 0.7}
    flags, verdicts = [], []
    for (graph, decision), (population, labels) in cells.items():
        by_graph[graph][decision] = {'flags': population, 'per_day': population / 10}
        for index, label in enumerate(labels):
            flag = {'graph': graph, 'edge_uuid': f'{graph}-{decision}-{index}',
                    'episode_uuid': f'ep-{decision}-{index // 2}', 'fact': f'fact {decision} {index}',
                    'decision': decision, 'handed_at': IN_WINDOW.isoformat()}
            flags.append(flag)
            verdicts.append({'graph': graph, 'episode_uuid': flag['episode_uuid'], 'edge_uuid': flag['edge_uuid'],
                             'fact': flag['fact'], 'label': label, 'rationale': f"span of {flag['edge_uuid']}"})
    for counts in by_graph.values():
        counts['handed'] = sum(counts[d]['flags'] for d in ('retired', 'kept', 'superseded', 'gone'))
    sample = {
        'sample': COMPOSITE_DEFINITION.to_dict(), 'graphs': [DF, REIFY], 'sampled_at': IN_WINDOW.isoformat(),
        'census': {'window_days': 10.0, 'by_graph': by_graph, 'prompt_versions': [PROMPT_VERSION]},
        'flags': flags,
    }
    return sample, {'sample': COMPOSITE_DEFINITION.to_dict(), 'verdicts': verdicts}


def _run_composite_score(
    tmp_path: Path, sample: dict[str, Any], verdicts: dict[str, Any], *, judge: dict[str, Any] | None = None,
) -> tuple[int, Path]:
    sample_file, verdict_file = tmp_path / 'composite-sample.json', tmp_path / 'composite-verdicts.json'
    out = tmp_path / 'out' / 'composite-report.json'
    sample_file.write_text(json.dumps(sample))
    verdict_file.write_text(json.dumps(verdicts))
    argv = ['score', '--sample', str(sample_file), '--verdicts', str(verdict_file), '--out', str(out)]
    if judge is not None:
        judge_file = tmp_path / 'judge-eval.json'
        judge_file.write_text(json.dumps(judge))
        argv += ['--judge-eval', str(judge_file)]
    return mod.run(argv), out


def _composite_scored(tmp_path: Path, cells: Cells = FAILING, *, judge: dict[str, Any] | None = None) -> dict[str, Any]:
    code, out = _run_composite_score(tmp_path, *_composite(cells), judge=judge)
    assert code == 0
    return json.loads(out.read_text())


class TestOneCopyOfTheBars:
    def test_the_precision_bars_and_estimators_are_imported(self) -> None:
        assert mod.PRECISION_MIN is judge_eval.PRECISION_MIN
        assert mod.PRECISION_LOWER_MIN is judge_eval.PRECISION_LOWER_MIN
        assert mod.weighted_rate is audit.weighted_rate and mod.load_verdicts is audit.load_verdicts


class TestCompositeScore:
    def test_cell_weights_are_census_population_over_sampled(self, tmp_path: Path) -> None:
        assert _composite_scored(tmp_path)['weights'] == {
            DF: {'retired': 10.0, 'kept': 10.0}, REIFY: {'retired': 2.0, 'kept': 5.0},
        }

    def test_retire_precision_is_weighted_overreach_among_the_retired(self, tmp_path: Path) -> None:
        score = _composite_scored(tmp_path)
        assert score['retire_precision'] == audit.weighted_rate([(10.0, 2, 3), (2.0, 1, 2)])
        assert score['unweighted']['retire_precision'] == audit.rate(3, 5)

    def test_flag_precision_is_weighted_overreach_over_every_cell(self, tmp_path: Path) -> None:
        assert _composite_scored(tmp_path)['flag_precision'] == audit.weighted_rate(
            [(10.0, 2, 3), (10.0, 1, 5), (2.0, 1, 2), (5.0, 2, 2)],
        )

    def test_stage1_recall_is_weighted_retired_overreach_over_flagged_overreach(self, tmp_path: Path) -> None:
        score = _composite_scored(tmp_path)
        assert score['stage1_recall_on_flags'] == audit.weighted_rate(
            [(10.0, 2, 2), (10.0, 0, 1), (2.0, 1, 1), (5.0, 0, 2)],
        )
        assert score['stage1_recall_on_flags']['rate'] == round(22 / 42, 4)

    def test_what_stage1_retires_per_day(self, tmp_path: Path) -> None:
        score = _composite_scored(tmp_path)
        assert score['holding_retired_share'] == audit.weighted_rate([(10.0, 1, 3), (2.0, 0, 2)])
        assert (score['holdings_retired_per_day'], score['overreach_retired_per_day']) == (1.0, 2.2)

    def test_label_counts_per_cell(self, tmp_path: Path) -> None:
        counts = _composite_scored(tmp_path)['label_counts']
        assert counts[DF]['retired'] == {
            'holding': 1, 'bookkeeping': 0, 'context': 0, 'overreach': 2, 'misbound': 0, 'unjudgeable': 0,
        }
        assert counts[REIFY]['kept']['overreach'] == 2

    def test_composite_recall_is_a_labelled_product_of_two_frames(self, tmp_path: Path) -> None:
        score = _composite_scored(tmp_path, judge=JUDGE_EVAL)
        stage1, judge = score['stage1_recall_on_flags'], JUDGE_EVAL['recall']
        composite = score['composite_recall']
        assert composite['point'] == round(judge['rate'] * stage1['rate'], 4)
        assert composite['range'] == [round(judge['ci'][0] * stage1['ci'][0], 4),
                                      round(judge['ci'][1] * stage1['ci'][1], 4)]
        assert composite['label'] == mod.COMPOSITE_RECALL_LABEL
        assert composite['judge_recall'] == {'rate': judge['rate'], 'ci': judge['ci']}

    def test_without_a_judge_eval_there_is_no_composite_recall(self, tmp_path: Path) -> None:
        assert _composite_scored(tmp_path)['composite_recall'] is None


class TestCompositeDecision:
    def test_it_fails_below_the_precision_bar(self, tmp_path: Path) -> None:
        decision = _composite_scored(tmp_path, FAILING)['decision']
        assert decision['checks'] == {'retire_precision': False, 'retire_precision_lower_bound': False}
        assert decision['passes'] is False

    def test_it_passes_above_both_bars(self, tmp_path: Path) -> None:
        score = _composite_scored(tmp_path, PASSING)
        assert score['retire_precision']['rate'] == 1.0 and score['retire_precision']['ci'][0] >= 0.65
        assert score['decision']['checks'] == {'retire_precision': True, 'retire_precision_lower_bound': True}
        assert score['decision']['passes'] is True

    def test_the_decision_states_its_bars(self, tmp_path: Path) -> None:
        assert _composite_scored(tmp_path)['decision']['bars'] == {
            'precision_min': 0.80, 'precision_lower_min': 0.65,
        }


class TestCompositeScoreFailsClosed:
    def _assert_rejected(self, tmp_path: Path, sample: dict[str, Any], verdicts: dict[str, Any]) -> None:
        code, out = _run_composite_score(tmp_path, sample, verdicts)
        assert code == 2
        assert not out.exists() and not out.parent.exists()

    def test_a_sampled_flag_without_a_verdict(self, tmp_path: Path) -> None:
        sample, verdicts = _composite(FAILING)
        verdicts['verdicts'] = verdicts['verdicts'][1:]
        self._assert_rejected(tmp_path, sample, verdicts)

    def test_a_verdict_outside_the_sample(self, tmp_path: Path) -> None:
        sample, verdicts = _composite(FAILING)
        stray = dict(verdicts['verdicts'][0], edge_uuid='ghost')
        verdicts['verdicts'].append(stray)
        self._assert_rejected(tmp_path, sample, verdicts)

    def test_an_unknown_label(self, tmp_path: Path) -> None:
        sample, verdicts = _composite(FAILING)
        verdicts['verdicts'][0]['label'] = 'maybe'
        self._assert_rejected(tmp_path, sample, verdicts)

    def test_a_verdict_file_adjudicated_under_another_sample(self, tmp_path: Path) -> None:
        sample, verdicts = _composite(FAILING)
        verdicts['sample'] = dict(verdicts['sample'], cap=8)
        self._assert_rejected(tmp_path, sample, verdicts)
