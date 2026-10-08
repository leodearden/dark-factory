"""Contract tests for the frozen FalkorDB index-provisioning activation run (PRD ζ).

`plans/falkordb-index-activation-run/` holds one live, read-only run of
``scripts/falkordb_index_activation_probe.py`` (task 3711) and a verbatim copy
of the E1 retrieval-health probe taken after it. Every verdict here is
RECOMPUTED from the recorded raw rows with the probe script's own pure
functions, never read from a recorded boolean, so a hand-edited verdict
cannot pass.

**Lane discipline.** File reads, one ``git`` subprocess, and importlib loads of
the two scripts: no network, no FalkorDB, no MCP server, and no
``integration`` marker, so the merge lane runs it.
"""
from __future__ import annotations

import functools
import json
import shutil
import subprocess
import types
from datetime import UTC, datetime
from pathlib import Path

import pytest
from _fm_helpers import load_script_module
from shared.memory_eval_metrics import (
    load_metric_series,
    parse_metric_series,
    serialize_metric_series,
)

from fused_memory.reconciliation.index_health import summarize_index_health

REPO_ROOT = Path(__file__).parents[2]
RUN_ROOT = REPO_ROOT / 'plans' / 'falkordb-index-activation-run'
SCRIPTS = Path(__file__).parent.parent / 'scripts'

E1_EVAL_ID = 'e1-retrieval-health'
BRIEFING_TASK_SEMANTIC_ITEM = 't-briefing-task-semantic'
RETAINED_IDS = frozenset({'2286', '3127', '1157'})
TASK_NAMED_PROJECTS = frozenset({
    'autopilot_video', 'autotrade', 'dark_factory', 'know_live', 'mission_control',
    'pump_web_ui', 'reify', 'solar_challenge', 'solar_challenge_platform',
})


@functools.cache
def _probe() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'falkordb_index_activation_probe.py', mod_name='falkordb_index_activation_probe',
    )


@functools.cache
def _e1() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'memory_eval_retrieval_probe.py', mod_name='memory_eval_retrieval_probe',
    )


def _activation_path() -> Path:
    found = sorted(RUN_ROOT.glob('activation-*.json'))
    assert len(found) == 1, f'expected exactly one activation record under {RUN_ROOT}, found {found!r}'
    return found[0]


@functools.cache
def _record() -> dict:
    return json.loads(_activation_path().read_text(encoding='utf-8'))


def _e1_run() -> tuple[Path, Path]:
    """``(metrics, report)`` of the single frozen E1 run."""
    found = sorted((RUN_ROOT / E1_EVAL_ID).glob('metrics-*.json'))
    assert len(found) == 1, f'expected exactly one frozen E1 run under {RUN_ROOT}, found {found!r}'
    metrics = found[0]
    stamp = metrics.name.removeprefix('metrics-').removesuffix('.json')
    return metrics, metrics.with_name(f'report-{stamp}.txt')


def _edges(raw: list[dict]) -> list:
    return [_probe().FulltextEdge(**edge) for edge in raw]


def _hits(raw: list[dict]) -> list:
    return [_probe().SearchHit(**result) for result in raw]


def _present_statuses() -> list[dict]:
    return [status for status in _record()['index_statuses'] if status['present']]


def _recomputed_healthy(status: dict) -> bool:
    health = summarize_index_health(
        {tuple(spec) for spec in status['actual']},
        {tuple(spec) for spec in _record()['expected_index_set']},
    )
    return health['healthy'] and status['unsettled'] == []


def _examined_table() -> dict[str, list]:
    return {
        candidate['task_id']: _edges(candidate['edges'])
        for selection in _record()['replacements']
        for candidate in selection['examined']
    }


def _chosen() -> dict[str, str]:
    return {selection['anchor']: selection['chosen'] for selection in _record()['replacements']}


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    """Run git at the repo root, or skip where git cannot answer at all."""
    if shutil.which('git') is None:
        pytest.skip('git is not available; cannot check tracked-ness')
    inside = subprocess.run(
        ['git', 'rev-parse', '--is-inside-work-tree'],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    if inside.returncode != 0 or inside.stdout.strip() != 'true':
        pytest.skip('not a git working tree; cannot check tracked-ness')
    return subprocess.run(
        ['git', *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )


class TestTheRunIsDurablyCommitted:
    def test_one_activation_record_and_the_readme_are_present(self):
        assert _activation_path().is_file()
        assert (RUN_ROOT / 'README.md').is_file()

    def test_every_file_is_tracked_by_git(self):
        for path in (_activation_path(), RUN_ROOT / 'README.md', *_e1_run()):
            result = _git('ls-files', '--error-unmatch', '--', str(path))
            assert result.returncode == 0, (
                f'{path.relative_to(REPO_ROOT)} is NOT tracked by git: {result.stderr.strip()}'
            )


class TestTheRegistryIsNotNarrowed:
    def test_every_task_named_project_is_registered(self):
        assert set(_record()['registry']) >= TASK_NAMED_PROJECTS


class TestEveryRegisteredGraphIsIndexed:
    def test_the_expected_set_is_not_empty(self):
        assert _record()['expected_index_set']

    def test_the_primary_graph_is_present(self):
        assert 'dark_factory' in {status['group_id'] for status in _present_statuses()}

    def test_every_present_registered_graph_is_complete_and_operational(self):
        for status in _present_statuses():
            assert _recomputed_healthy(status), (
                f"{status['group_id']}: missing {status['missing']}, unsettled {status['unsettled']}"
            )


class TestTheReplacementsFollowTheDeclaredRule:
    def test_exactly_the_drifted_ids_are_replaced(self):
        assert set(_chosen()) == set(_probe().DRIFTED_IDS) == {'877', '3600'}

    def test_each_replacement_is_new_and_has_enough_live_genuine_edges(self):
        probe = _probe()
        table = _examined_table()

        for chosen in _chosen().values():
            assert chosen not in probe.ORIGINAL_PROBE_IDS
            assert probe.genuine_live_count(table[chosen], chosen) >= probe.REPLACEMENT_FLOOR

    def test_the_rule_rerun_over_the_recorded_counts_reproduces_the_choice(self):
        selections = _probe().select_replacements(_examined_table().__getitem__)

        assert {selection.anchor: selection.chosen for selection in selections} == _chosen()


class TestThePrimaryProbe:
    def _probe_run(self) -> dict:
        return _record()['asserted_probe']

    def test_it_fires_the_retained_and_replacement_ids_graphiti_scoped(self):
        probe = _probe()
        run = self._probe_run()

        assert {query['task_id'] for query in run['queries']} == RETAINED_IDS | set(_chosen().values())
        assert run['template'] == probe.RETIRED_TASK_TEMPLATE
        assert run['stores'] == ['graphiti']
        assert run['limit'] == probe.PROBE_LIMIT == 5
        for query in run['queries']:
            assert query['query'] == probe.RETIRED_TASK_TEMPLATE.format(task_id=query['task_id'])

    def test_no_query_was_degraded_or_over_the_limit(self):
        for query in self._probe_run()['queries']:
            assert query['degraded'] is False
            assert len(query['results']) <= 5

    def test_the_recomputed_verdict_meets_the_floor(self):
        probe = _probe()
        hits_by_id = {
            query['task_id']: probe.query_hit(_hits(query['results']), query['task_id'])
            for query in self._probe_run()['queries']
        }

        verdict = probe.briefing_verdict(hits_by_id, floor=probe.PROBE_FLOOR)

        assert verdict.passed, f'{verdict.hits}/{verdict.total} below floor {verdict.floor}; missed {verdict.missed}'

    def test_the_original_ids_and_unscoped_variants_are_recorded(self):
        probe = _probe()
        record = _record()

        assert [q['task_id'] for q in record['original_ids_probe']['queries']] == list(probe.ORIGINAL_PROBE_IDS)
        assert record['unscoped_probe']['stores'] is None
        assert {q['task_id'] for q in record['unscoped_probe']['queries']} == (
            {q['task_id'] for q in self._probe_run()['queries']}
        )


class TestTheE1PreflightProvedMaintenanceANoop:
    def test_every_listed_graph_has_a_zero_dup_uuid_count(self):
        preflight = _record()['e1_preflight']

        assert {entry['group_id'] for entry in preflight['dup_uuid_groups']} == set(_record()['listed_graphs'])
        assert all(entry['groups'] == 0 for entry in preflight['dup_uuid_groups'])

    def test_no_present_registered_graph_would_be_provisioned(self):
        assert [s['group_id'] for s in _present_statuses() if not _recomputed_healthy(s)] == []


class TestTheE1SecondaryIsAValidM1Artifact:
    def test_one_metrics_file_and_its_report_are_present(self):
        metrics, report = _e1_run()

        assert metrics.is_file()
        assert report.is_file()

    def test_the_metrics_artifact_validates_as_e1(self):
        assert load_metric_series(_e1_run()[0]).eval_id == E1_EVAL_ID

    def test_reemitting_the_artifact_is_byte_identical(self):
        text = _e1_run()[0].read_text(encoding='utf-8')

        assert serialize_metric_series(parse_metric_series(json.loads(text))) == text

    def test_the_briefing_task_semantic_item_is_adjudicated(self):
        series = load_metric_series(_e1_run()[0])
        (tripwire,) = [m for m in series.metrics if m.metric_id == _e1().METRIC_TOPIC_CANONICAL_PRESENT]

        assert BRIEFING_TASK_SEMANTIC_ITEM in {item.item_key for item in tripwire.items or []}

    def test_the_run_root_is_disjoint_from_the_live_artifact_root(self):
        frozen = RUN_ROOT.resolve()
        live = _e1().DEFAULT_OUT_ROOT.resolve()

        assert frozen != live
        assert not frozen.is_relative_to(live)
        assert not live.is_relative_to(frozen)

    def test_the_e1_run_was_taken_after_the_preflight(self):
        stamp = _e1_run()[0].name.removeprefix('metrics-').removesuffix('.json')
        e1_at = datetime.strptime(stamp, '%Y%m%dT%H%M%SZ').replace(tzinfo=UTC)

        assert e1_at > datetime.fromisoformat(_record()['measured_at'])


class TestTheCaseFoldTripwireIsRecorded:
    def test_the_primary_graph_has_keys_nodes_and_split_groups(self):
        (dark_factory,) = [c for c in _record()['case_fold'] if c['group_id'] == 'dark_factory']

        assert 'keys' in dark_factory
        assert 'nodes' in dark_factory
        assert isinstance(dark_factory['split_groups'], list)
