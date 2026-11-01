"""ResultStore owns the persisted eval cell: its type, on-disk name and directory.

Driven only through the public names of ``orchestrator.evals.results`` against
real files under ``tmp_path``; the B1 boundary case drives a real executor and
reads its cell back through the store. The scripts loader's half of B1 lives in
``scripts/tests/test_eval_results_io.py``, where a scripts-only change runs it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from _campaign_gate_helpers import eval_base_config

import orchestrator.evals.results as results_module
from orchestrator.evals import runner
from orchestrator.evals.architect_eval import ArchitectEvalDeps
from orchestrator.evals.configs import EvalConfig
from orchestrator.evals.metrics import EvalMetrics, RoleUnderTest
from orchestrator.evals.results import (
    CellName,
    EvalResult,
    MissingResultsDirError,
    ResultLoadError,
    ResultStore,
    load_result,
    normalize_legacy_metrics,
)
from orchestrator.workflow import WorkflowOutcome

_TASK_PATH = Path('/fake/task.json')


def _result(
    task_id: str = 'df_task_1',
    config_name: str = 'cfg-a',
    run_id: str = 'abcd1234',
    **overrides,
) -> EvalResult:
    fields = {
        'outcome': 'done',
        'metrics': {'tests_passed': True, 'cost_usd': 1.5, 'role_under_test': 'implementer'},
        'worktree_path': '/fake/wt',
        'wall_clock_ms': 1200,
        'trial': 1,
    }
    fields.update(overrides)
    return EvalResult(
        task_id=task_id, config_name=config_name, run_id=run_id, **fields
    )


@pytest.fixture
def store(tmp_path: Path) -> ResultStore:
    return ResultStore(tmp_path / 'results')


class TestRoundTrip:
    def test_save_writes_the_cell_under_its_grammar_name(self, store: ResultStore):
        r = _result()
        path = store.save(r)
        assert path == store.path_for(r)
        assert path == store.directory / 'df_task_1__cfg-a__abcd1234.json'
        assert json.loads(path.read_text()) == r.to_dict()

    def test_load_all_returns_saved_cells_in_path_order(self, store: ResultStore):
        later = _result(task_id='df_task_2', run_id='00000002')
        earlier = _result(task_id='df_task_1', run_id='00000001')
        store.save(later)
        store.save(earlier)
        assert store.load_all() == [earlier, later]


class TestCellNameFromPath:
    def test_from_path_inverts_path_for(self, store: ResultStore):
        r = _result()
        assert CellName.from_path(store.path_for(r)) == CellName(
            r.task_id, r.config_name, r.run_id
        )

    def test_from_path_inverts_filename(self, tmp_path: Path):
        name = CellName('df_task_1', 'cfg-a', 'abcd1234')
        assert CellName.from_path(tmp_path / name.filename) == name

    def test_a_combo_label_with_plus_round_trips(self, store: ResultStore):
        r = _result(config_name='arch-a+impl-b')
        assert CellName.from_path(store.path_for(r)) == CellName(
            'df_task_1', 'arch-a+impl-b', 'abcd1234'
        )

    @pytest.mark.parametrize(
        'name',
        [
            'df_task_12__claude-opus-high.json',
            'truncated.json',
            'a__b__c__d.json',
            'a___b__c.json',
            'df_task_1__cfg-a__abcd1234.txt',
        ],
    )
    def test_from_path_rejects_names_outside_the_grammar(
        self, tmp_path: Path, name: str
    ):
        path = tmp_path / name
        with pytest.raises(ValueError) as excinfo:
            CellName.from_path(path)
        assert str(path) in str(excinfo.value)


class TestCellNameInvariant:
    @pytest.mark.parametrize('bad', ['a__b', 'a/b', 'cfg_', '_cfg'])
    def test_a_component_that_would_not_round_trip_is_rejected(self, bad: str):
        with pytest.raises(ValueError) as excinfo:
            CellName('t', bad, 'r')
        message = str(excinfo.value)
        assert 'config_name' in message
        assert bad in message

    def test_save_refuses_an_unaddressable_cell_and_writes_nothing(
        self, store: ResultStore
    ):
        with pytest.raises(ValueError):
            store.save(_result(task_id='t', config_name='a__b', run_id='r'))
        assert not store.directory.exists() or not any(store.directory.iterdir())


class TestExistence:
    def test_a_saved_pair_exists_and_matches_its_path(self, store: ResultStore):
        path = store.save(_result(task_id='t', config_name='cfg-high'))
        assert store.exists('t', 'cfg-high')
        assert store.paths_matching('t', 'cfg-high') == [path]

    @pytest.mark.parametrize(
        ('task_id', 'config_name'),
        [('t', 'other'), ('t', 'cfg'), ('u', 'cfg-high')],
    )
    def test_other_pairs_and_config_prefixes_do_not_match(
        self, store: ResultStore, task_id: str, config_name: str
    ):
        store.save(_result(task_id='t', config_name='cfg-high'))
        assert not store.exists(task_id, config_name)
        assert store.paths_matching(task_id, config_name) == []

    def test_a_missing_directory_holds_nothing(self, store: ResultStore):
        assert not store.exists('t', 'cfg')
        assert store.paths_matching('t', 'cfg') == []

    def test_a_legacy_two_segment_cell_does_not_count(self, store: ResultStore):
        store.directory.mkdir(parents=True)
        (store.directory / 't__cfg.json').write_text(
            json.dumps(_result(task_id='t', config_name='cfg').to_dict())
        )
        assert not store.exists('t', 'cfg')
        assert store.paths_matching('t', 'cfg') == []


class TestCellPaths:
    def test_only_top_level_json_files_are_cells(self, store: ResultStore):
        cell = store.save(_result())
        (store.directory / 're_reviews').mkdir()
        (store.directory / 're_reviews' / 'x.json').write_text('{}')
        (store.directory / 'notes.md').write_text('notes')
        assert store.cell_paths() == [cell]

    def test_a_missing_directory_lists_no_cells(self, store: ResultStore):
        assert store.cell_paths() == []


class TestLoudLoad:
    def test_a_missing_directory_is_named(self, store: ResultStore):
        with pytest.raises(MissingResultsDirError) as excinfo:
            store.load_all()
        assert isinstance(excinfo.value, ResultLoadError)
        assert excinfo.value.path == store.directory
        assert str(store.directory) in str(excinfo.value)

    @pytest.mark.parametrize(
        ('name', 'text', 'error_type'),
        [
            ('truncated.json', '{"task_id": "x", "conf', 'JSONDecodeError'),
            ('a_list.json', '[1, 2, 3]', 'AttributeError'),
            (
                'missing_field.json',
                json.dumps({'task_id': 't', 'config_name': 'c'}),
                'TypeError',
            ),
        ],
    )
    def test_a_bad_cell_is_named(
        self, store: ResultStore, name: str, text: str, error_type: str
    ):
        store.save(_result())
        bad = store.directory / name
        bad.write_text(text)
        with pytest.raises(ResultLoadError) as excinfo:
            store.load_all()
        assert excinfo.value.path == bad
        assert str(bad) in str(excinfo.value)
        assert excinfo.value.detail.startswith(error_type)


class TestLoadResult:
    def test_a_saved_cell_reads_back_equal_metrics_included(self, store: ResultStore):
        saved = _result(
            metrics={'tests_pass': True, 'review_blocking_issues': 2, 'role_under_test': 'implementer'},
        )
        assert load_result(store.save(saved)) == saved

    def test_a_missing_cell_is_named(self, store: ResultStore):
        missing = store.directory / 'absent.json'
        with pytest.raises(ResultLoadError) as excinfo:
            load_result(missing)
        assert excinfo.value.path == missing
        assert str(missing) in str(excinfo.value)

    @pytest.mark.parametrize(
        'text', ['[1, 2, 3]', '{"task_id": "x", "conf'], ids=['a-list', 'truncated'],
    )
    def test_an_unreadable_cell_is_named_never_a_bare_error(
        self, store: ResultStore, text: str
    ):
        store.directory.mkdir(parents=True)
        bad = store.directory / 'bad.json'
        bad.write_text(text)
        with pytest.raises(ResultLoadError) as excinfo:
            load_result(bad)
        assert excinfo.value.path == bad
        assert str(bad) in str(excinfo.value)


class TestRewrite:
    def test_the_change_lands_and_keys_eval_result_does_not_declare_survive(
        self, store: ResultStore
    ):
        path = store.save(_result())
        cell = json.loads(path.read_text())
        cell['retired_field'] = 'kept'
        path.write_text(json.dumps(cell))

        def add_legacy_metric(cell: dict) -> dict:
            cell['metrics']['legacy_sonnet_review_blocking_issues'] = 3
            return cell

        assert store.rewrite(path, add_legacy_metric) is True
        rewritten = json.loads(path.read_text())
        assert rewritten['retired_field'] == 'kept'
        assert rewritten['metrics'] == {
            **_result().metrics, 'legacy_sonnet_review_blocking_issues': 3,
        }

    def test_a_rewrite_that_changes_nothing_leaves_the_file_untouched(
        self, store: ResultStore
    ):
        path = store.save(_result())
        os.utime(path, ns=(1_000_000_000, 1_000_000_000))
        before = path.read_text()
        assert store.rewrite(path, lambda cell: cell) is False
        assert path.read_text() == before
        assert path.stat().st_mtime_ns == 1_000_000_000

    @pytest.mark.parametrize(
        'text', ['{"task_id": "x", "conf', '[1, 2, 3]', None],
        ids=['truncated', 'not-an-object', 'missing'],
    )
    def test_an_unreadable_cell_is_named_and_left_as_it_was(
        self, store: ResultStore, text: str | None
    ):
        store.directory.mkdir(parents=True)
        bad = store.directory / 'bad.json'
        if text is not None:
            bad.write_text(text)
        with pytest.raises(ResultLoadError) as excinfo:
            store.rewrite(bad, lambda cell: {**cell, 'outcome': 'blocked'})
        assert excinfo.value.path == bad
        assert str(bad) in str(excinfo.value)
        assert (bad.read_text() if bad.exists() else None) == text

    def test_a_cell_of_another_store_is_refused(
        self, store: ResultStore, tmp_path: Path
    ):
        path = ResultStore(tmp_path / 'elsewhere').save(_result())
        before = path.read_text()
        with pytest.raises(ValueError) as excinfo:
            store.rewrite(path, lambda cell: {**cell, 'outcome': 'blocked'})
        assert str(path) in str(excinfo.value)
        assert path.read_text() == before


class TestSchemaDriftTolerance:
    def test_a_legacy_cell_without_run_id_or_trial_loads_with_defaults(
        self, store: ResultStore
    ):
        store.directory.mkdir(parents=True)
        legacy = _result(task_id='df_task_12', config_name='claude-opus-high').to_dict()
        del legacy['run_id']
        del legacy['trial']
        (store.directory / 'df_task_12__claude-opus-high.json').write_text(
            json.dumps(legacy)
        )
        [loaded] = store.load_all()
        assert loaded.run_id == ''
        assert loaded.trial == 1
        assert loaded.task_id == 'df_task_12'

    def test_an_unknown_top_level_key_is_dropped(self, store: ResultStore):
        store.directory.mkdir(parents=True)
        data = _result().to_dict()
        data['retired_field'] = 'whatever'
        (store.directory / 'df_task_1__cfg-a__abcd1234.json').write_text(
            json.dumps(data)
        )
        assert store.load_all() == [_result()]


def test_normalize_legacy_metrics_returns_an_equal_new_dict():
    metrics = {'cost_usd': 2.0, 'nested': {'a': 1}, 'role_under_test': 'implementer'}
    normalized = normalize_legacy_metrics(metrics)
    assert normalized == metrics
    assert normalized is not metrics


_CANONICAL_LEGACY_REASON = (
    'Architect eval reify_task_2324 × architect-opus-max: '
    'all accounts capped after 1'
)


class TestLegacyInvocationErrorRule:
    """D3: a pre-6734 ``invocation_error`` string loads as typed records."""

    def test_the_canonical_corpus_marker_becomes_one_record(self):
        metrics = {
            'invocation_error': f'architect:cap_exhausted: {_CANONICAL_LEGACY_REASON}',
            'cap_tainted': True,
        }
        normalized = normalize_legacy_metrics(metrics)
        assert normalized['invocation_errors'] == [{
            'stage': 'architect',
            'cause': 'cap_exhausted',
            'reason': _CANONICAL_LEGACY_REASON,
        }]
        assert 'invocation_error' not in normalized

    def test_a_judge_marker_keeps_its_stage(self):
        normalized = normalize_legacy_metrics(
            {'invocation_error': 'judge:api_error: HTTP 429'}
        )
        assert normalized['invocation_errors'] == [
            {'stage': 'judge', 'cause': 'api_error', 'reason': 'HTTP 429'},
        ]

    def test_a_two_marker_string_is_one_record_driven_by_the_first(self):
        from orchestrator.evals.report import format_invocation_errors

        marker = 'architect:timeout: no answer within 60m; judge:raised: RuntimeError: x'
        normalized = normalize_legacy_metrics({'invocation_error': marker})
        [record] = normalized['invocation_errors']
        assert (record['stage'], record['cause']) == ('architect', 'timeout')
        assert format_invocation_errors(normalized['invocation_errors']) == marker

    def test_a_clipped_reason_keeps_its_trailing_space(self):
        from orchestrator.evals.report import format_invocation_errors

        marker = 'architect:cap_exhausted: all accounts capped after 1 '
        normalized = normalize_legacy_metrics({'invocation_error': marker})
        assert normalized['invocation_errors'][0]['reason'] == (
            'all accounts capped after 1 '
        )
        assert format_invocation_errors(normalized['invocation_errors']) == marker

    @pytest.mark.parametrize(
        'marker', ['architect:', 'architect:bogus_cause: x', 'garbage'],
    )
    def test_an_unrecognised_marker_is_legacy_unclassified(self, marker):
        normalized = normalize_legacy_metrics({'invocation_error': marker})
        assert normalized['invocation_errors'] == [{
            'stage': 'architect', 'cause': 'legacy_unclassified', 'reason': marker,
        }]

    def test_a_null_marker_drops_the_old_key(self):
        normalized = normalize_legacy_metrics({'invocation_error': None})
        assert normalized['invocation_errors'] is None
        assert 'invocation_error' not in normalized

    def test_a_tainted_cell_without_a_marker_gets_an_unclassified_record(self):
        normalized = normalize_legacy_metrics({'cap_tainted': True})
        assert normalized['invocation_errors'] == [{
            'stage': 'architect', 'cause': 'legacy_unclassified', 'reason': '',
        }]

    def test_a_new_shape_cell_passes_through_unchanged(self):
        metrics = {
            'role_under_test': 'architect',
            'cap_tainted': True,
            'invocation_errors': [
                {'stage': 'architect', 'cause': 'cap_hit', 'reason': 'r'},
            ],
        }
        assert normalize_legacy_metrics(metrics) == metrics

    def test_the_input_is_not_mutated(self):
        metrics = {'invocation_error': 'architect:cap_hit: r', 'cap_tainted': True}
        before = dict(metrics)
        normalize_legacy_metrics(metrics)
        assert metrics == before


class TestLegacyRoleRule:
    """ε: a cell persisted before task 6738 with no role was a run_eval implementer cell."""

    @pytest.mark.parametrize(
        'metrics', [{}, {'role_under_test': None}], ids=['absent', 'null'],
    )
    def test_a_roleless_cell_is_an_implementer_cell(self, metrics: dict):
        normalized = normalize_legacy_metrics(metrics)
        assert normalized['role_under_test'] is RoleUnderTest.IMPLEMENTER

    @pytest.mark.parametrize(
        ('stored', 'role'),
        [('architect', RoleUnderTest.ARCHITECT), ('judge', RoleUnderTest.JUDGE)],
    )
    def test_a_stored_role_is_parsed_to_its_member(self, stored: str, role: RoleUnderTest):
        assert normalize_legacy_metrics({'role_under_test': stored})['role_under_test'] is role

    def test_the_input_is_not_mutated(self):
        metrics = {'cost_usd': 1.0}
        normalize_legacy_metrics(metrics)
        assert metrics == {'cost_usd': 1.0}

    def test_an_unknown_role_fails_the_load_naming_the_cell(self, store: ResultStore):
        bad = store.save(_result(metrics={'role_under_test': 'reviewer'}))
        with pytest.raises(ResultLoadError) as excinfo:
            store.load_all()
        assert excinfo.value.path == bad


def test_the_packaged_store_lives_beside_the_results_module():
    assert ResultStore.packaged().directory == (
        Path(results_module.__file__).parent / 'results'
    )


def _executor_task(tmp_path: Path) -> dict:
    return {
        'id': 'df_task_b1',
        'project_root': str(tmp_path),
        'pre_task_commit': 'basecommit',
        'task_definition': {'title': 'W', 'description': 'd'},
        'modules': ['pkg/mod.py'],
        'plan': {'steps': [{'id': 's1', 'status': 'pending'}]},
    }


async def _fake_create_wt(*_a, **_k):
    return Path('/fake/wt'), 'run-b1'


async def _fake_collect(_wf, _wt, _task, *, role, replay_frame=None):
    return EvalMetrics(
        composite_score=0.9, tests_pass=True, role_under_test=role, replay_frame=replay_frame,
    )


@pytest.mark.asyncio
async def test_b1_an_executor_cell_reads_back_through_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = ResultStore(tmp_path / 'results')
    workflow = MagicMock()
    workflow.run = AsyncMock(return_value=SimpleNamespace(outcome=WorkflowOutcome.DONE))
    monkeypatch.setattr(runner, 'create_eval_worktree', _fake_create_wt)
    monkeypatch.setattr(runner, 'build_workflow', lambda **_kw: workflow)
    monkeypatch.setattr(runner, 'collect_metrics', AsyncMock(side_effect=_fake_collect))
    monkeypatch.setattr(runner, 'load_task', lambda _p: _executor_task(tmp_path))

    result = await runner.run_eval(
        _TASK_PATH,
        EvalConfig('claude-sonnet-max', 'claude', 'sonnet', 'max'),
        eval_base_config(tmp_path),
        usage_gate=None,
        store=store,
    )

    assert store.load_all() == [result]

    truncated = store.directory / 'zz_truncated.json'
    truncated.write_text('{"task_id": "x", "conf')
    with pytest.raises(ResultLoadError) as store_error:
        store.load_all()
    assert store_error.value.path == truncated


_UNADDRESSABLE = EvalConfig('cfg__bad', 'claude', 'sonnet', 'max')
_ADDRESSABLE = EvalConfig('cfg-fine', 'claude', 'sonnet', 'max')


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'executor', ['run_eval', 'run_architect_eval', 'run_end_to_end'],
)
async def test_an_unaddressable_cell_fails_before_the_run_spends_anything(
    executor: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = ResultStore(tmp_path / 'results')
    config_builder = MagicMock(spec=runner.build_eval_orch_config)
    monkeypatch.setattr(runner, 'create_eval_worktree', _fake_create_wt)
    monkeypatch.setattr(runner, 'load_task', lambda _p: _executor_task(tmp_path))
    monkeypatch.setattr(runner, 'build_eval_orch_config', config_builder)
    architect_deps = ArchitectEvalDeps(
        store=store,
        create_worktree=_fake_create_wt,
        cleanup_worktree=AsyncMock(),
        invoke=AsyncMock(),
        read_reference_diff=AsyncMock(return_value=''),
    )
    runs = {
        'run_eval': lambda: runner.run_eval(
            _TASK_PATH, _UNADDRESSABLE, usage_gate=None, store=store,
        ),
        'run_architect_eval': lambda: runner.run_architect_eval(
            _TASK_PATH, _UNADDRESSABLE, usage_gate=None, deps=architect_deps,
        ),
        'run_end_to_end': lambda: runner.run_end_to_end(
            _TASK_PATH, _UNADDRESSABLE, _ADDRESSABLE, usage_gate=None, store=store,
        ),
    }

    with pytest.raises(ValueError, match='cfg__bad'):
        await runs[executor]()

    config_builder.assert_not_called()
    assert store.cell_paths() == []
