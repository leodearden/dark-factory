"""ResultStore owns the persisted eval cell: its type, on-disk name and directory.

Driven only through the public names of ``orchestrator.evals.results`` against
real files under ``tmp_path``; the B1 boundary case drives a real executor and
reads its cell back through both loaders.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from _campaign_gate_helpers import eval_base_config

import orchestrator.evals.results as results_module
from orchestrator.evals import runner
from orchestrator.evals.configs import EvalConfig
from orchestrator.evals.results import (
    CellName,
    EvalResult,
    MissingResultsDirError,
    ResultLoadError,
    ResultStore,
    normalize_legacy_metrics,
)
from orchestrator.workflow import WorkflowOutcome

_SCRIPTS_LOADER = Path(__file__).resolve().parents[2] / 'scripts' / '_eval_results_io.py'


def _result(
    task_id: str = 'df_task_1',
    config_name: str = 'cfg-a',
    run_id: str = 'abcd1234',
    **overrides,
) -> EvalResult:
    fields = {
        'outcome': 'done',
        'metrics': {'tests_passed': True, 'cost_usd': 1.5},
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


class TestParseName:
    def test_parse_name_inverts_path_for(self, store: ResultStore):
        r = _result()
        assert ResultStore.parse_name(store.path_for(r)) == CellName(
            r.task_id, r.config_name, r.run_id
        )

    def test_a_combo_label_with_plus_round_trips(self, store: ResultStore):
        r = _result(config_name='arch-a+impl-b')
        assert ResultStore.parse_name(store.path_for(r)) == CellName(
            'df_task_1', 'arch-a+impl-b', 'abcd1234'
        )

    @pytest.mark.parametrize(
        'name',
        [
            'df_task_12__claude-opus-high.json',
            'truncated.json',
            'a__b__c__d.json',
            'df_task_1__cfg-a__abcd1234.txt',
        ],
    )
    def test_parse_name_rejects_names_outside_the_grammar(
        self, tmp_path: Path, name: str
    ):
        path = tmp_path / name
        with pytest.raises(ValueError) as excinfo:
            ResultStore.parse_name(path)
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
    metrics = {'cost_usd': 2.0, 'nested': {'a': 1}}
    normalized = normalize_legacy_metrics(metrics)
    assert normalized == metrics
    assert normalized is not metrics


def test_the_packaged_store_lives_beside_the_results_module():
    assert ResultStore.packaged().directory == (
        Path(results_module.__file__).parent / 'results'
    )


def _load_scripts_loader():
    spec = importlib.util.spec_from_file_location('_eval_results_io', _SCRIPTS_LOADER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_b1_an_executor_cell_reads_back_through_both_loaders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = ResultStore(tmp_path / 'results')
    task = {
        'id': 'df_task_b1',
        'project_root': str(tmp_path),
        'pre_task_commit': 'basecommit',
        'task_definition': {'title': 'W', 'description': 'd'},
        'modules': ['pkg/mod.py'],
        'plan': {'steps': [{'id': 's1', 'status': 'pending'}]},
    }

    async def fake_create_wt(*_a, **_k):
        return Path('/fake/wt'), 'run-b1'

    workflow = MagicMock()
    workflow.run = AsyncMock(return_value=SimpleNamespace(outcome=WorkflowOutcome.DONE))
    metrics = MagicMock()
    metrics.to_dict.return_value = {'composite_score': 0.9, 'tests_pass': True}
    monkeypatch.setattr(runner, 'create_eval_worktree', fake_create_wt)
    monkeypatch.setattr(runner, 'build_workflow', lambda **_kw: workflow)
    monkeypatch.setattr(runner, 'collect_metrics', AsyncMock(return_value=metrics))
    monkeypatch.setattr(runner, 'load_task', lambda _p: task)

    result = await runner.run_eval(
        Path('/fake/task.json'),
        EvalConfig('claude-sonnet-max', 'claude', 'sonnet', 'max'),
        eval_base_config(tmp_path),
        usage_gate=None,
        store=store,
    )

    scripts_loader = _load_scripts_loader()
    assert store.load_all() == [result]
    assert scripts_loader.load_results_from_dir(store.directory) == [result]

    truncated = store.directory / 'zz_truncated.json'
    truncated.write_text('{"task_id": "x", "conf')
    with pytest.raises(ResultLoadError) as store_error:
        store.load_all()
    assert store_error.value.path == truncated
    with pytest.raises(SystemExit) as scripts_error:
        scripts_loader.load_results_from_dir(store.directory)
    assert str(truncated) in str(scripts_error.value)
