"""The orchestrator reads its store paths and project id from shared (S14 β).

Contract: plans/review-all-s14-shared-leaves-prd.md (decisions 2, 3, 4; B2, B3).
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from _escalation_server_capture import create_server_kwargs
from _orch_helpers import _init_harness_state_for_test
from _sandbox_soak_fixtures import (
    build_events_db,
    build_tasks_db,
    commit_probe_report,
    init_repo_on_main,
)
from click.testing import CliRunner
from escalation.queue import EscalationQueue
from pydantic import ValidationError
from shared import project_layout
from shared.cli_invoke import AgentResult
from shared.project_id import project_id_for_root
from shared.project_layout import (
    ESCALATION_QUEUE_RELPATH,
    default_project_root,
    escalation_queue_dir,
    park_eviction_requests_db,
    runs_db,
    scheduler_overrides_db,
    tasks_db,
)

from orchestrator import b3_gate, sandbox_soak
from orchestrator import resume_age_bound as rab
from orchestrator.config import OrchestratorConfig, load_config
from orchestrator.config_census_ignore import read_task_statuses
from orchestrator.evals.prompt_opt.__main__ import cli as prompt_opt_cli
from orchestrator.evals.reviewer_trial.__main__ import cli as reviewer_trial_cli
from orchestrator.flake_ledger import ledger_db_path
from orchestrator.harness import Harness
from orchestrator.mcp import markup_sink


@pytest.fixture
def relocated_layout(monkeypatch) -> None:
    """Move the layout, so a consumer that hand-spells a path is left behind."""
    monkeypatch.setattr(project_layout, 'ESCALATION_QUEUE_RELPATH', 'data/relocated-escalations')
    monkeypatch.setattr(
        project_layout, 'orchestrator_data_dir', lambda root: Path(root) / 'relocated-orchestrator',
    )


def _load(monkeypatch, root: Path, body: dict) -> OrchestratorConfig:
    for key in [k for k in os.environ if k.startswith('ORCH_')]:
        monkeypatch.delenv(key)
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / 'dark-factory-orchestrator.yaml'
    config_path.write_text(yaml.safe_dump({'project_root': str(root), **body}))
    return load_config(config_path)


class TestEscalationQueueDirIsTheLayoutValue:
    @pytest.mark.parametrize('queue_dir', ['data/esc2', 'ABSOLUTE'])
    def test_a_non_layout_queue_dir_is_refused_naming_the_layout(
        self, monkeypatch, tmp_path, queue_dir,
    ):
        if queue_dir == 'ABSOLUTE':
            queue_dir = str(tmp_path / 'elsewhere')
        with pytest.raises(ValidationError) as excinfo:
            _load(monkeypatch, tmp_path, {'escalation': {'queue_dir': queue_dir}})
        message = str(excinfo.value)
        assert 'shared.project_layout.escalation_queue_dir' in message
        assert queue_dir in message
        assert ESCALATION_QUEUE_RELPATH in message

    @pytest.mark.parametrize('escalation', [None, {'queue_dir': ESCALATION_QUEUE_RELPATH}])
    def test_the_layout_value_loads_whether_omitted_or_declared(
        self, monkeypatch, tmp_path, escalation,
    ):
        body = {} if escalation is None else {'escalation': escalation}
        config = _load(monkeypatch, tmp_path, body)
        assert config.escalation.queue_dir == ESCALATION_QUEUE_RELPATH


class TestFusedMemoryProjectIdDerivesFromRoot:
    def test_an_omitted_project_id_derives_from_the_root_basename(self, monkeypatch, tmp_path):
        root = tmp_path / 'x' / 'Know-Live'
        config = _load(monkeypatch, root, {})
        assert config.fused_memory.project_id == 'know_live'
        assert config.fused_memory.project_id == project_id_for_root(root)

    def test_a_declared_project_id_wins(self, monkeypatch, tmp_path):
        root = tmp_path / 'reify-fix-3'
        config = _load(monkeypatch, root, {'fused_memory': {'project_id': 'reify'}})
        assert config.fused_memory.project_id == 'reify'

    def test_the_derived_id_survives_the_reload_round_trip(self, monkeypatch, tmp_path):
        config = _load(monkeypatch, tmp_path / 'x' / 'Know-Live', {})
        reloaded = OrchestratorConfig.model_validate(config.model_dump())
        assert reloaded.fused_memory.project_id == 'know_live'


def test_scheduler_store_paths_are_the_layout_paths(tmp_path):
    config = OrchestratorConfig(project_root=tmp_path)
    assert config.overrides_db_path == scheduler_overrides_db(tmp_path)
    assert config.park_eviction_requests_db_path == park_eviction_requests_db(tmp_path)


@pytest.mark.usefixtures('relocated_layout')
class TestHarnessFollowsTheLayout:
    @pytest.mark.asyncio
    async def test_the_escalation_server_serves_the_layout_queue(
        self, mock_orch_config, tmp_path,
    ):
        mock_orch_config.max_concurrent_tasks = 2
        mock_orch_config.fused_memory.project_id = 'test-project'
        mock_orch_config.escalation.queue_dir = str(tmp_path / 'legacy-esc')
        mock_orch_config.escalation.host = '127.0.0.1'
        mock_orch_config.escalation.port = 0
        with patch('orchestrator.harness.McpLifecycle'), \
             patch('orchestrator.harness.Scheduler'), \
             patch('orchestrator.harness.BriefingAssembler'):
            h = Harness(mock_orch_config)
        h.review_checkpoint = None

        kwargs = await create_server_kwargs(h)

        assert kwargs['store_identity'].queue_dir == escalation_queue_dir(tmp_path).resolve()

    @pytest.mark.asyncio
    async def test_the_watcher_banner_names_the_layout_queue(self, tmp_path):
        h = Harness.__new__(Harness)
        _init_harness_state_for_test(h)
        h.config = OrchestratorConfig(project_root=tmp_path)
        h._watcher_supervisor_task = None
        h._watcher_unclean_exits = deque()
        h.usage_gate = None
        h.cost_store = MagicMock()
        h._run_id = 'run-test-layout-rotation'
        h.mcp = MagicMock()
        h.mcp.mcp_config_json.return_value = {'mcp': 'config-sentinel'}
        captured: dict = {}

        async def fake_invoke(usage_gate, label, *, invoke_fn, **kwargs):
            captured['prompt'] = kwargs.get('prompt', '')
            return AgentResult(success=True, output='')

        with (
            patch('orchestrator.harness.invoke_with_cap_retry', fake_invoke),
            patch('orchestrator.harness.resolve_dark_factory_root', return_value=tmp_path / 'df'),
        ):
            await h._run_watcher_rotation()

        expected = escalation_queue_dir(h.config.project_root)
        assert f'--queue-dir {expected}' in captured['prompt']
        assert f'Escalation queue: {expected}' in captured['prompt']


@pytest.mark.usefixtures('relocated_layout')
def test_the_flake_ledger_lives_in_the_layout_runs_db(tmp_path):
    assert ledger_db_path(tmp_path) == runs_db(tmp_path)


@pytest.mark.usefixtures('relocated_layout')
class TestOperatorToolsFollowTheLayout:
    def test_the_b3_state_file_lands_in_the_layout_queue(self, tmp_path):
        assert b3_gate.main(['charge', '--task-id', '42', '--project-root', str(tmp_path)]) == 0
        assert (escalation_queue_dir(tmp_path) / 'b3-state.json').is_file()

    def test_the_sandbox_soak_defaults_read_the_layout_stores(self, tmp_path, capsys):
        repo = tmp_path / 'repo'
        init_repo_on_main(repo)
        commit_probe_report(repo)
        runs_db(repo).parent.mkdir(parents=True)
        build_events_db(runs_db(repo))
        tasks_db(repo).parent.mkdir(parents=True)
        build_tasks_db(tasks_db(repo), [(1, 'done'), (2, 'done'), (3, 'done')])

        rc = sandbox_soak.main(['--repo-root', str(repo), '--min-done', '3'])

        assert rc == 0, capsys.readouterr()

    @pytest.mark.asyncio
    async def test_the_markup_guard_files_into_the_layout_queue(self, tmp_path):
        spec = markup_sink.MarkupSinkSpec(
            server_label='plan-tools',
            agent_role='plan-tools-markup-guard',
            residue_anchor_task_id='plan-tools-markup-residue',
            storm_anchor_task_id='plan-tools-markup-storm',
            refusal_consequence='Nothing was written to the plan.',
            storm_consequence='every later leaked call is written through.',
            attribution_source='read the test journal',
        )
        sink = markup_sink.make_escalation_sink(
            worktree=tmp_path / 'proj' / '.worktrees' / '6714',
            spec=spec,
            subject_task_id=lambda: '6714',
            resolve_root=lambda _wt: tmp_path,
        )

        filed = await sink({
            'error_type': markup_sink.MARKUP_STORM_ERROR_TYPE,
            'count': 7,
            'threshold': 5,
            'window_seconds': 3600,
            'outcome': 'repaired',
            'project': None,
            'callers': [],
        })

        assert [e.id for e in EscalationQueue(escalation_queue_dir(tmp_path)).get_pending()] == [
            filed,
        ]


def test_the_census_ignore_audit_reads_the_layout_task_store(tmp_path):
    db = tasks_db(tmp_path)
    db.parent.mkdir(parents=True)
    conn = sqlite3.connect(db)
    conn.execute('CREATE TABLE tasks (tag TEXT, id INTEGER, status TEXT, metadata TEXT)')
    conn.execute(
        'INSERT INTO tasks VALUES (?, ?, ?, ?)',
        ('master', 7, 'deferred', json.dumps({'do_not_complete': True})),
    )
    conn.commit()
    conn.close()

    statuses = read_task_statuses(tmp_path)

    assert statuses is not None
    assert statuses[7].status == 'deferred'
    assert statuses[7].do_not_complete is True


@pytest.mark.usefixtures('relocated_layout')
class TestHomePathDefaultsDeriveFromTheMainCheckout:
    def test_the_resume_age_bound_reads_the_layout_runs_db(self, monkeypatch):
        monkeypatch.delenv(rab.RUNS_DB_ENV_VAR, raising=False)
        assert rab.default_runs_db_path() == runs_db(default_project_root())

    def test_the_prompt_opt_canary_reads_the_layout_runs_db(self, monkeypatch):
        monkeypatch.delenv('PROMPT_OPT_RUNS_DB', raising=False)

        result = CliRunner().invoke(prompt_opt_cli, [
            'canary', '--deploy-at', '2026-06-15T00:00:00+00:00', '--project-id', 'dark_factory',
        ])

        assert result.exit_code == 2, result.output
        assert f'runs.db not found at {runs_db(default_project_root())}' in result.output

    def test_the_reviewer_trial_miner_reads_the_layout_runs_db(self, monkeypatch):
        monkeypatch.delenv('REVIEWER_TRIAL_RUNS_DB', raising=False)

        result = CliRunner().invoke(reviewer_trial_cli, ['mine'])

        assert result.exit_code == 1, result.output
        assert f'runs.db not found at {runs_db(default_project_root())}' in result.output
