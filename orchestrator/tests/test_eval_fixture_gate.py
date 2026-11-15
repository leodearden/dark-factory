"""An eval corpus cell's task gate is its fixture's own verify_commands (task 9918).

Production's module registry and scoping describe today's checkout, not the
tree at the fixture's ``pre_task_commit``. A corpus cell therefore gates each
verify attempt on the fixture's declared commands, run verbatim, which are the
same commands the scoring verify runs.
"""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _briefing_helpers import MEMORY_TRANSPORT
from _workflow_helpers import PLAN, AgentStub, _init_repo

from orchestrator.config import load_config
from orchestrator.evals.configs import EvalConfig
from orchestrator.evals.results import ResultStore
from orchestrator.evals.runner import run_eval


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.run(
        ['git', *args], cwd=str(cwd), check=True, capture_output=True, text=True,
    ).stdout.strip()


@pytest.mark.asyncio
async def test_a_corpus_cell_gates_on_its_fixture_verify_commands_not_the_live_module_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    proj = tmp_path / 'proj'
    _init_repo(proj)
    pre_task_commit = _git(['rev-parse', 'HEAD'], cwd=proj)
    marker = tmp_path / 'gate.log'

    module_yaml = proj / 'lib' / 'orchestrator.yaml'
    module_yaml.parent.mkdir()
    module_yaml.write_text(''.join(
        f'{key}: "echo module-{gate} >> {marker} && exit 1"\n'
        for key, gate in (
            ('test_command', 'test'),
            ('lint_command', 'lint'),
            ('type_check_command', 'typecheck'),
        )
    ))
    _git(['add', 'lib/orchestrator.yaml'], cwd=proj)
    _git(['commit', '-m', "today's red module gate"], cwd=proj)

    config_path = tmp_path / 'orch.yaml'
    config_path.write_text(f'project_root: {proj}\n')
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(config_path))
    monkeypatch.setenv('ORCH_PROJECT_ROOT', str(proj))
    base = load_config(config_path)
    assert base.for_module('lib') is not None, 'the live registry must carry the red module'

    task_path = tmp_path / 'df_task_9918.json'
    task_path.write_text(json.dumps({
        'id': 'df_task_9918',
        'name': 'fixture gate cell',
        'project_root': str(proj),
        'pre_task_commit': pre_task_commit,
        'modules': ['lib'],
        'judge_after_each_iteration': False,
        'timeout_minutes': 5,
        'task_definition': {'title': PLAN['title'], 'description': PLAN['analysis']},
        'plan': {**copy.deepcopy(PLAN), 'task_id': 'df_task_9918'},
        'verify_commands': {
            gate: f'echo fixture-{gate} >> {marker}'
            for gate in ('test', 'lint', 'typecheck')
        },
    }))

    monkeypatch.setattr('orchestrator.workflow.invoke_agent', AgentStub().invoke_agent)
    monkeypatch.setattr(
        MEMORY_TRANSPORT,
        AsyncMock(side_effect=ConnectionRefusedError(111, 'Connection refused')),
    )

    result = await run_eval(
        task_path, EvalConfig('t', 'claude', 'sonnet', 'high'), base,
        usage_gate=None, store=ResultStore(tmp_path / 'results'),
    )

    assert result.outcome == 'done'
    assert set(marker.read_text().split()) == {
        'fixture-test', 'fixture-lint', 'fixture-typecheck',
    }
