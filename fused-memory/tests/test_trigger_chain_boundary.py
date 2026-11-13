"""``trigger_chain`` at the ``submit_task`` tool boundary, on real persisted rows.

PRD plans/completion-driven-triggers-prd.md C2: row 7's guard half (a second
open chain for one skill is refused, and accepted once the first is superseded
or cancelled) and row 15 (a slashed skill is refused with the bare-name hint).
Shapes follow skills/_shared/filing-the-trigger-chain.md §3, with the
predicate script rooted in the project rather than the factory.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _fastmcp_tool_result import tool_result_dict
from _hermetic_task_stack import hermetic_task_stack

pytestmark = pytest.mark.asyncio


@pytest.fixture
def project_root(tmp_path: Path) -> str:
    script = tmp_path / 'scripts' / 'check.py'
    script.parent.mkdir()
    script.write_text('#!/usr/bin/env python3\nraise SystemExit(0)\n')
    os.chmod(script, 0o755)
    return str(tmp_path)


def _chain(role: str, run_id: str, skill: str = 'review') -> dict[str, Any]:
    return {'role': role, 'skill': skill, 'run_id': run_id}


def _completion_gate(root: str, run_id: str) -> dict[str, Any]:
    first_check = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    return {
        'project_root': root,
        'title': f'Completion gate: {run_id} weighted share >= 0.7',
        'description': f'Re-arms daily until the tasks filed by {run_id} reach 0.7.',
        'priority': 'medium',
        'planning_mode': True,
        'task_kind': 'deterministic',
        'agent_id': f'review-{run_id}',
        'metadata': {
            'source': 'trigger-chain',
            'trigger_chain': _chain('completion_gate', run_id),
            'milestone': {'mode': 'dated', 'at': first_check},
            'before_done': {
                'kind': 'predicate',
                'script': 'scripts/check.py',
                'args': ['--run', run_id, '--threshold', '0.7', '--project-root', root],
                'timeout_secs': 120,
                'recheck_secs': 86400,
            },
        },
    }


def _human_gate(root: str, run_id: str, skill: str = 'review') -> dict[str, Any]:
    return {
        'project_root': root,
        'title': f'Run /review on p ({run_id} landed, {skill})',
        'description': 'Human gate: the escalation watcher spawns an attended session.',
        'priority': 'medium',
        'planning_mode': True,
        'task_kind': 'deterministic',
        'agent_id': f'review-{run_id}',
        'metadata': {
            'source': 'trigger-chain',
            'trigger_chain': _chain('human_gate', run_id, skill),
            'always_escalates': True,
        },
    }


async def _call(server, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return tool_result_dict(await server._tool_manager.call_tool(tool, arguments))


async def _file_chain(server, root: str, run_id: str) -> tuple[str, str]:
    completion = await _call(server, 'submit_task', _completion_gate(root, run_id))
    human = await _call(server, 'submit_task', _human_gate(root, run_id))
    assert completion.get('task_id'), completion
    assert human.get('task_id'), human
    return str(completion['task_id']), str(human['task_id'])


async def _titles(server, root: str) -> list[str]:
    listing = await _call(server, 'get_tasks', {'project_root': root})
    return [task['title'] for task in listing['tasks']]


async def test_a_second_open_chain_is_refused_until_the_first_is_superseded(
    tmp_path: Path, project_root: str,
) -> None:
    async with hermetic_task_stack(tmp_path) as (server, _interceptor):
        r1_ids = await _file_chain(server, project_root, 'review-p-1')

        refused = await _call(server, 'submit_task', _completion_gate(project_root, 'review-p-2'))
        assert refused.get('error_type') == 'TriggerChainOpen', refused
        assert {entry['id'] for entry in refused['open_chain']} == set(r1_ids)
        assert 'task_id' not in refused

        for task_id, role in zip(r1_ids, ('completion_gate', 'human_gate'), strict=True):
            updated = await _call(server, 'update_task', {
                'id': task_id,
                'project_root': project_root,
                'metadata': {
                    'trigger_chain': {**_chain(role, 'review-p-1'), 'superseded_by': 'review-p-2'},
                },
            })
            assert 'error' not in updated, updated

        accepted = await _call(server, 'submit_task', _completion_gate(project_root, 'review-p-2'))
        assert accepted.get('task_id'), accepted


async def test_cancelling_the_open_chain_clears_the_guard(
    tmp_path: Path, project_root: str,
) -> None:
    async with hermetic_task_stack(tmp_path) as (server, _interceptor):
        r1_ids = await _file_chain(server, project_root, 'review-p-1')
        refused = await _call(server, 'submit_task', _completion_gate(project_root, 'review-p-2'))
        assert refused.get('error_type') == 'TriggerChainOpen', refused

        for task_id in r1_ids:
            cancelled = await _call(server, 'set_task_status', {
                'id': task_id, 'status': 'cancelled', 'project_root': project_root,
            })
            assert 'error' not in cancelled, cancelled

        accepted = await _call(server, 'submit_task', _completion_gate(project_root, 'review-p-2'))
        assert accepted.get('task_id'), accepted


async def test_a_slashed_skill_is_refused_with_the_bare_name(
    tmp_path: Path, project_root: str,
) -> None:
    async with hermetic_task_stack(tmp_path) as (server, _interceptor):
        submission = _human_gate(project_root, 'review-p-1')
        submission['metadata']['trigger_chain']['skill'] = '/review'

        refused = await _call(server, 'submit_task', submission)

        assert refused.get('error_type') == 'ValidationError', refused
        assert (
            "trigger_chain.skill='/review' must be the bare skill name 'review' "
            '(no leading slash)'
        ) in refused['error']
        assert submission['title'] not in await _titles(server, project_root)


async def test_a_chain_task_carrying_x_finding_run_is_refused(
    tmp_path: Path, project_root: str,
) -> None:
    async with hermetic_task_stack(tmp_path) as (server, _interceptor):
        submission = _completion_gate(project_root, 'review-p-1')
        submission['metadata']['x_finding_run'] = 'review-p-1'

        refused = await _call(server, 'submit_task', submission)

        assert refused.get('error_type') == 'ValidationError', refused
        assert 'x_finding_run' in refused['error']
        assert submission['title'] not in await _titles(server, project_root)
