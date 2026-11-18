"""Recon stage prompts and payload footers ask only for what each stage can do (task 10491).

A stage finishes by calling ``complete`` and records findings through the
recon-report tools; it has no "structured report" response channel. Stage 1
proposes consolidation clusters rather than merging by hand, and Stage 2
cancels tasks rather than deleting them. These tests assert the absence of the
contradicting tokens; they do not pin prose (task 5192). Helpers are copied
rather than imported across test modules (house norm).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import AssembledPayload, StageId, Watermark
from fused_memory.models.scope import ProjectId, ProjectRoot, ProjectScope
from fused_memory.reconciliation.prompts.stage2 import build_stage2_system_prompt
from fused_memory.reconciliation.stages.task_knowledge_sync import (
    IntegrityCheck,
    TaskKnowledgeSync,
)
from reconciliation.consolidator_fixtures import make_consolidator

PROJECT_ID = 'test_proj'
PROJECT_ROOT = '/tmp/test_proj'
STAGE1_PAYLOADS = ('stage1_legacy', 'stage1_assembled', 'stage1_remediation')
ALL_PAYLOADS = (*STAGE1_PAYLOADS, 'stage2', 'stage3')


def _deps(project_id: str = PROJECT_ID) -> dict:
    memory_service = AsyncMock()
    memory_service.get_episodes = AsyncMock(return_value=[])
    memory_service.mem0 = AsyncMock()
    memory_service.mem0.get_all = AsyncMock(return_value={'results': []})
    memory_service.get_status = AsyncMock(return_value={})
    memory_service.earliest_unfinished_graphiti_write_start = MagicMock(return_value=None)
    taskmaster = AsyncMock()
    taskmaster.get_tasks.return_value = {'tasks': []}
    return {
        'memory_service': memory_service,
        'taskmaster': taskmaster,
        'journal': AsyncMock(),
        'config': ReconciliationConfig(enabled=True, explore_codebase_root='/tmp/test'),
        'scope': ProjectScope(ProjectId(project_id), ProjectRoot(PROJECT_ROOT)),
    }


def _stage1():
    return make_consolidator(project_id=PROJECT_ID, project_root=PROJECT_ROOT)


def _stage2(project_id: str = PROJECT_ID):
    stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **_deps(project_id))
    stage._current_run_id = 'test-run'
    return stage


async def _assembled_payloads() -> dict[str, str]:
    assembled = _stage1()
    assembled.assembled_payload = AssembledPayload(events=[], context_items={})
    remediation = _stage1()
    remediation.remediation_findings = [{'description': 'test finding'}]
    stages = {
        'stage1_legacy': _stage1(),
        'stage1_assembled': assembled,
        'stage1_remediation': remediation,
        'stage2': _stage2(),
        'stage3': IntegrityCheck(StageId.integrity_check, **_deps()),
    }
    return {
        name: await stage.assemble_payload([], Watermark(project_id=PROJECT_ID), [])
        for name, stage in stages.items()
    }


def _section(text: str, heading: str) -> str:
    start = text.index(heading)
    end = text.find('\n## ', start + len(heading))
    return text[start:] if end == -1 else text[start:end]


@pytest.mark.asyncio
@pytest.mark.parametrize('payload_name', ALL_PAYLOADS)
async def test_payload_asks_for_no_structured_report(payload_name):
    payload = (await _assembled_payloads())[payload_name]

    assert 'structured report' not in payload


@pytest.mark.asyncio
@pytest.mark.parametrize('payload_name', ('stage1_legacy', 'stage1_assembled'))
async def test_stage1_payload_does_not_ask_for_hand_merging(payload_name):
    payload = (await _assembled_payloads())[payload_name]

    assert 'Merge/delete' not in payload


@pytest.mark.asyncio
async def test_stage2_payload_task_steps_do_not_ask_to_delete_tasks():
    payload = (await _assembled_payloads())['stage2']

    assert 'delete tasks' not in payload[payload.index('## Your Task'):]


def test_stage2_task_accounting_rules_flag_through_no_structured_report():
    section = _section(
        build_stage2_system_prompt('dark_factory'), '## Verifying Task Operations',
    )

    assert 'structured report' not in section


@pytest.mark.parametrize('project_id', ('dark_factory', 'autopilot_video'))
def test_stage2_prompt_asks_for_no_git_history_or_working_tree(project_id):
    prompt = build_stage2_system_prompt(project_id)

    assert 'git log --all' not in prompt
    assert 'current working tree' not in prompt


def test_no_stage_prompt_asks_for_the_retired_memories_written_counter():
    stages = {
        'stage1': _stage1(),
        **{f'stage2_{pid}': _stage2(pid) for pid in ('dark_factory', 'autopilot_video')},
        'stage3': IntegrityCheck(StageId.integrity_check, **_deps()),
    }

    for name, stage in stages.items():
        assert 'memories_written' not in stage.get_system_prompt(), name
