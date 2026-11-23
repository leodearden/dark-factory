"""Recon stage prompts and payload footers ask only for what each stage can do (task 10491).

A stage finishes by calling ``complete`` and records findings through the
recon-report tools; it has no "structured report" response channel. Stage 1
proposes consolidation clusters rather than merging by hand, and Stage 2
cancels tasks rather than deleting them. These tests assert the absence of the
contradicting tokens; they do not pin prose (task 5192). Helpers are copied
rather than imported across test modules (house norm).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from unittest.mock import AsyncMock

import pytest

from fused_memory.models.reconciliation import AssembledPayload, StageId, Watermark
from fused_memory.models.scope import ProjectId, ProjectRoot, ProjectScope
from fused_memory.reconciliation.prompts.stage2 import build_stage2_system_prompt
from fused_memory.reconciliation.stages.base import BaseStage
from fused_memory.reconciliation.stages.task_knowledge_sync import (
    IntegrityCheck,
    TaskKnowledgeSync,
)
from fused_memory.reconciliation.task_filter import (
    is_count_snapshot,
    is_mixed_temporal_framing,
)
from reconciliation.consolidator_fixtures import make_consolidator
from reconciliation.harness_fixtures import (
    make_memory_service,
    make_reconciliation_config,
)

PROJECT_ID = 'test_proj'
SCOPE_ROOT = '/tmp/test_proj'


def _deps(project_id: str = PROJECT_ID) -> dict:
    taskmaster = AsyncMock()
    taskmaster.get_tasks.return_value = {'tasks': []}
    return {
        'memory_service': make_memory_service(),
        'taskmaster': taskmaster,
        'journal': AsyncMock(),
        'config': make_reconciliation_config(),
        'scope': ProjectScope(ProjectId(project_id), ProjectRoot(SCOPE_ROOT)),
    }


def _stage1():
    return make_consolidator(project_id=PROJECT_ID, project_root=SCOPE_ROOT)


def _stage1_assembled():
    stage = _stage1()
    stage.assembled_payload = AssembledPayload(events=[], context_items={})
    return stage


def _stage1_remediation():
    stage = _stage1()
    stage.remediation_findings = [{'description': 'test finding'}]
    return stage


def _stage2(project_id: str = PROJECT_ID):
    stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **_deps(project_id))
    stage._current_run_id = 'test-run'
    return stage


def _stage3():
    return IntegrityCheck(StageId.integrity_check, **_deps())


_STAGE_FACTORIES: dict[str, Callable[[], BaseStage]] = {
    'stage1_legacy': _stage1,
    'stage1_assembled': _stage1_assembled,
    'stage1_remediation': _stage1_remediation,
    'stage2': _stage2,
    'stage3': _stage3,
}


async def _assembled_payload(payload_name: str) -> str:
    stage = _STAGE_FACTORIES[payload_name]()
    return await stage.assemble_payload([], Watermark(project_id=PROJECT_ID), [])


def _section(text: str, heading: str) -> str:
    start = text.index(f'\n{heading}') + 1
    end = text.find('\n## ', start + len(heading))
    return text[start:] if end == -1 else text[start:end]


@pytest.mark.asyncio
@pytest.mark.parametrize('payload_name', tuple(_STAGE_FACTORIES))
async def test_payload_asks_for_no_structured_report(payload_name):
    payload = await _assembled_payload(payload_name)

    assert 'structured report' not in payload


@pytest.mark.asyncio
@pytest.mark.parametrize('payload_name', ('stage1_legacy', 'stage1_assembled'))
async def test_stage1_payload_does_not_ask_for_hand_merging(payload_name):
    payload = await _assembled_payload(payload_name)

    assert 'Merge/delete' not in payload


@pytest.mark.asyncio
async def test_stage2_payload_task_steps_do_not_ask_to_delete_tasks():
    payload = await _assembled_payload('stage2')

    assert 'delete tasks' not in payload[payload.index('## Your Task'):]


def test_stage2_task_accounting_rules_flag_through_no_structured_report():
    section = _section(
        build_stage2_system_prompt('dark_factory'), '## Verifying Task Operations',
    )

    assert 'structured report' not in section


@pytest.mark.parametrize('project_id', ('dark_factory', 'autopilot_video'))
def test_stage2_prompt_asks_for_no_git_history_or_working_tree(project_id):
    prompt = build_stage2_system_prompt(project_id)
    claim_rules = _section(prompt, '## Verifying specific commit/diff/stamp claims')

    assert 'current working tree' not in prompt
    assert 'git grep' not in claim_rules
    assert 'git log' not in claim_rules


def test_no_stage_prompt_asks_for_the_retired_memories_written_counter():
    stages = {
        'stage1': _stage1(),
        **{f'stage2_{pid}': _stage2(pid) for pid in ('dark_factory', 'autopilot_video')},
        'stage3': _stage3(),
    }

    for name, stage in stages.items():
        assert 'memories_written' not in stage.get_system_prompt(), name


def test_stage1_update_edge_example_fact_passes_the_add_memory_write_guard():
    section = _section(
        make_consolidator().get_system_prompt(), '## update_edge Temporal Limitation',
    )
    examples = re.findall(r'`"(As of [^"`]+)"`', section)

    assert examples
    for example in examples:
        assert is_count_snapshot(example) is False, example
        assert is_mixed_temporal_framing(example) is False, example
