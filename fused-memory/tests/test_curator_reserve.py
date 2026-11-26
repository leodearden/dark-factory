"""The memory service's task-filing calls are essential, so the allowance reserve serves them.

Contract: plans/allowance-reserve-by-account-prd.md §6.1 (the curator, the pack judge and the
path-scope adjudicator are essential) and §7's B3. Each test drives a real UsageGate in the B2
state: every account but the reserve capped, the reserve open, its protected window. The invoke
double stands in for the CLI spawn only; it opens the real gate's slot with the class its
caller passed, forwarded only when given, as ``invoke_with_cap_retry`` does.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import aiosqlite
import pytest
from _curator_helpers import agent_result, batch_create_result, empty_corpus, make_config, prepared
from shared.cli_invoke import AgentResult
from shared.cost_store import CostStore
from shared.prompt_artifact import PromptArtifactStore
from shared.testing_reserve import (
    PROTECTED,
    RESERVE_ACCOUNT,
    ReserveClock,
    cap_all_but_reserve,
    reserve_gate,
)
from shared.usage_gate import UsageGate

from fused_memory.config.schema import PathScopeAdjudicatorConfig
from fused_memory.middleware import pack_judge
from fused_memory.middleware.followup_packing import CandidatePack
from fused_memory.middleware.path_scope_adjudicator import PathScopeAdjudicator
from fused_memory.middleware.task_curator import TaskCurator

PACK = CandidatePack('parent', '1', (1, 2, 3, 4, 5), ('a/x.py',), survivor=1, pack_key='0' * 16)
RENDERED = tuple(f'[Task {task_id}] rendered member block {task_id}' for task_id in PACK.members)
WHOLE_PACK = {
    'verdict': 'pack',
    'groups': [{
        'member_ids': [1, 2, 3, 4, 5],
        'title': 'One coherent task',
        'description': 'what changes',
        'details': 'how',
        'justification': 'one concern',
    }],
    'justification': 'one concern',
}

GateInvoke = Callable[..., Awaitable[AgentResult]]


@contextlib.asynccontextmanager
async def b2_gate(tmp_path: Path) -> AsyncIterator[tuple[UsageGate, Path]]:
    """A real gate in the B2 state, its events landing in a real store at the yielded path."""
    gate = reserve_gate(tmp_path, clock=ReserveClock(PROTECTED))
    cap_all_but_reserve(gate)
    db_path = tmp_path / 'events.db'
    store = CostStore(db_path)
    await store.open()
    gate.attach_cost_store(store)
    try:
        yield gate, db_path
    finally:
        await gate.shutdown()
        await store.close()


def gate_faithful_invoke(answer: AgentResult, leased: list[str]) -> GateInvoke:
    """An invoke that is served by the real gate, then answers *answer* from the leased account."""

    async def invoke(**kwargs: Any) -> AgentResult:
        gate: UsageGate = kwargs['usage_gate']
        forwarded = {'work_class': kwargs['work_class']} if 'work_class' in kwargs else {}
        async with gate.invoke_slot(park=False, **forwarded) as slot:
            slot.confirm(0.25)
            name = slot.account_name
        leased.append(name)
        return dataclasses.replace(answer, account_name=name)

    return invoke


async def landed_served_details(db_path: Path) -> list[dict[str, Any]]:
    """The details of every ``reserve_served`` row, once the gate's fire-and-forget write landed.

    The per-test timeout is the only hang backstop, so a slow loop cannot fail a write that landed.
    """
    while True:
        async with aiosqlite.connect(db_path) as conn:
            cursor = await conn.execute(
                "SELECT details FROM account_events WHERE event_type = 'reserve_served'"
            )
            details = [json.loads(row[0]) for row in await cursor.fetchall()]
        if details:
            return details
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_a_curator_batch_is_served_by_the_reserve(tmp_path: Path) -> None:
    leased: list[str] = []
    async with b2_gate(tmp_path) as (gate, db_path):
        curator = TaskCurator(
            config=make_config(), taskmaster=None, usage_gate=gate, config_dir_base=tmp_path,
            invoke=gate_faithful_invoke(batch_create_result(2), leased),
        )
        with patch.object(curator, '_build_corpus', side_effect=empty_corpus):
            decisions = await curator.curate_prepared(
                [prepared('Alpha'), prepared('Beta')], 'p', '/p',
            )
        served = await landed_served_details(db_path)

    assert [decision.action for decision in decisions] == ['create', 'create']
    assert leased == [RESERVE_ACCOUNT]
    assert [(row['kind'], row['role'], row['project_id']) for row in served] == [
        ('essential', 'task_curator_batch', 'p'),
    ]


@pytest.mark.asyncio
async def test_the_pack_judge_is_served_by_the_reserve(tmp_path: Path) -> None:
    leased: list[str] = []
    async with b2_gate(tmp_path) as (gate, _db_path):
        curator = TaskCurator(
            make_config(), usage_gate=gate, prompt_store=PromptArtifactStore(tmp_path / 'prompts'),
            config_dir_base=tmp_path, invoke=gate_faithful_invoke(agent_result(WHOLE_PACK), leased),
        )
        verdict, _call = await pack_judge.judge_pack(curator, PACK, RENDERED, project_id='p')

    assert verdict.verdict == 'pack'
    assert leased == [RESERVE_ACCOUNT]


@pytest.mark.asyncio
async def test_the_path_scope_adjudicator_is_served_by_the_reserve(tmp_path: Path) -> None:
    leased: list[str] = []
    allow = agent_result({'verdict': 'allow', 'reason': 'ok'})
    async with b2_gate(tmp_path) as (gate, _db_path):
        adjudicator = PathScopeAdjudicator(
            config=PathScopeAdjudicatorConfig(), usage_gate=gate, cwd=tmp_path,
        )
        with patch(
            'fused_memory.middleware.path_scope_adjudicator.invoke_with_cap_retry',
            new=gate_faithful_invoke(allow, leased),
        ):
            verdict = await adjudicator.adjudicate(
                title='Update orchestrator/harness.py to fix a bug',
                description='Fix a crash in the harness',
                matched_paths=('orchestrator/',),
                project_id='p',
                suggested_project='orchestrator',
                project_root='/some/project',
            )

    assert not verdict.failed, verdict.reason
    assert leased == [RESERVE_ACCOUNT]
