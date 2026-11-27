"""Every markup detection plan-tools makes, inbound or stored, reaches plan-tools.jsonl.

Task 11978, the task 3382 shape. Its ``design_decisions[1].decision`` was
written before the boundary guard existed, and on re-dispatch the read-time
repair classified it unrepairable on a ``mark_step_done`` response only, so no
journal line named it. Both halves are pinned here through the REAL
``plan_tools.create_server`` over a ``fastmcp.Client``, the only caller that
traverses middleware: an inbound replay is refused and journalled as
``markup_detected``, and stored damage read by an accepted call is journalled
as :data:`~orchestrator.mcp.stored_markup_journal.STORED_MARKUP_FACT`.

Every specimen is built from ``_markup_helpers``, never spelled raw
(``tests/scripts/test_no_raw_envelope_literal.py``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from _markup_helpers import INVOKE_CLOSER, closer, invoke_opener, param_opener
from _markup_sink_rig import RecordingQueue, seed_plan
from escalation.models import Escalation
from fastmcp import Client
from fastmcp.exceptions import ToolError

from orchestrator.artifacts import TaskArtifacts
from orchestrator.mcp import markup_journal, plan_markup_repair, plan_markup_stamp, plan_tools
from orchestrator.mcp.stored_markup_journal import STORED_MARKUP_FACT

SUBJECT = '3382'

_DECISION_PROSE = 'Carry the lane plan forward into the task meta directory untouched.'
_ABSORBED_RATIONALE_PROSE = 'A re-dispatched lane must resume from the very plan it confirmed.'
_NEXT_DECISION_PROSE = 'Forward completion memories through the existing workflow hook.'
_STORED_RATIONALE_PROSE = 'The architect chose this over a fresh plan to keep step history.'

#: decision, then the rationale it swallowed, then the opening of a NEXT
#: add_design_decision call: the absorbed tail re-declares ``decision``.
INBOUND_DECISION = (
    _DECISION_PROSE + closer('decision') + '\n'
    + param_opener('rationale') + _ABSORBED_RATIONALE_PROSE + closer('rationale') + '\n'
    + INVOKE_CLOSER + '\n'
    + invoke_opener('mcp__plan-tools__add_design_decision') + '\n'
    + param_opener('decision') + _NEXT_DECISION_PROSE
)

#: The trailing mis-close a last parameter leaves behind: repairable on read.
STORED_RATIONALE = _STORED_RATIONALE_PROSE + closer('rationale') + '\n' + INVOKE_CLOSER + '\n'

DAMAGED_PROSE = (
    _DECISION_PROSE, _ABSORBED_RATIONALE_PROSE, _NEXT_DECISION_PROSE, _STORED_RATIONALE_PROSE,
)


@pytest.fixture(autouse=True)
def _clear_reported_refusals():
    plan_markup_repair.forget_reported_refusals()
    yield
    plan_markup_repair.forget_reported_refusals()


@pytest.fixture(autouse=True)
def _clear_pending_refusals():
    plan_markup_stamp.clear_pending()
    yield
    plan_markup_stamp.clear_pending()


class Lane:
    """The real plan-tools server over a temp worktree whose plan belongs to 3382."""

    def __init__(self, monkeypatch, tmp_path: Path) -> None:
        self.artifacts = TaskArtifacts(tmp_path)
        self.artifacts.init(SUBJECT, 'Test task', 'A test')
        seed_plan(self.artifacts, SUBJECT)
        self.root = tmp_path
        monkeypatch.setattr(plan_tools, '_markup_project_root', lambda worktree: tmp_path)
        monkeypatch.setattr(
            plan_tools, '_escalation_channel', lambda root: (Escalation, RecordingQueue())
        )
        self.server = plan_tools.create_server(self.artifacts)
        self._steps = 0

    async def call(self, tool: str, arguments: dict[str, Any]) -> Any:
        async with Client(self.server) as client:
            return await client.call_tool(tool, arguments)

    async def accepted_call(self) -> dict[str, Any]:
        self._steps += 1
        result = await self.call('add_plan_step', {
            'step_id': f'step-{self._steps}',
            'step_type': 'test',
            'description': 'A clean step description.',
        })
        return result.structured_content

    async def refused_replay(self) -> dict[str, Any]:
        with pytest.raises(ToolError) as excinfo:
            await self.call(
                'add_design_decision',
                {'decision': INBOUND_DECISION},
            )
        return json.loads(str(excinfo.value))

    def store_3382_damage(self) -> None:
        plan = self.artifacts.read_plan()
        plan['design_decisions'] = [
            {'decision': 'A clean stored decision.', 'rationale': STORED_RATIONALE},
            {'decision': INBOUND_DECISION, 'rationale': STORED_RATIONALE},
        ]
        self.artifacts.write_plan(plan)

    @property
    def journal_file(self) -> Path:
        return markup_journal.journal_path(self.root, 'plan-tools')

    def lines(self, fact: str) -> list[dict[str, Any]]:
        if not self.journal_file.exists():
            return []
        parsed = [
            json.loads(line)
            for line in self.journal_file.read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]
        return [line for line in parsed if line.get('fact') == fact]


@pytest.fixture()
def lane(monkeypatch, tmp_path) -> Lane:
    return Lane(monkeypatch, tmp_path)


def locators(facts: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [
        (fact['collection'], fact['index'], fact['field'], fact['outcome']) for fact in facts
    ]


class TestTheInboundReplayIsRefusedAndJournalled:
    @pytest.mark.asyncio
    async def test_the_boundary_refuses_and_journals_one_markup_detected_line(self, lane: Lane):
        payload = await lane.refused_replay()

        assert payload['error_type'] == 'mcp_markup_unrepairable'
        [line] = lane.lines('markup_detected')
        assert (line['tool'], line['param'], line['outcome']) == (
            'add_design_decision', 'decision', 'unrepairable',
        )
        assert line['subject_task_id'] == SUBJECT
        assert lane.artifacts.read_plan().get('design_decisions', []) == []


class TestStoredDamageReadByAnAcceptedCallIsJournalled:
    @pytest.mark.asyncio
    async def test_the_journal_holds_exactly_the_disclosed_facts(self, lane: Lane):
        lane.store_3382_damage()

        response = await lane.accepted_call()

        disclosed = response[plan_markup_repair.MARKUP_REPAIRS_KEY]
        stored = lane.lines(STORED_MARKUP_FACT)
        assert locators(stored) == locators(disclosed)
        assert sorted(locators(stored)) == [
            ('design_decisions', 0, 'rationale', 'repaired'),
            ('design_decisions', 1, 'decision', 'unrepairable'),
            ('design_decisions', 1, 'rationale', 'repaired'),
        ]
        assert {line['subject_task_id'] for line in stored} == {SUBJECT}

    @pytest.mark.asyncio
    async def test_a_second_accepted_call_journals_no_new_stored_line(self, lane: Lane):
        lane.store_3382_damage()
        await lane.accepted_call()
        journalled = lane.lines(STORED_MARKUP_FACT)

        await lane.accepted_call()

        assert journalled
        assert lane.lines(STORED_MARKUP_FACT) == journalled

    @pytest.mark.asyncio
    async def test_a_clean_plan_creates_no_journal_file(self, lane: Lane):
        await lane.accepted_call()

        assert not lane.journal_file.exists()

    @pytest.mark.asyncio
    async def test_no_line_carries_the_damaged_prose(self, lane: Lane):
        lane.store_3382_damage()
        await lane.refused_replay()
        await lane.accepted_call()

        text = lane.journal_file.read_text(encoding='utf-8')
        assert lane.lines(STORED_MARKUP_FACT)
        assert [prose for prose in DAMAGED_PROSE if prose in text] == []


class TestARefusedCallDefersTheStoredReport:
    @pytest.mark.asyncio
    async def test_the_refusal_journals_only_its_inbound_line(self, lane: Lane):
        lane.store_3382_damage()

        await lane.refused_replay()

        assert len(lane.lines('markup_detected')) == 1
        assert lane.lines(STORED_MARKUP_FACT) == []

        await lane.accepted_call()

        assert ('design_decisions', 1, 'decision', 'unrepairable') in locators(
            lane.lines(STORED_MARKUP_FACT)
        )
