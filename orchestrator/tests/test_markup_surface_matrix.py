"""The markup guard's user-observable contract, held across every server it guards.

Task 5055. plan-tools, verdict-tools and the escalation server each register
the one shared guard,
``shared/src/shared/mcp_markup_middleware.py::MarkupGuardMiddleware``, and
each has its own suite. This file is the matrix across them: the tool and
field combinations a leak really hits, each driven through its REAL server by
a fastmcp ``Client``, which is the only in-process call that runs registered
middleware.

ACCEPTANCE: a repairable leak is never lost. plan-tools (REJECT_WITH_REPAIR)
bounces it with a ``repaired_call`` that lands when resubmitted verbatim;
verdict-tools and the escalation server (FORWARD_REPAIR) forward it repaired.
Either way the destination holds the clean value and the swallowed sibling.

ATTRIBUTION: an unrepairable leak is refused, and the residue record filed for
it names its caller through the one shared rendering, ``residue_header_lines``,
and states ``MARKUP_REFUSAL_CUSTODY_CAVEAT`` rather than claiming to hold the
only copy.

CUSTODY OF A DROP: on the FORWARD_REPAIR surfaces a swallowed value that cannot
be typed is dropped from the forwarded call and filed instead. Nothing bounced
that caller, so its record states ``MARKUP_UNRECOVERED_PARAM_CUSTODY_CAVEAT``
beside ``MARKUP_UNRECOVERED_PARAM_OUTCOME_CAVEAT``, whichever sink filed it.

Every envelope specimen is built from ``_markup_helpers``, never spelled raw;
that module's header says why.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import pytest
from _markup_helpers import closer, param_opener
from _markup_sink_rig import seed_plan
from escalation.queue import EscalationQueue
from escalation.server import create_server as create_escalation_server
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError
from shared.mcp_markup_middleware import (
    MARKUP_REFUSAL_CUSTODY_CAVEAT,
    MARKUP_UNRECOVERED_PARAM_CUSTODY_CAVEAT,
    MARKUP_UNRECOVERED_PARAM_OUTCOME_CAVEAT,
    residue_header_lines,
)
from shared.project_layout import escalation_queue_dir

from orchestrator.artifacts import TaskArtifacts
from orchestrator.mcp import (
    plan_markup_repair,
    plan_markup_stamp,
    plan_tools,
    verdict_tools,
)

PLAN_TASK_ID = 'test-1'
CALLER_ROLE = 'architect'

#: Not a singleton verdict role, so verdict-tools registers ``submit_review_verdict``.
REVIEWER_ROLE = 'test_analyst'

CLEAN = 'The value the caller meant to send, ending here.'

ESCALATE_INFO_ARGUMENTS = {
    'task_id': '3441',
    'agent_role': 'implementer',
    'category': 'cleanup_needed',
    'summary': 'Eight near-duplicate entries restate the same gotcha.',
}

RESIDUE_CATEGORY = 'mcp_markup_residue'


def swallowing(field: str, sibling: str, sibling_value: str) -> str:
    """*field* mis-closed with its own name, then the whole *sibling* swallowed."""
    return CLEAN + closer(field) + '\n' + param_opener(sibling) + sibling_value


def unrepairable(field: str) -> str:
    """*field* mis-closed, then a pseudo-parameter no tool declares (boundary row B8)."""
    return CLEAN + closer(field) + '\n' + param_opener('not_a_declared_parameter') + 'stray'


async def refusal_of(server: FastMCP, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    async with Client(server) as client:
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool(tool, arguments)
    return json.loads(str(excinfo.value))


def pending_details(queue: EscalationQueue, anchor: str) -> list[str]:
    return [esc.detail for esc in queue.get_by_task(anchor, status='pending')]


@pytest.fixture(autouse=True)
def _forget_plan_tools_process_state():
    """plan-tools keeps refusal memos and stamp buffers per process; isolate each row."""
    plan_markup_repair.forget_reported_refusals()
    plan_markup_stamp.clear_pending()
    yield
    plan_markup_repair.forget_reported_refusals()
    plan_markup_stamp.clear_pending()


class Surface(Protocol):
    server: FastMCP

    async def deliver(self, tool: str, arguments: dict[str, Any]) -> Mapping[str, Any]:
        """Send *arguments* the way this server's caller recovers a leak; return what landed."""
        ...

    def residue_details(self) -> list[str]: ...


class PlanTools:
    """plan-tools as a ``--caller-role`` launch serves it, filing into a tmp main checkout."""

    _DESTINATION = {'add_design_decision': 'design_decisions', 'add_reuse_item': 'reuse'}

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.artifacts, self.root = seeded_checkout(tmp_path)
        monkeypatch.setattr(plan_tools, '_markup_project_root', lambda _worktree: self.root)
        self.server = plan_tools.server_from_argv([
            '--worktree', str(self.artifacts.worktree),
            '--caller-role', CALLER_ROLE,
            '--grants-enforced',
            *(flag for tool in self._DESTINATION for flag in ('--grant', tool)),
        ])

    async def deliver(self, tool: str, arguments: dict[str, Any]) -> Mapping[str, Any]:
        refusal = await refusal_of(self.server, tool, arguments)
        async with Client(self.server) as client:
            await client.call_tool(tool, refusal['repaired_call'])
        return self.artifacts.read_plan()[self._DESTINATION[tool]][-1]

    def residue_details(self) -> list[str]:
        return pending_details(
            EscalationQueue(escalation_queue_dir(self.root)),
            plan_tools.MARKUP_SINK_SPEC.residue_anchor_task_id,
        )


class VerdictTools:
    """verdict-tools serving a reviewer, filing into a tmp main checkout."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.artifacts, self.root = seeded_checkout(tmp_path)
        self.server = verdict_tools.create_server(
            self.artifacts, REVIEWER_ROLE, resolve_project_root=lambda _worktree: self.root,
        )

    async def deliver(self, tool: str, arguments: dict[str, Any]) -> Mapping[str, Any]:
        async with Client(self.server) as client:
            await client.call_tool(tool, arguments)
        envelope = self.artifacts.read_verdict(REVIEWER_ROLE)
        assert envelope is not None, 'the forwarded verdict never landed'
        return envelope['verdict']

    def residue_details(self) -> list[str]:
        return pending_details(
            EscalationQueue(escalation_queue_dir(self.root)),
            verdict_tools.MARKUP_SINK_SPEC.residue_anchor_task_id,
        )


class EscalationServer:
    """The escalation server over a tmp queue, called in process with no identity header."""

    _ADDRESSES_A_RECORD = frozenset({'stamp_triage', 'resolve_issue'})

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.queue = EscalationQueue(tmp_path / 'esc')
        self.server = create_escalation_server(self.queue, startup_sweep=False)

    async def deliver(self, tool: str, arguments: dict[str, Any]) -> Mapping[str, Any]:
        async with Client(self.server) as client:
            if tool in self._ADDRESSES_A_RECORD:
                seeded = await client.call_tool('escalate_info', ESCALATE_INFO_ARGUMENTS)
                written = seeded.data['id']
                await client.call_tool(tool, {'escalation_id': written, **arguments})
            else:
                written = (await client.call_tool(tool, arguments)).data['id']
        stored = self.queue.get(written)
        assert stored is not None, f'{written} is not on the queue'
        return stored.to_dict()

    def residue_details(self) -> list[str]:
        return [esc.detail for esc in self.queue.get_pending() if esc.category == RESIDUE_CATEGORY]


def seeded_checkout(tmp_path: Path) -> tuple[TaskArtifacts, Path]:
    """A task worktree holding a plan for ``PLAN_TASK_ID``, beside its main checkout."""
    worktree = tmp_path / 'worktree'
    root = tmp_path / 'main'
    worktree.mkdir()
    root.mkdir()
    artifacts = TaskArtifacts(worktree)
    artifacts.init(PLAN_TASK_ID, 'Test task', 'A test')
    seed_plan(artifacts, PLAN_TASK_ID)
    return artifacts, root


ACCEPTANCE = [
    pytest.param(
        PlanTools, 'add_design_decision', 'rationale', 'decision', 'Guard every surface.', {},
        id='plan-tools.add_design_decision.rationale',
    ),
    pytest.param(
        PlanTools, 'add_reuse_item', 'how', 'where', 'shared/src/shared/toolcall_markup.py',
        {'what': 'The shared repairer'},
        id='plan-tools.add_reuse_item.how',
    ),
    pytest.param(
        VerdictTools, 'submit_review_verdict', 'summary', 'verdict', 'PASS',
        {'reviewer': REVIEWER_ROLE, 'issues': []},
        id='verdict-tools.submit_review_verdict.summary',
    ),
    pytest.param(
        EscalationServer, 'escalate_info', 'detail', 'suggested_action', 'Fold them into one.',
        ESCALATE_INFO_ARGUMENTS,
        id='escalation.escalate_info.detail',
    ),
    pytest.param(
        EscalationServer, 'stamp_triage', 'triage_note', 'triaged_by', 'l2-escalation-watcher', {},
        id='escalation.stamp_triage.triage_note',
    ),
    pytest.param(
        EscalationServer, 'resolve_issue', 'resolution', 'resolved_by', 'steward', {},
        id='escalation.resolve_issue.resolution',
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('surface', 'tool', 'field', 'sibling', 'sibling_value', 'sent'), ACCEPTANCE,
)
async def test_a_repairable_leak_lands_clean(
    tmp_path, monkeypatch, surface, tool, field, sibling, sibling_value, sent,
):
    stored = await surface(tmp_path, monkeypatch).deliver(
        tool, {**sent, field: swallowing(field, sibling, sibling_value)},
    )

    assert stored[field] == CLEAN
    assert stored[sibling] == sibling_value


ATTRIBUTION = [
    pytest.param(
        PlanTools, 'add_design_decision', 'decision', {'rationale': 'Because.'},
        {'subject_task_id': PLAN_TASK_ID, 'subject_agent_role': CALLER_ROLE},
        id='plan-tools',
    ),
    pytest.param(
        VerdictTools, 'submit_review_verdict', 'summary',
        {'reviewer': REVIEWER_ROLE, 'verdict': 'PASS', 'issues': []},
        {'subject_task_id': PLAN_TASK_ID, 'subject_agent_role': REVIEWER_ROLE},
        id='verdict-tools',
    ),
    pytest.param(
        EscalationServer, 'escalate_info', 'detail', ESCALATE_INFO_ARGUMENTS,
        {'subject_task_id': '3441', 'subject_agent_role': 'implementer'},
        id='escalation',
    ),
]


async def refused_residue(
    surface: Surface, tool: str, field: str, sent: dict[str, Any],
) -> str:
    refusal = await refusal_of(surface.server, tool, {**sent, field: unrepairable(field)})
    assert refusal['error_type'] == 'mcp_markup_unrepairable'
    [detail] = surface.residue_details()
    return detail


@pytest.mark.asyncio
@pytest.mark.parametrize(('surface', 'tool', 'field', 'sent', 'subject'), ATTRIBUTION)
async def test_the_residue_names_its_caller(
    tmp_path, monkeypatch, surface, tool, field, sent, subject,
):
    detail = await refused_residue(surface(tmp_path, monkeypatch), tool, field, sent)

    caller = {'agent_id': None, 'project': None, 'override_swallowed': False, **subject}
    missing = set(residue_header_lines(caller)) - set(detail.splitlines())
    assert not missing, f'the residue detail lacks caller lines {sorted(missing)}'


@pytest.mark.asyncio
@pytest.mark.parametrize(('surface', 'tool', 'field', 'sent', 'subject'), ATTRIBUTION)
async def test_the_residue_states_what_its_custody_proves(
    tmp_path, monkeypatch, surface, tool, field, sent, subject,
):
    detail = await refused_residue(surface(tmp_path, monkeypatch), tool, field, sent)

    assert MARKUP_REFUSAL_CUSTODY_CAVEAT in detail
    assert 'only surviving copy' not in detail


UNRECOVERED = [
    pytest.param(
        VerdictTools, 'submit_review_verdict', 'summary', 'metadata', 'not a JSON object',
        {'reviewer': REVIEWER_ROLE, 'verdict': 'PASS', 'issues': []},
        id='verdict-tools.submit_review_verdict.metadata',
    ),
    pytest.param(
        EscalationServer, 'escalate_info', 'detail', 'evidence', '[{"observation": "cut off',
        ESCALATE_INFO_ARGUMENTS,
        id='escalation.escalate_info.evidence',
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('surface', 'tool', 'field', 'sibling', 'sibling_value', 'sent'), UNRECOVERED,
)
async def test_a_dropped_value_states_what_its_custody_proves(
    tmp_path, monkeypatch, surface, tool, field, sibling, sibling_value, sent,
):
    served = surface(tmp_path, monkeypatch)
    async with Client(served.server) as client:
        await client.call_tool(tool, {**sent, field: swallowing(field, sibling, sibling_value)})

    [detail] = served.residue_details()
    assert MARKUP_UNRECOVERED_PARAM_OUTCOME_CAVEAT in detail
    assert MARKUP_UNRECOVERED_PARAM_CUSTODY_CAVEAT in detail
    assert MARKUP_REFUSAL_CUSTODY_CAVEAT not in detail
