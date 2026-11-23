"""The boundary lets a quoted HTML closer through and keeps catching real leaks.

Task 10396. ``description``, ``details`` and ``content`` are envelope closer
names AND ordinary markup element names. A value quoting an HTML block, such as
a collapsible ``details`` section, carries one of those closers. When
:func:`shared.toolcall_markup.repair` has nothing to accept and
:func:`shared.toolcall_markup.only_inert_html_closers` says every needle is such
a closer, :class:`MarkupGuardMiddleware` treats the argument as if it carried no
needle: the call passes unchanged, with no fact, no storm count and no residue,
and later arguments are still scanned.

Driven through a ``Client``, the one harness that runs registered middleware,
over a small toy server of its own. The toy tools record the arguments they
received.

## Sentinel-literal hazard — DO NOT "helpfully" un-escape these

Every tag in this file is built from the module's public constants,
:func:`closer_for` or the ``\\x3c`` escape for ``<``; the rule and its
rationale are owned by ``shared/src/shared/toolcall_markup.py``
("Sentinel-literal hazard").
"""
from __future__ import annotations

import json
from typing import Any

import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

from shared.mcp_markup_middleware import (
    MARKUP_UNREPAIRABLE_ERROR_TYPE,
    MarkupGuardMiddleware,
    RepairPolicy,
)
from shared.toolcall_markup import CANONICAL_OPENER_PREFIX, INVOKE_CLOSER, closer_for

BOTH_POLICIES = pytest.mark.parametrize(
    'policy',
    [RepairPolicy.REJECT_WITH_REPAIR, RepairPolicy.FORWARD_REPAIR],
    ids=lambda p: p.name,
)


def _opener(name: str) -> str:
    return '\x3c' + name + '>'


def _element(name: str, body: str) -> str:
    return _opener(name) + body + closer_for(name)


_DETAILS_BLOCK = (
    _opener('details') + _element('summary', 'Measured evidence')
    + _element('pre', 'rc=0') + closer_for('details')
)
_BLOCK_IN_PROSE = 'The finding:\n' + _DETAILS_BLOCK + '\nThat is the whole record.'
_BLOCK_AT_END = 'The finding:\n' + _DETAILS_BLOCK


class _Harness:
    """A toy server, the guard under test, and what each channel received."""

    def __init__(self, policy: RepairPolicy) -> None:
        self.calls: list[dict[str, Any]] = []
        self.facts: list[dict[str, Any]] = []
        self.escalations: list[dict[str, Any]] = []
        self.mcp = FastMCP('inert-closer-harness')
        self._register_tools()
        self.guard = MarkupGuardMiddleware(
            policy,
            fact_sink=self.facts.append,
            escalation_sink=self._file_escalation,
        )
        self.mcp.add_middleware(self.guard)

    def _file_escalation(self, record: dict[str, Any]) -> str:
        self.escalations.append(record)
        return 'esc-markup-residue-1'

    def _register_tools(self) -> None:
        calls = self.calls

        @self.mcp.tool
        def add_plan_step(step_id: str, step_type: str, description: str) -> str:
            calls.append({
                'tool': 'add_plan_step', 'step_id': step_id,
                'step_type': step_type, 'description': description,
            })
            return 'ok'

        @self.mcp.tool
        def file_task(title: str, description: str, details: str = '') -> str:
            calls.append({
                'tool': 'file_task', 'title': title,
                'description': description, 'details': details,
            })
            return 'ok'

        @self.mcp.tool
        def add_design_decision(decision: str, rationale: str) -> str:
            calls.append({
                'tool': 'add_design_decision', 'decision': decision, 'rationale': rationale,
            })
            return 'ok'

    async def call(self, tool: str, arguments: dict[str, Any]) -> Any:
        async with Client(self.mcp) as client:
            return await client.call_tool(tool, arguments)

    async def refusal(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        with pytest.raises(ToolError) as excinfo:
            await self.call(tool, arguments)
        return json.loads(str(excinfo.value))

    def assert_non_event(self) -> None:
        assert self.facts == []
        assert self.escalations == []
        assert self.guard.storm_escape.tracked_keys == frozenset()


class TestAQuotedHtmlBlockPassesUnchanged:

    @BOTH_POLICIES
    @pytest.mark.parametrize(
        'description',
        [pytest.param(_BLOCK_IN_PROSE, id='inside-prose'), pytest.param(_BLOCK_AT_END, id='at-end')],
    )
    async def test_a_name_outside_the_schema(self, policy, description):
        h = _Harness(policy)

        await h.call('add_plan_step', {'step_id': 'step-1', 'step_type': 'impl', 'description': description})

        assert h.calls == [{
            'tool': 'add_plan_step', 'step_id': 'step-1',
            'step_type': 'impl', 'description': description,
        }]
        h.assert_non_event()

    @BOTH_POLICIES
    async def test_an_in_schema_name_followed_by_prose(self, policy):
        h = _Harness(policy)

        await h.call('file_task', {'title': 't', 'description': _BLOCK_IN_PROSE})

        assert h.calls == [{'tool': 'file_task', 'title': 't', 'description': _BLOCK_IN_PROSE, 'details': ''}]
        h.assert_non_event()


class TestARealLeakUsingTheSameNameIsStillCaught:

    _SWALLOWED = _element('p', 'The ' + _element('b', 'full') + ' log.')
    _LEAKED = (
        'File the follow-up.' + closer_for('description')
        + CANONICAL_OPENER_PREFIX + '"details">' + _SWALLOWED + closer_for('parameter')
    )

    async def test_reject_tier_bounces_with_the_swallowed_parameter(self):
        h = _Harness(RepairPolicy.REJECT_WITH_REPAIR)

        payload = await h.refusal('file_task', {'title': 't', 'description': self._LEAKED})

        assert payload['recovered_params'] == ['details']
        assert payload['repaired_call'] == {
            'title': 't', 'description': 'File the follow-up.', 'details': self._SWALLOWED,
        }
        assert h.calls == []

    async def test_forward_tier_repairs_in_place(self):
        h = _Harness(RepairPolicy.FORWARD_REPAIR)

        await h.call('file_task', {'title': 't', 'description': self._LEAKED})

        assert h.calls == [{
            'tool': 'file_task', 'title': 't',
            'description': 'File the follow-up.', 'details': self._SWALLOWED,
        }]
        assert [fact['outcome'] for fact in h.facts] == ['repaired']

    @BOTH_POLICIES
    async def test_an_invoke_closer_beside_the_block_is_unrepairable(self, policy):
        h = _Harness(policy)
        description = _DETAILS_BLOCK + INVOKE_CLOSER + 'junk'

        payload = await h.refusal(
            'add_plan_step', {'step_id': 'step-1', 'step_type': 'impl', 'description': description},
        )

        assert payload['error_type'] == MARKUP_UNREPAIRABLE_ERROR_TYPE
        assert payload['field'] == 'description'
        assert [record['raw_value'] for record in h.escalations] == [description]
        assert h.calls == []


class TestLaterArgumentsAreStillScanned:

    @BOTH_POLICIES
    async def test_a_leak_after_an_inert_quote_is_refused(self, policy):
        h = _Harness(policy)
        rationale = 'Because it is measured.' + closer_for('rationale') + INVOKE_CLOSER + 'junk'

        payload = await h.refusal(
            'add_design_decision', {'decision': _BLOCK_IN_PROSE, 'rationale': rationale},
        )

        assert payload['error_type'] == MARKUP_UNREPAIRABLE_ERROR_TYPE
        assert payload['field'] == 'rationale'
        assert [record['field'] for record in h.escalations] == ['rationale']
        assert h.calls == []
