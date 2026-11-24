"""A host that knows its caller fills the markup records' empty caller axes.

Task 5055. :class:`MarkupGuardMiddleware` reads a call's attribution from the
call's own arguments (``agent_id``, ``project_root`` / ``project_id``,
``task_id``, ``agent_role``). Many guarded tools declare none of them, so
their records named no caller, although the HOST serving them often knows it:
a per-role stdio server knows its role, and an HTTP server sees an identity
header. The ``host_attribution`` provider supplies that knowledge as a
:class:`HostAttribution`. It fills only the axes the arguments left ``None``;
arguments always win, and a provider that fails changes nothing.

Also pins :func:`residue_header_lines`, the one rendering of a residue
record's header lines, its caller first, that every residue filer shares.

Driven through a ``Client``, the one harness that runs registered middleware,
over a small toy server.

## Sentinel-literal hazard — DO NOT "helpfully" un-escape these

Every tag in this file is built from the module's public constants,
:func:`closer_for` or the ``\\x3c`` escape for ``<``; the rule and its
rationale are owned by ``shared/src/shared/toolcall_markup.py``
("Sentinel-literal hazard").
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

from shared.mcp_markup_middleware import (
    MARKUP_STORM_ERROR_TYPE,
    MARKUP_UNREPAIRABLE_ERROR_TYPE,
    RESIDUE_CALLER_KEYS,
    RESIDUE_HEADER_KEYS,
    HostAttribution,
    MarkupGuardMiddleware,
    RepairPolicy,
    residue_header_lines,
)
from shared.toolcall_markup import CANONICAL_OPENER_PREFIX, closer_for

BOTH_POLICIES = pytest.mark.parametrize(
    'policy',
    [RepairPolicy.REJECT_WITH_REPAIR, RepairPolicy.FORWARD_REPAIR],
    ids=lambda p: p.name,
)

_INTENDED = 'The host knows who is calling.'
_HOST = HostAttribution(agent_id='watcher-x', agent_role='architect')

#: Mis-closed on its own name, then a pseudo-parameter naming no parameter of
#: the tool, so repair() refuses it (PRD boundary row B8).
_UNREPAIRABLE = (
    _INTENDED + closer_for('detail')
    + CANONICAL_OPENER_PREFIX + '"not_a_parameter">stray' + closer_for('parameter')
)
#: Mis-closed on its own name with nothing after it: repairable, recovering
#: nothing (PRD boundary row B4).
_REPAIRABLE = _INTENDED + closer_for('detail')


def _host() -> HostAttribution:
    return _HOST


def _broken_host() -> HostAttribution:
    raise RuntimeError('the host cannot say who is calling')


class _Harness:
    """A toy server, the guard under test, and what each channel received."""

    def __init__(
        self,
        policy: RepairPolicy,
        host_attribution: Callable[[], HostAttribution] | None = None,
        *,
        storm_threshold: int = 3,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.facts: list[dict[str, Any]] = []
        self.escalations: list[dict[str, Any]] = []
        self.mcp = FastMCP('host-attribution-harness')
        calls = self.calls

        @self.mcp.tool
        def add_note(title: str, detail: str) -> str:
            calls.append({'title': title, 'detail': detail})
            return 'ok'

        @self.mcp.tool
        def add_signed_note(agent_id: str, agent_role: str, detail: str) -> str:
            calls.append({'agent_id': agent_id, 'agent_role': agent_role, 'detail': detail})
            return 'ok'

        self.guard = MarkupGuardMiddleware(
            policy,
            fact_sink=self.facts.append,
            escalation_sink=self._file_escalation,
            storm_threshold=storm_threshold,
            host_attribution=host_attribution,
        )
        self.mcp.add_middleware(self.guard)

    def _file_escalation(self, record: dict[str, Any]) -> str:
        self.escalations.append(record)
        return 'esc-markup-residue-1'

    @property
    def residues(self) -> list[dict[str, Any]]:
        return [r for r in self.escalations if r['error_type'] == MARKUP_UNREPAIRABLE_ERROR_TYPE]

    @property
    def storms(self) -> list[dict[str, Any]]:
        return [r for r in self.escalations if r['error_type'] == MARKUP_STORM_ERROR_TYPE]

    async def call(self, tool: str, arguments: dict[str, Any]) -> Any:
        async with Client(self.mcp) as client:
            return await client.call_tool(tool, arguments)

    async def refusal(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        with pytest.raises(ToolError) as excinfo:
            await self.call(tool, arguments)
        return json.loads(str(excinfo.value))


class TestTheHostFillsAxesTheArgumentsLeftEmpty:

    @BOTH_POLICIES
    async def test_on_the_unrepairable_residue_and_its_fact(self, policy):
        h = _Harness(policy, _host)

        await h.refusal('add_note', {'title': 't', 'detail': _UNREPAIRABLE})

        [residue] = h.residues
        assert residue['agent_id'] == 'watcher-x'
        assert residue['subject_agent_role'] == 'architect'
        assert residue['subject_task_id'] is None
        assert residue['project'] is None
        assert [fact['agent_id'] for fact in h.facts] == ['watcher-x']

    @BOTH_POLICIES
    async def test_on_the_fired_storm_record(self, policy):
        h = _Harness(policy, _host, storm_threshold=1)

        await h.refusal('add_note', {'title': 't', 'detail': _UNREPAIRABLE})

        [storm] = h.storms
        assert storm['crossing_agent_id'] == 'watcher-x'
        assert storm['crossing_subject_agent_role'] == 'architect'
        assert any("'watcher-x'" in label and "'architect'" in label for label in storm['callers'])

    async def test_on_a_reject_tier_repairable_fact(self):
        h = _Harness(RepairPolicy.REJECT_WITH_REPAIR, _host)

        await h.refusal('add_note', {'title': 't', 'detail': _REPAIRABLE})

        assert [(fact['outcome'], fact['agent_id']) for fact in h.facts] == [('rejected', 'watcher-x')]

    async def test_on_a_forward_tier_repairable_fact(self):
        h = _Harness(RepairPolicy.FORWARD_REPAIR, _host)

        await h.call('add_note', {'title': 't', 'detail': _REPAIRABLE})

        assert h.calls == [{'title': 't', 'detail': _INTENDED}]
        assert [(fact['outcome'], fact['agent_id']) for fact in h.facts] == [('repaired', 'watcher-x')]


class TestArgumentsWin:

    @BOTH_POLICIES
    async def test_a_caller_declaring_its_own_axes_keeps_them(self, policy):
        h = _Harness(policy, _host, storm_threshold=1)

        await h.refusal('add_signed_note', {
            'agent_id': 'caller-a', 'agent_role': 'reviewer', 'detail': _UNREPAIRABLE,
        })

        [residue] = h.residues
        assert (residue['agent_id'], residue['subject_agent_role']) == ('caller-a', 'reviewer')
        assert [fact['agent_id'] for fact in h.facts] == ['caller-a']
        [storm] = h.storms
        assert (storm['crossing_agent_id'], storm['crossing_subject_agent_role']) == ('caller-a', 'reviewer')


class TestAFailingProviderChangesNothing:

    @BOTH_POLICIES
    async def test_the_unrepairable_refusal(self, policy):
        plain, broken = _Harness(policy), _Harness(policy, _broken_host)
        arguments = {'title': 't', 'detail': _UNREPAIRABLE}

        assert await broken.refusal('add_note', arguments) == await plain.refusal('add_note', arguments)
        assert broken.residues == plain.residues
        assert broken.facts == plain.facts

    async def test_the_reject_bounce(self):
        plain = _Harness(RepairPolicy.REJECT_WITH_REPAIR)
        broken = _Harness(RepairPolicy.REJECT_WITH_REPAIR, _broken_host)
        arguments = {'title': 't', 'detail': _REPAIRABLE}

        assert await broken.refusal('add_note', arguments) == await plain.refusal('add_note', arguments)
        assert broken.facts == plain.facts

    async def test_the_forward(self):
        plain = _Harness(RepairPolicy.FORWARD_REPAIR)
        broken = _Harness(RepairPolicy.FORWARD_REPAIR, _broken_host)
        arguments = {'title': 't', 'detail': _REPAIRABLE}

        await plain.call('add_note', arguments)
        await broken.call('add_note', arguments)

        assert broken.calls == plain.calls == [{'title': 't', 'detail': _INTENDED}]
        assert broken.facts == plain.facts


class TestWithNoProviderAttributionIsArgumentOnly:

    @BOTH_POLICIES
    async def test_an_undeclared_caller_stays_unnamed(self, policy):
        h = _Harness(policy, storm_threshold=1)

        await h.refusal('add_note', {'title': 't', 'detail': _UNREPAIRABLE})

        [residue] = h.residues
        assert {key: residue[key] for key in RESIDUE_CALLER_KEYS} == {
            'subject_task_id': None, 'subject_agent_role': None, 'agent_id': None, 'project': None,
        }
        [storm] = h.storms
        assert storm['callers'] == []


class TestHostAttribution:

    def test_defaults_to_naming_nobody(self):
        assert HostAttribution() == HostAttribution(agent_id=None, agent_role=None)


class TestResidueHeaderLines:

    def test_the_caller_keys_and_their_order(self):
        assert RESIDUE_CALLER_KEYS == ('subject_task_id', 'subject_agent_role', 'agent_id', 'project')

    def test_the_header_is_the_caller_then_the_leak(self):
        assert RESIDUE_HEADER_KEYS == (*RESIDUE_CALLER_KEYS, 'override_swallowed')

    def test_one_key_equals_repr_line_per_key_in_order(self):
        record = {
            'subject_task_id': '5055', 'subject_agent_role': 'architect',
            'agent_id': 'watcher-x', 'project': '/srv/p', 'override_swallowed': True,
            'raw_value': 'never rendered here',
        }

        assert list(residue_header_lines(record)) == [
            "subject_task_id='5055'",
            "subject_agent_role='architect'",
            "agent_id='watcher-x'",
            "project='/srv/p'",
            'override_swallowed=True',
        ]

    def test_an_absent_key_renders_as_none(self):
        assert list(residue_header_lines({})) == [f'{key}=None' for key in RESIDUE_HEADER_KEYS]

    def test_a_newline_in_a_value_cannot_inject_a_line(self):
        lines = list(residue_header_lines({'agent_id': 'a\nproject=forged'}))

        assert len(lines) == len(RESIDUE_HEADER_KEYS)
        assert all('\n' not in line for line in lines)
