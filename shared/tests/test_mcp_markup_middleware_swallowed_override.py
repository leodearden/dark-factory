"""A deliberate-quoting override the leak swallowed is reported, never honoured.

Task 5055. The override is ``metadata={MARKUP_OVERRIDE_KEY: True}``. A leak
swallows the parameters that FOLLOW the leaking one, so an override sent after
the leaking string arrives as text inside that string's value, where the guard
cannot honour it and must not: a genuine quotation never makes the harness
over-consume, so finding the flag there proves a real leak. The guard reports
it as ``override_swallowed`` with a hint naming the remedy, which is to send
``metadata`` FIRST, ahead of the argument that carries the markup.

Driven through a ``Client``, the one harness that runs registered middleware,
over a small toy server whose tool declares ``metadata`` through
``shared/src/shared/markup_override_param.py::accepts_markup_override``.

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

from shared.markup_override_param import accepts_markup_override
from shared.mcp_markup_middleware import (
    MARKUP_UNREPAIRABLE_ERROR_TYPE,
    MarkupGuardMiddleware,
    RepairPolicy,
)
from shared.toolcall_markup import CANONICAL_OPENER_PREFIX, MARKUP_OVERRIDE_KEY, closer_for

BOTH_POLICIES = pytest.mark.parametrize(
    'policy',
    [RepairPolicy.REJECT_WITH_REPAIR, RepairPolicy.FORWARD_REPAIR],
    ids=lambda p: p.name,
)

_DECISION = 'Report a swallowed override.'
_INTENDED = 'The leak swallows only the parameters after the leaking one.'
_OVERRIDE_MAP = {MARKUP_OVERRIDE_KEY: True}


def _canonical(name: str, body: str) -> str:
    return CANONICAL_OPENER_PREFIX + '"' + name + '">' + body + closer_for('parameter')


_SWALLOWED_METADATA = _canonical('metadata', json.dumps(_OVERRIDE_MAP))
#: A pseudo-parameter naming no parameter of the tool, so repair() refuses the
#: value (PRD boundary row B8) and the call is unrepairable.
_OUT_OF_SCHEMA = _canonical('not_a_parameter', 'stray')


def _leaked(*tail: str) -> str:
    """*_INTENDED* mis-closed on its own name, followed by what it swallowed."""
    return _INTENDED + closer_for('rationale') + ''.join(tail)


_UNREPAIRABLE_WITH_OVERRIDE = _leaked(_SWALLOWED_METADATA, _OUT_OF_SCHEMA)
_UNREPAIRABLE_WITHOUT_OVERRIDE = _leaked(_OUT_OF_SCHEMA)
_REPAIRABLE_WITH_OVERRIDE = _leaked(_SWALLOWED_METADATA)
_REPAIRABLE_WITHOUT_OVERRIDE = _leaked()


def _names_the_swallowed_override(hint: str) -> bool:
    lowered = hint.lower()
    return 'swallowed by this leak' in lowered and 'metadata as the first argument' in lowered


class _Harness:
    """A toy server, the guard under test, and what each channel received."""

    def __init__(self, policy: RepairPolicy) -> None:
        self.calls: list[dict[str, Any]] = []
        self.facts: list[dict[str, Any]] = []
        self.escalations: list[dict[str, Any]] = []
        self.mcp = FastMCP('swallowed-override-harness')
        calls = self.calls

        @self.mcp.tool
        @accepts_markup_override
        def add_design_decision(decision: str, rationale: str) -> str:
            calls.append({'decision': decision, 'rationale': rationale})
            return 'ok'

        self.guard = MarkupGuardMiddleware(
            policy,
            fact_sink=self.facts.append,
            escalation_sink=self._file_escalation,
        )
        self.mcp.add_middleware(self.guard)

    def _file_escalation(self, record: dict[str, Any]) -> str:
        self.escalations.append(record)
        return 'esc-markup-residue-1'

    async def call(self, arguments: dict[str, Any]) -> Any:
        async with Client(self.mcp) as client:
            return await client.call_tool('add_design_decision', arguments)

    async def refusal(self, arguments: dict[str, Any]) -> dict[str, Any]:
        with pytest.raises(ToolError) as excinfo:
            await self.call(arguments)
        return json.loads(str(excinfo.value))


class TestAnUnrepairableLeakThatSwallowedTheOverride:

    @BOTH_POLICIES
    async def test_is_refused_and_names_the_swallowed_override(self, policy):
        h = _Harness(policy)

        payload = await h.refusal({'decision': _DECISION, 'rationale': _UNREPAIRABLE_WITH_OVERRIDE})

        assert payload['error_type'] == MARKUP_UNREPAIRABLE_ERROR_TYPE
        assert payload['override_swallowed'] is True
        assert _names_the_swallowed_override(payload['hint'])
        assert h.calls == []

    @BOTH_POLICIES
    async def test_its_residue_record_says_so(self, policy):
        h = _Harness(policy)

        await h.refusal({'decision': _DECISION, 'rationale': _UNREPAIRABLE_WITH_OVERRIDE})

        assert [record['override_swallowed'] for record in h.escalations] == [True]
        assert [record['raw_value'] for record in h.escalations] == [_UNREPAIRABLE_WITH_OVERRIDE]


class TestARepairableLeakThatSwallowedTheOverride:

    async def test_the_bounce_names_it_and_restores_the_map(self):
        h = _Harness(RepairPolicy.REJECT_WITH_REPAIR)

        payload = await h.refusal({'decision': _DECISION, 'rationale': _REPAIRABLE_WITH_OVERRIDE})

        assert payload['override_swallowed'] is True
        assert _names_the_swallowed_override(payload['hint'])
        assert payload['repaired_call'] == {
            'decision': _DECISION, 'rationale': _INTENDED, 'metadata': _OVERRIDE_MAP,
        }
        assert h.calls == []

    async def test_resubmitting_repaired_call_verbatim_is_accepted(self):
        h = _Harness(RepairPolicy.REJECT_WITH_REPAIR)
        payload = await h.refusal({'decision': _DECISION, 'rationale': _REPAIRABLE_WITH_OVERRIDE})

        await h.call(payload['repaired_call'])

        assert h.calls == [{'decision': _DECISION, 'rationale': _INTENDED}]


class TestALeakWithNoOverrideReportsFalse:

    @BOTH_POLICIES
    async def test_on_the_unrepairable_refusal_and_its_residue(self, policy):
        h = _Harness(policy)

        payload = await h.refusal({'decision': _DECISION, 'rationale': _UNREPAIRABLE_WITHOUT_OVERRIDE})

        assert payload['override_swallowed'] is False
        assert not _names_the_swallowed_override(payload['hint'])
        assert [record['override_swallowed'] for record in h.escalations] == [False]

    async def test_on_the_reject_bounce(self):
        h = _Harness(RepairPolicy.REJECT_WITH_REPAIR)

        payload = await h.refusal({'decision': _DECISION, 'rationale': _REPAIRABLE_WITHOUT_OVERRIDE})

        assert payload['override_swallowed'] is False
        assert not _names_the_swallowed_override(payload['hint'])


class TestAnOverrideSentAheadOfTheLeakStillBypassesTheGuard:
    """The remedy the hint advertises: ``metadata`` first cannot be swallowed."""

    @BOTH_POLICIES
    async def test_no_fact_no_residue_and_the_value_arrives_unchanged(self, policy):
        h = _Harness(policy)

        await h.call({
            'metadata': _OVERRIDE_MAP,
            'decision': _DECISION,
            'rationale': _UNREPAIRABLE_WITHOUT_OVERRIDE,
        })

        assert h.calls == [{'decision': _DECISION, 'rationale': _UNREPAIRABLE_WITHOUT_OVERRIDE}]
        assert h.facts == []
        assert h.escalations == []
