"""A deliberate-quoting override the leak swallowed is reported, never honoured.

Task 5055. The override is ``metadata={MARKUP_OVERRIDE_KEY: True}``. A leak
swallows the parameters that FOLLOW the leaking one, so an override sent after
the leaking string arrives as text inside that string's value, where the guard
cannot honour it and must not: a genuine quotation never makes the harness
over-consume, so finding the flag there proves a real leak. The guard reports
it as ``override_swallowed`` and extends the hint it would give for the same
leak without the override; the extension names the remedy, which is to send
``metadata`` FIRST, ahead of the argument that carries the markup. A bounce's
``repaired_call`` restores the swallowed ``metadata`` without the flag, so the
verbatim resubmission it invites is guarded again.

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
    MARKUP_UNRECOVERED_PARAM_ERROR_TYPE,
    MARKUP_UNREPAIRABLE_ERROR_TYPE,
    MarkupGuardMiddleware,
    RepairPolicy,
)
from shared.toolcall_markup import (
    CANONICAL_OPENER_PREFIX,
    MARKUP_OVERRIDE_KEY,
    closer_for,
    markup_override_requested,
    swallowed_markup_override,
)

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
#: A declared list that does not decode, so FORWARD_REPAIR drops and files it.
_UNTYPABLE = _canonical('alternatives', 'not a JSON list')


def _leaked(*tail: str) -> str:
    """*_INTENDED* mis-closed on its own name, followed by what it swallowed."""
    return _INTENDED + closer_for('rationale') + ''.join(tail)


_UNREPAIRABLE_WITH_OVERRIDE = _leaked(_SWALLOWED_METADATA, _OUT_OF_SCHEMA)
_UNREPAIRABLE_WITHOUT_OVERRIDE = _leaked(_OUT_OF_SCHEMA)
_REPAIRABLE_WITH_OVERRIDE = _leaked(_SWALLOWED_METADATA)
_REPAIRABLE_WITHOUT_OVERRIDE = _leaked()
_UNRECOVERED_WITH_OVERRIDE = _leaked(_SWALLOWED_METADATA, _UNTYPABLE)
_UNRECOVERED_WITHOUT_OVERRIDE = _leaked(_UNTYPABLE)

#: Intended text carrying far more inert opener-shaped tokens than the
#: non-validating scanner inspects, so it gives up on the whole value.
_SCANNER_BLIND_INTENDED = 'Rendered table: ' + '\x3cb>x' * 1024
_SCANNER_BLIND_WITH_OVERRIDE = (
    _SCANNER_BLIND_INTENDED + closer_for('rationale') + _SWALLOWED_METADATA
)
_SCANNER_BLIND_WITHOUT_OVERRIDE = _SCANNER_BLIND_INTENDED + closer_for('rationale')


def _swallowed_override_remedy(swallowed: dict[str, Any], plain: dict[str, Any]) -> str:
    """What a swallowed override adds to the hint the same leak gets without one.

    Compared rather than phrase-matched, so the remedy's wording is free to
    change: it must EXTEND the plain hint, and the plain hint must not carry it.
    """
    assert swallowed['hint'].startswith(plain['hint'])
    remedy = swallowed['hint'][len(plain['hint']):]
    assert remedy
    assert remedy not in plain['hint']
    return remedy


class _Harness:
    """A toy server, the guard under test, and what each channel received."""

    def __init__(self, policy: RepairPolicy) -> None:
        self.calls: list[dict[str, Any]] = []
        self.notes: list[dict[str, Any]] = []
        self.facts: list[dict[str, Any]] = []
        self.escalations: list[dict[str, Any]] = []
        self.mcp = FastMCP('swallowed-override-harness')
        calls, notes = self.calls, self.notes

        @self.mcp.tool
        @accepts_markup_override
        def add_design_decision(
            decision: str, rationale: str, alternatives: list[str] | None = None,
        ) -> str:
            calls.append({'decision': decision, 'rationale': rationale})
            return 'ok'

        @self.mcp.tool
        def record_note(note: str, metadata: dict[str, Any] | None = None) -> str:
            """Declares ``metadata`` natively, so what it receives is observable."""
            notes.append({'note': note, 'metadata': metadata})
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

    async def call(self, arguments: dict[str, Any], tool: str = 'add_design_decision') -> Any:
        async with Client(self.mcp) as client:
            return await client.call_tool(tool, arguments)

    async def refusal(self, arguments: dict[str, Any]) -> dict[str, Any]:
        with pytest.raises(ToolError) as excinfo:
            await self.call(arguments)
        return json.loads(str(excinfo.value))


class TestAnUnrepairableLeakThatSwallowedTheOverride:

    @BOTH_POLICIES
    async def test_is_refused_with_the_remedy_added_to_its_hint(self, policy):
        h = _Harness(policy)

        payload = await h.refusal({'decision': _DECISION, 'rationale': _UNREPAIRABLE_WITH_OVERRIDE})
        plain = await h.refusal({'decision': _DECISION, 'rationale': _UNREPAIRABLE_WITHOUT_OVERRIDE})

        assert payload['error_type'] == MARKUP_UNREPAIRABLE_ERROR_TYPE
        assert payload['override_swallowed'] is True
        _swallowed_override_remedy(payload, plain)
        assert h.calls == []

    @BOTH_POLICIES
    async def test_its_residue_record_says_so(self, policy):
        h = _Harness(policy)

        await h.refusal({'decision': _DECISION, 'rationale': _UNREPAIRABLE_WITH_OVERRIDE})

        assert [record['override_swallowed'] for record in h.escalations] == [True]
        assert [record['raw_value'] for record in h.escalations] == [_UNREPAIRABLE_WITH_OVERRIDE]


class TestARepairableLeakThatSwallowedTheOverride:

    async def test_the_bounce_adds_the_remedy_to_its_hint(self):
        h = _Harness(RepairPolicy.REJECT_WITH_REPAIR)

        payload = await h.refusal({'decision': _DECISION, 'rationale': _REPAIRABLE_WITH_OVERRIDE})
        plain = await h.refusal({'decision': _DECISION, 'rationale': _REPAIRABLE_WITHOUT_OVERRIDE})

        assert payload['override_swallowed'] is True
        _swallowed_override_remedy(payload, plain)
        assert h.calls == []

    async def test_the_bounce_restores_the_map_without_the_override(self):
        h = _Harness(RepairPolicy.REJECT_WITH_REPAIR)

        payload = await h.refusal({'decision': _DECISION, 'rationale': _REPAIRABLE_WITH_OVERRIDE})

        assert payload['repaired_call'] == {
            'decision': _DECISION, 'rationale': _INTENDED, 'metadata': {},
        }
        assert not markup_override_requested(payload['repaired_call']['metadata'])

    async def test_a_forwarded_call_files_an_unrecovered_residue_that_says_so(self):
        h = _Harness(RepairPolicy.FORWARD_REPAIR)

        await h.call({'decision': _DECISION, 'rationale': _UNRECOVERED_WITH_OVERRIDE})

        assert [(r['error_type'], r['override_swallowed']) for r in h.escalations] == [
            (MARKUP_UNRECOVERED_PARAM_ERROR_TYPE, True),
        ]
        assert h.calls == [{'decision': _DECISION, 'rationale': _INTENDED}]

    async def test_resubmitting_repaired_call_verbatim_is_accepted(self):
        h = _Harness(RepairPolicy.REJECT_WITH_REPAIR)
        payload = await h.refusal({'decision': _DECISION, 'rationale': _REPAIRABLE_WITH_OVERRIDE})

        await h.call(payload['repaired_call'])

        assert h.calls == [{'decision': _DECISION, 'rationale': _INTENDED}]


class TestAnOverrideTheScannerCannotSeeIsStillStripped:
    """The strip keys on the validated recovery, never on the scanner's verdict."""

    def test_the_scanner_gives_up_on_this_value(self):
        # If the scanner's cap ever grows past these 1024 tokens, grow them too.
        assert swallowed_markup_override(_SCANNER_BLIND_WITH_OVERRIDE) is False

    async def test_the_bounce_restores_the_map_without_the_override(self):
        h = _Harness(RepairPolicy.REJECT_WITH_REPAIR)

        payload = await h.refusal({'decision': _DECISION, 'rationale': _SCANNER_BLIND_WITH_OVERRIDE})

        assert payload['repaired_call'] == {
            'decision': _DECISION, 'rationale': _SCANNER_BLIND_INTENDED, 'metadata': {},
        }
        assert not markup_override_requested(payload['repaired_call']['metadata'])

    async def test_the_bounce_reports_what_it_stripped(self):
        h = _Harness(RepairPolicy.REJECT_WITH_REPAIR)

        payload = await h.refusal({'decision': _DECISION, 'rationale': _SCANNER_BLIND_WITH_OVERRIDE})
        plain = await h.refusal({'decision': _DECISION, 'rationale': _SCANNER_BLIND_WITHOUT_OVERRIDE})

        assert payload['override_swallowed'] is True
        _swallowed_override_remedy(payload, plain)

    @pytest.mark.parametrize(
        'intended',
        ['A short note.', _SCANNER_BLIND_INTENDED],
        ids=['scanner-visible', 'scanner-blind'],
    )
    async def test_a_forwarded_call_receives_no_live_override(self, intended):
        h = _Harness(RepairPolicy.FORWARD_REPAIR)

        await h.call({'note': intended + closer_for('note') + _SWALLOWED_METADATA}, tool='record_note')

        assert [entry['note'] for entry in h.notes] == [intended]
        assert not markup_override_requested(h.notes[0]['metadata'])


class TestALeakWithNoOverrideReportsFalse:

    @BOTH_POLICIES
    async def test_on_the_unrepairable_refusal_and_its_residue(self, policy):
        h = _Harness(policy)

        payload = await h.refusal({'decision': _DECISION, 'rationale': _UNREPAIRABLE_WITHOUT_OVERRIDE})

        assert payload['override_swallowed'] is False
        assert [record['override_swallowed'] for record in h.escalations] == [False]

    async def test_on_an_unrecovered_residue(self):
        h = _Harness(RepairPolicy.FORWARD_REPAIR)

        await h.call({'decision': _DECISION, 'rationale': _UNRECOVERED_WITHOUT_OVERRIDE})

        assert [(r['error_type'], r['override_swallowed']) for r in h.escalations] == [
            (MARKUP_UNRECOVERED_PARAM_ERROR_TYPE, False),
        ]

    async def test_on_the_reject_bounce(self):
        h = _Harness(RepairPolicy.REJECT_WITH_REPAIR)

        payload = await h.refusal({'decision': _DECISION, 'rationale': _REPAIRABLE_WITHOUT_OVERRIDE})

        assert payload['override_swallowed'] is False


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
