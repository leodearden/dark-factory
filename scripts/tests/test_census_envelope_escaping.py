"""Census payloads that QUOTE tool-call envelope literals must survive the
boundary markup guard (task 5907).

Regression origin: the 2026-09-25 reify census (reify escalations
esc-markup-residue-4 to -10). Verified clusters whose sighting evidence quoted
the invoke, content and rationale closers were turned by
``census.py::build_task_payloads`` into ``submit_task`` kwargs carrying those
literals raw. The fused-memory boundary guard
(``shared.mcp_markup_middleware.MarkupGuardMiddleware``) refused each call as
``mcp_markup_unrepairable`` and queued a residue escalation, so the candidates
never reached the curator.

These tests drive the REAL guard on an in-process FastMCP ``submit_task`` whose
parameters are exactly the payload's keys, so "submits" means what the live
server would do, not a restatement of its predicate. The file is deliberately
SELF-CONTAINED: cross-test-module imports are fragile under the repo-wide
``--import-mode=importlib`` addopts (see scripts/tests/conftest.py).

## Sentinel-literal hazard — DO NOT "helpfully" un-escape these

Every envelope literal here is built from the public constants of
``shared/src/shared/toolcall_markup.py``, which owns the rule and its rationale
("Sentinel-literal hazard"). Expected ESCAPED text is spelled with a doubled
backslash, so the four-character escape text is what the source holds.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, NamedTuple

import census as mod
import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError
from shared.mcp_markup_middleware import MarkupGuardMiddleware, RepairPolicy
from shared.toolcall_markup import INVOKE_CLOSER, closer_for, detect_for

_ESCAPE = '\\x3c'
_DOWNSTREAM_SCHEMA = ('rationale', 'how', 'decision', 'what')
_QUOTED_EVIDENCE = (
    'agent emitted ' + INVOKE_CLOSER + ', then ' + closer_for('content')
    + ' and ' + closer_for('rationale')
)
_SELF_NAME_TITLE = 'Agents mis-close ' + closer_for('title') + ' before ' + INVOKE_CLOSER


class _GuardedSubmit(NamedTuple):
    refusal: ToolError | None
    escalations: list[dict[str, Any]]
    received: dict[str, Any]


def _submit_through_guard(payload: dict[str, Any]) -> _GuardedSubmit:
    """Call a toy ``submit_task`` behind the real guard with *payload* as its
    arguments. ``received`` stays empty when the tool body never ran."""
    escalations: list[dict[str, Any]] = []
    received: dict[str, Any] = {}

    def escalation_sink(record: dict[str, Any]) -> str:
        escalations.append(record)
        return 'esc-markup-residue-1'

    mcp = FastMCP('census-envelope-harness')

    @mcp.tool
    def submit_task(
        project_root: str,
        title: str,
        description: str,
        task_kind: str = 'normal',
        priority: str = 'medium',
        metadata: dict | None = None,
    ) -> dict:
        received.update(
            project_root=project_root,
            title=title,
            description=description,
            task_kind=task_kind,
            priority=priority,
            metadata=metadata,
        )
        return {'ticket': 'tkt_1'}

    mcp.add_middleware(
        MarkupGuardMiddleware(
            RepairPolicy.REJECT_WITH_REPAIR, escalation_sink=escalation_sink,
        )
    )

    async def call() -> None:
        async with Client(mcp) as client:
            await client.call_tool('submit_task', payload)

    try:
        asyncio.run(call())
    except ToolError as refusal:
        return _GuardedSubmit(refusal, escalations, received)
    return _GuardedSubmit(None, escalations, received)


def _quoting_cluster(**overrides: Any) -> dict[str, Any]:
    cluster = {
        'title': 'Agents quote envelope closers in their reports',
        'summary': 'Sighting evidence quotes the literals an agent emitted.',
        'evidence': [_QUOTED_EVIDENCE],
        'severity': 'high',
        'sightings': [
            {'origin_phase': 'implement', 'manifested_phase': 'verify', 'session': 'sess-1'},
        ],
    }
    cluster.update(overrides)
    return cluster


def _payload_for(cluster: dict[str, Any]) -> dict[str, Any]:
    return mod.build_task_payloads([cluster], project_root='/r', project_id='reify')[0]


def test_evidence_quoting_envelope_closers_submits_and_round_trips_escaped():
    payload = _payload_for(_quoting_cluster())

    outcome = _submit_through_guard(payload)

    assert outcome.refusal is None
    assert outcome.escalations == []
    description = outcome.received['description']
    assert description == payload['description']
    assert '\\x3c/invoke>' in description
    assert '\\x3c/content>' in description
    assert '\\x3c/rationale>' in description
    assert _QUOTED_EVIDENCE in description.replace(_ESCAPE, chr(60))
    assert 'Observed in 1 sighting(s)' in description


def test_title_quoting_a_self_name_closer_submits():
    payload = _payload_for(_quoting_cluster(title=_SELF_NAME_TITLE))

    outcome = _submit_through_guard(payload)

    assert outcome.refusal is None
    assert outcome.escalations == []
    assert outcome.received['title'] == payload['title']


@pytest.mark.parametrize('field', ['title', 'description'])
def test_the_harness_refuses_that_payload_with_one_field_left_raw(field):
    """Negative control: the tests above mean "the guard let it through" only
    if the harness really reaches the guard."""
    payload = _payload_for(_quoting_cluster(title=_SELF_NAME_TITLE))
    raw_payload = {**payload, field: payload[field].replace(_ESCAPE, chr(60))}

    outcome = _submit_through_guard(raw_payload)

    assert outcome.refusal is not None
    assert outcome.escalations != []
    assert outcome.received == {}


def test_every_string_field_is_safe_to_requote_downstream():
    title = 'Agents mis-close ' + closer_for('what') + ' in plans'
    payload = _payload_for(_quoting_cluster(title=title))

    for key, value in payload.items():
        if isinstance(value, str):
            assert detect_for(value, key, _DOWNSTREAM_SCHEMA) is None, key


def test_this_module_spells_no_raw_envelope_literal():
    """This file's own SOURCE must never contain a raw ``chr(60)`` + ``/``."""
    needle = chr(60) + '/'
    source = Path(__file__).read_text(encoding='utf-8')
    assert needle not in source, (
        'A raw envelope literal was written into this test file. Spell it with '
        'the \\x3c escape instead — see this module\'s docstring for why.'
    )
