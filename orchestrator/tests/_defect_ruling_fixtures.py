"""Reviewer defect rulings for tests, in the wire shape submit_review_verdict takes.

The wire shape is orchestrator/review_payload.py::DefectRuling.to_record. A
field it gains is added to contract_ruling here once, and every ruling fixture
built from it follows.
"""

from __future__ import annotations

from shared.defects import (
    MalformedDefectRef,
    RulingSubject,
    mint_redmain_ref,
    mint_review_ref,
    parse_ref,
)

RULED_REF = mint_review_ref(7048, 'a.py:1', 'x is wrong')
SECOND_RULED_REF = mint_redmain_ref('0123456789abcdef')


def contract_ruling(**overrides: object) -> dict[str, object]:
    """An accepted closure of RULED_REF, with *overrides* set over it."""
    return {
        'ref': RULED_REF,
        'subject': RulingSubject.CLOSURE.value,
        'accepted': True,
        **overrides,
    }


def contract_ruling_without(key: str, **overrides: object) -> dict[str, object]:
    """``contract_ruling(**overrides)`` with *key* absent."""
    ruling = contract_ruling(**overrides)
    del ruling[key]
    return ruling


def parse_ref_message(value: object) -> str:
    """The message shared.defects.parse_ref refuses *value* with."""
    try:
        parse_ref(value)
    except MalformedDefectRef as exc:
        return str(exc)
    raise AssertionError(f'{value!r} parsed as a defect ref')
