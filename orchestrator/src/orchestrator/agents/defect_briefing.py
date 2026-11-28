"""Renders a task's defect records for briefings (plans/defect-tests-prd.md decision 17)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from functools import partial
from typing import Any

from shared.defects import (
    DEFECT_TEST_RETIREMENTS_METADATA_KEY,
    DEFECTS_METADATA_KEY,
    DefectDeclaration,
    Exemption,
    Retirement,
    RulingSubject,
    Withdrawal,
)
from shared.governed_exceptions import TaskRef, TicketRef

from orchestrator.agents.metadata_list_block import MetadataListBlock
from orchestrator.defect_records import closures_awaiting_ruling, parse_record

_DECLARED_DEFECTS_BLOCK = MetadataListBlock(
    title='Declared Defects',
    key=DEFECTS_METADATA_KEY,
    contents='declarations',
    intro='These are the defects declared on this task, each identified by its ref.',
)

DECLARED_DEFECTS_HEADING = _DECLARED_DEFECTS_BLOCK.heading

CLOSURES_AWAITING_RULING_HEADING = (
    f'## Defect Closures Awaiting Your Ruling (metadata.{DEFECTS_METADATA_KEY})'
)

_CLOSURES_INTRO = (
    'Each of these defects carries a declared closure that awaits your ruling. Answer each '
    f'one in `defect_rulings` with `subject` `{RulingSubject.CLOSURE}` and its ref exactly as shown.'
)

_RETIREMENTS_BLOCK = MetadataListBlock(
    title='Defect-Test Retirements',
    key=DEFECT_TEST_RETIREMENTS_METADATA_KEY,
    contents='retirements',
    intro=(
        "These are the retirements of proven defects' tests declared on this task, each "
        'identified by the ref of the defect whose test this change removes. Answer each '
        f'pending one in `defect_rulings` with `subject` `{RulingSubject.RETIREMENT}` and its ref '
        'exactly as shown.'
    ),
)

DEFECT_TEST_RETIREMENTS_HEADING = _RETIREMENTS_BLOCK.heading


def _one_line(text: str) -> str:
    return ' '.join(text.split())


def _coded(code: StrEnum, ref: str | None) -> str:
    return f'`{code}`' if ref is None else f'`{code}` `{ref}`'


def _owner(exemption: Exemption) -> str:
    match exemption.owner.owner:
        case TaskRef(id=task_id):
            return f'task {task_id}'
        case TicketRef(id=ticket_id):
            return f'ticket `{ticket_id}`'


def _closure(closure: Exemption | Withdrawal) -> str:
    if isinstance(closure, Exemption):
        return f'exemption: `{closure.reason}`, owed by {_owner(closure)}'
    return f'withdrawal: {_coded(closure.reason.reason, closure.reason.ref)}'


def _bullets(declarations: Sequence[DefectDeclaration]) -> Iterable[str]:
    for declaration in declarations:
        yield (
            f'- `{declaration.ref}` — origin: `{declaration.origin}`, '
            f'status: `{declaration.status}` — {_one_line(declaration.summary)}'
        )


def _awaiting_ruling(metadata: Mapping[str, Any]) -> Iterable[str]:
    for declaration, closure in closures_awaiting_ruling(metadata):
        yield f'- `{declaration.ref}` — {_closure(closure)} — {_one_line(declaration.summary)}'


def _retirement_bullets(retirements: Sequence[Retirement]) -> Iterable[str]:
    for retirement in retirements:
        yield (
            f'- `{retirement.ref}` — reason: {_coded(retirement.reason.reason, retirement.reason.ref)} '
            f'— state: `{retirement.state}`'
        )


def format_declared_defects(metadata: Mapping[str, Any]) -> str:
    """The ``metadata.defects`` block: '' when none are declared, malformed entries counted visibly."""
    return _DECLARED_DEFECTS_BLOCK.render(
        metadata.get(DEFECTS_METADATA_KEY), partial(parse_record, DefectDeclaration), _bullets,
    )


def format_closures_awaiting_ruling(metadata: Mapping[str, Any]) -> str:
    """The pending closures on ``metadata.defects`` for the reviewer to rule on; '' when none is pending.

    Malformed entries are skipped here; the declared-defects block counts them.
    """
    listed = '\n'.join(_awaiting_ruling(metadata))
    if not listed:
        return ''
    return f'{CLOSURES_AWAITING_RULING_HEADING}\n\n{_CLOSURES_INTRO}\n\n{listed}'


def format_defect_test_retirements(metadata: Mapping[str, Any]) -> str:
    """The ``metadata.defect_test_retirements`` block: '' when none, malformed entries counted visibly."""
    return _RETIREMENTS_BLOCK.render(
        metadata.get(DEFECT_TEST_RETIREMENTS_METADATA_KEY), partial(parse_record, Retirement),
        _retirement_bullets,
    )
