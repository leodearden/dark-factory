"""Renders a task's declared defects for a task-preamble briefing (plans/defect-tests-prd.md decision 17)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError
from shared.defects import DEFECTS_METADATA_KEY, DefectDeclaration

DECLARED_DEFECTS_HEADING = f'## Declared Defects (metadata.{DEFECTS_METADATA_KEY})'

_INTRO = 'These are the defects declared on this task, each identified by its ref.'


def _bullet(declaration: DefectDeclaration) -> str:
    summary = ' '.join(declaration.summary.split())
    return (
        f'- `{declaration.ref}` — origin: `{declaration.origin}`, '
        f'status: `{declaration.status}` — {summary}'
    )


def _readable(raw: list[object]) -> list[DefectDeclaration]:
    declarations = []
    for entry in raw:
        try:
            declarations.append(DefectDeclaration.model_validate(entry))
        except ValidationError:
            continue
    return declarations


def format_declared_defects(metadata: Mapping[str, Any]) -> str:
    """The ``metadata.defects`` block: '' when none are declared, malformed entries counted visibly."""
    raw = metadata.get(DEFECTS_METADATA_KEY)
    if not raw:
        return ''
    if not isinstance(raw, list):
        return (
            f'{DECLARED_DEFECTS_HEADING}\n\n'
            f'⚠ `metadata.{DEFECTS_METADATA_KEY}` is malformed (a {type(raw).__name__}, not a '
            'list), so its declarations cannot be shown here; read them from the task record.'
        )
    declarations = _readable(raw)
    lines = [_bullet(declaration) for declaration in declarations]
    malformed = len(raw) - len(declarations)
    if malformed:
        noun = 'entry' if malformed == 1 else 'entries'
        lines.append(
            f'⚠ {malformed} malformed {noun} on `metadata.{DEFECTS_METADATA_KEY}` could not be '
            'rendered here; read the full list from the task record.'
        )
    listed = '\n'.join(lines)
    return f'{DECLARED_DEFECTS_HEADING}\n\n{_INTRO}\n\n{listed}'
