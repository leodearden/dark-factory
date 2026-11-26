"""Renders a task's declared defects for a task-preamble briefing (plans/defect-tests-prd.md decision 17)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import ValidationError
from shared.defects import DEFECTS_METADATA_KEY, DefectDeclaration

from orchestrator.agents.metadata_list_block import MetadataListBlock

_DECLARED_DEFECTS_BLOCK = MetadataListBlock(
    title='Declared Defects',
    key=DEFECTS_METADATA_KEY,
    contents='declarations',
    intro='These are the defects declared on this task, each identified by its ref.',
)

DECLARED_DEFECTS_HEADING = _DECLARED_DEFECTS_BLOCK.heading


def _parse(entry: object) -> DefectDeclaration | None:
    try:
        return DefectDeclaration.model_validate(entry)
    except ValidationError:
        return None


def _bullets(declarations: Sequence[DefectDeclaration]) -> Iterable[str]:
    for declaration in declarations:
        summary = ' '.join(declaration.summary.split())
        yield (
            f'- `{declaration.ref}` — origin: `{declaration.origin}`, '
            f'status: `{declaration.status}` — {summary}'
        )


def format_declared_defects(metadata: Mapping[str, Any]) -> str:
    """The ``metadata.defects`` block: '' when none are declared, malformed entries counted visibly."""
    return _DECLARED_DEFECTS_BLOCK.render(metadata.get(DEFECTS_METADATA_KEY), _parse, _bullets)
