"""Reads a task's stored defect records fail-open: an entry that does not validate is skipped, a value that is not a list holds none.

A briefing block that must say what it skipped counts that itself
(agents/metadata_list_block.py::MetadataListBlock); every other reader just
drops it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError
from shared.defects import (
    DEFECT_TEST_RETIREMENTS_METADATA_KEY,
    DEFECTS_METADATA_KEY,
    DefectDeclaration,
    DefectStatus,
    Exemption,
    Retirement,
    RetirementState,
    Withdrawal,
)

_Record = TypeVar('_Record', bound=BaseModel)


def parse_record(model: type[_Record], entry: object) -> _Record | None:
    """*entry* read as *model*, or None when it does not validate."""
    try:
        return model.model_validate(entry)
    except ValidationError:
        return None


def readable_records(model: type[_Record], raw: object) -> tuple[_Record, ...]:
    """The entries of the stored list *raw* that read as *model*."""
    if not isinstance(raw, list):
        return ()
    parsed = (parse_record(model, entry) for entry in raw)
    return tuple(record for record in parsed if record is not None)


def closures_awaiting_ruling(
    metadata: Mapping[str, Any],
) -> tuple[tuple[DefectDeclaration, Exemption | Withdrawal], ...]:
    """Each pending ``metadata.defects`` entry, paired with the closure that awaits a ruling."""
    return tuple(
        (declaration, declaration.closure)
        for declaration in readable_records(DefectDeclaration, metadata.get(DEFECTS_METADATA_KEY))
        if declaration.status is DefectStatus.PENDING and declaration.closure is not None
    )


def retirements_awaiting_ruling(metadata: Mapping[str, Any]) -> tuple[Retirement, ...]:
    """The ``metadata.defect_test_retirements`` records no ruling has settled yet."""
    return tuple(
        retirement
        for retirement in readable_records(Retirement, metadata.get(DEFECT_TEST_RETIREMENTS_METADATA_KEY))
        if retirement.state is RetirementState.PENDING
    )


def awaits_a_ruling(metadata: Mapping[str, Any]) -> bool:
    """Whether a declared closure or a defect-test retirement on the task still awaits its ruling."""
    return bool(closures_awaiting_ruling(metadata) or retirements_awaiting_ruling(metadata))
