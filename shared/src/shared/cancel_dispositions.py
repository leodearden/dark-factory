"""The one home of the cancel-with-children contract: shapes, validation and the decision table.

Contract: plans/cancel-with-children-prd.md §4 and §5.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Final, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    StringConstraints,
)
from pydantic_core import PydanticCustomError

__all__ = [
    'CALLER_ACTIONS',
    'CHILD_CANCELLING_ACTIONS',
    'ENTRY_ONLY_ACTIONS',
    'REASON_MAX_CHARS',
    'CancelAction',
    'ChildDisposition',
    'ChildDispositionVariant',
    'FoldDisposition',
    'KeepDisposition',
    'MootDisposition',
    'RepointDisposition',
]


class CancelAction(StrEnum):
    KEEP = 'keep'
    REPOINT = 'repoint'
    FOLD = 'fold'
    MOOT = 'moot'
    CO_CANCELLED = 'co_cancelled'


CALLER_ACTIONS: Final = frozenset(
    {CancelAction.KEEP, CancelAction.REPOINT, CancelAction.FOLD, CancelAction.MOOT}
)
ENTRY_ONLY_ACTIONS: Final = frozenset({CancelAction.CO_CANCELLED})
CHILD_CANCELLING_ACTIONS: Final = frozenset({CancelAction.FOLD, CancelAction.MOOT})

REASON_MAX_CHARS: Final = 500


def _distinct(ids: tuple[int, ...]) -> tuple[int, ...]:
    repeated = sorted({task_id for task_id in ids if ids.count(task_id) > 1})
    if repeated:
        raise PydanticCustomError(
            'duplicate_target', 'names {repeated} more than once', {'repeated': repeated}
        )
    return ids


TaskId = Annotated[int, Field(strict=True, gt=0)]
ReasonText = Annotated[str, StringConstraints(max_length=REASON_MAX_CHARS)]
DistinctTaskIds = Annotated[tuple[TaskId, ...], Field(min_length=1), AfterValidator(_distinct)]


class _DispositionBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    reason: ReasonText | None = None


class KeepDisposition(_DispositionBase):
    action: Literal[CancelAction.KEEP] = CancelAction.KEEP


class RepointDisposition(_DispositionBase):
    action: Literal[CancelAction.REPOINT] = CancelAction.REPOINT
    onto: DistinctTaskIds


class FoldDisposition(_DispositionBase):
    action: Literal[CancelAction.FOLD] = CancelAction.FOLD
    into: TaskId


class MootDisposition(_DispositionBase):
    action: Literal[CancelAction.MOOT] = CancelAction.MOOT


_DispositionUnion = KeepDisposition | RepointDisposition | FoldDisposition | MootDisposition
ChildDispositionVariant = Annotated[_DispositionUnion, Field(discriminator='action')]


class ChildDisposition(RootModel[ChildDispositionVariant]):
    model_config = ConfigDict(frozen=True)
