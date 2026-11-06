"""Which tasks declare a defect ref: the liveness read behind ``duplicate_of`` and ``superseded_by``.

``plans/defect-tests-prd.md`` decision 13 lets a closure or retirement name
another ref only when that ref is live, meaning some task, of any status,
declares it in ``metadata.defects``. The read pages any
:class:`MetadataLookupSource`, the one-method surface both the task backend
and the interceptor expose.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import ValidationError
from shared.defects import (
    DEFECTS_METADATA_KEY,
    DeclareEvent,
    DefectDeclaration,
    DefectStatus,
    OutcomeEvent,
    parse_ref,
)

from fused_memory.backends.defect_records import DefectRefNotLive, refs_required_live
from fused_memory.backends.metadata_lookup import DEFAULT_LOOKUP_LIMIT
from fused_memory.backends.task_backend_types import MetadataLookupResult, MetadataLookupRow

__all__ = [
    'DefectRefCarrier',
    'DefectRefLookup',
    'MetadataLookupSource',
    'lookup_defect_ref',
    'require_refs_live',
]

_PAGE_SIZE = 100


class MetadataLookupSource(Protocol):
    async def find_tasks_by_metadata(
        self,
        project_root: str,
        key: str,
        value: str | int | None = None,
        *,
        statuses: list[str] | None = None,
        limit: int = DEFAULT_LOOKUP_LIMIT,
        offset: int = 0,
        tag: str | None = None,
        include_value: bool = False,
    ) -> MetadataLookupResult: ...


@dataclass(frozen=True)
class DefectRefCarrier:
    """A task that declares the ref, and where the ref stands on it."""

    task_id: int
    task_status: str
    defect_status: DefectStatus


@dataclass(frozen=True)
class DefectRefLookup:
    """The tasks declaring *ref*, and those whose defect entries could not be read."""

    ref: str
    carriers: tuple[DefectRefCarrier, ...]
    unreadable_task_ids: tuple[int, ...]

    @property
    def is_live(self) -> bool:
        return bool(self.carriers)

    def as_dict(self) -> dict[str, Any]:
        return {
            'ref': self.ref,
            'live': self.is_live,
            'carriers': [
                {
                    'task_id': carrier.task_id,
                    'task_status': carrier.task_status,
                    'defect_status': carrier.defect_status.value,
                }
                for carrier in self.carriers
            ],
            'unreadable_task_ids': list(self.unreadable_task_ids),
        }


def _declarations(row: MetadataLookupRow) -> tuple[DefectDeclaration, ...] | None:
    """The row's entries, or None when any of them does not read as a declaration."""
    entries = row.get('matched_value')
    if not isinstance(entries, list):
        return None
    try:
        return tuple(DefectDeclaration.model_validate(entry) for entry in entries)
    except ValidationError:
        return None


async def _pages(
    source: MetadataLookupSource, project_root: str, *, tag: str | None, page_size: int
) -> AsyncIterator[MetadataLookupResult]:
    offset = 0
    while True:
        page = await source.find_tasks_by_metadata(
            project_root,
            DEFECTS_METADATA_KEY,
            None,
            include_value=True,
            limit=page_size,
            offset=offset,
            tag=tag,
        )
        yield page
        offset += len(page['results'])
        if not page['results'] or offset >= page['total']:
            return


async def lookup_defect_ref(
    source: MetadataLookupSource,
    project_root: str,
    ref: str,
    *,
    tag: str | None = None,
    page_size: int = _PAGE_SIZE,
) -> DefectRefLookup:
    """Every task, of any status, whose ``metadata.defects`` declares *ref*.

    Raises:
        MalformedDefectRef: *ref* is outside the ref grammar.
    """
    parse_ref(ref)
    carriers: list[DefectRefCarrier] = []
    unreadable: set[int] = set()
    async for page in _pages(source, project_root, tag=tag, page_size=page_size):
        unreadable.update(page['malformed_metadata_ids'])
        for row in page['results']:
            declarations = _declarations(row)
            if declarations is None:
                unreadable.add(row['task_id'])
                continue
            carriers.extend(
                DefectRefCarrier(row['task_id'], row['status'], declaration.status)
                for declaration in declarations
                if declaration.ref == ref
            )
    return DefectRefLookup(ref, tuple(carriers), tuple(sorted(unreadable)))


async def require_refs_live(
    source: MetadataLookupSource,
    project_root: str,
    events: Iterable[DeclareEvent | OutcomeEvent],
    *,
    tag: str | None = None,
) -> None:
    """Refuse *events* when a ``duplicate_of`` or ``superseded_by`` ref they name is not live.

    Raises:
        DefectRefNotLive: for the first such ref no task declares.
    """
    for requirement in refs_required_live(events):
        lookup = await lookup_defect_ref(source, project_root, requirement.ref, tag=tag)
        if not lookup.is_live:
            raise DefectRefNotLive(
                requirement.ref,
                requirement.relation.value,
                unreadable_task_ids=lookup.unreadable_task_ids,
            )
