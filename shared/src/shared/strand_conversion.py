"""``metadata.strand_converted``: the stamp left on a row the stranded-task sweep moved in-progress -> blocked.

Contract: plans/task-escalation-state-graph-prd.md D2; docs/task-escalation-state-spec.md §7.2 and §7.8.
Producer: orchestrator/src/orchestrator/strand_conversion.py::record_strand_conversion. The schema lives
in shared so fused-memory's write boundary validates the same class.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_serializer

__all__ = ['STRAND_CONVERTED_METADATA_KEY', 'StrandConverted']

STRAND_CONVERTED_METADATA_KEY = 'strand_converted'


class StrandConverted(BaseModel):
    """One conversion of a stranded row to ``blocked``.

    at: the conversion instant, timezone-aware.
    shape: the ``recovery_shape_str`` the recovery table classified the row on.
    pinning_escalation_ids: the open records that pin the converted row — its exit owners.
    """

    model_config = ConfigDict(frozen=True, extra='allow')

    at: AwareDatetime
    shape: str = Field(min_length=1)
    pinning_escalation_ids: tuple[str, ...] = Field(min_length=1)

    @field_serializer('at', when_used='json')
    def _at_as_isoformat(self, at: datetime) -> str:
        return at.isoformat()

    def to_metadata(self) -> dict[str, dict]:
        """The ``update_task(metadata_mode='merge')`` fragment."""
        return {STRAND_CONVERTED_METADATA_KEY: self.model_dump(mode='json')}
