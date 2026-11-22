"""``metadata.followup_holder`` and ``metadata.followup_hold``: the follow-up holder's marker and a held task's record.

Contract: plans/follow-up-filing-shape-prd.md §4.3 and §3 decisions 9-14. A held task is one with a
dependency edge on its project's holder; the edge is the authority for "held now" and
``followup_hold`` is its history. Both keys are machine-written only, by
fused-memory/src/fused_memory/middleware/followup_hold.py. The schema lives in shared so
fused-memory's write boundary and the orchestrator parse the same classes.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

__all__ = [
    'FOLLOWUP_HOLDER_METADATA_KEY',
    'FOLLOWUP_HOLD_METADATA_KEY',
    'FollowupHold',
    'FollowupHolder',
    'PromotionCause',
    'PromotionRecord',
]

FOLLOWUP_HOLDER_METADATA_KEY = 'followup_holder'
FOLLOWUP_HOLD_METADATA_KEY = 'followup_hold'

_RECORD_CONFIG = ConfigDict(frozen=True, extra='forbid')


class PromotionCause(StrEnum):
    DEPENDENT = 'dependent'
    CITATION = 'citation'
    COMBINE = 'combine'
    PACK = 'pack'
    RIDER = 'rider'
    HUMAN = 'human'


class FollowupHolder(BaseModel):
    """The marker that makes a task its project's holder; ``minted_at`` is ISO-8601 UTC."""

    model_config = _RECORD_CONFIG

    project_id: str
    minted_at: str


class PromotionRecord(BaseModel):
    """Why and when a held task was released; ``ref`` names the dependent, finding key, ticket or run."""

    model_config = _RECORD_CONFIG

    by: PromotionCause
    at: str
    ref: str | None = None


class FollowupHold(BaseModel):
    """A task's hold: the holder it was parked behind, when, and its promotion once released."""

    model_config = _RECORD_CONFIG

    holder: int
    held_at: str
    promoted: PromotionRecord | None = None
