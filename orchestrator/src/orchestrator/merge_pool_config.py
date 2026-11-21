"""The ``merge_pool`` config file: the merge pool's order, promotion period and bounce cap."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

MergePoolOrder = Literal['arrival', 'priority']


class MergePoolConfig(BaseModel):
    """The merge pool's knobs (plans/priority-at-later-gates-prd.md D9, D14 and 6.4).

    Nothing reads them until the merge pool orders by priority (the PRD's γ).
    """

    order: MergePoolOrder = Field(
        default='arrival',
        description="'arrival' keeps today's pick order; 'priority' sorts by task_priority.order_key.",
    )
    promote_after_secs: float | None = Field(
        default=None, gt=0,
        description='Waiting this long promotes a request one tier, never past high; None promotes nothing.',
    )
    bounce_cap: int = Field(
        default=20, ge=1,
        description='Most mechanical rebases of one branch before the lane escalates instead; to replace MERGE_BOUNCE_CAP.',
    )
