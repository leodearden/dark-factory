"""The quality series producer: one row per landing, and the statistics read over those rows.

Contract: plans/quality-dashboard-prd.md §Contract (B + H). Every module is stdlib only.
"""

from shared.quality_series.landings import (
    CursorNotOnMain,
    GitCrossCheck,
    Landing,
    LandingBatch,
    UnrecognisedTrainEvent,
    landings,
)
from shared.quality_series.rows import LandingRow, rows_for, unit_heads

__all__ = [
    'CursorNotOnMain',
    'GitCrossCheck',
    'Landing',
    'LandingBatch',
    'LandingRow',
    'UnrecognisedTrainEvent',
    'landings',
    'rows_for',
    'unit_heads',
]
