"""The quality series producer: one row per landing, and the statistics read over those rows.

Contract: plans/quality-dashboard-prd.md §Contract (B + H). Every module is stdlib only.
"""

from shared.quality_series.landings import (
    CursorNotOnMain,
    GitCrossCheck,
    Landing,
    LandingBatch,
    landings,
)

__all__ = [
    'CursorNotOnMain',
    'GitCrossCheck',
    'Landing',
    'LandingBatch',
    'landings',
]
