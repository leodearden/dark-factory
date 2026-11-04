"""The ``## `` headings of a brief ``gen_brief.py`` writes, importable without the CLI's imports."""
from __future__ import annotations

import enum


class Section(enum.StrEnum):
    """The brief's ``## `` headings."""

    DECOMPOSE = 'Decompose'
    STREAM = 'Stream'
    RULINGS = 'Rulings binding the stream'
    UPSTREAM = 'Upstream program tasks'
    GATE = 'Gate'
    RELEASE_DUTIES = 'Release duties'
    INVESTIGATE = 'Investigate'
    ITEM = 'Item'
    FINDINGS = 'Findings'
    FILINGS = 'Phase 5 filings'
    XREFS = 'Cross-references to the stream'
    SITTING = 'Design questions go to the sitting'
    RETURN = 'Return'
    SPAWN = 'Spawn'
