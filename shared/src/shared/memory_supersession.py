"""The wire spelling of the memory supersession mark.

fused-memory writes it
(``fused-memory/src/fused_memory/services/supersession_mark.py::mark_superseded``)
onto a record that a later ``add_memory`` names in ``metadata.supersedes``. Its
value is the superseding Mem0 record's id, and ``add_memory`` discards any value
a caller supplies, so nothing else writes it. Readers such as
``orchestrator/src/orchestrator/agents/memory_recall.py`` render it; the two
packages cannot import each other, so the spelling lives here.
"""

from typing import Final

__all__ = ['SUPERSEDED_BY_METADATA_KEY']

SUPERSEDED_BY_METADATA_KEY: Final = 'x_superseded_by'
