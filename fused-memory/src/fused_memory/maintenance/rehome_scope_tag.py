"""Explicit source-project scope tagging for CGL-eta cross_target_rehome entries.

The CGL-eta leak migration (task 2273) wrote Mem0 entries whose true project
scope (``src_project``/``dst_project``/``src_entity``/``dst_entity``) lives
only in metadata, never inline in the human-readable content. Because a
rehomed fact physically lives in its ``dst_project``'s Mem0 collection but
references ``src_project``'s task numbers, a plain/semantic search over the
dst collection cannot distinguish the stale foreign fact from the dst
project's own same-numbered task without inspecting metadata.

This module is the reusable, importable gate: any future rehome path (or
the one-shot maintenance script, ``scripts/tag_cgl_eta_rehome_scope.py``)
routes content through :func:`apply_scope_tag` so the origin project is
always disambiguated inline, not just in metadata.
"""

from __future__ import annotations

# The metadata `kind` value the task-2273 CGL-eta migration wrote onto every
# cross_target_rehome entry. Verified nowhere else in the repo (the writer
# was never committed) -- this constant is the canonical, single source of
# truth for identifying those entries going forward.
CGL_ETA_REHOME_KIND = 'cgl_eta_cross_target_rehome'
