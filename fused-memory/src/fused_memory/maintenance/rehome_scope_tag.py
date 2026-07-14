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

from typing import Any

# The metadata `kind` value the task-2273 CGL-eta migration wrote onto every
# cross_target_rehome entry. Verified nowhere else in the repo (the writer
# was never committed) -- this constant is the canonical, single source of
# truth for identifying those entries going forward.
CGL_ETA_REHOME_KIND = 'cgl_eta_cross_target_rehome'


def scope_tag_for(metadata: dict[str, Any]) -> str | None:
    """Build the explicit source-project scope tag for a rehomed entry.

    Returns ``"[<src_project>:<entity>]"`` where ``<entity>`` is
    ``src_entity`` (falling back to ``dst_entity``) when either is present,
    or the bare ``"[<src_project>]"`` when neither entity ref is present.
    Returns ``None`` when ``src_project`` is missing or empty -- there is no
    origin project to disambiguate against.
    """
    src = (metadata.get('src_project') or '').strip()
    if not src:
        return None
    ent = (metadata.get('src_entity') or metadata.get('dst_entity') or '').strip()
    return f'[{src}:{ent}]' if ent else f'[{src}]'


def apply_scope_tag(content: str, metadata: dict[str, Any]) -> str:
    """Prepend the source-project scope tag to *content*, idempotently.

    Returns *content* unchanged when ``scope_tag_for`` cannot compute a tag
    (no ``src_project``) or when *content* already starts with this
    ``src_project``'s tag prefix (``"[<src_project>:"`` or
    ``"[<src_project>]"``) -- this prefix check, not an exact-tag match, is
    what makes re-running the maintenance script over already-tagged
    entries a no-op.
    """
    tag = scope_tag_for(metadata)
    if tag is None:
        return content
    src = (metadata.get('src_project') or '').strip()
    if content.startswith(f'[{src}:') or content.startswith(f'[{src}]'):
        return content
    return f'{tag} {content}'
