"""The excerpt a pre-service guard block echoes back to the writer it rejected.

Every guard in ``server/tools.py``, ``server/near_duplicate_guard.py`` and
``server/entities_gate.py`` spreads :func:`excerpt_fields` into its block, so the
excerpt width and its truncation disclosure have one home.
"""

from __future__ import annotations

from typing import Any

from fused_memory.utils.clip_disclosure import clip_text, truncation_fields

GUARD_EXCERPT_CHARS = 200


def excerpt_fields(*, content_excerpt: str, matched_excerpt: str | None = None) -> dict[str, Any]:
    """The excerpt keys, each cut unmarked to ``GUARD_EXCERPT_CHARS``, plus a truncation fact per cut."""
    clips = {'content_excerpt': clip_text(content_excerpt, GUARD_EXCERPT_CHARS, field='content_excerpt')}
    if matched_excerpt is not None:
        clips['matched_excerpt'] = clip_text(
            matched_excerpt, GUARD_EXCERPT_CHARS, field='matched_excerpt'
        )
    return {
        **{key: clip.text for key, clip in clips.items()},
        **truncation_fields(*(clip.fact for clip in clips.values())),
    }
