"""The O4 handoff of the overreach shadow's flags to recon Stage 1 (design.md §6.1, task 6137).

Each full recon cycle hands Stage 1 a deterministic list of the edges the
shadow judge labelled overreach since the previous run started. Stage 1 may
retire a flag it confirms; nothing here filters reads.
"""
from __future__ import annotations

import logging
from datetime import datetime

from fused_memory.services.overreach_shadow import OverreachFlag, OverreachFlagPage

logger = logging.getLogger(__name__)

FLAGS_SECTION_HEADER = '### Overreach Shadow Flags'
FLAGS_SECTION_LIMIT = 40
_TEXT_CHARS = 300

_INSTRUCTIONS = (
    'The overreach shadow judge flagged these edges as asserting more than the decision '
    'record they were extracted from decided. They are NOT filtered from reads. Confirm '
    'each one against the span the judge quotes, using the rubric in '
    'docs/ruling-overreach-guard-design-2026-10-01/design.md §2.6. If the edge does '
    'over-assert what the record decided, retire it with '
    '`update_edge(edge_uuid=..., invalid_at=<now>)`. That is the only sanctioned action: '
    'leave a flag you cannot confirm alone, and never rewrite the fact.'
)


async def fetch_overreach_flags(
    memory_service: object, project_id: str, *, since: datetime | None,
) -> OverreachFlagPage | None:
    """*project_id*'s flags judged after *since*, or None (warned) when the shadow cannot answer."""
    shadow = getattr(memory_service, 'overreach_shadow', None)
    if shadow is None:
        logger.warning('overreach flags unavailable for project_id=%r: no overreach shadow', project_id)
        return None
    try:
        page = await shadow.overreach_flags_since(project_id, since, FLAGS_SECTION_LIMIT)
    except Exception as exc:
        logger.warning('overreach flags fetch failed for project_id=%r: %r', project_id, exc)
        return None
    if not isinstance(page, OverreachFlagPage):
        logger.warning('overreach flags fetch for project_id=%r returned a %s, not a flag page',
                       project_id, type(page).__name__)
        return None
    return page


def _clip(text: str) -> str:
    flat = ' '.join(text.split())
    return flat if len(flat) <= _TEXT_CHARS else flat[:_TEXT_CHARS - 1] + '…'


def _flag_line(flag: OverreachFlag) -> str:
    return (
        f'- edge {flag.edge_uuid} (episode {flag.episode_uuid}): "{_clip(flag.fact)}"'
        f' — judge: {_clip(flag.rationale)}'
    )


def render_overreach_flags_section(page: OverreachFlagPage | None) -> str:
    """Stage 1's section for *page*; '' when there is nothing to hand over."""
    if page is None or not page.flags:
        return ''
    lines = [
        f'\n{FLAGS_SECTION_HEADER} ({len(page.flags) + page.omitted})',
        _INSTRUCTIONS,
        *(_flag_line(flag) for flag in page.flags),
    ]
    if page.omitted:
        lines.append(f'+{page.omitted} more')
    return '\n'.join(lines) + '\n'
