"""The O4 handoff of the overreach shadow's flags to recon Stage 1 (design.md §6.1, task 6137).

Each full recon cycle hands Stage 1 a deterministic list of the oldest edges
the shadow judge labelled overreach that no completed cycle has been shown yet,
less those whose edge is gone or already retired. A completed cycle
acknowledges its page, so a flag past the page limit waits for the next cycle
and a failed cycle's flags are offered again. Stage 1 may retire a flag it
confirms; nothing here filters reads.
"""
from __future__ import annotations

import logging
from typing import Protocol

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


class OverreachFlagSource(Protocol):
    """What the handoff needs of services/overreach_shadow.py::OverreachShadow."""

    async def overreach_flag_page(self, graph: str, limit: int) -> OverreachFlagPage: ...

    async def acknowledge(self, page: OverreachFlagPage) -> None: ...


async def fetch_overreach_flags(
    shadow: OverreachFlagSource | None, project_id: str,
) -> OverreachFlagPage | None:
    """*project_id*'s next page of flags, or None (warned) when the shadow cannot answer."""
    if shadow is None:
        logger.warning('overreach flags unavailable for project_id=%r: no overreach shadow', project_id)
        return None
    try:
        return await shadow.overreach_flag_page(project_id, FLAGS_SECTION_LIMIT)
    except Exception as exc:
        logger.warning('overreach flags fetch failed for project_id=%r: %r', project_id, exc)
        return None


async def acknowledge_overreach_flags(
    shadow: OverreachFlagSource | None, page: OverreachFlagPage | None,
) -> None:
    """Record that Stage 1 was shown *page*, once complete_run has made the cycle durable.

    A failure (warned) re-offers the page next cycle.
    """
    if shadow is None or page is None:
        return
    try:
        await shadow.acknowledge(page)
    except Exception as exc:
        logger.warning('overreach flags acknowledgement failed for project_id=%r: %r', page.graph, exc)


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
        lines.append(f'+{page.omitted} more, held for the next cycle')
    return '\n'.join(lines) + '\n'
