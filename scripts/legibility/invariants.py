"""A project's invariant slugs, read from its own design-invariants doc.

This module is the one home of the heading shape that declares an invariant:
``## INV-<id> `<slug>``` on its own line, where ``<id>`` is ``<n>`` or
``<FAMILY>-<n>``. The census coder lists the slugs in its prompt, the codebook
merger checks sightings against them, and
scripts/tests/test_design_invariants_consistency.py parses dark-factory's doc
through it (plans/census-incremental-prd.md D10).

Stdlib-only, so it imports under ``uv run --project shared``.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger('legibility.invariants')

DOC_RELPATH = Path('docs/legibility/design-invariants.md')

_HEADING_RE = re.compile(
    r"^## INV-(?:(?P<family>[A-Z]+)-)?(?P<number>\d+) `(?P<slug>[a-z0-9][a-z0-9-]*)`$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class Invariant:
    """One declared invariant; *family* is None for the unprefixed ``INV-<n>`` form."""

    family: str | None
    number: int
    slug: str


def parse_headings(md_text: str) -> tuple[Invariant, ...]:
    """Every invariant heading in *md_text*, in document order, unvalidated."""
    return tuple(
        Invariant(
            family=match['family'],
            number=int(match['number']),
            slug=match['slug'],
        )
        for match in _HEADING_RE.finditer(md_text)
    )


def read_slugs(project_root: str | os.PathLike[str]) -> tuple[str, ...]:
    """The distinct slugs *project_root*'s doc declares, first occurrence first.

    A project without the doc declares none. A doc that exists but yields no
    heading is a parse drift, not an empty family, so it is logged.
    """
    path = Path(project_root) / DOC_RELPATH
    try:
        text = path.read_text(encoding='utf-8')
    except FileNotFoundError:
        return ()
    headings = parse_headings(text)
    if not headings:
        logger.warning(
            '%s exists but has no `## INV-<id> `<slug>`` heading; the coder '
            'will be told this project declares no invariant slugs',
            path,
        )
        return ()
    return tuple(dict.fromkeys(heading.slug for heading in headings))
