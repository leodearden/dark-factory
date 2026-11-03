"""Read-time resolution of the commit SHAs an escalation cites to the commits
they landed as; a record is never rewritten (plans/plan-deviation-recording-prd.md D12).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from escalation.models import Escalation

CITED_SHA = re.compile(r'\b[0-9a-f]{7,40}\b')
MAX_CITED_SHAS_RESOLVED = 10


class CitationResolver(Protocol):
    async def resolve_rebased_citation(self, sha: str) -> Mapping[str, object] | None: ...


@dataclass(frozen=True)
class CitedShaResolution:
    """``resolved``: one ``{cited, **landed twin}`` entry per looked-up SHA that
    has a twin, in cited order.  ``skipped``: the cited SHAs past
    ``MAX_CITED_SHAS_RESOLVED``, never looked up."""

    resolved: tuple[dict[str, object], ...]
    skipped: tuple[str, ...]


def cited_shas(esc: Escalation) -> list[str]:
    """Distinct SHA-shaped tokens in summary, detail and evidence observations, first seen first."""
    observations = [entry.get('observation') for entry in esc.evidence]
    texts = [esc.summary, esc.detail, *(o for o in observations if isinstance(o, str))]
    return list(dict.fromkeys(sha for text in texts for sha in CITED_SHA.findall(text)))


async def resolve_cited_shas(esc: Escalation, resolver: CitationResolver) -> CitedShaResolution:
    """Look up the first ``MAX_CITED_SHAS_RESOLVED`` of *esc*'s cited SHAs, one at a time."""
    shas = cited_shas(esc)
    resolved: list[dict[str, object]] = []
    for sha in shas[:MAX_CITED_SHAS_RESOLVED]:
        found = await resolver.resolve_rebased_citation(sha)
        if found is not None:
            resolved.append({'cited': sha, **found})
    return CitedShaResolution(
        resolved=tuple(resolved), skipped=tuple(shas[MAX_CITED_SHAS_RESOLVED:]),
    )
