"""Read-time resolution of the commit SHAs an escalation cites to the commits
they landed as; a record is never rewritten (plans/plan-deviation-recording-prd.md D12).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Protocol

from escalation.models import Escalation

CITED_SHA = re.compile(r'\b[0-9a-f]{7,40}\b')


class CitationResolver(Protocol):
    async def resolve_rebased_citation(self, sha: str) -> Mapping[str, object] | None: ...


def cited_shas(esc: Escalation) -> list[str]:
    """Distinct SHA-shaped tokens in summary, detail and evidence observations, first seen first."""
    observations = [entry.get('observation') for entry in esc.evidence]
    texts = [esc.summary, esc.detail, *(o for o in observations if isinstance(o, str))]
    return list(dict.fromkeys(sha for text in texts for sha in CITED_SHA.findall(text)))


async def resolve_cited_shas(
    esc: Escalation, resolver: CitationResolver,
) -> list[dict[str, object]]:
    """One ``{cited, **landed twin}`` entry per cited SHA the resolver maps, in cited order."""
    resolved: list[dict[str, object]] = []
    for sha in cited_shas(esc):
        found = await resolver.resolve_rebased_citation(sha)
        if found is not None:
            resolved.append({'cited': sha, **found})
    return resolved
