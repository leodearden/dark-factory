"""Read-time resolution of the commit SHAs an escalation cites to the commits
they landed as; a record is never rewritten (plans/plan-deviation-recording-prd.md D12).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from escalation.models import Escalation

logger = logging.getLogger(__name__)

CITED_SHA = re.compile(r'\b[0-9a-f]{7,40}\b')
MAX_CITED_SHAS_RESOLVED = 10


class CitationResolver(Protocol):
    """A git authority for cited tokens; both calls raise when git could not answer, so an omitted token is always a fact."""

    async def commits_named(self, tokens: Sequence[str]) -> Mapping[str, str]: ...

    async def resolve_rebased_citation(self, sha: str) -> Mapping[str, object] | None: ...


@dataclass(frozen=True)
class CitedShaResolution:
    """``resolved``: one ``{cited, **landed twin}`` entry per looked-up SHA that
    has a twin, in cited order.  ``failed``: ``{cited, error}`` per token git
    could not answer for, in cited order; a fault, not a fact.  ``skipped``:
    the tokens past ``MAX_CITED_SHAS_RESOLVED``, never looked up.  Tokens
    naming no commit appear in none of the three and never count against the
    cap; when git could not tell which tokens name a commit, every token
    counts."""

    resolved: tuple[dict[str, object], ...]
    failed: tuple[dict[str, str], ...]
    skipped: tuple[str, ...]


def cited_shas(esc: Escalation) -> list[str]:
    """Distinct SHA-shaped tokens in summary, detail and evidence observations, first seen first."""
    observations = [entry.get('observation') for entry in esc.evidence]
    texts = [esc.summary, esc.detail, *(o for o in observations if isinstance(o, str))]
    return list(dict.fromkeys(sha for text in texts for sha in CITED_SHA.findall(text)))


def _failure(token: str, exc: Exception) -> dict[str, str]:
    return {'cited': token, 'error': f'{type(exc).__name__}: {exc}'}


async def resolve_cited_shas(esc: Escalation, resolver: CitationResolver) -> CitedShaResolution:
    """Look up the first ``MAX_CITED_SHAS_RESOLVED`` of *esc*'s cited tokens
    that name a commit, one at a time.

    Any exception from *resolver* means git could not answer (the Protocol's
    contract), so it is logged at WARNING with its traceback and recorded as
    failed, never read as a miss.  When classification itself fails, the
    first ``MAX_CITED_SHAS_RESOLVED`` tokens are failed and the rest skipped.
    """
    tokens = cited_shas(esc)
    try:
        commits = await resolver.commits_named(tokens)
    except Exception as exc:
        logger.warning(
            'could not tell which of the %d tokens %s cites name a commit; '
            'reporting them failed', len(tokens), esc.id, exc_info=True,
        )
        return CitedShaResolution(
            resolved=(),
            failed=tuple(_failure(token, exc) for token in tokens[:MAX_CITED_SHAS_RESOLVED]),
            skipped=tuple(tokens[MAX_CITED_SHAS_RESOLVED:]),
        )
    cited_commits = [token for token in tokens if token in commits]
    resolved: list[dict[str, object]] = []
    failed: list[dict[str, str]] = []
    for sha in cited_commits[:MAX_CITED_SHAS_RESOLVED]:
        try:
            found = await resolver.resolve_rebased_citation(sha)
        except Exception as exc:
            logger.warning(
                'could not resolve %s cited by %s; reporting it failed',
                sha, esc.id, exc_info=True,
            )
            failed.append(_failure(sha, exc))
        else:
            if found is not None:
                resolved.append({'cited': sha, **found})
    return CitedShaResolution(
        resolved=tuple(resolved),
        failed=tuple(failed),
        skipped=tuple(cited_commits[MAX_CITED_SHAS_RESOLVED:]),
    )
