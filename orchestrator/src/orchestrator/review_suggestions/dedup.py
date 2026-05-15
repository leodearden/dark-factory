"""Dedup helpers for the review-suggestion direct-to-curator path.

These three functions are the single canonical owner of the
``#hash:<hex16>#`` format used both in the steward escalation fallback
(``_escalate_suggestions`` in workflow.py) and the new fire-and-forget
curator path (``_route_review_suggestions_to_curator``).

Import site contract:  every call site that participates in review-suggestion
dedup **must** import from this module — that is the acceptance criterion for
the shared-helper requirement.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Iterable

from orchestrator.agents.triage import sha256_16

if TYPE_CHECKING:
    pass


def review_suggestion_payload_hash(suggestions: list) -> str:
    """Return a stable 16-char hex hash over *suggestions*.

    Byte-identical to the inline expression in workflow.py:4835-4837::

        hashlib.sha256(
            json.dumps(suggestions, sort_keys=True).encode()
        ).hexdigest()[:16]

    Routed through :func:`orchestrator.agents.triage.sha256_16` to honour the
    canonical 16-char sha256-hex shape contract documented in that function's
    docstring.

    :raises ValueError: when *suggestions* is empty — hashing an empty
        suggestion list has no meaningful dedup value and likely indicates a
        caller bug.  (The guard mirrors sha256_16's own empty-input rejection.)
    """
    if not suggestions:
        raise ValueError(
            "review_suggestion_payload_hash requires a non-empty suggestions list"
        )
    return sha256_16(json.dumps(suggestions, sort_keys=True))


def hash_marker(content_hash: str) -> str:
    """Return the ``#hash:<content_hash>#`` prefix string.

    Used as the leading bytes of the ``detail`` field so that dedup lookups
    can use a cheap ``str.startswith`` check instead of parsing JSON.
    """
    return f'#hash:{content_hash}#'


def find_prior_review_suggestion(
    prior_escalations: Iterable[Any],
    content_hash: str,
) -> Any | None:
    """Return the first prior escalation that matches *content_hash*, or None.

    Predicate extracted verbatim from workflow.py:4840-4851::

        for prev in existing:
            if (
                prev.category == 'review_suggestions'
                and prev.detail
                and prev.detail.startswith(f'#hash:{content_hash}#')
            ):
                return prev

    :param prior_escalations: iterable of Escalation-like objects with
        ``category`` and ``detail`` attributes.
    :param content_hash: the 16-char hex hash to look for.
    :returns: the first matching record, or None if none found.
    """
    marker = hash_marker(content_hash)
    for prev in prior_escalations:
        if (
            getattr(prev, 'category', None) == 'review_suggestions'
            and prev.detail
            and prev.detail.startswith(marker)
        ):
            return prev
    return None
