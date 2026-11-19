"""Which sink one ``TaskWorkflow._route_review_suggestions_to_curator`` call
handed a batch of review suggestions to — the ``[suggestions → <value>]``
suffix a steward reads on a ``review_issues`` escalation.
"""

from __future__ import annotations

from enum import StrEnum


class SuggestionDisposition(StrEnum):
    """What each value tells the steward about where the suggestions went.

    NONE: the review carried no suggestions to route.
    DEDUPED: this TaskWorkflow instance had already DELIVERED a
    byte-identical set as its most recent batch, so nothing was re-sent.
    CURATOR: every suggestion was accepted by ``submit_task`` and holds a
    curator ticket.
    UNDELIVERED: ``submit_task`` was attempted but at least one suggestion
    got no curator ticket; a WARNING per suggestion names the reason, and the
    ``review_issues`` detail carries the content.
    ESCALATION_QUEUE: no MCP transport, so the batch was filed as an info
    escalation for steward triage.
    DROPPED: no MCP transport and no escalation queue; only a WARNING log
    records the batch.
    ERROR: scoping or routing raised before submission, so the
    ``review_issues`` detail is its only copy.  Out-of-delta suggestions,
    which the amendment-delta scope routes on its own, are not part of the
    batch and may already be with the curator.
    """

    NONE = 'none'
    DEDUPED = 'deduped'
    CURATOR = 'curator'
    UNDELIVERED = 'undelivered'
    ESCALATION_QUEUE = 'escalation_queue'
    DROPPED = 'dropped'
    ERROR = 'error'
