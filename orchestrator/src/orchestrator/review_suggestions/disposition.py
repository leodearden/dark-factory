"""Which sink one ``TaskWorkflow._route_review_suggestions_to_curator`` call
handed a batch of review suggestions to — the ``[suggestions → <value>]``
suffix a steward reads on a ``review_issues`` escalation.
"""

from __future__ import annotations

from enum import StrEnum


class SuggestionDisposition(StrEnum):
    """What each value tells the steward about where the suggestions went.

    NONE: the review carried no suggestions to route.
    DEDUPED: this TaskWorkflow instance had already routed a byte-identical
    set as its most recent batch, so nothing was re-sent.
    CURATOR: one ``submit_task`` post per suggestion was scheduled,
    fire-and-forget; a delivery failure is logged by ``shared.mcp_post``
    (task 4023), never reported back.
    ESCALATION_QUEUE: no MCP transport, so the batch was filed as an info
    escalation for steward triage.
    DROPPED: no MCP transport and no escalation queue; only a WARNING log
    records the batch.
    ERROR: scoping or scheduling raised before the batch reached any sink,
    so the ``review_issues`` detail is its only copy.  Out-of-delta
    suggestions, which the amendment-delta scope routes on its own, are not
    part of the batch and may already be with the curator.
    """

    NONE = 'none'
    DEDUPED = 'deduped'
    CURATOR = 'curator'
    ESCALATION_QUEUE = 'escalation_queue'
    DROPPED = 'dropped'
    ERROR = 'error'
