"""Which sink one ``TaskWorkflow._route_review_suggestions_to_curator`` call
handed a batch of review suggestions to.

``CURATOR`` means the ``submit_task`` posts were SCHEDULED, fire-and-forget:
a delivery failure is logged by ``shared.mcp_post`` (task 4023), never
reported back here.  ``ERROR`` means routing raised, so whether anything was
delivered is unknown.
"""

from __future__ import annotations

from enum import StrEnum


class SuggestionDisposition(StrEnum):
    NONE = 'none'
    DEDUPED = 'deduped'
    CURATOR = 'curator'
    ESCALATION_QUEUE = 'escalation_queue'
    DROPPED = 'dropped'
    ERROR = 'error'
