"""The typed facts a merge outcome carries about why it did not land (task 7302)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from escalation.categories import EscalationCategory


class MergeBlockCategory(StrEnum):
    """The escalation category the workflow files a generic blocked merge under.

    Each value is taken from :class:`EscalationCategory`, so a category the
    escalation registry does not know cannot be spelled here.
    """

    POST_MERGE_VERIFY = EscalationCategory.POST_MERGE_VERIFY.value
    MERGE_NEEDS_REBASE = EscalationCategory.MERGE_NEEDS_REBASE.value
    MERGE_ERROR = EscalationCategory.MERGE_ERROR.value


MAX_RECORDED_CONFLICTED_PATHS = 100


@dataclass(frozen=True)
class ConflictEvidence:
    """The paths a merge conflicted on and, when known, whom it conflicted with.

    ``counterpart_task_ids`` is ``None`` when the conflict was not attributed
    to any landed-ahead work, and ``()`` when it conflicted against main itself.
    """

    conflicted_paths: tuple[str, ...]
    counterpart_task_ids: tuple[str, ...] | None = None

    def to_event_payload(self) -> dict[str, Any]:
        """The runs.db fields this evidence adds to a merge event's data.

        ``conflicted_paths`` is bounded at ``MAX_RECORDED_CONFLICTED_PATHS``;
        ``conflicted_paths_total`` always carries the true count, so a reader
        never has to know the cap to tell a complete list from a truncated one.
        """
        payload: dict[str, Any] = {
            'conflicted_paths': list(self.conflicted_paths[:MAX_RECORDED_CONFLICTED_PATHS]),
            'conflicted_paths_total': len(self.conflicted_paths),
        }
        if self.counterpart_task_ids is not None:
            payload['counterpart_task_ids'] = list(self.counterpart_task_ids)
        return payload
