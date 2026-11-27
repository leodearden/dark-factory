"""The lane-buffered requests a deep chain holds for its tip verify, and when that hold ends.

A reservation is provisional from the chain snapshot until the build finishes,
then confirmed onto the links the build kept, with a deadline. The contract is
``plans/deep-merge-ahead-prd.md`` (dispatch contract).
"""
from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Collection, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator.merge_lane.types import MergeRequest

logger = logging.getLogger('orchestrator.merge_lane')


@dataclasses.dataclass(frozen=True)
class ChainReservation:
    """One chain's hold: the links in land order, and the deadline once confirmed."""

    owner_request_id: str
    owner_task_id: str
    links: tuple[tuple[str, str], ...]
    tip: str | None = None
    deadline: float | None = None

    @property
    def link_request_ids(self) -> tuple[str, ...]:
        return tuple(request_id for request_id, _ in self.links)

    @property
    def link_task_ids(self) -> tuple[str, ...]:
        return tuple(task_id for _, task_id in self.links)


def _past_deadline(record: ChainReservation, now: float) -> bool:
    return record.deadline is not None and now >= record.deadline


class ChainReservations:
    """Every live chain reservation, keyed by the owning (tip) request id."""

    def __init__(
        self,
        *,
        monotonic: Callable[[], float],
        on_release: Callable[[], None],
    ) -> None:
        self._monotonic = monotonic
        self._on_release = on_release
        self._by_owner: dict[str, ChainReservation] = {}
        self._owner_of_link: dict[str, str] = {}

    def holds(self, request_id: str) -> bool:
        """True iff *request_id* is a link some chain reserves."""
        return request_id in self._owner_of_link

    def reserve_candidates(
        self, owner: MergeRequest, candidates: Sequence[MergeRequest],
    ) -> ChainReservation:
        """Provisionally reserve *candidates* for *owner*'s chain build.

        A request another owner already holds is refused (INV-2: a request
        belongs to at most one round).
        """
        for candidate in candidates:
            self._refuse_foreign_hold(owner, candidate)
        record = ChainReservation(
            owner_request_id=owner.request_id,
            owner_task_id=owner.task_id,
            links=tuple((c.request_id, c.task_id) for c in candidates),
        )
        self._put(record)
        return record

    def confirm(
        self,
        owner_request_id: str,
        *,
        link_request_ids: Collection[str],
        tip: str,
        timeout_secs: float,
    ) -> ChainReservation | None:
        """Narrow *owner_request_id*'s hold to the built links and start its deadline."""
        record = self._by_owner.get(owner_request_id)
        if record is None:
            return None
        kept = tuple(link for link in record.links if link[0] in link_request_ids)
        confirmed = dataclasses.replace(
            record, links=kept, tip=tip, deadline=self._monotonic() + timeout_secs,
        )
        self._put(confirmed)
        if len(kept) < len(record.links):
            self._on_release()
        return confirmed

    def release(self, owner_request_id: str) -> ChainReservation | None:
        """End *owner_request_id*'s hold; None when it holds nothing (idempotent)."""
        record = self._drop(owner_request_id)
        if record is not None:
            self._on_release()
        return record

    def expire(self) -> list[ChainReservation]:
        """Release every confirmed hold past its deadline, one WARNING each."""
        now = self._monotonic()
        expired = [r for r in self._by_owner.values() if _past_deadline(r, now)]
        for record in expired:
            self._drop(record.owner_request_id)
            logger.warning(
                'Deep chain reservation of task %s expired: links %s go back to '
                'the ordinary pipeline while its tip verify continues',
                record.owner_task_id, ', '.join(record.link_task_ids),
            )
        if expired:
            self._on_release()
        return expired

    def snapshot(self) -> list[dict]:
        """One row per live reservation, for the lane snapshot."""
        now = self._monotonic()
        return [
            {
                'owner_task_id': record.owner_task_id,
                'link_task_ids': list(record.link_task_ids),
                'tip': record.tip,
                'seconds_left': (
                    None if record.deadline is None else max(0.0, record.deadline - now)
                ),
            }
            for record in self._by_owner.values()
        ]

    def clear(self) -> None:
        """Drop every reservation without a WARNING (lane stop)."""
        self._by_owner.clear()
        self._owner_of_link.clear()

    def _put(self, record: ChainReservation) -> None:
        self._drop(record.owner_request_id)
        self._by_owner[record.owner_request_id] = record
        for request_id in record.link_request_ids:
            self._owner_of_link[request_id] = record.owner_request_id

    def _drop(self, owner_request_id: str) -> ChainReservation | None:
        record = self._by_owner.pop(owner_request_id, None)
        if record is not None:
            for request_id in record.link_request_ids:
                self._owner_of_link.pop(request_id, None)
        return record

    def _holder_of(self, request_id: str) -> ChainReservation | None:
        owner_request_id = self._owner_of_link.get(request_id)
        return None if owner_request_id is None else self._by_owner[owner_request_id]

    def _refuse_foreign_hold(self, owner: MergeRequest, candidate: MergeRequest) -> None:
        holder = self._holder_of(candidate.request_id)
        if holder is None or holder.owner_request_id == owner.request_id:
            return
        raise ValueError(
            f'request {candidate.request_id} (task {candidate.task_id}) is already '
            f'reserved by task {holder.owner_task_id}\'s chain; task '
            f'{owner.task_id}\'s chain cannot reserve it too',
        )
