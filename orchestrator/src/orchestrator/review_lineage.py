"""A branch-swapping quarantine ends the review record's lineage (task 3663).

The task-root review record, meaning ``review_state.json``'s lifetime
counters and the ``reviews-<label>/`` round archives, describes one branch
lineage. A quarantine that renames ``task/<id>`` aside ends it: the next
dispatch cuts a fresh branch from main, so the counters no longer describe
the branch under review, and the amend-round responses committed on the
parked branch are gone from it.

:func:`retire_quarantined_lineages` runs at dispatch setup. For each
``_quarantined_from`` stamp not yet acknowledged it moves the round archives
under ``retired-reviews/<parking-basename>/`` and zeroes the counters, keeping
the counts in a ``retired_lineages`` record. :func:`retired_reviews_since` is
the carry-forward view a replanning architect is briefed from.

Built on ``orchestrator.artifacts``, ``orchestrator.quarantine_provenance``
and ``orchestrator.review_suggestions.prior_round``; the workflow and the
briefing import this module, never the other way round.
"""
from __future__ import annotations

import logging
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from types import NoneType
from typing import Any

from shared.task_deferral import parse_utc_timestamp

from orchestrator.artifacts import TaskArtifacts
from orchestrator.quarantine_provenance import QUARANTINED_FROM_KEY, quarantined_from_of
from orchestrator.review_suggestions.prior_round import load_prior_round_suggestions

logger = logging.getLogger(__name__)

_RECORD_FIELDS: Mapping[str, tuple[type, ...]] = {
    'branch': (str, NoneType),
    'tip_sha': (str, NoneType),
    'parking': (str,),
    'quarantined_at': (str,),
    'amendment_rounds_total': (int,),
    'review_cycles_total': (int,),
    'archived': (bool,),
}


@dataclass(frozen=True)
class RetiredReview:
    """One review lineage a quarantine ended, as retired at the next dispatch.

    ``branch`` is None when the quarantine renamed no branch aside, so nothing
    was retired. ``archive`` is the directory the round archives moved into,
    ``TaskArtifacts.retired_review_dir`` of the parking's basename; None when
    none moved. ``suggestions`` are the archived amend rounds' non-blocking
    issues.
    """

    branch: str | None
    tip_sha: str | None
    parking: str
    quarantined_at: str
    amendment_rounds_total: int
    review_cycles_total: int
    archive: Path | None
    suggestions: tuple[Mapping[str, Any], ...] = ()

    def to_record(self) -> dict[str, Any]:
        """The ``retired_lineages`` entry; ``archive`` is kept as whether one exists, and re-derived on read."""
        return {
            'branch': self.branch,
            'tip_sha': self.tip_sha,
            'parking': self.parking,
            'quarantined_at': self.quarantined_at,
            'amendment_rounds_total': self.amendment_rounds_total,
            'review_cycles_total': self.review_cycles_total,
            'archived': self.archive is not None,
        }


def _archive_dir(artifacts: TaskArtifacts, parking: str) -> Path:
    return artifacts.retired_review_dir(Path(parking).name)


def _from_record(artifacts: TaskArtifacts, record: Mapping[str, Any]) -> RetiredReview | None:
    """The record as a :class:`RetiredReview`; None when it does not match ``_RECORD_FIELDS``."""
    if any(key not in record or type(record[key]) not in accepted for key, accepted in _RECORD_FIELDS.items()):
        return None
    return RetiredReview(
        branch=record['branch'],
        tip_sha=record['tip_sha'],
        parking=record['parking'],
        quarantined_at=record['quarantined_at'],
        amendment_rounds_total=record['amendment_rounds_total'],
        review_cycles_total=record['review_cycles_total'],
        archive=_archive_dir(artifacts, record['parking']) if record['archived'] else None,
    )


def _retired_reviews(artifacts: TaskArtifacts) -> tuple[RetiredReview, ...]:
    parsed = (_from_record(artifacts, record) for record in artifacts.retired_review_records())
    return tuple(review for review in parsed if review is not None)


def _with_suggestions(review: RetiredReview) -> RetiredReview:
    if review.archive is None:
        return review
    return replace(review, suggestions=tuple(load_prior_round_suggestions(review.archive)))


def retire_quarantined_lineages(artifacts: TaskArtifacts) -> tuple[RetiredReview, ...]:
    """Retire the review lineage of each unacknowledged ``_quarantined_from`` stamp. Never raises.

    Returns one :class:`RetiredReview` per stamp acknowledged by this call, in
    stamp order. Each branch-swapping stamp retires whatever review record is
    current, so with several pending stamps the counts and archives go to the
    first and the rest retire zero. A stamp whose ``branch`` is None is
    acknowledged with nothing retired.
    """
    try:
        return _retire_unacknowledged(artifacts)
    except Exception:
        logger.exception('review lineage: could not retire the quarantined review record under %s', artifacts.task_root)
        return ()


def _retire_unacknowledged(artifacts: TaskArtifacts) -> tuple[RetiredReview, ...]:
    stamps = quarantined_from_of(artifacts.peek_plan()).get(QUARANTINED_FROM_KEY, [])
    acknowledged = {(review.parking, review.quarantined_at) for review in _retired_reviews(artifacts)}
    retired = []
    for stamp in stamps:
        identity = (stamp['parking'], stamp['quarantined_at'])
        if identity in acknowledged:
            continue
        acknowledged.add(identity)
        retired.append(_retire(artifacts, stamp))
    return tuple(retired)


def _retire(artifacts: TaskArtifacts, stamp: Mapping[str, Any]) -> RetiredReview:
    review = RetiredReview(
        branch=stamp['branch'],
        tip_sha=stamp['tip_sha'],
        parking=stamp['parking'],
        quarantined_at=stamp['quarantined_at'],
        amendment_rounds_total=0,
        review_cycles_total=0,
        archive=None,
    )
    if review.branch is None:
        artifacts.record_retired_lineage(review.to_record(), zero_counters=False)
        return review
    archive = _archive_dir(artifacts, review.parking)
    review = replace(
        review,
        amendment_rounds_total=artifacts.get_amendment_rounds_total(),
        review_cycles_total=artifacts.get_review_cycles_total(),
        archive=archive if _move_round_archives(artifacts, archive) else None,
    )
    artifacts.record_retired_lineage(review.to_record(), zero_counters=True)
    if review.amendment_rounds_total or review.review_cycles_total:
        logger.warning(
            'review lineage: retired %d amendment round(s) and %d review cycle(s) answered on quarantined '
            'branch %s (tip %s); this branch gets a fresh review allowance',
            review.amendment_rounds_total, review.review_cycles_total, review.branch, review.tip_sha,
        )
    return _with_suggestions(review)


def _move_round_archives(artifacts: TaskArtifacts, destination_root: Path) -> bool:
    """Move every ``reviews-<label>`` archive under *destination_root*; whether that then exists."""
    for archive in artifacts.review_archive_dirs():
        destination = destination_root / archive.name
        if destination.exists():
            logger.warning('review lineage: %s already exists; leaving %s in place', destination, archive)
            continue
        destination_root.mkdir(parents=True, exist_ok=True)
        shutil.move(archive, destination)
    return destination_root.is_dir()


def retired_reviews_since(artifacts: TaskArtifacts, since: datetime | None) -> tuple[RetiredReview, ...]:
    """The retired lineages that still carry suggestions and were quarantined after *since*. Never raises.

    *since* is when an architect last confirmed the plan; None, for a fresh
    plan, keeps every retirement. A record whose ``quarantined_at`` does not
    parse is kept, failing toward briefing the architect.
    """
    try:
        pending = (
            _with_suggestions(review) for review in _retired_reviews(artifacts)
            if review.archive is not None and _quarantined_after(review.quarantined_at, since)
        )
        return tuple(review for review in pending if review.suggestions)
    except (OSError, ValueError):
        logger.warning('review lineage: could not read the retired review record under %s', artifacts.task_root, exc_info=True)
        return ()


def _quarantined_after(quarantined_at: str, since: datetime | None) -> bool:
    if since is None:
        return True
    try:
        return parse_utc_timestamp(quarantined_at) > parse_utc_timestamp(since)
    except ValueError:
        return True
