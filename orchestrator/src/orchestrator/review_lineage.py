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
from pathlib import Path
from types import NoneType
from typing import Any

from orchestrator.artifacts import TaskArtifacts
from orchestrator.quarantine_provenance import QUARANTINED_FROM_KEY, quarantined_from_of
from orchestrator.review_suggestions.prior_round import load_prior_round_suggestions

logger = logging.getLogger(__name__)

RETIRED_REVIEWS_DIR = 'retired-reviews'

_RECORD_FIELDS: Mapping[str, tuple[type, ...]] = {
    'branch': (str, NoneType),
    'tip_sha': (str, NoneType),
    'parking': (str,),
    'quarantined_at': (str,),
    'amendment_rounds_total': (int,),
    'review_cycles_total': (int,),
    'archive': (str, NoneType),
}


@dataclass(frozen=True)
class RetiredReview:
    """One review lineage a quarantine ended, as retired at the next dispatch.

    ``branch`` is None when the quarantine renamed no branch aside, so nothing
    was retired. ``archive`` is the ``retired-reviews/<parking-basename>/``
    directory the round archives moved into; None when none moved.
    ``suggestions`` are the archived amend rounds' non-blocking issues.
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
        """The ``retired_lineages`` entry; the suggestions stay on disk in ``archive``."""
        return {
            'branch': self.branch,
            'tip_sha': self.tip_sha,
            'parking': self.parking,
            'quarantined_at': self.quarantined_at,
            'amendment_rounds_total': self.amendment_rounds_total,
            'review_cycles_total': self.review_cycles_total,
            'archive': str(self.archive) if self.archive is not None else None,
        }


def _from_record(record: Mapping[str, Any]) -> RetiredReview | None:
    """The record as a :class:`RetiredReview`; None when it does not match ``_RECORD_FIELDS``."""
    for key, accepted in _RECORD_FIELDS.items():
        value = record.get(key)
        if key not in record or not isinstance(value, accepted) or isinstance(value, bool):
            return None
    archive = record['archive']
    return RetiredReview(
        branch=record['branch'],
        tip_sha=record['tip_sha'],
        parking=record['parking'],
        quarantined_at=record['quarantined_at'],
        amendment_rounds_total=record['amendment_rounds_total'],
        review_cycles_total=record['review_cycles_total'],
        archive=Path(archive) if archive is not None else None,
    )


def _retired_reviews(artifacts: TaskArtifacts) -> tuple[RetiredReview, ...]:
    parsed = (_from_record(record) for record in artifacts.retired_review_records())
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
    review = replace(
        review,
        amendment_rounds_total=artifacts.get_amendment_rounds_total(),
        review_cycles_total=artifacts.get_review_cycles_total(),
        archive=_move_round_archives(artifacts, Path(review.parking).name),
    )
    artifacts.record_retired_lineage(review.to_record(), zero_counters=True)
    if review.amendment_rounds_total or review.review_cycles_total:
        logger.warning(
            'review lineage: retired %d amendment round(s) and %d review cycle(s) answered on quarantined '
            'branch %s (tip %s); this branch gets a fresh review allowance',
            review.amendment_rounds_total, review.review_cycles_total, review.branch, review.tip_sha,
        )
    return _with_suggestions(review)


def _move_round_archives(artifacts: TaskArtifacts, parking_name: str) -> Path | None:
    """Move every ``reviews-<label>`` archive under ``retired-reviews/<parking_name>/``; None when that holds none."""
    destination_root = artifacts.artifact_path(f'{RETIRED_REVIEWS_DIR}/{parking_name}')
    for archive in artifacts.review_archive_dirs():
        destination = destination_root / archive.name
        if destination.exists():
            logger.warning('review lineage: %s already exists; leaving %s in place', destination, archive)
            continue
        destination_root.mkdir(parents=True, exist_ok=True)
        shutil.move(archive, destination)
    return destination_root if destination_root.is_dir() else None
