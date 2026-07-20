"""Suppress re-flagging of suggestions settled in a PRIOR amendment round (task 2523).

After an in-workflow amendment round, ``reviewer_comprehensive`` re-runs on the
full task diff and can re-surface a suggestion the team already SETTLED — tried
and resolved in an earlier round (applied, or deliberately reverted and kept
as-is) — burning implementer + curator cycles re-litigating a closed decision.
This module is the TEMPORAL half of the fix, a sibling of the SPATIAL
:mod:`orchestrator.review_suggestions.amendment_scope`:

- :func:`load_prior_round_suggestions` reconstructs the prior-round
  suggestion-severity set from the archived ``reviews-amend-*/`` review JSONs,
  mirroring :meth:`orchestrator.artifacts.TaskArtifacts.aggregate_reviews`'
  partition (skip ``verdict == 'ERROR'`` reviews; keep only
  ``severity != 'blocking'`` issues).
- :func:`build_resettled_adjudicator_prompt` builds the batched adjudication
  prompt comparing the current suggestions against that prior-round set.
- :func:`partition_by_decisions` applies the adjudicator's per-index verdict,
  suppressing a current suggestion ONLY on an explicit :data:`SETTLED` — every
  other value fails SAFE toward emit.

Pure functions — no git invocation, no workflow state.  Parsing is tolerant:
a missing root, a malformed / non-JSON archive file, or a review lacking an
``issues`` list are each skipped without raising.  The fail-safe posture
(default to EMITTING a suggestion on any ambiguity or failure) is the single
invariant shared with the async workflow layer that consumes these helpers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Per-index adjudication vocabulary.  Only SETTLED suppresses a suggestion;
# NOT_SETTLED and INCONCLUSIVE both keep it (fail-safe toward emit).
SETTLED = 'settled'
NOT_SETTLED = 'not_settled'
INCONCLUSIVE = 'inconclusive'


def load_prior_round_suggestions(artifacts_root: Path) -> list[dict]:
    """Return the flattened prior-round suggestion-severity issues.

    Globs ``reviews-amend-*/`` under *artifacts_root* and reads each ``*.json``
    review file (written as ``{reviewer_name}.json`` by
    :meth:`orchestrator.artifacts.TaskArtifacts.write_review`).  Mirrors
    ``aggregate_reviews``' partition exactly: reviews with
    ``verdict == 'ERROR'`` are skipped and only non-blocking
    (``severity != 'blocking'``) issues are collected, each enriched with the
    originating ``reviewer`` name (the file stem) like ``aggregate_reviews``.

    Tolerant of the messy realities of an on-disk archive: a missing / non-dir
    root, a non-``reviews-amend-*`` sibling, a malformed / unreadable / non-JSON
    file, a review that is not a dict, or a review lacking an ``issues`` list
    are each skipped without raising.  Returns ``[]`` when nothing qualifies.
    """
    if not isinstance(artifacts_root, Path) or not artifacts_root.is_dir():
        return []
    collected: list[dict] = []
    for archive_dir in sorted(artifacts_root.glob('reviews-amend-*')):
        if not archive_dir.is_dir():
            continue
        for review_path in sorted(archive_dir.glob('*.json')):
            try:
                review = json.loads(review_path.read_text())
            except (OSError, ValueError):
                # Non-JSON / unreadable archive file — a debugging aid, not
                # load-bearing; skip rather than fail the suppression pass.
                continue
            if not isinstance(review, dict):
                continue
            if review.get('verdict') == 'ERROR':
                continue
            issues = review.get('issues')
            if not isinstance(issues, list):
                continue
            for issue in issues:
                if not isinstance(issue, dict):
                    continue
                if issue.get('severity') == 'blocking':
                    continue
                collected.append({**issue, 'reviewer': review_path.stem})
    return collected
