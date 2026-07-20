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


def partition_by_decisions(
    current: list[dict], decisions: list[str],
) -> tuple[list[dict], list[dict]]:
    """Partition *current* into ``(kept, suppressed)`` by adjudicator verdict.

    A current suggestion is suppressed ONLY when its positionally-aligned
    decision (``decisions[i]``) is exactly :data:`SETTLED`.  Every other outcome
    keeps the suggestion (fail-safe toward emit): any other value, an unknown
    string, or a missing index (``decisions`` shorter than *current*).  Input
    order is preserved within each partition.
    """
    kept: list[dict] = []
    suppressed: list[dict] = []
    for i, suggestion in enumerate(current):
        decision = decisions[i] if i < len(decisions) else NOT_SETTLED
        if decision == SETTLED:
            suppressed.append(suggestion)
        else:
            kept.append(suggestion)
    return kept, suppressed


def _format_suggestion_block(
    suggestions: list[dict], *, enumerated: bool
) -> str:
    """Render *suggestions* as a human-readable block for the adjudicator.

    When *enumerated* is True each entry is prefixed with its 0-based index
    (``[i]``) so the adjudicator can address a per-index verdict back; otherwise
    entries are bulleted.  Field access is tolerant — a non-dict or a missing
    field renders as empty rather than raising.
    """
    lines: list[str] = []
    for i, s in enumerate(suggestions):
        get = s.get if isinstance(s, dict) else (lambda _k, _d='': '')
        location = get('location', '')
        category = get('category', '')
        description = get('description', '')
        suggested_fix = get('suggested_fix', '')
        header = f'[{i}] ' if enumerated else '- '
        lines.append(
            f'{header}location={location} | category={category}\n'
            f'    description: {description}\n'
            f'    suggested_fix: {suggested_fix}'
        )
    return '\n'.join(lines) if lines else '(none)'


def build_resettled_adjudicator_prompt(
    current: list[dict], prior_settled: list[dict],
) -> str:
    """Build the batched prior-round-resolution adjudication prompt (pure).

    Embeds an enumerated (0-indexed) block of the *current* suggestions
    (location + category + description + suggested_fix) and a block of the
    *prior_settled* suggestions, and asks for a per-index verdict drawn from the
    {:data:`SETTLED`, :data:`NOT_SETTLED`, :data:`INCONCLUSIVE`} vocabulary —
    with explicit guidance to answer ``inconclusive`` when unsure so the
    caller's fail-safe partition keeps anything ambiguous.
    """
    current_block = _format_suggestion_block(current, enumerated=True)
    prior_block = _format_suggestion_block(prior_settled, enumerated=False)
    return (
        'You are adjudicating whether each CURRENT review suggestion merely '
        're-flags a concern that was already SETTLED in a PRIOR amendment round '
        'of this same task — a concern the team already tried and resolved '
        '(applied a fix, or deliberately reverted and kept the code as-is). '
        'Re-flagging a settled concern wastes cycles re-litigating a closed '
        'decision.\n\n'
        'PRIOR-ROUND SETTLED SUGGESTIONS:\n'
        f'{prior_block}\n\n'
        'CURRENT SUGGESTIONS (0-indexed):\n'
        f'{current_block}\n\n'
        'For EACH current suggestion index, return exactly one decision:\n'
        f'  - "{SETTLED}": substantively the same concern as a prior-round '
        'settled suggestion — it re-litigates that closed decision.\n'
        f'  - "{NOT_SETTLED}": a new or still-open concern — emit it.\n'
        f'  - "{INCONCLUSIVE}": you cannot tell. Answer this whenever you are '
        'unsure — do NOT guess "' + SETTLED + '".\n\n'
        'Only "' + SETTLED + '" suppresses a suggestion; every other answer '
        'emits it. When in doubt, prefer "' + INCONCLUSIVE + '". Return one '
        'decision per current index.'
    )
