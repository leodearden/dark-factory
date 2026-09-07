"""File a residual-duplicate escalation for the sqlite backend's v3->v4
self-gating migration (fm-task-dedup W8 task A2).

When ``_migrate_v3_to_v4`` (:mod:`fused_memory.backends.sqlite_task_backend`)
finds non-cancelled duplicate ``candidate_key`` groups still present at
connection-open, it skips building the partial UNIQUE index and calls
:func:`emit_residual_candidate_key_escalation` so an operator sees the
residuals surfaced as an escalation, not just a log line.

The filer skeleton it uses lives in
:mod:`fused_memory.middleware._folded_escalation`, which is the ONE home for
that shape (task 4854).  NOT
:mod:`fused_memory.middleware.scope_violation_escalator`, which this docstring
used to name: that module belongs to a DIFFERENT family, deduping on a content
fingerprint via ``escalation.dedupe.submit_or_dedupe`` over a cached queue
rather than on a pending anchor.  What the shared helper provides here:

* Defensive import of the optional ``escalation`` workspace package — when
  missing (minimal envs, tests without escalation infra), this becomes a
  logged no-op so the migration's fail-safe guarantee is never at risk.
* Escalations land in ``{project_root}/data/escalations`` — the affected
  project's own queue.
* Queue CONSTRUCTION and ``submit`` are both wrapped so a queue I/O failure
  never raises — the migration's self-gating skip has already happened by the
  time this is called; escalation is purely additive.  The construction guard
  is new (task 4854): this module used to construct the queue unguarded, which
  contradicted its own "NEVER raises" promise on a read-only ``project_root``.
* Dedups against an already-open escalation for the same anchor before
  filing a new one (review amendment) — a connection (and therefore this
  migration step) runs at most once per project_root per process, so a
  residual-dup condition that outlives a single process restart would
  otherwise mint a brand-new escalation on every restart.
"""

from __future__ import annotations

import logging
from typing import Any

from fused_memory.middleware._folded_escalation import file_folded_escalation

logger = logging.getLogger(__name__)

# Anchor task_id threaded through ``EscalationQueue.make_id`` AND the dedupe
# fold by ``file_folded_escalation``, so the resulting escalation IDs (e.g.
# ``esc-candidate-key-migration-1``) are easily greppable and distinct from the
# path-guard's ``task-path-guard`` series.
#
# STAYS A CONSTANT OF THIS MODULE even though the filer body is now shared: a
# filer deduping against an anchor somebody else keeps open never files again,
# and that silence is indistinguishable from health. `tests/server/
# test_write_triage.py` and the pairwise regression in
# `tests/test_folded_escalation.py` both read this FROM HERE, which is what
# makes a colliding rename fail a test instead of going silent in production.
_ANCHOR_TASK_ID: str = 'candidate-key-migration'

_AGENT_ROLE: str = 'fused-memory/candidate-key-migration'
_CATEGORY: str = 'candidate_key_residual_duplicates'


def emit_residual_candidate_key_escalation(
    project_root: str,
    residual_groups: list[dict[str, Any]],
) -> str | None:
    """File a ``candidate_key_residual_duplicates`` escalation naming the
    residual non-cancelled duplicate ``candidate_key`` groups a v3->v4
    connection-open audit found.

    ``residual_groups`` entries carry ``tag``, ``candidate_key``,
    ``task_ids`` (list[str]), ``count``, and ``reason`` (``'mixed_status'``
    | ``'title_divergent'``) — the migration (fm-task-dedup self-heal
    amendment) auto-heals genuine content-duplicate groups itself and only
    escalates the AMBIGUOUS remainder, so every group reaching here needs a
    human, and ``reason`` says why.

    Returns the escalation id — either a freshly filed one, or the id of an
    already-open escalation for this condition when one exists (dedup, see
    below) — or ``None`` when neither is possible (the ``escalation``
    package is unavailable, or the queue write failed). NEVER raises: this
    is called from connection-open migration code, and a raise here would
    defeat the self-gating step's own fail-safe guarantee.
    """
    groups_desc = '; '.join(
        f'tag={g.get("tag")!r} candidate_key={g.get("candidate_key")!r} '
        f'task_ids={g.get("task_ids")!r} count={g.get("count")!r} '
        f'reason={g.get("reason")!r}'
        for g in residual_groups
    )
    detail_lines = [
        f'project_root={project_root!r}',
        f'residual_group_count={len(residual_groups)}',
        f'groups={groups_desc}',
        '',
        'The v3->v4 schema migration (fm-task-dedup W8 task A2, self-heal '
        'amendment) auto-healed every genuine content-duplicate group it '
        'found (every row recomputes the same candidate_key, none done) — '
        'the group(s) above are AMBIGUOUS and were left for a human: '
        'reason=mixed_status means the group contains a done row '
        '(cancelling completed work needs sign-off); '
        'reason=title_divergent means the stored candidate_key no longer '
        "matches a fresh recompute of that row's (title, files) (a stale "
        'key, not a real content match). It skipped building the partial '
        'UNIQUE index ux_tasks_candidate_key while any group above remains. '
        'Resolve each (cancel or merge the extras) — the next '
        'connection-open will re-audit and land the index automatically.',
    ]

    # The dedup described in the module docstring, the defensive import, the
    # guarded queue and the never-raise submit all live in
    # `middleware/_folded_escalation`. `_ANCHOR_TASK_ID` is passed explicitly
    # because the helper has NO default for it — sharing an anchor with another
    # filer would silence one of them behind the other's open record.
    return file_folded_escalation(
        project_root,
        anchor_task_id=_ANCHOR_TASK_ID,
        agent_role=_AGENT_ROLE,
        category=_CATEGORY,
        severity='blocking',
        summary=(
            f'{len(residual_groups)} residual duplicate candidate_key '
            f'group(s) blocking the UNIQUE index build'
        ),
        detail='\n'.join(detail_lines),
        suggested_action='clean up residual duplicates (cancel or merge)',
        logger=logger,
        log_label='candidate_key_escalation',
        context=(
            f'{len(residual_groups)} residual duplicate group(s) in '
            f'project_root={project_root!r}'
        ),
        level=1,
    )
