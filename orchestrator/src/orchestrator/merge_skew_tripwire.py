"""Merge-skew δ: pipeline-landing tripwire (task 2382).

M3 of ``plans/merge-skew-attribution-prd.md`` (PRD task δ, invariant I6,
boundary rows 5-6) — the proactive tripwire. On each successful merge
landing, if the landing's changed files trip a project-configured
load-bearing oracle, emit exactly ONE advisory info escalation naming the
landing sha and the in-flight tasks whose branch diffs OVERLAP the landing's
changed set (the ones whose own edits must be PORTED, not merely rebased),
and attach a steward-visible note to those tasks' metadata.

Advisory-only: never blocks/reorders the queue, ≤1 escalation per landing,
oracle absent/erroring → logged no-op, never delays advance. This module
holds the testable, injectable logic; ``Harness._maybe_pipeline_landing_tripwire``
(orchestrator/harness.py) is the thin fail-open adapter that wires it to the
real config/scheduler/git_ops/escalation_queue.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class TripwireHit:
    """One in-flight task whose branch diff overlaps a load-bearing landing.

    ``overlap_files`` is the sorted intersection of this task's own branch
    diff with the landing's changed files — the files whose landed edits
    this task's branch must PORT (not merely rebase over).
    """

    task_id: str
    branch: str
    overlap_files: tuple[str, ...]


async def _run_load_bearing_oracle(
    project_root: Path,
    oracle_cmd: list[str] | None,
    changed_files: list[str],
) -> bool:
    """Return True iff the project-configured load-bearing oracle says
    *changed_files* are load-bearing.

    Shells out to ``[*oracle_cmd, *changed_files]`` in *project_root* and
    returns True when the command exits 0 (conventional Unix predicate: exit
    0 ⟹ condition is true ⟹ load-bearing).

    Fail-open contract — returns False for ANY of:
    - ``oracle_cmd`` is ``None`` or empty (tripwire disabled / misconfigured).
    - ``changed_files`` is empty.
    - The command exits non-zero (oracle says not load-bearing).
    - The command is missing, non-executable, or any other exception is
      raised (an oracle hiccup must never wedge the merge-landed hot path —
      log WARNING, return False).

    Mirrors ``verify._verify_pipeline_guard_requires_full_gate`` exactly,
    but takes a config-driven command list instead of a hardcoded
    ``scripts/verify-pipeline-guard.sh`` path.
    """
    try:
        if not oracle_cmd or not changed_files:
            return False
        from orchestrator.git_ops import _run  # noqa: PLC0415, I001 — lazy, mirrors _verify_pipeline_guard_requires_full_gate
        rc, _out, _err = await _run(
            [*oracle_cmd, *changed_files],
            cwd=project_root,
        )
        return rc == 0
    except Exception:
        logger.warning(
            '_run_load_bearing_oracle: unexpected error for %s',
            project_root, exc_info=True,
        )
        return False


def compute_tripwire_overlap(
    landing_changed_files: list[str],
    inflight_diffs: list[tuple[str, str, list[str]]],
) -> list[TripwireHit]:
    """Return a :class:`TripwireHit` for each in-flight task whose branch
    diff overlaps *landing_changed_files*.

    *inflight_diffs* is ``[(task_id, branch, branch_changed_files), ...]``.
    A task with ZERO overlap is silently absent from the result (not a
    zero-``overlap_files`` hit) — Open Q2's resolution: the landing's own
    changed files ARE the concrete load-bearing set the oracle just
    flagged, so intersecting each branch's diff against them yields exactly
    the files whose landed edits that branch must port.

    Deterministic: preserves *inflight_diffs* input order; each hit's
    ``overlap_files`` is sorted independently.
    """
    landing_set = set(landing_changed_files)
    if not landing_set:
        return []
    hits: list[TripwireHit] = []
    for task_id, branch, branch_changed_files in inflight_diffs:
        overlap = tuple(sorted(set(branch_changed_files) & landing_set))
        if overlap:
            hits.append(TripwireHit(task_id=task_id, branch=branch, overlap_files=overlap))
    return hits


async def emit_pipeline_landing_tripwire(
    *,
    project_root: Path,
    oracle_cmd: list[str] | None,
    escalation_queue: Any,
    landing_sha: str,
    landing_task_id: str,
    landing_changed_files: list[str],
    inflight: list[tuple[str, str]],
    get_branch_diff: Callable[[str], Awaitable[list[str] | None]],
    update_task: Callable[..., Awaitable[bool]],
) -> None:
    """Advisory pipeline-landing tripwire (PRD task δ, boundary rows 5-6).

    If *landing_changed_files* trips the project-configured load-bearing
    oracle (*oracle_cmd*), emit exactly ONE advisory info escalation naming
    *landing_sha* and every in-flight task (from *inflight*, a list of
    ``(task_id, branch)`` pairs — the just-landed task already excluded by
    the caller) whose branch diff — fetched via *get_branch_diff* —
    overlaps *landing_changed_files*, and attach a steward-visible note to
    each such task's metadata via *update_task*.

    Fully fail-open (I6): *oracle_cmd* absent → logged no-op; oracle
    negative → no-op; zero overlapping in-flight tasks → no-op (nothing
    actionable to report). Never raises.
    """
    if not oracle_cmd:
        logger.debug('emit_pipeline_landing_tripwire: no oracle_cmd configured, no-op')
        return
    if not landing_changed_files:
        return
    load_bearing = await _run_load_bearing_oracle(
        project_root, oracle_cmd, landing_changed_files,
    )
    if not load_bearing:
        return

    inflight_diffs: list[tuple[str, str, list[str]]] = []
    for task_id, branch in inflight:
        branch_files = await get_branch_diff(branch)
        if branch_files is None:
            continue
        inflight_diffs.append((task_id, branch, branch_files))

    hits = compute_tripwire_overlap(landing_changed_files, inflight_diffs)
    if not hits:
        return

    from escalation.models import Escalation  # noqa: PLC0415, I001 — local import, escalation optional dep

    sentinel = f'pipeline-landing-tripwire-{landing_sha[:12]}'
    hit_lines = '\n'.join(
        f'- task {hit.task_id} (branch {hit.branch}): {", ".join(hit.overlap_files)}'
        for hit in hits
    )
    overlapping_task_ids = ', '.join(hit.task_id for hit in hits)
    summary = (
        f'Pipeline landing {landing_sha[:12]} touches load-bearing files also '
        f'present on {len(hits)} in-flight branch(es): {overlapping_task_ids}'
    )
    detail = (
        f'Landing task: {landing_task_id} (sha {landing_sha[:12]})\n'
        f'Load-bearing changed files: {", ".join(landing_changed_files)}\n'
        '\n'
        f'In-flight branches with overlapping edits:\n{hit_lines}'
    )
    esc = Escalation(
        id=escalation_queue.make_id(sentinel),
        task_id=sentinel,
        agent_role='orchestrator-merge-skew-tripwire',
        severity='info',
        level=0,
        category='risk_identified',
        summary=summary,
        detail=detail,
        suggested_action=(
            "Port the landed change into each named in-flight branch's own "
            'edits (not merely rebase) before it merges, to avoid silently '
            'reverting or conflicting with the load-bearing change.'
        ),
    )
    escalation_queue.submit(esc)

    for hit in hits:
        await update_task(
            hit.task_id,
            {
                'merge_skew_tripwire': {
                    'landing_sha': landing_sha,
                    'overlap_files': list(hit.overlap_files),
                },
            },
            metadata_mode='merge',
        )
