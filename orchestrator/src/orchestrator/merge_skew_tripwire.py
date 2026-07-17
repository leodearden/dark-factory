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

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


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
