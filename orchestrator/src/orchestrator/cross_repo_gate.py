"""cross_repo_gate — dispatch-time cross-repo admission gate.

Classifies a task as *foreign-owned* (its declared work belongs to another
project) BEFORE an agent spins up, so the harness can block it and file a
human-reviewed L1 rather than burning an architect + implementer turn on work
this orchestrator cannot legitimately land, and then paying for an L2 when the
branch comes back empty.

Task 3121.  Incident shape: reify-5638 — a task whose ``metadata.files`` are
all absolute paths under another project's root reached the architect, because
the ONLY consumer of the cross-repo signal (``merge_gates.is_cross_repo_task``,
reached from the pre-merge Decision-1 gate) sits at MERGE time, which such a
task never reaches.  This module closes that gap at the other end of the
lifecycle.

Dependency-light: stdlib + logging only, mirroring ``substrate_gate``'s
charter — this module runs in-process inside the harness's ``_run_slot``, on
every dispatch, so it must not drag imports into that hot path.  The one
heavier read (``merge_gates.is_cross_repo_task``, for path containment) is a
function-local import, mirroring workflow.py's call site.

Note the deliberate asymmetry with the fused-memory submit-time guard: the
orchestrator owns exactly ONE ``project_root`` and has no cross-project
registry (fused-memory is not a runtime dependency of ``orchestrator``), so it
can classify an ABSOLUTE path by containment alone and must otherwise rely on
the marker the submit path wrote.  That is precisely the two-signal contract
``merge_gates.is_cross_repo_task`` already documents, which is why this module
reuses that function instead of forking the definition of "foreign".
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLOCK = 'block'   # foreign-owned: block dispatch + escalate
ALLOW = 'allow'   # no cross-repo evidence: dispatch may proceed
SKIP = 'skip'     # evidence UNREADABLE (degenerate metadata) — never "verified clean"


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------


def _extract_metadata(task: dict[str, Any]) -> dict[str, Any] | None:
    """Return ``task['metadata']`` as a dict, or None when unreadable.

    Applies the same JSON-string→dict coercion as
    ``substrate_gate.carries_substrate_probe`` / ``extract_probe_set`` (and
    ``Scheduler._normalize_task_metadata``) so both dispatch gates read task
    metadata through identical rules — a wire-format change cannot make one
    gate see a marker the other misses.

    Returns None for: absent metadata, ``None``, a non-dict value, a string
    that fails to parse, and a string that decodes to a non-dict.  Callers
    MUST distinguish that None ("no evidence readable") from an empty dict
    ("read fine, carries nothing").
    """
    raw = task.get('metadata')

    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        if not isinstance(parsed, dict):
            return None
        raw = parsed

    if not isinstance(raw, dict):
        return None

    return raw


# ---------------------------------------------------------------------------
# Dispatch predicate
# ---------------------------------------------------------------------------


def carries_cross_repo_signal(task: dict[str, Any]) -> bool:
    """Return True iff *task* carries anything the cross-repo gate should weigh.

    Used by ``Harness._run_slot`` to decide whether to run the gate at all, so
    a task with no cross-repo signal pays nothing.

    True when the task's metadata (dict, or JSON string decoding to a dict)
    carries ANY of:

    * the ``'cross_repo'`` KEY — regardless of its value;
    * the ``'possible_scope_mismatch'`` KEY — regardless of its value;
    * a non-empty ``'files'`` value.

    KEY-presence, deliberately, NOT value validity.  This is the lesson task
    2121 already paid for on the substrate gate: gating dispatch on a stricter
    predicate (one requiring a well-formed marker) would let a MALFORMED marker
    skip the gate entirely, when the whole point is that it should enter the
    gate and fail CLOSED there.  The observed markers are caller-authored and
    untyped (``true``, ``"dark-factory"``,
    ``"dark-factory:orchestrator/src/orchestrator/offline_lane.py"``, often
    with no ``cross_repo_project`` companion), so a strict predicate would miss
    every real instance.  See ``substrate_gate.carries_substrate_probe``.

    An EMPTY ``files`` list is NOT a signal: it carries no path evidence, so
    there is nothing for the gate to weigh (and ``is_cross_repo_task`` returns
    False for it by design).  A marked task with empty ``files`` still enters
    the gate via the ``cross_repo`` key.

    Never raises — it runs on every dispatch and must not take down a slot.
    """
    meta = _extract_metadata(task)
    if meta is None:
        return False

    if 'cross_repo' in meta or 'possible_scope_mismatch' in meta:
        return True

    return bool(meta.get('files'))
