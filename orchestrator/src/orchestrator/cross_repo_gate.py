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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLOCK = 'block'   # foreign-owned: block dispatch + escalate
ALLOW = 'allow'   # no cross-repo evidence: dispatch may proceed
SKIP = 'skip'     # evidence UNREADABLE (degenerate metadata) — never "verified clean"

# Signal names recorded on a blocking verdict.
SIGNAL_MARKER = 'cross_repo_marker'


# ---------------------------------------------------------------------------
# Verdict dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrossRepoVerdict:
    """Result of a dispatch-time cross-repo classification.

    Mirrors ``substrate_gate.SubstrateVerdict`` / ``.flipped``.

    Attributes:
        verdict: One of BLOCK / ALLOW / SKIP.
        owner_project: The project that owns the declared work, when it can be
            RESOLVED from the task's own metadata.  ``None`` when nothing names
            it — the orchestrator has no cross-project registry and must not
            guess a project name from a path.  An honest "unresolved" in the L1
            beats a placeholder a human would have to disbelieve.
        signals: The evidence legs that fired, e.g. ``('cross_repo_marker',)``.
        foreign_paths: The declared entries the path-containment leg judged
            foreign (empty when no path evidence was weighed).
        reason: Human-readable explanation, rendered into the L1 detail.
    """

    verdict: str
    owner_project: str | None
    signals: tuple[str, ...]
    foreign_paths: tuple[str, ...]
    reason: str

    @property
    def blocked(self) -> bool:
        """True when the verdict is BLOCK (gate should block dispatch)."""
        return self.verdict == BLOCK


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


# ---------------------------------------------------------------------------
# Owner resolution
# ---------------------------------------------------------------------------


def _as_project_name(value: Any) -> str | None:
    """Return *value* as a non-empty project name, or None.

    Rejects non-strings (including ``bool``) and blank/whitespace strings, so a
    malformed companion never becomes a placeholder owner in an L1.
    """
    if not isinstance(value, str):
        return None
    name = value.strip()
    return name or None


def _resolve_owner(meta: dict[str, Any]) -> str | None:
    """Resolve the owning project from *meta*, or None when nothing names it.

    Precedence, most-authoritative first:

    1. ``cross_repo_project`` — the typed companion the fused-memory submit
       path writes alongside the marker.
    2. ``cross_repo`` itself when it is a non-empty string.  The observed
       caller-authored spellings are ``'dark-factory'`` and
       ``'dark-factory:orchestrator/src/orchestrator/offline_lane.py'``, so the
       pre-``':'`` field is taken.
    3. ``possible_scope_mismatch['suggested_project']`` — the prose matcher's
       suggestion; last because it is advisory.

    Returns None rather than inventing a name.  The orchestrator owns exactly
    one project_root and has no cross-project registry, so it genuinely cannot
    name the owner of a bare ``cross_repo: true``.
    """
    owner = _as_project_name(meta.get('cross_repo_project'))
    if owner:
        return owner

    marker = meta.get('cross_repo')
    if isinstance(marker, str):
        owner = _as_project_name(marker.split(':', 1)[0])
        if owner:
            return owner

    stamp = meta.get('possible_scope_mismatch')
    if isinstance(stamp, dict):
        owner = _as_project_name(stamp.get('suggested_project'))
        if owner:
            return owner

    return None


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_cross_repo(
    *,
    task: dict[str, Any],
    project_root: Path | str,
) -> CrossRepoVerdict:
    """Classify *task* as foreign-owned (BLOCK), local (ALLOW) or unreadable (SKIP).

    Leg (A) — ``metadata.cross_repo`` truthy.  The marker the fused-memory
    submit path writes when its ``ProjectPrefixRegistry`` recognizes that every
    declared file is owned by one other project.  This is the only leg that can
    see the RELATIVE-foreign shape, which the orchestrator cannot classify
    unaided.  A FALSY value is not evidence: key-presence is what got the task
    into the gate (see ``carries_cross_repo_signal``), truth is what blocks.

    Legs (B) and (C) — declared-files containment and a containment-confirmed
    ``possible_scope_mismatch`` — land in later steps.

    Never raises for any metadata shape; the caller (``_run_cross_repo_gate``)
    additionally fails CLOSED should that ever prove wrong.
    """
    meta = _extract_metadata(task)
    if meta is None:
        # step-8 turns this into a LOUD SKIP.  ALLOW for now.
        return CrossRepoVerdict(
            verdict=ALLOW,
            owner_project=None,
            signals=(),
            foreign_paths=(),
            reason='task metadata is not a readable dict',
        )

    signals: list[str] = []
    reasons: list[str] = []

    marker = meta.get('cross_repo')
    if marker:
        signals.append(SIGNAL_MARKER)
        reasons.append(
            f'metadata.cross_repo is set ({marker!r}) — the submit path classified '
            f'this task\'s declared files as owned by another project'
        )

    if not signals:
        return CrossRepoVerdict(
            verdict=ALLOW,
            owner_project=None,
            signals=(),
            foreign_paths=(),
            reason='no cross-repo evidence in task metadata',
        )

    return CrossRepoVerdict(
        verdict=BLOCK,
        owner_project=_resolve_owner(meta),
        signals=tuple(signals),
        foreign_paths=(),
        reason='; '.join(reasons),
    )
