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
SIGNAL_ALL_FILES_FOREIGN = 'all_files_foreign'
SIGNAL_SCOPE_MISMATCH_CONFIRMED = 'scope_mismatch_confirmed'


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
# Path containment
# ---------------------------------------------------------------------------


def _path_entries(value: Any) -> list[str]:
    """Return the usable path entries in *value* — non-empty strings only.

    A bare string is NOT treated as a one-element list: ``files`` is declared
    as a list, and silently splitting or wrapping a malformed value would
    invent evidence the task never carried.

    Junk entries (empty strings, ``None``, ints) are DROPPED rather than kept,
    because ``is_cross_repo_task`` reads a non-absolute entry as "local" and
    would let one piece of junk veto an otherwise-unanimous foreign verdict.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        return []
    return [entry for entry in value if isinstance(entry, str) and entry]


def _all_entries_foreign(entries: list[str], root: Path) -> bool:
    """True iff every entry is an ABSOLUTE path resolving outside *root*.

    Delegates to ``merge_gates.is_cross_repo_task`` so dispatch time and merge
    time share ONE definition of "foreign" rather than forking it.

    ``metadata=None`` is load-bearing, not incidental:

    * it isolates the pure path-containment leg, so the ``cross_repo`` marker
      is scored exactly ONCE — in leg (A), where the owner-naming lives — and
      is not double-counted here;
    * with metadata passed, that function's ``if not plan_files: return False``
      early return (merge_gates.py:997) precedes its marker check, so a MARKED
      task with no declared files would silently classify False.  Leg (A)
      reading the marker directly is immune to that ordering.

    Imported function-locally (mirroring workflow.py's call site) so this
    module stays import-light at module scope — it runs on every dispatch.
    """
    from orchestrator.merge_gates import is_cross_repo_task  # noqa: PLC0415

    return is_cross_repo_task(entries, root, None)


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

    Leg (B) — every declared ``metadata.files`` entry is an ABSOLUTE path
    resolving OUTSIDE ``project_root``.  The reify-5638 shape, and the only
    one the orchestrator can judge unaided.  Conservative by ``is_cross_repo_
    task``'s documented contract: an empty list, any relative entry, or any
    entry resolving in-tree yields no block, so a still-undelivered NEW local
    file is never mistaken for a foreign deliverable.

    Leg (C) — a ``possible_scope_mismatch`` stamp whose ``matched_paths`` are
    CONFIRMED foreign by the same containment test leg (B) uses.  The
    confirmation requirement is the whole point: the stamp is written by the
    fused-memory prose matcher (``'source': 'prose'``) and is documented to
    over-fire — task 3120 landed a right-boundary fix for exactly that and
    records a KNOWN FAIL-OPEN residue.  Blocking on unconfirmed prose would
    convert a false-positive advisory into a stalled task PLUS a spurious L1,
    strictly worse than today.  Confirmation makes leg (C) a genuine second
    evidence source rather than a restatement of the heuristic.

    Degenerate metadata — unreadable (non-dict, or a string that fails to parse
    or decodes to a non-dict) — yields SKIP and a WARNING.  SKIP means "no
    evidence readable", NEVER "verified clean": a silent ALLOW here would wave
    a task through with no trace that its markers could not be read, which is
    precisely the no-silent-fail-soft failure this gate exists to prevent.

    ``project_root`` may be a ``Path`` or a ``str``; it is coerced here.

    Never raises for any metadata shape; the caller (``_run_cross_repo_gate``)
    additionally fails CLOSED should that ever prove wrong.
    """
    root = Path(project_root)
    meta = _extract_metadata(task)
    if meta is None:
        raw = task.get('metadata')
        logger.warning(
            'cross_repo_gate: task %s metadata is not a readable dict '
            '(type=%s, repr=%.200r) — classifying SKIP; markers, if any, were '
            'NOT read, so this is not a clean bill of health',
            task.get('id'), type(raw).__name__, raw,
        )
        return CrossRepoVerdict(
            verdict=SKIP,
            owner_project=None,
            signals=(),
            foreign_paths=(),
            reason=(
                f'task metadata is not a readable dict '
                f'(type={type(raw).__name__}) — no evidence could be read'
            ),
        )

    signals: list[str] = []
    reasons: list[str] = []
    foreign_paths: tuple[str, ...] = ()

    # ── Leg (A): the metadata.cross_repo marker ──────────────────────────
    marker = meta.get('cross_repo')
    if marker:
        signals.append(SIGNAL_MARKER)
        reasons.append(
            f'metadata.cross_repo is set ({marker!r}) — the submit path classified '
            f'this task\'s declared files as owned by another project'
        )

    # ── Leg (B): every declared file resolves outside project_root ───────
    entries = _path_entries(meta.get('files'))
    if entries and _all_entries_foreign(entries, root):
        signals.append(SIGNAL_ALL_FILES_FOREIGN)
        foreign_paths = tuple(entries)
        reasons.append(
            f'all {len(entries)} declared metadata.files entries are absolute paths '
            f'resolving outside project_root ({root})'
        )

    # ── Leg (C): a CONTAINMENT-CONFIRMED possible_scope_mismatch stamp ───
    stamp = meta.get('possible_scope_mismatch')
    if isinstance(stamp, dict):
        matched = _path_entries(stamp.get('matched_paths'))
        if matched and _all_entries_foreign(matched, root):
            signals.append(SIGNAL_SCOPE_MISMATCH_CONFIRMED)
            foreign_paths = foreign_paths + tuple(
                path for path in matched if path not in foreign_paths
            )
            reasons.append(
                f'metadata.possible_scope_mismatch (an advisory prose stamp) is '
                f'CONFIRMED by path containment: all {len(matched)} matched_paths '
                f'resolve outside project_root ({root})'
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
        foreign_paths=foreign_paths,
        reason='; '.join(reasons),
    )
