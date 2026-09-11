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

Exactly TWO legs can block; see :func:`classify_cross_repo`.  A third leg —
blocking on a containment-CONFIRMED ``possible_scope_mismatch`` stamp — was
implemented and then REMOVED during review, because no producer emits the shape
it matched; the removal is pinned by ``TestScopeMismatchStampIsNotABlockingLeg``
and explained in :func:`classify_cross_repo`'s docstring.  The stamp survives
here only as an owner-NAMING fallback (:func:`_resolve_owner`), which is reached
only once one of the two real legs has already fired.

KNOWN DUPLICATION: :func:`_extract_metadata` is the fourth hand-copy of the
dict-or-JSON-string metadata coercion (``Scheduler._normalize_task_metadata``,
``substrate_gate.carries_substrate_probe`` / ``extract_probe_set``,
``TaskInterceptor._extract_metadata_dict``).  Unifying them into one shared
dependency-light helper needs edits in packages this task does not hold, so it
is filed as a follow-up; until it lands, a wire-format change is a four-site
audit and this copy's tri-state contract (below) is the one that differs.
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
ALLOW = 'allow'   # metadata read fine, no cross-repo evidence: dispatch may proceed
SKIP = 'skip'     # metadata PRESENT but UNREADABLE — never "verified clean"

# Signal names recorded on a blocking verdict.
SIGNAL_MARKER = 'cross_repo_marker'
SIGNAL_ALL_FILES_FOREIGN = 'all_files_foreign'


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
    """Return ``task['metadata']`` as a dict, or None when PRESENT-but-unreadable.

    Applies the same JSON-string→dict coercion as
    ``substrate_gate.carries_substrate_probe`` / ``extract_probe_set`` (and
    ``Scheduler._normalize_task_metadata``) so both dispatch gates read task
    metadata through identical rules — a wire-format change cannot make one
    gate see a marker the other misses.  (See the module docstring's KNOWN
    DUPLICATION note: "identical rules" is currently maintained by hand.)

    Three-way, and the distinction is load-bearing for both callers:

    * ``{}`` when metadata is ABSENT or ``None`` — readable, declares nothing.
      This is the ordinary shape of most tasks, so it must be SILENT: it is not
      a defect, and treating it as one would warn on every metadata-free
      dispatch and admit every such task to the gate.
    * a dict — read fine, weigh whatever it carries.
    * ``None`` when metadata is PRESENT but cannot be read as a dict: a non-dict
      value, a string that fails to parse, or a string decoding to a non-dict.
      Callers must treat this as "no evidence readable", never "carries
      nothing" — that is the whole of the SKIP contract.

    Note this differs from ``Scheduler._normalize_task_metadata``, which
    collapses BOTH the absent and the unreadable case to ``{}`` (loudly, since
    task 3121).  The distinction is kept here because the gate's SKIP verdict
    is defined by it.
    """
    raw = task.get('metadata')

    if raw is None:
        return {}

    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed

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

    True when:

    * the task's metadata is PRESENT but UNREADABLE — so the gate runs, warns
      and returns SKIP (see below); or
    * the readable metadata carries the ``'cross_repo'`` KEY, regardless of its
      value; or
    * the readable metadata carries a non-empty ``'files'`` value.

    KEY-presence, deliberately, NOT value validity.  This is the lesson task
    2121 already paid for on the substrate gate: gating dispatch on a stricter
    predicate (one requiring a well-formed marker) would let a MALFORMED marker
    skip the gate entirely, when the whole point is that it should enter the
    gate and fail CLOSED there.  The observed markers are caller-authored and
    untyped (``true``, ``"dark-factory"``,
    ``"dark-factory:orchestrator/src/orchestrator/offline_lane.py"``, often
    with no ``cross_repo_project`` companion), so a strict predicate would miss
    every real instance.  See ``substrate_gate.carries_substrate_probe``.

    UNREADABLE metadata admits the task for exactly the same reason a malformed
    marker does.  Returning False there would be the sharper version of the same
    defect — the gate would silently skip a task whose markers it could not
    read, which is precisely the outcome :data:`SKIP` exists to make loud.  So
    the tri-state of :func:`_extract_metadata` is honored end to end: absent
    metadata (``{}``) is the ordinary shape and stays out of the gate silently;
    present-but-unreadable enters it.

    ``'possible_scope_mismatch'`` is NOT a signal here.  It cannot contribute to
    a block (see :func:`classify_cross_repo` — there is no leg for it), so
    admitting a task on it alone would guarantee an ALLOW and buy nothing.  The
    stamp already fires its own advisory escalation at submit time, and it still
    NAMES an owner (:func:`_resolve_owner`) once a real leg has fired.

    An EMPTY ``files`` list is NOT a signal: it carries no path evidence, so
    there is nothing for the gate to weigh (and ``is_cross_repo_task`` returns
    False for it by design).  A marked task with empty ``files`` still enters
    the gate via the ``cross_repo`` key.

    Never raises — it runs on every dispatch and must not take down a slot.
    """
    meta = _extract_metadata(task)
    if meta is None:
        # Present but unreadable — admit it so the gate reports a loud SKIP.
        return True

    if 'cross_repo' in meta:
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
       suggestion; last because it is advisory.  This is the stamp's ONLY
       remaining role: it cannot itself block (see :func:`classify_cross_repo`),
       so it is read here only after a real leg has already fired, where an
       advisory guess at the owner's NAME is strictly better than the
       "unresolved" the L1 would otherwise have to say.

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

    There is deliberately NO leg for ``possible_scope_mismatch``.  One was
    implemented (block when the stamp's ``matched_paths`` are containment-
    confirmed foreign) and removed during review, because NO PRODUCER EMITS
    THAT SHAPE:

    * the stamp's only writer is ``TaskInterceptor._attach_possible_scope_
      mismatch``, called from exactly one site with the PROSE verdict;
    * that verdict's ``matched_paths`` come from ``path_scope_guard.find_paths``
      over registry PREFIXES, so they are bare relative directory names with a
      trailing slash (``'orchestrator/'``, ``'fused_memory/'``) — never file
      paths;
    * ``is_cross_repo_task``'s first per-entry test is ``if not os.path.isabs
      (entry): return False``, so a relative prefix short-circuits the
      containment check to False, always;
    * the FILES-certain guard variant, which DOES put absolute file paths in
      ``matched_paths``, hard-REJECTS the submission instead of stamping it, so
      that shape never reaches a dispatched task either.

    And the confirmation requirement could not simply be relaxed: the stamp is a
    prose heuristic documented to over-fire (task 3120 landed a right-boundary
    fix for exactly that, with a KNOWN FAIL-OPEN residue).  Blocking on it
    unconfirmed would convert a false-positive advisory into a stalled task PLUS
    a spurious L1 — strictly worse than today.  With confirmation unattainable
    and relaxation unacceptable, the honest resolution is no leg at all: a stamp
    is advisory, it already fires its own escalation at submit time, and it
    still NAMES an owner here via :func:`_resolve_owner`.  Pinned by
    ``TestScopeMismatchStampIsNotABlockingLeg``, which builds the stamp in the
    real producer shape so a naive re-add cannot go green on a synthetic one.

    Metadata that is PRESENT but unreadable (a non-dict, or a string that fails
    to parse or decodes to a non-dict) yields SKIP and a WARNING.  SKIP means
    "no evidence readable", NEVER "verified clean": a silent ALLOW here would
    wave a task through with no trace that its markers could not be read, which
    is precisely the no-silent-fail-soft failure this gate exists to prevent.
    ABSENT metadata is a different thing entirely — it is readable and simply
    declares nothing, so it is a silent ALLOW (see :func:`_extract_metadata`).

    In production the scheduler normalizes ``task['metadata']`` to a dict at the
    wire boundary (``Scheduler._normalize_task_metadata``, itself loud since
    task 3121), so SKIP is defence in depth for callers that bypass that
    normalization rather than the primary report of an unreadable blob.  It is
    kept so the gate does not have to TRUST an upstream normalizer to honor its
    own no-silent-fail-soft guarantee.

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

    # NB: metadata.possible_scope_mismatch is deliberately NOT a third leg —
    # see this function's docstring.  Its only role is naming the owner below.

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
