"""Multi-project path-scope guard.

Generalises the original task-1088 dark-factory-only guard (its back-compat
re-export shim module was retired in task 2208 / PRD D2) to a per-project
prefix registry built from configured ``known_project_roots``.

A candidate is rejected when its title / description / details / files
cite a path prefix owned by a project other than the one being filed
into.  "Cite" is two-sided (task 3120): the prefix must sit at a left word
boundary AND be followed by something that looks like a path segment, so a
bare MENTION ("see ``fused-memory/``") or an English slash-construction
("not a ``backend/``timeout error") is not a hit.  The verdict's
``suggested_project`` field carries the owning
project_id of the first mismatched prefix so the caller can resubmit (or
the LLM can re-route under the multi-project Stage 2 prompt).

The guard runs on TWO signals with different certainties.  DECLARED files
(``metadata.files`` &c.) are classified exactly, via
:meth:`ProjectPrefixRegistry.project_for_path`; PROSE is classified
heuristically, via :func:`find_paths`.  Because prose cannot distinguish
"modifies X" from "merely cites X", the prose branch is gated on the declared
signal (task 3106).

That yields THREE outcomes at the interceptor's ``submit_task`` seam
(``TaskInterceptor._path_guard_or_skip``), in evaluation order — recorded
here so a future reader need not re-derive the taxonomy from the call site:

1. FILES-CERTAIN REJECT (:func:`check_files_for_scope`, task 2206) — a
   DECLARED file is owned by another project.  Hard reject, no LLM
   adjudication: the owner is either known and different, or it isn't.
2. CROSS-REPO ALLOW-AND-TAG (:func:`all_files_foreign_owner`, task 3004) —
   EVERY owned declared file belongs to ONE other REGISTERED project, and
   the filer is registered too.  A legitimate cross-repo deliverable: allowed
   and tagged (``metadata.cross_repo``), not rejected.
3. PROSE ADVISORY (:func:`check_text_for_scope`) — a heuristic hit in
   title/description/details with no files-level mismatch.  NEVER rejects:
   the task is created, stamped ``metadata.possible_scope_mismatch``, and a
   non-blocking advisory escalation fires.  ITSELF GATED on attribution —
   :func:`local_attesting_signals` (task 3106) asks whether the DECLARED
   deliverables (``metadata.files`` ∪ ``files_to_modify`` ∪ ``modules``)
   attest work in the FILING project, i.e. at least one is owned by it and
   NONE is owned by another; when they do, the interceptor suppresses BOTH
   the stamp and the escalation, logging the decision at INFO instead.  With
   no declared deliverable, with only unowned ones, or with a MIXED
   declaration, the advisory fires unchanged.

(1) and (2) partition the DECLARED signal; (3) applies only once neither
fired.  (2) and (3)'s suppression are logical complements — one needs every
owned file foreign, the other needs at least one local and none foreign — so
they can never both apply.

THIS IS THE CANONICAL STATEMENT of the taxonomy: the interceptor's
``_path_guard_or_skip`` and the individual functions below cross-reference
it by outcome number rather than restating it, so there is one place to edit
when the ordering or the attribution inputs change.

Wired into :class:`fused_memory.middleware.task_interceptor.TaskInterceptor`
at the ``submit_task`` entry point.  Path-guard
rejections also fire a ``scope_violation`` escalation via
:class:`fused_memory.middleware.scope_violation_escalator.ScopeViolationEscalator`
when one is configured.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from fused_memory.middleware.project_prefix_registry import ProjectPrefixRegistry

if TYPE_CHECKING:
    from fused_memory.middleware.task_curator import CandidateTask


# ---------------------------------------------------------------------------
# Pattern cache
# ---------------------------------------------------------------------------

_PATTERN_CACHE: dict[tuple[str, ...], re.Pattern[str]] = {}

# One path segment: the chars that may legally appear in a single path
# component (no '/').  Used only inside the right-context lookahead.
_SEG: str = r'[A-Za-z0-9_.\-]+'

# Right-context assertion (task 3120): the token after '<prefix>/' must
# look like a path segment — either another '/' follows, or it carries a
# file extension.  NOTE: the interval quantifier is written {{1,6}} here
# because this is an f-string; braces in the RENDERED value are not
# re-processed when it is interpolated into _build_pattern below.
_RIGHT_CONTEXT: str = rf'(?={_SEG}/|[A-Za-z0-9_\-]*\.[A-Za-z0-9]{{1,6}}(?![A-Za-z0-9]))'


def _build_pattern(prefixes: tuple[str, ...]) -> re.Pattern[str]:
    """Build (and cache) a compiled regex for the given prefix tuple.

    Each prefix is anchored on BOTH sides: a left word boundary, and a right
    context that must look like a path segment.

    LEFT boundary — start-of-string OR a character that is NOT
    ``[A-Za-z0-9_\\-/.]``.  This means a prefix matches ONLY as a *leading*
    path component — at start-of-string or after whitespace, punctuation,
    quotes, colons, etc. — but NOT when preceded by a path separator (``/``)
    or a dot (``.``).

    Excluding ``/`` and ``.`` from the boundary class ensures that mid-path
    occurrences like ``vendor/corpus/expr.txt`` or ``x.corpus/foo`` do NOT
    match the bare prefix ``corpus/``, eliminating the false-positive routing
    of tasks whose text merely passes *through* a known-project directory.

    Note: ``\\-`` is kept escaped (literal hyphen, no range); ``/`` and ``.``
    are plain literals inside the class.

    RIGHT boundary (task 3120) — the token following ``<prefix>/`` must look
    like a path segment: either another ``/`` follows (``crates/reify/src.rs``)
    or the token carries a file extension (``gui/package.json``).  Without
    this, any English slash-construction whose left half happens to name a
    registered top-level directory lexed as a path — "not a ``backend/``
    timeout error", "``tools/``call", "``archive/``pause".

    A bare trailing ``<prefix>/`` at end-of-token deliberately does NOT match,
    because that is the bare-MENTION class ("see ``fused-memory/``", "the
    'corpus/' dir") and admitting it would re-admit almost all of the noise
    this assertion exists to remove.  For the ``submit_task`` path this loses
    no real protection: the interceptor classifies ``metadata.files`` through
    the FILES-certain check (:func:`check_files_for_scope` ->
    ``registry.project_for_path``), which still hard-rejects a genuinely
    DECLARED foreign file and is untouched by this change.

    That backstop is specific to that path, and the guarantee is NOT blanket:
    :func:`check_candidate_for_scope` folds ``files_to_modify`` into this same
    prose scan with no certain-check fallback, so a declared *directory* entry
    (``files_to_modify=['fused-memory/src']``) now goes undetected there —
    measured: that call returns ``ok`` where :func:`check_files_for_scope`
    returns ``rejection``.  Harmless today because that function has no
    production caller (the interceptor imports only ``check_files_for_scope``
    and ``check_text_for_scope``), but any future caller must route concrete
    file entries through :func:`check_files_for_scope` rather than rely on
    this heuristic scan.

    KNOWN FAIL-OPEN residue, deliberately not closed (each is pinned by a
    test so the gap stays visible rather than implied covered):

    - extensionless right context — ``<prefix>/<dir>`` and extensionless
      filenames (``fused-memory/src``, ``crates/reify-eval``,
      ``crates/README``) satisfy neither alternative.  This is the LARGEST
      miss class, since bare directory references are a common prose
      spelling; it is accepted anyway because admitting it would mean
      accepting any ``<word>/<word>``, which is exactly the English-
      punctuation shape this assertion exists to reject;
    - extensions longer than 6 chars (the ``{1,6}`` cap) — ``docs/.gitignore``,
      ``crates/x.markdown``, ``crates/x.properties`` fail the extension
      alternative.  The mechanism is the LENGTH cap, NOT the leading dot: a
      short-extension dotfile such as ``docs/.env`` still matches;
    - glob spellings — the metacharacter in ``plans/*.md`` is in neither the
      segment class nor the extension class.

    All three are MISSED detections, never false ones — the same conservative
    direction ``project_prefix_registry`` already takes for
    ``../other-repo/x.py`` and ``~user/...``.  Widening the character classes
    to admit them would start re-admitting the punctuation shapes above.

    Cache note: ``_PATTERN_CACHE`` is keyed by the prefix tuple; the boundary
    class and the right-context assertion are both compile-time constants
    (not runtime variables), so no cache invalidation or versioning is needed
    when the pattern template changes — only new prefixes cause a new compile.
    """
    if prefixes not in _PATTERN_CACHE:
        alts = '|'.join(re.escape(p) for p in prefixes)
        pattern = re.compile(rf'(?:^|(?<=[^A-Za-z0-9_\-/.]))({alts}){_RIGHT_CONTEXT}')
        _PATTERN_CACHE[prefixes] = pattern
    return _PATTERN_CACHE[prefixes]


def find_paths(text: str, prefixes: tuple[str, ...]) -> list[str]:
    """Scan *text* for any of *prefixes* and return ordered, deduplicated matches.

    Empty *text* or empty *prefixes* short-circuit to ``[]``.
    """
    if not text or not prefixes:
        return []
    pattern = _build_pattern(prefixes)
    seen: set[str] = set()
    result: list[str] = []
    for match in pattern.finditer(text):
        prefix = match.group(1)
        if prefix not in seen:
            seen.add(prefix)
            result.append(prefix)
    return result


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

# Kept stable for one merge cycle so existing callers branching on
# ``error_type='DarkFactoryPathScopeViolation'`` continue to work.  Renaming
# to a neutral name is a tracked followup.
_ERROR_TYPE: str = 'DarkFactoryPathScopeViolation'


@dataclass(frozen=True)
class PathGuardVerdict:
    """Outcome of a multi-project path-scope check.

    Mirrors the original dark-factory-only ``PathGuardVerdict`` shape so
    existing test fixtures keep working; adds ``suggested_project`` to
    carry the owning project_id of the first mismatched prefix.

    Fields:
        outcome: ``'ok'`` or ``'rejection'``.
        project_id: The project_id that was checked (the wrong one on rejection).
        matched_paths: Tuple of path prefixes found in the candidate that
            belonged to a project other than ``project_id``.
        suggested_project: project_id whose prefix matched first; ``None``
            when the matched prefix has no owner in the registry, or when
            multiple owners were matched and we cannot pick one.
        error_type: Stable error-type string for callers to branch on.
    """

    outcome: Literal['ok', 'rejection']
    project_id: str = ''
    matched_paths: tuple[str, ...] = field(default_factory=tuple)
    suggested_project: str | None = None
    error_type: str = _ERROR_TYPE

    @property
    def is_rejection(self) -> bool:
        return self.outcome == 'rejection'

    def to_error_dict(self) -> dict:
        """Return a structured MCP error dict, or ``{}`` on ok.

        The error message names the matched paths, the wrong project_id,
        and the suggested target project (when known) so the caller can
        re-submit without a second round-trip to figure out where the
        task belongs.
        """
        if not self.is_rejection:
            return {}
        paths_str = ', '.join(self.matched_paths)
        if self.suggested_project:
            tail = f'Resubmit to the {self.suggested_project} project instead.'
        else:
            tail = (
                'No single suggested target — the matched prefixes belong to '
                'multiple projects or none.  Inspect matched_paths and route '
                'manually.'
            )
        return {
            'error': (
                f'{self.error_type}: task references paths owned by another '
                f'project ({paths_str}) but was filed under project '
                f'{self.project_id!r}. {tail}'
            ),
            'error_type': self.error_type,
            'project_id': self.project_id,
            'matched_paths': list(self.matched_paths),
            'suggested_project': self.suggested_project,
        }


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------


def _aggregate_owner_mismatches(
    items: list[str],
    project_id: str,
    owner_of: Callable[[str], str | None],
) -> tuple[list[str], str | None]:
    """Classify *items* via *owner_of* and collect those owned by another project.

    Shared aggregation core for :func:`_resolve_mismatches` (matched
    prefixes from the regex-over-prose heuristic, classified via
    :meth:`ProjectPrefixRegistry.project_for_prefix`) and
    :func:`check_files_for_scope` (concrete file paths, classified via
    :meth:`ProjectPrefixRegistry.project_for_path`).  The two callers differ
    only in *what* ``owner_of`` resolves, not in how mismatches are
    aggregated into a suggested target — factored out here (task 2206
    review) so that logic can't drift out of sync between them.

    Returns ``(mismatched_items, suggested_project)``.  An item is
    mismatched when ``owner_of(item)`` returns a project_id other than
    *project_id*; items with no registered owner (``owner_of`` returns
    ``None``) are dropped from the result — an unclassifiable item is never
    grounds for rejection, keeping the guard conservative.
    ``suggested_project`` is the single shared owner of every mismatch, or
    ``None`` when mismatches span more than one project.
    """
    mismatched: list[str] = []
    owners: set[str] = set()
    first_owner: str | None = None
    for item in items:
        owner = owner_of(item)
        if owner is None or owner == project_id:
            continue
        mismatched.append(item)
        if first_owner is None:
            first_owner = owner
        owners.add(owner)
    suggested = first_owner if len(owners) == 1 else None
    return mismatched, suggested


def _resolve_mismatches(
    matched: list[str], project_id: str, registry: ProjectPrefixRegistry,
) -> tuple[list[str], str | None]:
    """Filter *matched* prefixes to those owned by some other project.

    Returns ``(mismatched_prefixes, suggested_project)``.  ``suggested_project``
    is the owner of the first mismatch when all mismatches share an owner;
    ``None`` when they span multiple projects.  Prefixes with no registered
    owner are dropped from the rejection (the registry could not classify
    them — ignoring them keeps the guard conservative).

    Thin wrapper around :func:`_aggregate_owner_mismatches` bound to the
    prefix-string lookup :meth:`ProjectPrefixRegistry.project_for_prefix`.
    """
    return _aggregate_owner_mismatches(matched, project_id, registry.project_for_prefix)


def check_candidate_for_scope(
    candidate: CandidateTask,
    project_id: str,
    registry: ProjectPrefixRegistry,
) -> PathGuardVerdict:
    """Reject *candidate* when it cites paths owned by another project.

    Scans ``title``, ``description``, ``details``, and
    ``files_to_modify``.  Returns ``ok`` when the registry is empty, when
    no prefixes match, or when every match is owned by ``project_id``.

    CAVEAT (task 3120): ``files_to_modify`` entries are folded into the same
    heuristic prose scan as the free text, so they inherit :func:`find_paths`'
    right-boundary requirement — a declared *directory* entry such as
    ``['fused-memory/src']`` is NOT detected here, and unlike the
    interceptor's ``submit_task`` path there is no FILES-certain backstop
    behind this function.  A caller that needs concrete file entries
    classified with certainty must call :func:`check_files_for_scope` on them
    as well; this function alone is a heuristic.
    """
    if not registry:
        return PathGuardVerdict(outcome='ok', project_id=project_id)

    parts: list[str] = [
        candidate.title or '',
        candidate.description or '',
        candidate.details or '',
    ]
    parts.extend(candidate.files_to_modify or [])
    combined = '\n'.join(parts)

    matched = find_paths(combined, registry.all_prefixes())
    mismatched, suggested = _resolve_mismatches(matched, project_id, registry)
    if mismatched:
        return PathGuardVerdict(
            outcome='rejection',
            project_id=project_id,
            matched_paths=tuple(mismatched),
            suggested_project=suggested,
        )
    return PathGuardVerdict(outcome='ok', project_id=project_id)


def check_files_for_scope(
    files: list[str] | None,
    project_id: str,
    registry: ProjectPrefixRegistry,
) -> PathGuardVerdict:
    """Reject when any of *files* is owned by a project other than *project_id*.

    CERTAIN counterpart to :func:`check_candidate_for_scope` /
    :func:`check_text_for_scope`: classifies each concrete file path via
    :meth:`ProjectPrefixRegistry.project_for_path` instead of the heuristic
    regex-over-prose :func:`find_paths`.  That lookup resolves ABSOLUTE
    entries (and ``~/``-prefixed ones) against the registry's known project
    roots and RELATIVE ones by exact leading-path-component match, so an
    absolute foreign path is classified identically to its repo-relative
    spelling (task 3109).  Spellings resolvable only against a base directory
    the registry is not given — ``'../other-repo/x.py'``, ``'~user/...'`` —
    remain UNOWNED and so fail open here, exactly as they did pre-3109; see
    :meth:`ProjectPrefixRegistry._owner_for_absolute_path` for the pinned
    residue.  Used by the interceptor's FILES-certain check (task 2206) to
    hard-reject a submission whose ``metadata.files`` name a path under a
    KNOWN other project's tree — no LLM adjudication, since an exact owner
    lookup leaves nothing to adjudicate.

    Returns ``ok`` when the registry is empty/falsy, *files* is empty/falsy,
    or every file is either unowned or owned by *project_id*.  On a
    mismatch, ``matched_paths`` carries the offending file paths (not
    prefixes) and ``suggested_project`` is the single owner when all
    mismatches share one, else ``None``.
    """
    if not registry or not files:
        return PathGuardVerdict(outcome='ok', project_id=project_id)

    mismatched: list[str] = []
    owners: set[str] = set()
    first_owner: str | None = None
    for f in files:
        owner = registry.project_for_path(f)
        if owner is None or owner == project_id:
            continue
        mismatched.append(f)
        if first_owner is None:
            first_owner = owner
        owners.add(owner)

    if not mismatched:
        return PathGuardVerdict(outcome='ok', project_id=project_id)

    suggested = first_owner if len(owners) == 1 else None
    return PathGuardVerdict(
        outcome='rejection',
        project_id=project_id,
        matched_paths=tuple(mismatched),
        suggested_project=suggested,
    )


def all_files_foreign_owner(
    files: list[str] | None,
    project_id: str,
    registry: ProjectPrefixRegistry,
) -> str | None:
    """Return the single foreign owner iff EVERY owned file belongs to ONE other project.

    Recognises the cross-repo deliverable shape (task 3004 / reify-task 5308):
    a task filed under *project_id* whose ``metadata.files`` are ALL owned by
    one OTHER registered project, so its own branch is legitimately empty (the
    deliverable lands on that project's branch).  The interceptor consumes the
    returned owner to TAG the submission (``metadata.cross_repo`` +
    ``cross_repo_project``) and ALLOW it, instead of hard-rejecting via
    :func:`check_files_for_scope`.

    FILER MUST BE REGISTERED (task 3004 / esc-3004 NARROW decision): the
    allow-and-tag path fires ONLY when *project_id* is itself a registered
    project (``registry.is_known(project_id)``).  A cross-repo deliverable is a
    relationship between two KNOWN projects (e.g. reify → dark_factory), not an
    open door for any namespace to declare another project's files.  An
    UNREGISTERED filer whose files are all foreign therefore returns ``None``
    here and falls through to the :func:`check_files_for_scope` hard reject,
    preserving task-2206's anti-bypass guard.

    Distinct from :func:`check_files_for_scope`, which rejects on ANY foreign
    file: here the submission must be ENTIRELY foreign under a SINGLE owner,
    with NO locally-owned file, before it qualifies.  Reuses
    :meth:`ProjectPrefixRegistry.project_for_path` exactly as
    ``check_files_for_scope`` does — including its resolution of ABSOLUTE
    (and ``~/``-prefixed) entries against the registry's known project roots,
    so an all-absolute foreign file set is tagged just like its repo-relative
    spelling (task 3109), and its residual fail-open spellings
    (``'../other-repo/x.py'``, ``'~user/...'``), which stay unowned.  Files
    with no registered owner (``project_for_path`` returns ``None``) are
    neutral — they neither block nor establish cross-repo (conservative,
    matching :func:`_aggregate_owner_mismatches`).

    Returns ``None`` when the registry/*files* are empty, when the FILER
    (*project_id*) is not a registered project, when ANY file is owned by
    *project_id* (a genuine same-repo or mixed scope, left to
    ``check_files_for_scope``'s hard-reject), when no file is foreign, or when
    foreign files span more than one owner.
    """
    if not registry or not files:
        return None
    if not registry.is_known(project_id):
        # Unregistered filer → task-2206 anti-bypass preserved; a cross-repo
        # deliverable is a relationship between two REGISTERED projects, not a
        # blanket allowance for any namespace to declare foreign files.
        return None
    foreign_owners: set[str] = set()
    first_owner: str | None = None
    for f in files:
        owner = registry.project_for_path(f)
        if owner is None:
            continue  # unowned → neutral, never grounds for classification
        if owner == project_id:
            return None  # a locally-owned file → not a pure cross-repo deliverable
        if first_owner is None:
            first_owner = owner
        foreign_owners.add(owner)
    # len 0 (no foreign file) or >1 (spans multiple owners) → not cross-repo.
    if len(foreign_owners) != 1:
        return None
    return first_owner


def local_attesting_signals(
    signals: list[str] | None,
    project_id: str,
    registry: ProjectPrefixRegistry,
) -> list[str]:
    """Return the declared deliverable signals that ATTEST *project_id* work.

    Gate for the module docstring's outcome (3) — the prose advisory —
    returning the WITNESS rather than a bare bool so the caller can name the
    attesting entries in its log without reclassifying.  Truthiness IS the
    predicate: :func:`local_deliverable_attested` is ``bool()`` of this, so
    the two cannot drift and every signal is classified exactly once.

    *signals* are the submission's DECLARED deliverables — the union of
    ``metadata.files``, ``metadata.files_to_modify`` and ``metadata.modules``
    (task 3106).  Attestation requires BOTH:

    1. at least one signal OWNED by *project_id* (the positive half the
       module docstring describes), and
    2. NO signal owned by a DIFFERENT project.

    (2) exists because the caller's upstream ``check_files_for_scope`` sees a
    NARROWER list than this one: only ``files``, or (when ``files`` is
    absent) ``files_to_modify`` — never ``modules``, and never a
    ``files_to_modify`` entry shadowed by a present ``files`` key.  A foreign
    entry in that non-overlapping remainder was therefore never classified by
    the hard reject, so a MIXED declaration (local ``files`` + foreign
    ``modules``) must not buy silence here.  It is advisory-only: declining
    to attest leaves such a submission on the unchanged
    stamp-plus-advisory path, and never creates a new rejection.

    Signals are classified with the CERTAIN
    :meth:`ProjectPrefixRegistry.project_for_path` lookup, never the
    heuristic regex-over-prose :func:`find_paths`.  That matters: a declared
    DIRECTORY (``'fused-memory/src'``, a ``modules`` lock key) and an
    ABSOLUTE spelling both classify correctly here, where the prose
    scanner's right-boundary assertion (task 3120) deliberately misses them.
    It also means attribution inherits ``project_for_path``'s documented
    fail-open residue (``'../other-repo/x.py'``, ``'~user/...'``) in the SAFE
    direction: an unclassifiable signal simply fails to attest, so the
    advisory still fires.

    Entries with no registered owner are NEUTRAL — they neither attest nor
    veto, matching :func:`_aggregate_owner_mismatches`' conservative
    direction.  Declaring only ``README.md`` is no evidence of local work.

    Returns ``[]`` when the registry is empty/falsy or *signals* is
    empty/falsy.
    """
    if not registry or not signals:
        return []
    local: list[str] = []
    for signal in signals:
        owner = registry.project_for_path(signal)
        if owner is None:
            continue  # unowned → neutral
        if owner != project_id:
            return []  # foreign entry anywhere in the union → no attestation
        local.append(signal)
    return local


def local_deliverable_attested(
    signals: list[str] | None,
    project_id: str,
    registry: ProjectPrefixRegistry,
) -> bool:
    """Return True iff the declared deliverables attest *project_id* work.

    Pure predicate form of :func:`local_attesting_signals` (see it for the
    two conditions and why the second exists); the interceptor uses the
    witness-returning form so it can log what attested.

    POSITIVE-attribution counterpart to :func:`all_files_foreign_owner`'s
    negative one, and its exact logical complement — that function needs
    EVERY owned entry foreign under ONE owner, this needs at least one LOCAL
    and none foreign — so the module docstring's outcomes (2) and (3)-
    suppressed can never both fire.
    """
    return bool(local_attesting_signals(signals, project_id, registry))


def is_routing_override(routing_override_reason: str | None) -> bool:
    """Return True when *routing_override_reason* constitutes a deliberate bypass.

    A reason is considered present (and the path guards should be skipped) when
    it is a non-empty string after stripping leading/trailing whitespace.
    Empty strings, None, and whitespace-only strings all return False —
    preserving the default no-override semantics.

    Use only when sure the task belongs to the submitting project.
    If unsure, escalate rather than risking a mis-filed task.
    """
    return bool((routing_override_reason or '').strip())


def check_text_for_scope(
    text: str | None,
    project_id: str,
    registry: ProjectPrefixRegistry,
) -> PathGuardVerdict:
    """Reject *text* when it cites paths owned by another project.

    Prompt-only counterpart to :func:`check_candidate_for_scope` for the
    ``prompt``-only ``submit_task`` branch where no ``title`` is supplied
    (and therefore no ``CandidateTask`` is built).
    """
    if not registry:
        return PathGuardVerdict(outcome='ok', project_id=project_id)
    matched = find_paths(text or '', registry.all_prefixes())
    mismatched, suggested = _resolve_mismatches(matched, project_id, registry)
    if mismatched:
        return PathGuardVerdict(
            outcome='rejection',
            project_id=project_id,
            matched_paths=tuple(mismatched),
            suggested_project=suggested,
        )
    return PathGuardVerdict(outcome='ok', project_id=project_id)
