"""Every function recon uses to decide whether two findings are the same flag.

A LEAF: it imports the stdlib, ``fused_memory.utils.validation`` and
``fused_memory.reconciliation.flag_shape`` only, so flag dedup, the recon
report server, scope freshness and the sweep guards all import it downward.

Persisted formats
-----------------
Four outputs are stored, so their bytes are frozen.
``tests/reconciliation/test_flag_identity.py`` pins each one as literals.

| Output | Stored by | A change requires |
|---|---|---|
| :func:`content_fingerprint` | ``recon_ledger`` ``stage1_flag_marker`` rows (``task_id`` = ``fp:…``, ``flag_type`` = :data:`CONTENT_FP_FLAG_TYPE`); ``scope_freshness`` snapshot metadata ``subject_description_fingerprint`` | a ledger re-key migration and a scope-snapshot re-stamp (else every snapshot reads changed for one cycle) |
| :func:`description_digest` | ``recon_report_state.db`` ``entry_json`` ``deschash_to_finding`` | re-keying persisted runs, or waiting out their TTL |
| :func:`compute_marker_signature` (lexicographic union, ``'<project>:<id>'`` for foreign citations) | ``recon_ledger`` ``stage1_flag_marker`` identity | a ledger re-key migration (else recurrence history is orphaned) |
| :func:`recon_signature_task_id` (numeric-aware sort, keeps ``fp:``) | ``recon_report_state.db`` ``signature_to_finding``, ``sig_anchor_slice`` | re-keying persisted runs, or waiting out their TTL |

The two task-id orders differ on purpose: each is a stored form, so neither
may change to match the other.

Invariants: every function is pure, and total over the LLM-authored flags and
task-id values it reads; malformed input yields an empty subject or a
``None`` signature, never an exception.  The marker-key predicates take the
``str`` key a signature emitted.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Final

from fused_memory.reconciliation.flag_shape import collect_str_values
from fused_memory.utils.validation import validate_project_id

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Descriptions
# --------------------------------------------------------------------------- #

#: Sentinel flag_type of a content-fingerprint signature whose flag carries no
#: flag_type.  Stored in ledger identity columns.
CONTENT_FP_FLAG_TYPE: Final = '__content_fp__'

#: Prefix of every :func:`content_fingerprint` key.
CONTENT_FP_PREFIX: Final = 'fp:'

#: Hex characters of the SHA-256 digest a fingerprint keeps (128 bits).
CONTENT_FP_HEXLEN: Final = 32

#: ONLY canonical fingerprint keys: the prefix and exactly CONTENT_FP_HEXLEN
#: lowercase hex digits.  ``hexdigest`` never emits uppercase, so accepting it
#: would widen the accept-set beyond what the emitter can produce.
_CONTENT_FP_RE: re.Pattern[str] = re.compile(
    rf'{re.escape(CONTENT_FP_PREFIX)}[0-9a-f]{{{CONTENT_FP_HEXLEN}}}\Z'
)


def normalize_description(text: str) -> str:
    """Collapse every whitespace run to one space, strip, and casefold."""
    return ' '.join(text.split()).casefold()


def description_digest(text: str) -> str:
    """SHA-256 hex digest (64 chars) of :func:`normalize_description` of *text*."""
    return hashlib.sha256(normalize_description(text).encode('utf-8')).hexdigest()


def content_fingerprint(text: str) -> str:
    """:data:`CONTENT_FP_PREFIX` plus the first :data:`CONTENT_FP_HEXLEN` chars of the digest.

    Deterministic across processes and PYTHONHASHSEED (unlike builtin
    ``hash()``), and always accepted by :func:`is_valid_marker_task_id`.
    """
    return CONTENT_FP_PREFIX + description_digest(text)[:CONTENT_FP_HEXLEN]


# --------------------------------------------------------------------------- #
# The flag subject: a finding's task_id value
# --------------------------------------------------------------------------- #


def signature_field(value: object) -> str | None:
    """Coerce a signature field (task_id or flag_type) to ``str``; ``None`` stays ``None``.

    An integer task_id is common in LLM output, and ``(1976, ft)`` must be
    the same signature as ``('1976', ft)``.  ``None`` is never coerced to
    ``'None'``, so an absent field stays absent.
    """
    return None if value is None else str(value)


@dataclass(frozen=True)
class FlagSubject:
    """A finding's ``task_id`` value, as carried and as its component task ids."""

    raw: str | None
    components: tuple[str, ...]

    @property
    def is_composite(self) -> bool:
        return len(self.components) >= 2

    def covered_by(self, others: Iterable[str]) -> frozenset[str]:
        """The components that *others* also holds."""
        return frozenset(self.components).intersection(others)


def parse_flag_subject(raw: object) -> FlagSubject:
    """Parse one finding ``task_id`` value.

    ``raw`` is :func:`signature_field` of the value, verbatim.  Components:

    - Only a ``str``, or an ``int`` straight off a task dict, has any
      (``3105`` -> ``('3105',)``).  ``None``, a ``bool`` (an ``int`` subclass,
      never a task id), a ``float``, ``bytes``, a list or a dict yields ``()``,
      so a malformed value is never stringified into a lookup.
    - Split on ``','``, the composite shape (``'3105,4223'``); each component
      stripped, blank components dropped, so ``','`` yields ``()``.
    - Deduped, keeping each component's FIRST position.
    - No usability judgement: ``'0'`` and ``'-5'`` come back as components.
      Whether a component could name a real task is the consumer's call.
    """
    components: tuple[str, ...] = ()
    if isinstance(raw, (str, int)) and not isinstance(raw, bool):
        components = tuple(dict.fromkeys(p.strip() for p in str(raw).split(',') if p.strip()))
    return FlagSubject(raw=signature_field(raw), components=components)


def _task_id_part_sort_key(part: str) -> tuple[int, Any]:
    try:
        return (0, int(part))
    except ValueError:
        return (1, part)


def recon_signature_task_id(subject: FlagSubject) -> str | None:
    """The recon report's stored task_id: components sorted, comma-joined.

    Integer components sort first, by value; the rest follow lexically, so
    ``'12,9'`` -> ``'9,12'`` and an ``fp:`` key is kept as it is.  A subject
    with no components keeps its :attr:`FlagSubject.raw` (``''`` stays ``''``,
    ``None`` stays ``None``).
    """
    if not subject.components:
        return subject.raw
    return ','.join(sorted(subject.components, key=_task_id_part_sort_key))


# --------------------------------------------------------------------------- #
# Marker keys
# --------------------------------------------------------------------------- #


def is_valid_marker_task_id(tid: str) -> bool:
    """Return True iff *tid* is a valid stage1_flag_marker key.

    Accepts:
    - A canonical content-fingerprint key, the only shape
      :func:`content_fingerprint` emits: ``'fp:'`` followed by exactly
      :data:`CONTENT_FP_HEXLEN` (32) lowercase hex digits, e.g.
      ``'fp:9216e85ac497b68d93043b64684eb049'``.
    - A bare non-negative integer string (e.g. ``'42'``, ``'0'``).
    - A comma-joined list of non-negative integers (e.g. ``'12,15'``), the
      shape :func:`compute_flag_signature`'s ``cited_tasks`` union produces
      for multi-task findings.
    - Any of those components written ``'<project>:<digits>'`` (e.g.
      ``'12,reify:650'``), the foreign cited id :func:`compute_marker_signature`
      emits; the project half must pass ``validate_project_id``.

    Rejects:
    - Falsy / empty input.
    - Malformed fp: forms: ``'fp:'`` (no hex), too-short or too-long hex bodies,
      uppercase hex, non-hex characters in the body.
    - Any component that is neither a non-negative integer nor
      ``'<project>:<digits>'`` after strip.
    - Trailing/leading commas that yield empty components (e.g. ``'12,'``).

    The check lives here, not beside
    ``shared/src/shared/task_submit.py::looks_like_task_id``, because every
    shape it accepts beyond a bare id (comma-joined keys, ``fp:`` keys,
    ``'<project>:<digits>'`` components) belongs to the marker format alone.
    """
    if not tid:
        return False
    if _CONTENT_FP_RE.fullmatch(tid):
        return True
    return all(_is_valid_marker_task_component(part.strip()) for part in tid.split(','))


def _is_valid_marker_task_component(component: str) -> bool:
    project, separator, task_id = component.partition(':')
    if not separator:
        return component.isdigit()
    return task_id.isdigit() and validate_project_id(project) is None


def is_content_fingerprint_task_id(tid: str) -> bool:
    """Return True iff *tid* is a canonical content-fingerprint marker key.

    The fp:-ONLY gate: unlike :func:`is_valid_marker_task_id`, it rejects
    every other shape, including otherwise-valid numeric and comma-joined
    marker keys.  ``task_knowledge_sync`` (task-2047 Gap 2) scopes the
    cross-cycle fingerprint-marker sweep with it, leaving numeric markers on
    the 14-day age-only GC.
    """
    return bool(tid) and bool(_CONTENT_FP_RE.fullmatch(tid))


# --------------------------------------------------------------------------- #
# Signatures
# --------------------------------------------------------------------------- #


def compute_flag_signature(flag: dict[str, Any]) -> tuple[str, str] | None:
    """Return a (task_id_str, flag_type_str) signature for *flag*, or ``None``.

    This is the project-agnostic in-run signature; a flag's ledger identity is
    :func:`compute_marker_signature`.

    Both ``task_id`` and ``flag_type`` must be present (i.e. not ``None``) for
    a signature to be computed.  Values are coerced by :func:`signature_field`.
    Falsy-but-valid values like ``task_id=0`` or ``flag_type=''`` are accepted
    — only ``None`` (absent key) triggers a ``None`` return.

    **cited_tasks union (task-2432 bullets 3/4; supersedes the PRD γ §9.3
    fallback):** whenever ``cited_tasks`` is a non-empty list, the task
    component is the *sorted set* of the UNION of {the top-level
    ``task_id``'s comma-split parts, if it is set} ∪ {every ``task_id`` value
    in ``cited_tasks``}, comma-joined with a plain lexicographic ``sorted()``.
    This collapses a shape mismatch between calls that describe the same task
    set differently: a null top-level task_id (the original fallback case), a
    single value equal to one of the cited tasks, or an already comma-joined
    top-level value all produce the SAME signature once cited_tasks agrees. It
    also prevents two findings that share only some of their cited tasks from
    colliding (reviewer finding dedup_correctness). When ``cited_tasks`` is
    absent (or not a non-empty list), the top-level ``task_id`` passes
    through UNCHANGED — single, comma-joined, or ``None``.  Callers that need
    precise single-task dedup should always set the top-level ``task_id``
    explicitly.

    The union's order is a stored form; see the module docstring's table.

    **Upgrade-boundary note (reviewer finding backward_compatibility):**
    widening this union to apply whenever ``cited_tasks`` is non-empty
    (rather than only when the top-level ``task_id`` is ``None``) changes
    the signature for any pre-existing finding that had BOTH a set top-level
    ``task_id`` and ``cited_tasks`` — e.g. ``task_id='5040', cited=[5040]``
    and ``task_id='5040', cited=[5040, 5149]`` used to collapse to the same
    signature and are now distinct. A ledger suppression marker written
    under the old signature will therefore miss once immediately after
    deploy; this is self-healing (the very next cycle writes suppression
    under the new signature), so no marker migration or dual-key lookup is
    required.

    **Payload-persistence hazard (task 4712):** because this union folds every
    ``cited_tasks`` entry's ``task_id`` into the task component, any value
    *derived from* ``cited_tasks`` is unconditionally stored in a row keyed by
    itself, and so can never be persisted as durable CROSS-CYCLE state in that
    row's own payload — a later cycle presenting a different citation set
    keys to a *different* row and can never read it back.  Measured:
    ``compute_flag_signature({'task_id': 598, 'flag_type': ft, 'cited_tasks':
    [{'project_id': 'dark_factory', 'task_id': '3839'}]})`` returns
    ``('3839,598', ft)`` while the citation-less ``compute_flag_signature({
    'task_id': 598, 'flag_type': ft})`` returns ``('598', ft)`` — two
    different rows for one logical flag.  This is why
    ``flag_dedup.py::dedup_flags`` no longer persists ``cited_tasks`` in the
    marker payload; see that module docstring's invariant for the general rule
    this instance falls out of.

    Returns ``None`` for flags without enough signal to deduplicate — these are
    passed through unchanged by ``dedup_flags``.
    """
    return _task_union_signature(flag, home_project_id=None)


def compute_marker_signature(
    flag: dict[str, Any], *, project_id: str
) -> tuple[str, str] | None:
    """Return *flag*'s ``stage1_flag_marker`` ledger identity in *project_id*, or ``None``.

    The task component is :func:`compute_flag_signature`'s union, except that
    a cited task from a project other than *project_id* contributes
    ``'<project>:<task_id>'``.  Every bare id names a task of the row's own
    ``project_id``.  Both halves are stripped, and a foreign citation whose
    qualified component :func:`is_valid_marker_task_id` would reject
    contributes its bare id, as before.  A flag with no task anchor falls back
    to :func:`compute_content_fingerprint_signature`.
    """
    return (
        _task_union_signature(flag, home_project_id=project_id)
        or compute_content_fingerprint_signature(flag)
    )


def _task_union_signature(
    flag: dict[str, Any], *, home_project_id: str | None
) -> tuple[str, str] | None:
    task_id = flag.get('task_id')

    # cited_tasks union (task-2432): whenever cited_tasks is present, the
    # task component is the sorted union of the top-level task_id's
    # comma-split parts (if any) and every cited task_id.  The top-level value
    # is str-coerced first, so a bool/float/list task_id splits as it always
    # has.  flag_type is still required at the top level.
    cited_tasks = flag.get('cited_tasks')
    if cited_tasks and isinstance(cited_tasks, list):
        task_id_parts = set(parse_flag_subject(signature_field(task_id)).components)
        task_id_parts.update(
            _cited_task_component(c, home_project_id)
            for c in cited_tasks
            if isinstance(c, dict) and c.get('task_id') is not None
        )
        if task_id_parts:
            task_id = ','.join(sorted(task_id_parts))

    sig_task_id = signature_field(task_id)
    sig_flag_type = signature_field(flag.get('flag_type'))
    if sig_task_id is None or sig_flag_type is None:
        return None
    return (sig_task_id, sig_flag_type)


def _cited_task_component(cited: dict[str, Any], home_project_id: str | None) -> str:
    bare = str(cited['task_id'])
    cited_project = cited.get('project_id')
    if home_project_id is None or not isinstance(cited_project, str):
        return bare
    project = cited_project.strip()
    if not project or project == home_project_id:
        return bare
    qualified = f'{project}:{bare.strip()}'
    if _is_valid_marker_task_component(qualified):
        return qualified
    return bare


def compute_content_fingerprint_signature(
    flag: dict[str, Any],
) -> tuple[str, str] | None:
    """Return a content-fingerprint (fp:<hex>, flag_type_or_sentinel) or None.

    Activated ONLY when ALL of the following hold:
    - ``task_id`` is None (no top-level task anchor)
    - ``cited_tasks`` yields no task_id values (empty list, absent, or all None)
    - ``description`` is a ``str`` whose normalized form is non-blank

    Returns None when any condition fails so callers can fall through to
    :func:`compute_flag_signature` (which handles the cited_tasks path) or
    pass the flag through unchanged (when both return None).

    When ``flag_type`` is falsy, the sentinel :data:`CONTENT_FP_FLAG_TYPE` is
    used so the 2-tuple shape is preserved for marker write/match in
    ``dedup_flags``.
    """
    if flag.get('task_id') is not None:
        return None

    cited_tasks = flag.get('cited_tasks')
    if cited_tasks and isinstance(cited_tasks, list) and any(
        isinstance(c, dict) and c.get('task_id') is not None
        for c in cited_tasks
    ):
        return None

    description = flag.get('description') or ''
    if not isinstance(description, str) or not normalize_description(description):
        return None

    ftype = flag.get('flag_type') or CONTENT_FP_FLAG_TYPE
    return (content_fingerprint(description), str(ftype))


# --------------------------------------------------------------------------- #
# flag_type families
# --------------------------------------------------------------------------- #

#: Splits on runs of non-alphanumeric characters for flag_type family
#: tokenization (task 2503) -- underscore, hyphen, and whitespace are all
#: treated as separators.
_FLAG_TYPE_TOKEN_SPLIT_RE: re.Pattern[str] = re.compile(r'[^a-z0-9]+')


def canonical_flag_type_family(flag_type: str) -> str:
    """Return a canonical "family" key for *flag_type* (task 2503).

    Collapses four normalization axes onto one deterministic key so a
    reworded/reordered flag_type for the SAME underlying finding is
    recognized as a match by ``flag_dedup.py::filter_suppressed`` and
    ``write_suppression_record``'s pre-write coverage check: case
    (``'Missing_Deliverable'``), separator (``'-'``/``'_'``/``' '``),
    whitespace, and word-order (``'deliverable_missing'`` ==
    ``'missing_deliverable'``).

    Algorithm: casefold *flag_type*, split on runs of non-alphanumeric
    characters (:data:`_FLAG_TYPE_TOKEN_SPLIT_RE`), drop empty tokens, then
    join ``sorted(set(tokens))`` with ``'_'``. Empty/whitespace/
    all-separator input (or input that yields zero tokens) returns ``''``.

    Deliberately does NOT stem, synonym-map, or tolerate an added/removed
    word -- only case/separator/whitespace/word-order variants collapse to
    the same family. Wider tolerance would risk false-positive
    over-suppression: the scoped-suppression PRD documents a
    genuinely-recurring finding (``live_workflow_recurrence_counter_needed``)
    hidden for 6+ cycles by an over-broad suppression record.
    Under-suppression here only costs one extra companion record, which the
    pre-write coverage check still catches for reorder/case/separator
    variants of an already-covered flag_type.

    Note the specific mechanism the word-order axis accepts: two flag_types
    that merely share the same TOKEN MULTISET (e.g. ``'user_missing'`` and
    ``'missing_user'``) collapse to the same family even if they describe
    different findings -- see ``filter_suppressed``'s "ACCEPTED RISK"
    docstring paragraph for the reviewed rationale and the audit-log
    mitigation it emits on such a collision.

    Idempotent: the output is already lowercase, separator-free, and
    ``'_'``-joined in sorted order, so re-applying this function to its own
    output is a no-op.
    """
    tokens = _FLAG_TYPE_TOKEN_SPLIT_RE.split(flag_type.casefold())
    return '_'.join(sorted({t for t in tokens if t}))


@dataclass(frozen=True)
class FlagTypeFamily:
    """The ``flag_type`` spellings of one Stage-1 finding that a filter acts on.

    ``flag_type`` is LLM-authored with no committed schema entry, so a filter
    keyed on it goes silently inert whenever Stage 1 coins a new spelling.
    :meth:`matches` tolerates what :func:`canonical_flag_type_family`
    collapses, and :meth:`log_drift` makes the rest observable.

    ``name`` labels the family in the drift log; ``spellings`` are the
    measured ones; a ``flag_type`` whose casefolded text contains any one of
    the ``drift_tokens`` is probably meant for this family.  A non-empty
    ``unseen_spelling_tokens`` also admits a never-observed spelling whose
    tokens include all of them.
    """

    name: str
    spellings: frozenset[str]
    drift_tokens: frozenset[str]
    unseen_spelling_tokens: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        tokens = self.drift_tokens
        if (
            not isinstance(tokens, frozenset)
            or not tokens
            or any(not t or t != t.casefold() for t in tokens)
        ):
            raise ValueError(
                f'FlagTypeFamily {self.name!r}: drift_tokens must be a non-empty frozenset '
                f'of non-empty casefolded substrings, got {tokens!r}'
            )

    @cached_property
    def _canonical_spellings(self) -> frozenset[str]:
        return frozenset(canonical_flag_type_family(s) for s in self.spellings)

    def matches(self, flag_type: Any) -> bool:
        """True iff *flag_type* is a spelling of this family; ``False`` for any non-``str``."""
        if not isinstance(flag_type, str) or not flag_type:
            return False
        if canonical_flag_type_family(flag_type) in self._canonical_spellings:
            return True
        tokens = {t for t in _FLAG_TYPE_TOKEN_SPLIT_RE.split(flag_type.casefold()) if t}
        return bool(self.unseen_spelling_tokens) and self.unseen_spelling_tokens <= tokens

    def _probably_meant_for_family(self, flag_type: str) -> bool:
        folded = flag_type.casefold()
        return any(token in folded for token in self.drift_tokens)

    def log_drift(self, flags: list[dict[str, Any]], *, log_event: str) -> None:
        """Log as *log_event* each ``flag_type`` carrying a drift token that does not match."""
        unmatched = [
            ft
            for flag in flags
            if isinstance(ft := flag.get('flag_type'), str)
            and self._probably_meant_for_family(ft)
            and not self.matches(ft)
        ]
        if unmatched:
            logger.info(
                '%s family=%s unmatched_flag_types=%s known_types=%s '
                '— add the spelling to that FlagTypeFamily if drift confirmed',
                log_event,
                self.name,
                unmatched,
                sorted(self.spellings),
            )


# --------------------------------------------------------------------------- #
# UUIDs a flag names
# --------------------------------------------------------------------------- #

#: Canonical 8-4-4-4-12 hex UUID, anchored on word boundaries so a UUID is only
#: extracted when it stands alone (not as a substring of a longer alnum run).
_UUID_RE: re.Pattern[str] = re.compile(
    r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-'
    r'[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b'
)


def extract_flag_uuids(flag: dict[str, Any]) -> set[str]:
    """Every UUID *flag* names in any ``str`` value, lowercased.

    Structured and free-text fields alike, at any nesting depth, since a
    stamps-omitted flag may cite a UUID wherever the LLM placed it; non-``str``
    values are ignored.  Upper- and lower-case spellings of one UUID, and
    repeats, collapse to one value.
    """
    text = ' '.join(collect_str_values(flag))
    return {m.lower() for m in _UUID_RE.findall(text)}
