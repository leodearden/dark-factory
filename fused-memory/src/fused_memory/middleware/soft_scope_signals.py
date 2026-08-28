"""Soft (non-structural) scope signals for FILELESS misfiles.

Every guard in :mod:`fused_memory.middleware.path_scope_guard` classifies by
DECLARED PATHS — ``metadata.files`` exactly (``project_for_path``) or
repo-relative prefixes lexed out of prose (``find_paths``).  Roughly half of
the real misfiles in the measured corpus declare NO files at all and cite no
repo-relative prefix, so for that class there is currently no mechanical
signal whatsoever.  This module produces one.

The signals here are deliberately SOFT: they are behavioural regularities in
how agents write task text, not structural invariants about what a task
touches.  None of them is fit to stamp a task on its own.  They exist to
TRIGGER a confirmation step — the otherwise-dormant
:class:`~fused_memory.middleware.path_scope_adjudicator.PathScopeAdjudicator`
— and the trigger is gated on ``strength='strong'`` precisely because the
adjudicator costs money (~$0.105 per firing at the documented floor).

Signal (a) — ``find_title_project_prefix``.  STRONG.  The leading
``<project>: ...`` title convention, by which an agent announces the repo a
task belongs to.  THE HONEST CAVEAT: this is agents self-labelling by an
UNENFORCED convention.  Measured over 8,719 tasks it fires 4 times, at 75%
precision overall and 3/3 in the reify -> dark_factory direction.  A positive
class of n=4 cannot support a stamp — one more false positive would move
measured precision by 25 points — which is exactly why it is wired as a
trigger for a confirmation step and never as a verdict.  The one measured
false positive ('Reify first census: ...', a dark_factory task ABOUT reify)
is pinned by a test so that cost stays visible.

Signal (b1) — ``find_absolute_foreign_roots``.  STRONG.  An ABSOLUTE path
under another project's known root, appearing anywhere in the task text.
This closes a blindness ``find_paths`` cannot: its left-boundary class
``[^A-Za-z0-9_\\-/.]`` excludes ``/`` on purpose (so ``vendor/corpus/x``
does not match the bare prefix ``corpus/``), which makes absolute paths
structurally invisible to it.  Widening that class would re-admit exactly
the mid-path false positives it exists to remove, so the more certain
evidence form gets its own scan, keyed on ``registry.project_to_root``.

Signal (b2) — ``find_foreign_project_names``.  WEAK.  A bare foreign project
NAME in prose.  Measured fire rate is 20.6% of dark_factory tasks and 3.2%
of reify tasks — far too broad to trigger a paid confirmation step, let
alone to stamp.  It is carried as adjudicator CONTEXT and census detail
only; :attr:`SoftScopeFinding.should_adjudicate` is True iff a STRONG signal
is present.

Pure module: no I/O, no LLM, no filesystem access.  Written as a direct
sibling of ``path_scope_guard`` — frozen dataclasses with derived boolean
properties, module-level regex caches keyed by the alias tuple.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from fused_memory.middleware.project_prefix_registry import ProjectPrefixRegistry


__all__ = [
    'SoftScopeSignal',
    'SoftScopeFinding',
    'project_name_aliases',
    'find_title_project_prefix',
    'find_absolute_foreign_roots',
    'find_foreign_project_names',
    'collect_soft_scope_signals',
    'soft_scope_enforced',
]


# ---------------------------------------------------------------------------
# Pattern cache
# ---------------------------------------------------------------------------

# Keyed by the (foreign) alias tuple the pattern was built from, mirroring
# ``path_scope_guard._PATTERN_CACHE``'s prefix-tuple keying: the alias set is
# loop-invariant for a given (registry, filing project) pair, so the compile
# happens once per distinct pair rather than once per submission.
_TITLE_PATTERN_CACHE: dict[tuple[str, ...], re.Pattern[str]] = {}

# How much text may sit between the announced project name and the colon.
# Measured from the corpus: the true positives carry either nothing
# ('dark-factory:') or a short parenthetical subsystem qualifier
# ('CROSS-REPO (dark-factory merge_queue):').  Wider than this and the rule
# starts matching ordinary sentences whose first word happens to be a
# project name.
_TITLE_QUALIFIER_MAX: int = 40


# ---------------------------------------------------------------------------
# Signal record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SoftScopeSignal:
    """One soft signal that a candidate may belong to another project.

    Fields:
        kind: Signal discriminator — ``'title_project_prefix'``,
            ``'absolute_foreign_root'`` or ``'foreign_project_name'``.
        project_id: The IMPLICATED foreign project_id (never the filer).
        evidence: The literal matched text, carried verbatim so the census
            line, the adjudicator prompt and any operator escalation all
            quote what was actually seen rather than a paraphrase.
        strength: ``'strong'`` signals may trigger the paid confirmation
            step; ``'weak'`` ones are context and census detail only.
    """

    kind: Literal[
        'title_project_prefix', 'absolute_foreign_root', 'foreign_project_name'
    ]
    project_id: str
    evidence: str
    strength: Literal['strong', 'weak'] = 'strong'

    @property
    def tagged_evidence(self) -> str:
        """``<short-tag>:<evidence>`` — self-describing evidence for consumers.

        The confirmation step
        (:class:`~fused_memory.middleware.path_scope_adjudicator.PathScopeAdjudicator`)
        takes a ``matched_paths`` tuple and renders it under the fixed label
        ``Flagged matched path prefixes``.  That label is TRUE for its
        original caller, whose evidence really is repo-relative path
        prefixes lexed out of prose — and FALSE for every signal in this
        module: a leading title run, an absolute root, and a bare project
        name are not path prefixes, and two of the three are not paths at
        all.  Handed over raw, ``dark-factory`` would be presented to the
        classifier as a path prefix, with nothing to say which rule found
        it or how much weight it carries.

        Tagging at the producer rather than reshaping the adjudicator's
        prompt builder or its structured-output schema is deliberate: that
        module's fail-safe contract and its 22 existing tests stay untouched,
        and the trigger adapts to the confirmation step rather than the
        reverse.  The tags double as operator-facing provenance in the
        escalation detail and in the ``possible_scope_mismatch`` marker task
        3121 consumes.
        """
        return f'{_EVIDENCE_TAGS[self.kind]}:{self.evidence}'


# Short, stable, operator-legible tag per signal kind.  Stable because they
# are written into task metadata (`possible_scope_mismatch.matched_paths`)
# and read back by downstream consumers — renaming one is a data migration,
# not a cosmetic edit.
_EVIDENCE_TAGS: dict[str, str] = {
    'title_project_prefix': 'title-prefix',
    'absolute_foreign_root': 'abs-root',
    'foreign_project_name': 'project-name',
}


# ---------------------------------------------------------------------------
# Alias derivation
# ---------------------------------------------------------------------------


def project_name_aliases(project_id: str, root: str | None) -> tuple[str, ...]:
    """Return the lowercased name spellings that denote *project_id*.

    The registry canonicalises its keys (``resolve_project_id_for_root``:
    lowercase, ``-`` -> ``_``), so it knows ``dark_factory`` — while agents
    typing prose overwhelmingly write the on-disk directory spelling,
    ``dark-factory``.  Nothing in the registry bridges the two: its
    ``project_to_root`` values are consumed only as ROOTS for absolute-file
    lookup, and project NAMES are never matched against prose at all.  This
    is that bridge.

    Yields, deduplicated and in stable order: the project_id itself, its
    ``_`` -> ``-`` transliteration, and the basename of *root* when a root
    is known.  Deliberately a free function rather than a
    :class:`ProjectPrefixRegistry` method — the registry is a frozen
    dataclass with a wide test surface and several production consumers, and
    this concern belongs to exactly one caller.
    """
    out: list[str] = []
    seen: set[str] = set()
    candidates = [project_id, (project_id or '').replace('_', '-')]
    if root:
        candidates.append(os.path.basename(str(root).rstrip('/')))
    for raw in candidates:
        alias = (raw or '').strip().lower()
        if not alias or alias in seen:
            continue
        seen.add(alias)
        out.append(alias)
    return tuple(out)


def _foreign_aliases(
    project_id: str, registry: ProjectPrefixRegistry
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Return ``(alias -> owning project_id, alias tuple)`` for FOREIGN projects.

    Aliases belonging to the FILING project are excluded outright, so
    self-reference ('reify: ...' filed under reify) can never match — it is
    an announcement of the task's own scope, not a misfile.  An alias
    claimed by more than one project is dropped: the same silence-beats-
    ambiguity rule ``ProjectPrefixRegistry.from_roots`` applies to colliding
    prefixes.

    The alias tuple is ordered LONGEST FIRST so the regex alternation
    prefers the most specific spelling (Python's ``|`` is first-match, not
    longest-match), with ties broken alphabetically for a stable cache key.
    """
    filer = (project_id or '').strip().lower()
    filer_aliases = set(
        project_name_aliases(filer, registry.root_for_project(project_id))
    )
    owners: dict[str, set[str]] = {}
    for pid, root in registry.project_to_root.items():
        if (pid or '').strip().lower() == filer:
            continue
        for alias in project_name_aliases(pid, root):
            if alias in filer_aliases:
                continue
            owners.setdefault(alias, set()).add(pid)
    alias_to_project = {
        alias: next(iter(pids)) for alias, pids in owners.items() if len(pids) == 1
    }
    ordered = tuple(sorted(alias_to_project, key=lambda a: (-len(a), a)))
    return alias_to_project, ordered


# ---------------------------------------------------------------------------
# Signal (a): leading "<project>:" title convention
# ---------------------------------------------------------------------------


def _build_title_pattern(aliases: tuple[str, ...]) -> re.Pattern[str]:
    """Build (and cache) the leading-announcement regex for *aliases*.

    ``^\\s*(?:cross[- ]repo\\s*\\(\\s*)?(<alias-alternation>)\\b[^:]{0,40}:``

    Anchored at start-of-string: the announcement shape is a PREFIX, so a
    mid-title mention ('teach dark-factory: about X') must not match.  The
    optional ``CROSS-REPO (`` lead-in is the second measured spelling of the
    same convention.  ``[^:]{0,40}`` admits a short subsystem qualifier
    while forbidding the colon itself, so the colon that terminates the run
    is the FIRST one — a title whose colon lands beyond the window (an
    ordinary sentence that merely opens with a project name) does not match.
    """
    cached = _TITLE_PATTERN_CACHE.get(aliases)
    if cached is not None:
        return cached
    alternation = '|'.join(re.escape(a) for a in aliases)
    pattern = re.compile(
        rf'^\s*(?:cross[- ]repo\s*\(\s*)?({alternation})\b[^:]{{0,{_TITLE_QUALIFIER_MAX}}}:',
        re.IGNORECASE,
    )
    _TITLE_PATTERN_CACHE[aliases] = pattern
    return pattern


def find_title_project_prefix(
    title: str | None,
    project_id: str,
    registry: ProjectPrefixRegistry | None,
) -> SoftScopeSignal | None:
    """Return the leading ``<foreign-project>:`` announcement signal, or None.

    STRONG (see module docstring for the measured precision and the n=4
    caveat).  Returns the match as a ``strength='strong'``
    :class:`SoftScopeSignal` whose ``project_id`` is the IMPLICATED foreign
    project and whose ``evidence`` is the matched leading run verbatim.

    Short-circuits to ``None`` on a falsy/whitespace-only title, on a falsy
    registry (an empty registry is falsy by construction), and when the
    registry knows no foreign project.
    """
    if not title or not title.strip() or not registry:
        return None
    alias_to_project, aliases = _foreign_aliases(project_id, registry)
    if not aliases:
        return None
    match = _build_title_pattern(aliases).search(title)
    if not match:
        return None
    owner = alias_to_project.get(match.group(1).strip().lower())
    if not owner:
        return None
    return SoftScopeSignal(
        kind='title_project_prefix',
        project_id=owner,
        evidence=match.group(0).strip(),
        strength='strong',
    )


# ---------------------------------------------------------------------------
# Signal (b1): an ABSOLUTE path under a foreign project root
# ---------------------------------------------------------------------------

# Right-hand component boundary for a matched root.  The match is REJECTED
# when the root is immediately continued by another path-name character —
# ``<root>-old``, ``<root>ish``, ``<root>.bak`` are different directories,
# not the root — which is the same boundary
# ``ProjectPrefixRegistry._owner_for_absolute_path`` enforces for declared
# file paths (``root + '/'``), stated here as its right-context complement
# so the BARE-root spelling is admitted too.
#
# A trailing ``.`` is the one character that has to be read in context: it
# begins a real suffix in ``<root>.bak/x.py`` (reject) but is ordinary
# sentence punctuation in "the work is in <root>." (accept).  Treating every
# ``.`` as a suffix would lose sentence-final citations, which in prose is
# the COMMON spelling of exactly the fileless case this signal exists to
# catch; treating none as a suffix would re-admit the sibling-directory
# false positive.  So ``.`` blocks only when it is itself followed by a
# path-name character.
_ROOT_SUFFIX_RE: re.Pattern[str] = re.compile(r'[A-Za-z0-9_\-]|\.[A-Za-z0-9_\-]')


def _foreign_roots_longest_first(
    project_id: str, registry: ProjectPrefixRegistry
) -> tuple[tuple[str, str], ...]:
    """Return FOREIGN ``(root, project_id)`` pairs, longest root first.

    Mirrors :attr:`ProjectPrefixRegistry._roots_longest_first`: nested roots
    must resolve to the MOST SPECIFIC owner deterministically, independent of
    dict insertion order.  Degenerate roots (empty, relative, or normalising
    to ``/``) are dropped for the same reason they are there — ``''`` would
    make every text a match.  Ties break alphabetically so the emitted signal
    order is stable across runs.
    """
    filer = (project_id or '').strip().lower()
    pairs: list[tuple[str, str]] = []
    for pid, raw in registry.project_to_root.items():
        if (pid or '').strip().lower() == filer:
            continue
        root = os.path.normpath(raw).rstrip('/') if raw else ''
        if not root or not root.startswith('/'):
            continue
        pairs.append((root, pid))
    pairs.sort(key=lambda pair: (-len(pair[0]), pair[0]))
    return tuple(pairs)


def find_absolute_foreign_roots(
    text: str | None,
    project_id: str,
    registry: ProjectPrefixRegistry | None,
) -> list[SoftScopeSignal]:
    """Return one STRONG signal per foreign project root cited in *text*.

    WHY THIS CANNOT BE FOLDED INTO ``path_scope_guard.find_paths``: that
    matcher's LEFT boundary class ``[^A-Za-z0-9_\\-/.]`` excludes ``/`` and
    ``.`` ON PURPOSE, so that ``vendor/corpus/expr.txt`` does not match the
    bare prefix ``corpus/``; and its ``_RIGHT_CONTEXT`` assertion (task
    3120) further requires the match be followed by ``<seg>/`` or a file
    extension, so an English slash-construction does not lex as a path.
    Both constraints are load-bearing, and both make an absolute path
    structurally invisible: the prefix in
    ``/home/leo/src/dark-factory/orchestrator/x.py`` is always preceded by
    ``/``.  Widening that class to admit absolute paths would re-admit
    precisely the mid-path false positives it exists to remove — so the MORE
    certain evidence form is the one the prose matcher cannot see, and it
    gets its own scan keyed on ``registry.project_to_root`` (a field
    consumed today only for absolute FILE lookup, never matched against
    prose).

    A bare root with no trailing segment DOES fire, unlike a bare relative
    prefix: nothing but the project spells that project's absolute root, so
    there is no bare-MENTION ambiguity to defend against — including at the
    end of a sentence, where the trailing ``.`` is punctuation rather than a
    directory suffix (see :data:`_ROOT_SUFFIX_RE`).

    Roots are scanned longest-first so nested roots resolve to the most
    specific owner; at most one signal is emitted per foreign project, in
    that same stable order.  Pure string work — no filesystem access.
    """
    if not text or not registry:
        return []
    signals: list[SoftScopeSignal] = []
    seen: set[str] = set()
    for root, owner in _foreign_roots_longest_first(project_id, registry):
        if owner in seen:
            continue
        start = text.find(root)
        while start != -1:
            end = start + len(root)
            tail = text[end : end + 2]
            if not tail or not _ROOT_SUFFIX_RE.match(tail):
                seen.add(owner)
                signals.append(
                    SoftScopeSignal(
                        kind='absolute_foreign_root',
                        project_id=owner,
                        evidence=root,
                        strength='strong',
                    )
                )
                break
            start = text.find(root, start + 1)
    return signals


# ---------------------------------------------------------------------------
# Signal (b2): a bare foreign project NAME in prose — WEAK
# ---------------------------------------------------------------------------

_NAME_PATTERN_CACHE: dict[tuple[str, ...], re.Pattern[str]] = {}


def _build_name_pattern(aliases: tuple[str, ...]) -> re.Pattern[str]:
    """Build (and cache) a word-boundary alternation over *aliases*."""
    cached = _NAME_PATTERN_CACHE.get(aliases)
    if cached is not None:
        return cached
    alternation = '|'.join(re.escape(a) for a in aliases)
    pattern = re.compile(rf'\b({alternation})\b', re.IGNORECASE)
    _NAME_PATTERN_CACHE[aliases] = pattern
    return pattern


def find_foreign_project_names(
    text: str | None,
    project_id: str,
    registry: ProjectPrefixRegistry | None,
) -> list[SoftScopeSignal]:
    """Return one WEAK signal per foreign project NAMED in *text*.

    THE RATE IS THE WHOLE POINT: measured over the corpus, the bare-name
    rule fires on 20.6% of dark_factory tasks and 3.2% of reify tasks.  At
    the adjudicator's documented ~$0.105-per-firing floor that is an
    unaffordable trigger, and it is far too broad to stamp — a task can
    mention another project because it INTEGRATES with it, because it cites
    a precedent from it, or simply because the two share an operator.  So
    these signals are ``strength='weak'``: they ride along as adjudicator
    context and census detail, and
    :attr:`SoftScopeFinding.should_adjudicate` deliberately ignores them.

    Word-boundary anchored on both sides, so ``dark-factoryish`` and
    ``darkfactory`` do not match ``dark-factory``.  At most one signal per
    implicated project; the filer's own aliases are excluded upstream in
    :func:`_foreign_aliases`, so self-mention never fires.
    """
    if not text or not registry:
        return []
    alias_to_project, aliases = _foreign_aliases(project_id, registry)
    if not aliases:
        return []
    signals: list[SoftScopeSignal] = []
    seen: set[str] = set()
    for match in _build_name_pattern(aliases).finditer(text):
        owner = alias_to_project.get(match.group(1).strip().lower())
        if not owner or owner in seen:
            continue
        seen.add(owner)
        signals.append(
            SoftScopeSignal(
                kind='foreign_project_name',
                project_id=owner,
                evidence=match.group(1),
                strength='weak',
            )
        )
    return signals


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SoftScopeFinding:
    """Every soft signal collected for one candidate, strong ones first."""

    signals: tuple[SoftScopeSignal, ...] = ()

    @property
    def should_adjudicate(self) -> bool:
        """True iff a STRONG signal is present.

        The trigger for the paid confirmation step.  Weak signals never
        reach it on their own — see :func:`find_foreign_project_names` for
        the measured fire rate that makes that non-negotiable.
        """
        return any(s.strength == 'strong' for s in self.signals)

    @property
    def suggested_project(self) -> str | None:
        """The single STRONG-implicated foreign project, or None on disagreement.

        Mirrors ``path_scope_guard._aggregate_owner_mismatches``' rule: when
        the evidence names more than one project there is no single target
        to suggest, and silence beats a coin flip.

        Agreement is computed over ``strength='strong'`` signals ONLY, for
        the same reason :attr:`should_adjudicate` ignores weak ones: the bare
        project-NAME rule fires on 20.6% of dark_factory tasks, so letting it
        vote would let one incidental mention of another repo veto a target
        that unambiguous strong evidence had already established.  Measured
        over the 4,791-task dark_factory corpus, all-signal agreement lost
        the suggestion on 10 of 111 strong firings (9.0%) purely to that
        dilution.  The loss is not cosmetic: this value is passed to
        ``PathScopeAdjudicator.adjudicate`` (so the classifier loses the
        target name) and is the only field of the ``possible_scope_mismatch``
        stamp any landed consumer reads
        (``orchestrator/src/orchestrator/cross_repo_gate.py::_resolve_owner``).

        Weak signals still travel in :attr:`matched_paths` and in the census
        line, tagged, so an operator sees every project the text mentioned.
        """
        owners = {s.project_id for s in self.signals if s.strength == 'strong'}
        return next(iter(owners)) if len(owners) == 1 else None

    @property
    def matched_paths(self) -> tuple[str, ...]:
        """Tagged evidence, strong-first — the transport shape for consumers.

        Named for the ``matched_paths`` parameter it feeds
        (``PathScopeAdjudicator.adjudicate``, ``PathGuardVerdict``,
        ``possible_scope_mismatch.matched_paths``) so the one conversion
        lives here rather than being re-derived at each call site.  See
        :attr:`SoftScopeSignal.tagged_evidence` for why the values are
        tagged rather than raw.
        """
        return tuple(sig.tagged_evidence for sig in self.signals)


def collect_soft_scope_signals(
    title: str | None,
    description: str | None,
    details: str | None,
    project_id: str,
    registry: ProjectPrefixRegistry | None,
) -> SoftScopeFinding:
    """Collect every soft scope signal for one candidate.

    The two PROSE signals scan title/description/details joined into one
    blob, matching ``check_candidate_for_scope``'s part-joining.  The TITLE
    convention runs against the TITLE ALONE: its rule is anchored at
    start-of-string, so against a joined blob it would only ever see the
    title anyway when the title comes first — and would silently start
    matching a description's opening words the moment the join order or an
    empty title changed it.  Anchoring it to the field it was measured on
    keeps the rule meaning what it was measured to mean.

    Strong signals are returned before weak ones so the census line, the
    adjudicator prompt and any operator escalation all lead with the best
    evidence.
    """
    strong: list[SoftScopeSignal] = []
    title_signal = find_title_project_prefix(title, project_id, registry)
    if title_signal is not None:
        strong.append(title_signal)
    blob = '\n'.join(part for part in (title, description, details) if part)
    strong.extend(find_absolute_foreign_roots(blob, project_id, registry))
    weak = find_foreign_project_names(blob, project_id, registry)
    return SoftScopeFinding(signals=tuple(strong) + tuple(weak))


# ---------------------------------------------------------------------------
# Staged rollout
# ---------------------------------------------------------------------------

# Truthy string values for FUSED_SOFT_SCOPE_ENFORCE (case-insensitive,
# whitespace-stripped). Unset, empty, or any other value -> warn-only (the
# default) — see soft_scope_enforced().  A deliberate literal mirror of
# routing_intent_guard._TRUTHY_ENV_VALUES: `shared/` has no truthy-env
# helper, and the local-copy shape is this codebase's established
# convention for staged-rollout flags.
_TRUTHY_ENV_VALUES: frozenset[str] = frozenset({'1', 'true', 'yes', 'on'})


def soft_scope_enforced() -> bool:
    """Read ``FUSED_SOFT_SCOPE_ENFORCE`` from the environment.

    Returns ``True`` (act on a confirmed misroute: stamp the advisory marker
    and fire the non-blocking escalation) only when the env var is set to a
    recognized truthy string.  Unset, empty, or any other value -> ``False``
    (warn-only, the default: adjudicate and log the census line, act on
    nothing).  Mirrors
    :func:`~fused_memory.middleware.routing_intent_guard.routing_intent_enforced`.

    Enforcement NEVER blocks creation in either mode — the measured prose
    precision this signal family lives beside is 10.7%, so the maximum
    enforced action is advisory.
    """
    raw = os.environ.get('FUSED_SOFT_SCOPE_ENFORCE', '')
    return raw.strip().lower() in _TRUTHY_ENV_VALUES
