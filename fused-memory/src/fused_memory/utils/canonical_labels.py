"""THE canonical task-label vocabulary — one normative site, no second copy.

This module owns the single description of what a canonical task label looks
like: the anchored node-name form ('Task 132'), the unanchored prose mention
('blocked on task 132'), the project-qualified cross-project form
('dark_factory:2500'), and the narrowings that keep colon-bearing noise out.

INV-5 (no lockstep duplication) / PRD resolved decision 5. Before task 3667
the vocabulary existed as separate compiled copies in utils/task_naming.py
and utils/cross_project_refs.py, which had already drifted: task_naming's
``^\\s*tasks?\\s+(\\d+)\\s*$`` requires whitespace between the word and the
digits and is therefore STRUCTURALLY unable to see 'task #1153' — one of the
PRD's 53 measured task-node variant splits — while cross_project_refs had
grown a second, colon-spelled mention pattern task_naming never got. Both of
those modules now call this one and carry NO compiled pattern of their own;
their test suites assert that structurally. Do not re-introduce a second copy
of the label pattern anywhere: a vocabulary that exists twice drifts, and the
drift is invisible until a destructive consumer acts on the stale half.

This module is a dependency-free leaf — it imports only stdlib and
utils/validation (itself a leaf) — so it can be imported from both the
services write path and any future reconciliation sweep without import
cycles. Mirrors utils/task_naming.py and utils/cross_project_refs.py, whose
shape it copies.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass

from fused_memory.utils.validation import (
    PathShapedProjectIdError,
    canonicalize_project_id,
)

#: The registry of referent kinds and the bare node-name label each renders.
#: Deliberately holds only 'task' today; 'escalation' is the PRD's next entry
#: and is NOT added speculatively. Extending the vocabulary means adding an
#: entry here, which is the whole point of routing every caller through one
#: module.
#:
#: NOTE for a reader hunting the caller that emits a non-'task' kind: there
#: isn't one. Both public entry points hardcode ``kind='task'``, and both
#: compiled patterns hardcode the literal word 'task', so ``__post_init__``'s
#: rejection is reachable only from a direct ``Referent(kind=...)`` call.
#: Adding 'escalation' will need new patterns and new parse/scan branches too —
#: this registry is where the LABEL lives, not a one-line extension point.
_KIND_LABELS: dict[str, str] = {'task': 'Task'}

# ANCHORED (whole-string) task-node name: a bare 'task(s) N', optionally padded
# with whitespace, case-insensitive. Anchoring means names that merely mention a
# task ('Task 42 orchestrator', 'reify task 12') or resemble but aren't a
# task-node name ('subtask 5', 'multitask 3', 'taskforce 9') never match, so
# they are never renamed.
#
# The separator alternation REQUIRES at least one of whitespace, '#' or ':'.
# That is load-bearing: 'task #1153' and 'Task: 132' — two of the PRD's 53
# measured variant splits, structurally invisible to this pattern's ancestor
# '^\s*tasks?\s+(\d+)\s*$' — now match, while 'task132' still does NOT. The
# obvious spelling 'tasks?\s*#?\s*(\d+)' would have matched 'task132' too,
# silently widening what _normalize_task_node_names renames.
#
# The '#'/':' branch is padded with '[ \t]', NOT '\s', so it cannot span a line
# break. '\s' matches '\n', which would let 'task\n#132' parse as a task label —
# a widening this task never intended and never measured (see
# _LOCAL_MENTION_PATTERN, where the same widening had a live consequence). The
# whitespace branch keeps '\s+' VERBATIM from the ancestor pattern: narrowing it
# would be an equally undeclared change in the opposite direction.
_TASK_NODE_NAME_PATTERN = re.compile(r'^\s*tasks?(?:[ \t]*[#:][ \t]*|\s+)(\d+)\s*$', re.IGNORECASE)

# The ANCHORED twin of _QUALIFIED_REF_PATTERN: a whole-string cross-project node
# name, '<project_id>:<task_number>'. Shares that pattern's letter-start and
# >=3-character qualifier rules, so clock times ('12:30') and short non-project
# tokens ('w6:2', 'py:3') never parse as a project. Case-SENSITIVE start class
# with no IGNORECASE flag needed, since [A-Za-z] already spans both cases.
_QUALIFIED_NODE_NAME_PATTERN = re.compile(r'^\s*([A-Za-z][A-Za-z0-9_-]{2,})\s*:\s*(\d+)\s*$')

# Task-vocabulary words are never project ids. Matched with fullmatch() against
# the CANONICALIZED qualifier, so every spelling ('Task', 'TASK', 'sub-task',
# 'Sub-Tasks') collapses onto this one check, while a real project id that
# merely starts with 'task' ('taskmaster') is not rejected.
#
# Moved verbatim from cross_project_refs, where it guards the ONE rejection the
# split consumer's decisive 'episode touched a node named Task N' guard cannot
# make: for 'Task: 2500' that node is exactly what extraction produces, so the
# guard is guaranteed to CO-OCCUR rather than to filter, and the split would
# move a LOCAL task's facts onto a bogus 'task:2500' entity.
_TASK_VOCABULARY_QUALIFIER = re.compile(r'(sub_?)?tasks?')

# An UNANCHORED mention of a task number in prose — the scan-side twin of
# _TASK_NODE_NAME_PATTERN, sharing its separator alternation so every spelling
# the anchored parser accepts ('task 5', 'task #5', 'Task: 5') is also seen
# here. This ONE pattern replaces BOTH of cross_project_refs' pre-3667
# _BARE_TASK_MENTION_PATTERN and _BARE_TASK_COLON_PATTERN, which differed only
# in their separator; keeping them apart would leave two vocabulary copies
# inside the very module that exists to eliminate copies (INV-5).
#
# - The '(?<![\w:-])' lookbehind refuses word-glued lookalikes ('subtask 5',
#   'multitask 3', 'reify-task 12') and — decisively for the colon form — the
#   'task: N' substring INSIDE 'subtask: N', which would otherwise suppress a
#   real foreign ref.
# - '(?!\d)' anchors the number's right edge so a truncated prefix is never
#   captured.
# - The '#'/':' branch is padded with '[ \t]', NOT '\s', so a mention can never
#   span a line break. This is measured, not stylistic: '\s' matches '\n', so
#   '\s*[#:]\s*' read a markdown heading across a paragraph break as a mention —
#   'Completed the task\n\n# 1153 retrospective' yielded a bare Task 1153, and
#   'The following tasks:\n\n1. fix the parser' yielded a bare Task 1. A phantom
#   bare mention lands in bare_numbers and pushes a genuine foreign ref into
#   .ambiguous, silently suppressing a real cross-project repair. Neither
#   spelling was reachable before task 3667 added this branch, so trimming it
#   restores the pre-3667 answer rather than inventing a new narrowing.
# - The whitespace branch deliberately keeps '\s+' VERBATIM from the pre-3667
#   _BARE_TASK_MENTION_PATTERN, newline-spanning and all. Tightening it to
#   '[ \t]+' would be an UNDECLARED narrowing of behaviour this task never
#   touched, and it narrows in the dangerous direction: fewer bare mentions
#   means fewer contests, which means MORE confident splits on prose that
#   pre-3667 refused. A false positive here only ever adds ambiguity, which the
#   consumer refuses; a false negative lets destructive surgery proceed.
_LOCAL_MENTION_PATTERN = re.compile(
    r'(?<![\w:-])tasks?(?:[ \t]*[#:][ \t]*|\s+)(\d+)(?!\d)', re.IGNORECASE
)

# A project-qualified task reference: '<qualifier>:<digits>'. Moved VERBATIM
# from cross_project_refs with its rationale — each clause encodes a specific
# measured false positive, so do NOT re-derive them.
#
# - The lookbehind rejects a qualifier glued to a preceding word character or
#   path/URL punctuation, which is what keeps source locations
#   ('graphiti_client.py:2091' — '.py' is preceded by '.'), URL authorities
#   ('example.com:8080'), colon-chained tokens ('a:b:3') and digit-prefixed
#   junk ('2500dark_factory:2500') out.
# - The qualifier must start with a letter, so clock times ('12:30') never
#   match, and must be at least 3 characters, so short non-project tokens
#   ('w6:2', 'py:3', 'a:1') never match.
# - '\s*' around the colon tolerates the spacing humans actually write.
# - '(?!\d)' anchors the number's right edge so a truncated prefix is never
#   captured.
_QUALIFIED_REF_PATTERN = re.compile(r'(?<![\w:/.-])([A-Za-z][A-Za-z0-9_-]{2,})\s*:\s*(\d+)(?!\d)')


@dataclass(frozen=True, kw_only=True)
class Referent:
    """One thing a canonical label refers to: a kind, a project, a number.

    Frozen because a referent is evidence for destructive graph surgery — a
    consumer must not be able to rewrite which node it names.

    Keyword-only so ``number`` can be the one REQUIRED field while ``kind``
    and ``project_id`` carry defaults, making the overwhelmingly common
    own-project task referent spell as ``Referent(number='132')``.
    """

    #: The number's digits VERBATIM — never int-normalized, so a referent
    #: never invents or reformats a task number ('0132' stays '0132', and is
    #: a DIFFERENT referent from '132').
    number: str
    #: Which registry entry this referent belongs to; see :data:`_KIND_LABELS`.
    kind: str = 'task'
    #: EMPTY means own-project / unqualified. A non-empty value is always
    #: ``canonicalize_project_id``-canonical and denotes a DIFFERENT project.
    #: Encoding local-ness in the referent itself is what lets
    #: :attr:`node_name` discriminate 'Task 132' from 'reify:132' from the
    #: referent alone, with no ambient group_id.
    project_id: str = ''

    def __post_init__(self) -> None:
        if self.kind not in _KIND_LABELS:
            raise ValueError(
                f'unregistered referent kind {self.kind!r}; registered kinds are '
                f'{sorted(_KIND_LABELS)}. Add it to canonical_labels._KIND_LABELS '
                'rather than rendering an unlabelled referent.'
            )

    # Derived name is a property rather than a field so it cannot drift out of
    # sync with the source-of-truth fields above (the CrossProjectRef.entity_name
    # precedent).
    @property
    def node_name(self) -> str:
        """The graph node name this referent denotes.

        ``'<project_id>:<number>'`` when the referent names a foreign project
        — the qualifier is a different-project signal and is NEVER normalized
        away to the bare form, because that collapse is precisely the bug
        utils/cross_project_refs.py exists to detect. Otherwise the canonical
        bare label for the kind, e.g. ``'Task 132'``.
        """
        if self.project_id:
            return f'{self.project_id}:{self.number}'
        return f'{_KIND_LABELS[self.kind]} {self.number}'


def is_task_vocabulary_qualifier(project_id: str) -> bool:
    """Is *project_id* a task-VOCABULARY word rather than a real project key?

    ``'task'``, ``'tasks'``, ``'subtask'``, ``'sub_task'`` and their spellings
    are local task vocabulary; none of them names a project. ``'taskmaster'``
    — a real project that merely starts with 'task' — is not one, which is why
    this is a ``fullmatch`` and not a prefix test.

    THE single public predicate for that question, so the rule lives at exactly
    one site (INV-5). Three callers now need it and they must agree: the two
    inside this module (:func:`parse_node_name`, :func:`scan_content`), and
    :func:`~fused_memory.utils.referent_resolution._declared_referents`, whose
    ``project_id`` arrives as a STRUCTURED caller-supplied field rather than as
    a spelling to be parsed. A second copy in the policy layer is the lockstep
    duplication this module exists to prevent — and the declared path skipping
    the rule outright is exactly how ``{'id': 2500, 'project_id': 'task'}``
    came to mint a foreign ``'task:2500'`` node that cannot exist, while the
    same spelling resolved to the LOCAL ``'Task 2500'`` through both other
    paths.

    Args:
        project_id: A CANONICALIZED project id (``canonicalize_project_id``
            output). Passing a raw qualifier is a caller bug: ``'Sub-Task'``
            only collapses onto the pattern once canonicalized to
            ``'sub_task'``.
    """
    return _TASK_VOCABULARY_QUALIFIER.fullmatch(project_id) is not None


def parse_node_name(name: str) -> Referent | None:
    """Parse an entity *name* as a whole task label, or return None.

    ANCHORED by design: this answers "is this entity NAME a task label?", not
    "does this text mention a task" (:func:`scan_content` answers that). So a
    name that merely contains a task reference — 'Task 42 orchestrator',
    'reify task 12' — returns None and its node is left untouched.

    Two forms parse:

    - The bare local form 'task(s) N', separated by whitespace, '#' or ':',
      in any case and tolerant of padding, yielding an OWN-project referent
      ('task #1153' -> ``Task 1153``).
    - The project-qualified form '<project_id>:N', yielding a FOREIGN referent
      whose qualifier is canonicalized but never normalized AWAY ('reify:132'
      stays ``reify:132``; flattening it to ``Task 132`` is precisely the
      cross-project collapse utils/cross_project_refs.py exists to detect).

    Digits are preserved VERBATIM, never int-normalized, so a referent never
    invents or reformats a task number ('task 0132' -> ``Task 0132``, and
    ``Referent(number='0132') != Referent(number='132')``).

    Returns None for a path-shaped qualifier rather than canonicalizing it: a
    mangled path must never be silently mapped into a new, wrong project key
    (RCA §4). Also None for a task-VOCABULARY qualifier ('subtask: 2500'),
    which is local task vocabulary, not a project id.
    """
    local = _TASK_NODE_NAME_PATTERN.match(name)
    if local is not None:
        # Ordering matters: the local pattern is tried FIRST, which is what
        # makes a vocabulary-qualified name like 'Task: 132' resolve to the
        # local referent it is, rather than to a project named 'task'.
        return Referent(kind='task', number=local.group(1))

    qualified = _QUALIFIED_NODE_NAME_PATTERN.match(name)
    if qualified is None:
        return None
    qualifier, number = qualified.group(1), qualified.group(2)
    try:
        project_id = canonicalize_project_id(qualifier)
    except PathShapedProjectIdError:
        # Defensive: the pattern cannot capture a '/' or a leading '-', so this
        # is unreachable today. Kept so a future pattern relaxation refuses the
        # name instead of raising into a caller's write path.
        return None
    if is_task_vocabulary_qualifier(project_id):
        # 'subtask: 2500' / 'sub-task: 2500'. The local pattern already claimed
        # every spelling that IS a local label ('Task: 132'), so anything
        # reaching here is a word-glued lookalike, not a task label at all.
        return None
    return Referent(kind='task', project_id=project_id, number=number)


@dataclass(frozen=True)
class LabelScan:
    """The referents found in one body of content.

    Frozen for the same reason :class:`Referent` is: a scan is evidence for
    destructive graph surgery, not a mutable accumulator.

    The two fields are TUPLES rather than lists, which is what makes that claim
    true. ``frozen=True`` only blocks attribute REBINDING, so list fields would
    leave ``scan.refs.append(...)`` wide open — a consumer could quietly add a
    referent the scanner refused to infer and the frozen-ness would not notice.
    Tuples also make the scan hashable, which a frozen dataclass holding lists
    silently is not.
    """

    #: Referents safe to act on, in first-seen positional order,
    #: de-duplicated on (kind, project_id, number).
    refs: tuple[Referent, ...] = ()
    #: Referents whose number is claimed by BOTH a BARE, unqualified
    #: own-project mention ('task 2500') and a foreign-qualified reference in
    #: the same content. Such content is genuinely ambiguous about which task
    #: the facts belong to, so a consumer must refuse to act on it and say so
    #: loudly rather than guess. An explicitly qualified reference — including
    #: a SELF-qualified one — already names its project and so never creates
    #: ambiguity; see :func:`scan_content`.
    ambiguous: tuple[Referent, ...] = ()


def _canonical_allowlist(known_project_ids: Collection[str] | None) -> frozenset[str] | None:
    """Canonicalize *known_project_ids*, or return None for permissive mode.

    Returns None when the registry is missing or empty. That is PERMISSIVE by
    design and mirrors ``validate_known_project_id``'s documented
    empty-registry mode: failing closed would silently disable this protection
    entirely on any deployment that never wires the registry, which is the
    exact silent-degradation failure mode this scanner exists to eliminate.

    A path-shaped registry key is skipped rather than normalized (normalizing a
    mangled path would mint a new, wrong canonical key — RCA §4) and rather
    than raised, so one bad entry never disables the whole allowlist.
    """
    if not known_project_ids:
        return None
    canonical = set()
    for raw in known_project_ids:
        try:
            canonical.add(canonicalize_project_id(raw))
        except PathShapedProjectIdError:
            continue
    return frozenset(canonical)


def scan_content(
    content: str,
    *,
    group_id: str,
    known_project_ids: Collection[str] | None = None,
) -> LabelScan:
    """Scan *content* for every task referent it mentions.

    The UNANCHORED counterpart to :func:`parse_node_name`: this answers "what
    does this prose refer to", reading the VERBATIM body rather than extracted
    node names or LLM-restated facts, because a project qualifier is exactly
    what entity extraction discards.

    PRECISION OVER RECALL. Consumers perform destructive edge surgery, so a
    false positive misattributes a fact — the same bug in the opposite
    direction. The narrowings that make that safe live in the compiled
    patterns above and are deliberately conservative.

    KNOWN BLIND SPOTS, so no reader assumes completeness: a node named with
    bare digits ('1251'), a reference made by task TITLE rather than number,
    and Greek-letter or codename aliases are all invisible here by design.
    Recall is the consumer's problem; precision is this module's.

    Args:
        content: The verbatim body. Falsy content yields an empty scan.
        group_id: The group the content belongs to (= the local
            ``project_id``). A reference qualified with this project is not
            cross-project: it is RECLASSIFIED to an own-project referent, since
            extraction collapsing 'reify:5181' onto 'Task 5181' inside the
            reify graph is correct, not a bug. Both sides are canonicalized
            before the comparison, so case and hyphen/underscore spelling
            differences still count as local.
        known_project_ids: Optional registry of known project ids (any
            collection; a ``{project_id: project_root}`` mapping works, since
            iterating it yields its keys). When non-empty, FOREIGN referents
            naming a project outside it are dropped — own-project referents
            are never dropped by it. When None or empty the filter is
            PERMISSIVE — see :func:`_canonical_allowlist`.

    Returns:
        A :class:`LabelScan`. ``refs`` is safe to act on; ``ambiguous`` must
        not be acted on silently. A number goes to ``ambiguous`` only when the
        content claims it BOTH bare ('task 2500', which never says which
        project) and foreign-qualified. Two QUALIFIED references — even when
        one of them is self-qualified — each already name their project and are
        never mutually ambiguous.

    A path-shaped *group_id* yields an empty scan: without a trustworthy local
    project id, local and foreign references cannot be told apart, and the
    conservative answer is to report nothing.
    """
    if not content:
        return LabelScan()
    try:
        local_project = canonicalize_project_id(group_id)
    except PathShapedProjectIdError:
        return LabelScan()

    allowlist = _canonical_allowlist(known_project_ids)
    # (offset, referent, arrived_bare). The third element records whether the
    # referent came from an UNQUALIFIED mention, which is what the ambiguity
    # partition below is decided on.
    found: list[tuple[int, Referent, bool]] = []

    for match in _LOCAL_MENTION_PATTERN.finditer(content):
        found.append((match.start(), Referent(kind='task', number=match.group(1)), True))

    for match in _QUALIFIED_REF_PATTERN.finditer(content):
        qualifier, number = match.group(1), match.group(2)
        try:
            project_id = canonicalize_project_id(qualifier)
        except PathShapedProjectIdError:
            # Defensive: the pattern cannot capture a '/' or a leading '-', so
            # this is unreachable today. Kept so a future pattern relaxation
            # skips the candidate instead of raising into the write path.
            continue
        if is_task_vocabulary_qualifier(project_id):
            # 'Task: 2500' / 'subtask: 2500'. The local pass above already
            # claimed every spelling that IS a local mention, so dropping this
            # candidate loses nothing and stops 'task' becoming a project id.
            continue
        if project_id == local_project:
            # Self-qualified: a genuine LOCAL referent, not a foreign one.
            # Reclassifying rather than dropping keeps the local referent set
            # complete for consumers that need it; the foreign-only filter in
            # cross_project_refs.find_cross_project_task_refs reproduces the
            # drop that module wants.
            found.append((match.start(), Referent(kind='task', number=number), False))
            continue
        if allowlist is not None and project_id not in allowlist:
            continue
        found.append(
            (match.start(), Referent(kind='task', project_id=project_id, number=number), False)
        )

    # Merge the two passes by OFFSET rather than concatenating them, so the
    # result reads in the order a human reads the content. Sorting is stable,
    # so the rare same-offset pair keeps its discovery order.
    found.sort(key=lambda item: item[0])

    deduped: list[Referent] = []
    seen: set[tuple[str, str, str]] = set()
    for _offset, referent, _arrived_bare in found:
        key = (referent.kind, referent.project_id, referent.number)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(referent)

    # A number claimed BOTH by a BARE, unqualified mention and by a
    # foreign-qualified reference cannot say which task the facts belong to.
    #
    # Only an UNQUALIFIED mention creates a contest. Ambiguity is a property of
    # what the prose FAILED to say: an explicitly project-qualified reference
    # already names its project, so two qualified references — even when one of
    # them is SELF-qualified and was reclassified into an own-project referent
    # above — are never mutually ambiguous. Contesting on own-project-ness
    # instead would make 'reify:2500 relates to dark_factory:2500' ambiguous in
    # group reify, suppressing a real cross-project repair that nothing in the
    # content actually cast doubt on.
    #
    # The partition is SYMMETRIC: when a contest IS real, both referents are
    # derived from the same ambiguous prose, so handing a consumer the local one
    # as clean evidence would be a confidently-wrong answer, not a safe
    # fallback. Refuse both rather than pick either.
    #
    # It is per-NUMBER, not per-content, so one contested number never
    # suppresses an unrelated clean ref elsewhere in the same body.
    #
    # The contest is keyed on (kind, number) — the same tuple dedup uses, minus
    # project_id, which is exactly the axis being contested. Keying on the bare
    # number alone happens to agree today only because _KIND_LABELS holds one
    # entry; the moment 'escalation' joins the registry, a bare 'escalation
    # 2500' mention would contest an unrelated foreign TASK ref
    # 'dark_factory:2500' and suppress a real repair. That failure could not be
    # caught by any test, since no test can construct a second kind, so the
    # guard has to be structural.
    #
    # bare_keys is computed over the PRE-dedup `found` list, NOT over
    # `deduped`, because bare-ness must be STICKY per number. Dedup is
    # first-seen-wins on (kind, project_id, number), so in
    # 'reify:2500 and task 2500 and dark_factory:2500' the self-qualified
    # spelling arrives first and the surviving own-project referent is the one
    # that arrived QUALIFIED — reading bare-ness off `deduped` would silently
    # lose a genuine contest the content does contain. (Dedup still collapses
    # 'reify:5181 and task 5181' to one own-project referent; that pair simply
    # has no foreign side to contest.)
    bare_keys = {(r.kind, r.number) for _offset, r, arrived_bare in found if arrived_bare}
    foreign_keys = {(r.kind, r.number) for r in deduped if r.project_id}
    contested = bare_keys & foreign_keys

    # Digits are compared as literals, never int-normalized, so '0250' and
    # '250' are different numbers and create no false contest.
    return LabelScan(
        refs=tuple(r for r in deduped if (r.kind, r.number) not in contested),
        ambiguous=tuple(r for r in deduped if (r.kind, r.number) in contested),
    )
