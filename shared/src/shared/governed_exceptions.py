"""INV-12's one disposition vocabulary and its one inline-marker parser.

**What this module is.**  INV-12 ("exceptions are owned or ratified") says
every entry of a governed exception list — every entry that silences a
detector — carries a *disposition*: a named owner who will remove it, or a
recorded operator ruling that it stays.  This module is where that vocabulary
is defined, and where the inline grammar that spells it in a comment is
parsed.  ``plans/inv12-exceptions-owned-or-ratified-prd.md``, Contract section
**Vocabulary**, decision **D2**.

**Why it lives in ``shared/src`` rather than a script.**  The declarations D4
places in each package's test tree (``orchestrator/tests/…``,
``fused-memory/tests/…``, ``shared/tests/…``) all import this vocabulary, and
so do the register and the scanner that read them.  A module under
``scripts/`` would be reachable by a bare-``python3`` checker but not by six
packages' suites; a module in one package's tests would be reachable by that
package alone.  ``shared`` is the only home every consumer can already import.

**What it deliberately does NOT do.**

* *No ratification-table load.*  ``Policy('some-row')`` constructs whether or
  not ``docs/legibility/exception-ratifications.yaml`` has that row.  D3 makes
  Policy a closed world checked by the register, hermetically and in one
  place; a dataclass that consulted the table would need a filesystem, would
  make the closed-world rule live in two homes, and would stop an agent
  writing down the very policy an operator is about to ratify.
* *No knowledge of suppression kinds.*  ``type: ignore``, ``noqa``,
  ``pyright: ignore``, ``pragma: no cover`` and ``nosec`` are D8's consumer
  model, which lives in the scanner.  :func:`parse_disposition_marker`
  therefore parses the disposition grammar and nothing else — see its
  docstring for the boundary and why it is drawn there.
* *No filesystem, and no import-time work at all.*  D4 requires that
  production modules and stdlib-only scripts gain no import-time raise from
  INV-12: a mis-edited disposition must fail a test, never an import.  This
  module's body defines constants, one exception family and the dataclasses,
  and runs nothing.

**Consumers.**  The declaration sites in every package's test tree; the static
register that reads those declarations by AST; the inline-suppression scanner
that reads markers from source; the nightly sweep that follows dead owners;
and the implementer-facing prompt block that publishes the accepted forms.
All of them render :data:`INLINE_MARKER_FORMS` or :data:`DECLARATION_FORMS`
rather than retyping the grammar.

Intentionally NOT re-exported from ``shared/__init__.py``.  Consumers import
via the fully-qualified path (``from shared.governed_exceptions import …``),
consistent with the ``task_statuses`` / ``mcp_envelope`` / ``neutral_cwd`` /
``config_dir`` sub-module convention that ``shared/tests/test_public_api.py``
pins.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

__all__ = [
    'DECLARATION_FORMS',
    'Debt',
    'Disposition',
    'GovernedList',
    'INLINE_MARKER_FORMS',
    'MalformedDeclaration',
    'MalformedDisposition',
    'Policy',
    'TaskRef',
    'TicketRef',
    'UndisposedException',
    'governed_exceptions',
    'parse_disposition_marker',
]

# A dotted, globally unique list id: non-empty segments joined by single dots.
#
# THE THREE ID VALIDATORS ARE APPLIED WITH `fullmatch`, NEVER `match` against an
# anchored pattern.  Python's `$` also matches immediately before a TRAILING
# NEWLINE, so an anchored `match` accepts 'inv12-x\n' and 'tkt_ABC\n' — ids
# carrying an invisible character, which then silently fail to match the
# ratification row or the ticket they name, and which defeat the global
# uniqueness the dotted list_id exists to give.  The patterns are therefore left
# unanchored and the whole-value requirement lives at the call site, the
# spelling `shared/src/shared/memory_eval_metrics.py::_STAMP_PATTERN` uses for
# the same reason.  The MARKER patterns further down are a separate case: they
# end `\s*$` deliberately, because trailing whitespace around a source comment
# is not part of the operand.
_LIST_ID_RE = re.compile(r'[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+')

# THE OPERAND PATTERNS, each written once and consumed twice — by the
# dataclass that validates a hand-written value, and by the marker form that
# recognises the same operand in a comment.  They are shared as pattern
# STRINGS rather than duplicated because a second copy of "what a ticket id
# looks like" would drift from the first, which is the whole reason the
# vocabulary is one module.

# A decimal task numeral, with no leading zero: `007` is not how anyone writes
# a task id, and admitting it would let two spellings of task 7 key the same
# debt.  This one has no dataclass twin — TaskRef takes an int, where the
# question of leading zeros does not arise — so the LEXICAL rule lives here and
# the SEMANTIC rule (positive, not a bool) lives in TaskRef.
_TASK_ID_PATTERN = r'[1-9][0-9]*'

# Crockford base32 — the alphabet fused-memory/src/fused_memory/middleware/
# ticket_store.py::_new_ticket_id mints from (0-9 and A-Z minus I, L, O, U,
# the four characters Crockford drops because they are read as 1, 1, 0 and V).
# The body length is deliberately NOT pinned: _new_ticket_id emits 26
# characters today, and pinning that would turn a future change there into a
# vocabulary violation at every disposition site in the repo.
_TICKET_ID_PATTERN = r'tkt_[0-9ABCDEFGHJKMNPQRSTVWXYZ]+'

# Kebab-case, as D3 requires of every ratification row id: lower-case
# alphanumeric segments joined by single hyphens, with no leading, trailing or
# doubled hyphen.
_POLICY_ID_PATTERN = r'[a-z0-9]+(?:-[a-z0-9]+)*'

# Unanchored on purpose, and applied with `fullmatch`; see _LIST_ID_RE above.
_TICKET_ID_RE = re.compile(_TICKET_ID_PATTERN)
_POLICY_ID_RE = re.compile(_POLICY_ID_PATTERN)

# The keyword probe.  Case-sensitive on purpose: the grammar D6 publishes is
# lower-case, so `# DEBT: task 5601` is not a marker at all and must return
# None rather than raise — otherwise every comment beginning with the word
# DEBT becomes an instrument failure.
_MARKER_KEYWORD_RE = re.compile(r'#\s*(?:debt|ratified):')

# One anchored form per INLINE_MARKER_FORMS entry, matched against the comment
# from the keyword onward.  The trailing `$` is what rejects extra prose after
# a disposition: `# debt: task 5601 (see also 5602)` is a note, not a marker,
# and honouring its prefix would silently disposition an entry by half a
# sentence.
_DEBT_TASK_RE = re.compile(rf'^#\s*debt:\s*task\s+({_TASK_ID_PATTERN})\s*$')
_DEBT_TICKET_RE = re.compile(rf'^#\s*debt:\s*ticket\s+({_TICKET_ID_PATTERN})\s*$')
_RATIFIED_RE = re.compile(rf'^#\s*ratified:\s*({_POLICY_ID_PATTERN})\s*$')

INLINE_MARKER_FORMS: tuple[str, ...] = (
    '# debt: task <task id>',
    '# debt: ticket tkt_<ticket id>',
    '# ratified: <ratification row id>',
)
"""The accepted inline forms, published once for every consumer to render.

What must not be retyped is the GRAMMAR.  Presentation legitimately differs
between :class:`MalformedDisposition`'s message (which renders these when a
marker does not parse), the inline-suppression scanner's per-line rejection
output, and the implementer-facing prompt block that tells an agent how to
spell a disposition before the gate does.  A single pre-rendered message block
would force all three into one layout and invite a second copy; a tuple of
forms lets each render its own and keeps one home for the grammar itself.

``shared/tests/test_governed_exceptions.py`` fills every form from a
placeholder table and feeds it to :func:`parse_disposition_marker`, so a form
published here that nothing implements — or an implemented form nobody
published — turns that suite red."""


class MalformedDisposition(Exception):
    """A disposition value or inline marker does not have a legal shape.

    Deliberately NOT a ``ValueError`` subclass, and deliberately sharing no
    base class with this module's other faults.  The three map to different
    exit codes and different operator actions — a malformed disposition is
    fixed at the site (exit 1), an undisposed entry is fixed by adding a
    disposition (exit 1), a malformed declaration is an instrument failure
    (exit 2) — so a single broad ``except`` must not be able to conflate them.
    The cost of a wrong catch here is a silently swallowed INV-12 breach.

    Attributes:
        value: The offending value, verbatim, for the caller to render.
    """

    def __init__(self, message: str, *, value: object) -> None:
        self.value = value
        super().__init__(message)


@dataclass(frozen=True)
class TaskRef:
    """A task id owning a piece of debt.

    Attributes:
        id: A positive task id.  ``bool`` is rejected even though it is an
            ``int`` subclass, so ``TaskRef(True)`` can never render as task 1.
    """

    id: int

    def __post_init__(self) -> None:
        if type(self.id) is not int:
            raise MalformedDisposition(
                f'TaskRef: id={self.id!r} must be an int, not {type(self.id).__name__} — '
                'a task id is a number, and bool is excluded even though it is an int '
                'subclass (True would otherwise render as task 1).',
                value=self.id,
            )
        if self.id < 1:
            raise MalformedDisposition(
                f'TaskRef: id={self.id!r} must be at least 1 — task ids start at 1, '
                'so a zero or negative id names no task and can never be followed up.',
                value=self.id,
            )


@dataclass(frozen=True)
class TicketRef:
    """A ticket id owning a piece of debt.

    D2's reason this type exists at all: no agent role holds ``resolve_ticket``
    (``orchestrator/src/orchestrator/agents/roles.py``), so an agent that files
    follow-up work via ``submit_task`` learns a ticket id and never a task id.
    Without ``TicketRef`` it would have nothing well-formed to write down.

    Attributes:
        id: The literal ``tkt_`` prefix plus one or more Crockford base32
            characters.  The minted length is not pinned; see
            :data:`_TICKET_ID_RE`.
    """

    id: str

    def __post_init__(self) -> None:
        if not isinstance(self.id, str):
            raise MalformedDisposition(
                f'TicketRef: id={self.id!r} must be a str, not {type(self.id).__name__}.',
                value=self.id,
            )
        if not _TICKET_ID_RE.fullmatch(self.id):
            raise MalformedDisposition(
                f'TicketRef: id={self.id!r} is not a ticket id — expected the literal '
                "prefix 'tkt_' followed by one or more Crockford base32 characters "
                '(0-9 and A-Z excluding I, L, O and U), as ticket_store.py mints them.',
                value=self.id,
            )


@dataclass(frozen=True)
class Policy:
    """An operator ruling that an entry stays.

    Attributes:
        ratified: The kebab-case id of a row in
            ``docs/legibility/exception-ratifications.yaml``.  Whether the row
            EXISTS is D3's closed-world check, performed by the register; this
            type checks only that the id is well formed, so that an agent can
            write down a policy the operator is about to ratify.
    """

    ratified: str

    def __post_init__(self) -> None:
        if not isinstance(self.ratified, str):
            raise MalformedDisposition(
                f'Policy: ratified={self.ratified!r} must be a str, '
                f'not {type(self.ratified).__name__}.',
                value=self.ratified,
            )
        if not _POLICY_ID_RE.fullmatch(self.ratified):
            raise MalformedDisposition(
                f'Policy: ratified={self.ratified!r} is not a ratification row id — '
                'expected kebab-case (lower-case alphanumeric segments joined by single '
                'hyphens, no leading, trailing or doubled hyphen).',
                value=self.ratified,
            )


@dataclass(frozen=True)
class Debt:
    """A named owner who will remove the entry.

    Attributes:
        owner: A :class:`TaskRef` or a :class:`TicketRef`.  A :class:`Policy`
            is a sibling disposition, never an owner: debt is owed by someone,
            and a ruling owes nothing.
    """

    owner: TaskRef | TicketRef

    def __post_init__(self) -> None:
        if not isinstance(self.owner, TaskRef | TicketRef):
            raise MalformedDisposition(
                f'Debt: owner={self.owner!r} must be a TaskRef or a TicketRef, '
                f'not {type(self.owner).__name__} — debt names who will remove the '
                'entry, so the owner has to be something the sweep can follow.',
                value=self.owner,
            )


Disposition = Debt | Policy
"""D2's three legal states, as two types: ``Debt(TaskRef | TicketRef)`` and
``Policy(row id)``.  A bare ref is not a disposition — it answers "who", not
"why this entry is legal"."""


def parse_disposition_marker(comment: str) -> Disposition | None:
    """Parse D6's inline grammar out of one whole COMMENT token.

    THE TWO-OUTCOME CONTRACT, and it is the whole interface:

    * ``None`` — the comment carries no disposition marker.  Never a raise:
      the overwhelming majority of comments in the tree are ordinary prose,
      and a parser that raised on them would report the tree as one long
      instrument failure.
    * a :data:`Disposition` — the marker is present and well formed.
    * :class:`MalformedDisposition` — the marker is present and does not
      parse.  Present-but-broken is the case worth being loud about, because
      the author plainly meant to disposition something and the entry is
      silently undisposed until someone is told.

    That split — absent is a value, present-and-broken is a raise — is the
    same one ``shared/src/shared/deploy_state.py::DeployState.from_metadata``
    draws, for the same reason.

    WHAT THIS FUNCTION DELIBERATELY DOES NOT CHECK: that a suppression
    actually precedes the marker on the line.  D6 names two violations — "a
    marker that does not parse, or that sits on a line with no suppression" —
    and only the first is a property of the disposition grammar.  The second
    needs the kind table (``type: ignore``, ``noqa``, ``pyright: ignore``,
    ``pragma: no cover``, ``nosec``) and D8's consumer model, both of which
    live in the inline-suppression scanner.  So ``x = 1  # debt: task 5601``
    parses here and is reported there; do not assume it is already covered.

    Every value is built through :class:`TaskRef`, :class:`TicketRef` and
    :class:`Policy` rather than validated first, so the operand rules have one
    home — the same home reached by the declarations that construct
    dispositions by hand with no parser in the path.
    """
    keywords = list(_MARKER_KEYWORD_RE.finditer(comment))
    if not keywords:
        return None

    # Hoisted so both raise sites publish the same help; the forms are rendered
    # from INLINE_MARKER_FORMS rather than retyped.
    rejected = (
        f'parse_disposition_marker: comment {comment!r} carries a disposition marker '
        'that does not parse. Accepted forms:\n  ' + '\n  '.join(INLINE_MARKER_FORMS)
    )

    if len(keywords) > 1:
        raise MalformedDisposition(
            f'{rejected}\nFound {len(keywords)} disposition keywords — one disposition '
            'covers every suppression in a comment, so exactly one is expected.',
            value=comment,
        )

    marker = comment[keywords[0].start() :]
    task = _DEBT_TASK_RE.match(marker)
    if task is not None:
        return Debt(TaskRef(int(task.group(1))))
    ticket = _DEBT_TICKET_RE.match(marker)
    if ticket is not None:
        return Debt(TicketRef(ticket.group(1)))
    policy = _RATIFIED_RE.match(marker)
    if policy is not None:
        return Policy(policy.group(1))
    raise MalformedDisposition(rejected, value=comment)


DECLARATION_FORMS: tuple[str, ...] = (
    'Debt(TaskRef(<task id>))',
    "Debt(TicketRef('tkt_<ticket id>'))",
    "Policy('<ratification row id>')",
)
"""The accepted declaration-side forms, published once for every consumer.

The twin of :data:`INLINE_MARKER_FORMS` for entries disposed in a declaration
rather than in a comment.  Rendered by :class:`UndisposedException`'s message
(so the agent reading a red gate is told how to fix it), by the register's
rejection output, and by the implementer-facing prompt block.  Nobody retypes
the grammar; ``shared/tests/test_governed_exceptions.py`` asserts the
exception's message really carries every form."""


class MalformedDeclaration(Exception):
    """A declaration is broken, so nothing about it can be judged.

    An INSTRUMENT failure — exit 2 downstream, and the operator fixes the
    declaration rather than the code it governs.  Deliberately shares no base
    class with :class:`UndisposedException`: that one is the invariant
    violation itself (exit 1), and a single ``except`` that caught both would
    report a real INV-12 breach as a broken tool, which is the conflation the
    two exit codes exist to prevent.

    Attributes:
        value: The offending value, verbatim, for the caller to render.
    """

    def __init__(self, message: str, *, value: object) -> None:
        self.value = value
        super().__init__(message)


class UndisposedException(Exception):
    """A governed entry has no disposition — the INV-12 violation itself.

    Exit 1 downstream, and the fix is an agent's: add an override, or increment
    the count beside the disposition the entry is borrowing.

    Attributes:
        list_id: The declaring list.
        keys: The undisposed keys, sorted.
        covered: How many keys actually have no override.
        default_covers: How many the declaration says the default covers, or
            ``None`` when there is no default.
    """

    def __init__(
        self,
        *,
        list_id: str,
        keys: tuple[str, ...],
        covered: int,
        default_covers: int | None,
    ) -> None:
        self.list_id = list_id
        self.keys = keys
        self.covered = covered
        self.default_covers = default_covers
        super().__init__(
            f'governed_exceptions: {list_id} declares default_covers={default_covers}, but '
            f'{covered} key(s) have no override — these are undisposed: {list(keys)}. Add an '
            'override for each, or increment default_covers beside the disposition they '
            'borrow. Accepted forms:\n  ' + '\n  '.join(DECLARATION_FORMS)
        )


@dataclass(frozen=True)
class GovernedList:
    """One governed exception list, with a disposition reachable for every key.

    A value, not a registry: it is what a declaration in a package's test tree
    evaluates to, and the report renders.  Constructed through
    :func:`governed_exceptions`, which is where every check lives — this type
    normalises and resolves, and refuses nothing.

    Attributes:
        list_id: The dotted, globally unique id of the list.
        rule: One sentence saying what the list's entries are allowed to be —
            the thing an operator judges a new entry against.
        keys: The declared entries, in declaration order.
        default: A disposition borrowed by every key without an override, or
            ``None``.
        default_covers: The literal COUNT of keys the default covers, or
            ``None``.  D4 makes it a count rather than a list on purpose: a new
            entry then needs either an explicit override or a visible increment
            beside the disposition it is borrowing.
        overrides: Per-key dispositions, read-only.  Each restates its entry's
            key, so drift between the two is loud by construction (D5).
    """

    list_id: str
    rule: str
    keys: tuple[str, ...]
    default: Disposition | None
    default_covers: int | None
    overrides: Mapping[str, Disposition]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'keys', tuple(self.keys))
        object.__setattr__(self, 'overrides', MappingProxyType(dict(self.overrides)))

    def disposition_for(self, key: str) -> Disposition | None:
        """Resolve *key*'s disposition: its override, else the list's default.

        THE SINGLE HOME of the override-else-default rule.  Every consumer that
        needs to know why an entry is legal — the report, the sweep that
        follows dead owners, the register's closed-world Policy check — asks
        here rather than reimplementing the two-line fallback, because a second
        copy that forgot the override would silently attribute an entry to the
        wrong owner.

        Raises:
            KeyError: *key* is not declared in this list.  Deliberately not a
                quiet ``None``: returning the default for an entry nobody
                declared would let a report print a disposition for something
                that does not exist.
        """
        if key not in self.keys:
            raise KeyError(
                f'GovernedList {self.list_id!r} does not declare the key {key!r}.'
            )
        return self.overrides.get(key, self.default)


def governed_exceptions(
    list_id: str,
    rule: str,
    keys: Iterable[str],
    *,
    default: Disposition | None = None,
    default_covers: int | None = None,
    dispositions: Mapping[str, Disposition] = MappingProxyType({}),
) -> GovernedList:
    """Declare a governed exception list and check every entry is disposed.

    Called at import time from a test module in the owning package's test tree
    (D4), so a mis-edited disposition fails that package's suite and nothing
    else: no production module and no stdlib-only script gains an import or an
    import-time raise from INV-12.

    D5's DIVISION OF KNOWLEDGE, which is why this function exists at all.  The
    static register reads ``list_id``, ``rule``, ``default``, ``default_covers``
    and the overrides straight out of the source, because all five are
    literals.  Only this runtime call also sees the KEY SET — the container's
    actual contents, which no AST can enumerate — so this is the one place keys
    and dispositions are reconciled, and the only place that can notice a list
    has quietly grown an entry.

    ``dispositions`` defaults to an empty ``MappingProxyType`` rather than
    ``{}``: ruff's B006 forbids a mutable default, and the immutable proxy is
    the shape the field stores anyway.

    THE ONE COUNT RULE, recorded here so a later reader does not re-split it
    into two.  Let ``covered`` be the number of keys with no override, and
    ``declared`` be ``default_covers`` when a default is present and ``0``
    otherwise.  Then:

    * ``declared == covered`` — every entry has a disposition.  Accept.
    * ``covered > declared`` — entries beyond the declared count are genuinely
      undisposed.  :class:`UndisposedException` (exit 1): an agent adds an
      override or increments the literal.
    * ``covered < declared`` — the literal over-claims.  No entry's disposition
      is broken; the list merely shrank without the count following.
      :class:`MalformedDeclaration` (exit 2): fix the instrument.

    That single invariant is what makes boundary scenario 12 — a defaulted list
    that gains a key with ``default_covers`` unchanged — fall out rather than
    be special-cased, and treating a list with no default as ``declared = 0``
    is what folds the plain missing-override case into the same arm.  Two
    separate checks with a bespoke branch for scenario 12 would put the same
    fact in two places and invite them to drift.

    Raises:
        MalformedDeclaration: the declaration itself is broken (exit 2).
        UndisposedException: the declaration is well formed and an entry has no
            disposition (exit 1).  The two are checked in that order, and the
            order is load-bearing: a duplicate key makes every count downstream
            of it meaningless, so reporting undisposed keys first would hand
            back a wall of noise whose one real cause is a line the reader can
            see.
    """
    if not isinstance(list_id, str) or not _LIST_ID_RE.fullmatch(list_id):
        raise MalformedDeclaration(
            f'governed_exceptions: list_id={list_id!r} must be a dotted, globally unique '
            "id such as 'orchestrator.tests.timeout_marker_grandfathered' — non-empty "
            'segments joined by single dots.',
            value=list_id,
        )
    if not isinstance(rule, str) or not rule.strip():
        raise MalformedDeclaration(
            f'governed_exceptions: {list_id} declares rule={rule!r}, but the rule is the '
            'one sentence an operator judges a NEW entry against, so it cannot be blank.',
            value=rule,
        )

    for key, disposition in dispositions.items():
        if not isinstance(disposition, Debt | Policy):
            raise MalformedDeclaration(
                f'governed_exceptions: {list_id} gives key {key!r} the override '
                f'{disposition!r}, which is not a Debt or a Policy. Accepted forms:\n  '
                + '\n  '.join(DECLARATION_FORMS),
                value=disposition,
            )
    if default is not None and not isinstance(default, Debt | Policy):
        raise MalformedDeclaration(
            f'governed_exceptions: {list_id} declares default={default!r}, which is not a '
            'Debt or a Policy. Accepted forms:\n  ' + '\n  '.join(DECLARATION_FORMS),
            value=default,
        )

    declared_keys = tuple(keys)
    duplicates = sorted(key for key, count in Counter(declared_keys).items() if count > 1)
    if duplicates:
        raise MalformedDeclaration(
            f'governed_exceptions: {list_id} declares {duplicates} more than once. A key '
            'appears once, so that its disposition has one home and the counts below mean '
            'something.',
            value=duplicates,
        )

    strays = sorted(set(dispositions) - set(declared_keys))
    if strays:
        raise MalformedDeclaration(
            f'governed_exceptions: {list_id} overrides {strays}, which the list does not '
            'declare. An override restates its entry key, so a stray one means the entry '
            'was removed or renamed and its disposition was left behind.',
            value=strays,
        )

    if (default is None) != (default_covers is None):
        raise MalformedDeclaration(
            f'governed_exceptions: {list_id} declares default={default!r} and '
            f'default_covers={default_covers!r}; both are present or both are absent. A '
            'default with no count would cover an unbounded number of future entries '
            'silently, which is the whole thing D4 makes the count a literal to prevent.',
            value=(default, default_covers),
        )

    covered = len(declared_keys) - len(dispositions)
    # A list with no default declares that it covers nothing by default, which is
    # what folds the plain missing-override case into the same arm as scenario 12.
    # Both-or-neither is enforced just above, so keying on default_covers here is
    # the same predicate as keying on default — and the one a type checker follows.
    declared = 0 if default_covers is None else default_covers
    if covered == declared:
        return GovernedList(
            list_id=list_id,
            rule=rule,
            keys=declared_keys,
            default=default,
            default_covers=default_covers,
            overrides=dispositions,
        )

    if covered < declared:
        raise MalformedDeclaration(
            f'governed_exceptions: {list_id} declares default_covers={default_covers} but '
            f'only {covered} key(s) have no override. The list shrank and the literal was '
            'not decremented — every remaining entry still has a disposition, so this is a '
            'stale count, not a violation.',
            value=default_covers,
        )

    raise UndisposedException(
        list_id=list_id,
        keys=tuple(sorted(set(declared_keys) - set(dispositions))),
        covered=covered,
        default_covers=default_covers,
    )
