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
from dataclasses import dataclass

__all__ = [
    'Debt',
    'Disposition',
    'INLINE_MARKER_FORMS',
    'MalformedDisposition',
    'Policy',
    'TaskRef',
    'TicketRef',
    'parse_disposition_marker',
]

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

_TICKET_ID_RE = re.compile(f'^{_TICKET_ID_PATTERN}$')
_POLICY_ID_RE = re.compile(f'^{_POLICY_ID_PATTERN}$')

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
        if not _TICKET_ID_RE.match(self.id):
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
        if not _POLICY_ID_RE.match(self.ratified):
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
