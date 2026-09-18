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
    'MalformedDisposition',
    'Policy',
    'TaskRef',
    'TicketRef',
]

# Crockford base32 — the alphabet fused-memory/src/fused_memory/middleware/
# ticket_store.py::_new_ticket_id mints from (0-9 and A-Z minus I, L, O, U,
# the four characters Crockford drops because they are read as 1, 1, 0 and V).
# The body length is deliberately NOT pinned: _new_ticket_id emits 26
# characters today, and pinning that would turn a future change there into a
# vocabulary violation at every disposition site in the repo.
_TICKET_ID_RE = re.compile(r'^tkt_[0-9ABCDEFGHJKMNPQRSTVWXYZ]+$')

# Kebab-case, as D3 requires of every ratification row id: lower-case
# alphanumeric segments joined by single hyphens, with no leading, trailing or
# doubled hyphen.
_POLICY_ID_RE = re.compile(r'^[a-z0-9]+(-[a-z0-9]+)*$')


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
