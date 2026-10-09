"""Decide each inline suppression site's ownership under a given class table.

A site is owned by an inline disposition (D6), ratified by a class row (D9), or
unowned, and only an unowned site contributes a key to the ratchet's multiset.
:func:`classify` takes the class table as an argument and reads no module
global; the operator's table lives in the entry module, which hands it in.
:func:`classify` also collects the two disposition faults D6 names.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from inline_suppression_consumers import Consumer, ConsumerModel
from inline_suppression_key import key_for
from inline_suppression_kinds import Kind, Site
from inline_suppression_refusal import import_shared
from inline_suppression_scan import Scan

if TYPE_CHECKING:
    from shared.governed_exceptions import Disposition, Policy


# D9 — the key a ratified class is written in.


class Scope(Enum):
    """Where a ratified class applies.

    ``SRC`` is defined as "not tests" rather than as its own predicate, so the
    two are a PARTITION and no path can fall outside both.  Two independent
    predicates could each miss a path, and a class row that silently covered
    nothing is the failure nobody would notice.
    """

    SRC = 'src'
    TESTS = 'tests'
    ANY = 'any'

    def covers(self, path: str) -> bool:
        """Whether *path* is in this scope.

        A path is ``tests`` iff one of its COMPONENTS is ``tests`` — not a
        filename pattern.
        """
        if self is Scope.ANY:
            return True
        in_tests = 'tests' in Path(path).parts
        return in_tests if self is Scope.TESTS else not in_tests


@dataclass(frozen=True)
class SuppressionClass:
    """D9's key: one kind, one code, one scope.

    The operator's table, ``scripts/inline_suppressions.py::RATIFIED_SUPPRESSION_CLASSES``,
    is keyed by this TYPE rather than by its rendering, so the operator's rows
    are type-checked at import and a malformed row cannot
    masquerade as a class nobody happens to match.  :meth:`render` exists only
    because the report publishes the key as a string.

    Attributes:
        kind: The suppression kind the row covers.
        code: The single rule code, or ``''`` for the codeless form — which
            matches a BARE marker rather than matching everything, because a
            row covering every code of a kind is a far bigger valve than D9
            describes and should be written out if it is ever wanted.
        scope: Where it applies.
    """

    kind: Kind
    code: str
    scope: Scope

    def render(self) -> str:
        """D9's published spelling, ``kind[code]@scope``."""
        return f'{self.kind.value}[{self.code}]@{self.scope.value}'

    def covers(self, site: Site) -> bool:
        """Whether this row ratifies *site*."""
        matches_code = self.code in site.codes if self.code else not site.codes
        return self.kind is site.kind and matches_code and self.scope.covers(site.path)


# ---------------------------------------------------------------------------
# Classification.


class Ownership(Enum):
    """How a site is accounted for.

    :attr:`UNOWNED` is the only value that contributes a key to the multiset;
    the other three are the three ways of being answered for.
    """

    DEBT = 'debt'
    POLICY = 'policy'
    CLASS = 'class'
    UNOWNED = 'unowned'


@dataclass(frozen=True)
class Classified:
    """One site, and everything the pipeline decided about it."""

    site: Site
    consumer: Consumer
    ownership: Ownership
    disposition: Disposition | None = None
    suppression_class: SuppressionClass | None = None


@dataclass(frozen=True)
class Violation:
    """One exit-1 finding, rendered as one line.

    Attributes:
        path: The file, repo-relative.
        line: 1-based physical line.
        kind: The suppression kind, or ``None`` when the finding is about a
            comment that carries no suppression at all.
        codes: The rule codes, for the reader who needs to find the marker.
        reason: What is wrong, in one clause.
        forms: The accepted disposition forms to publish, rendered from
            ``shared.governed_exceptions.INLINE_MARKER_FORMS`` rather than
            retyped.  EMPTY when the finding accepts no disposition — a
            consumer-less marker is deleted, not dispositioned, so offering
            forms there would be advice that does not work.
    """

    path: str
    line: int
    kind: Kind | None
    codes: tuple[str, ...]
    reason: str
    forms: tuple[str, ...] = ()

    def render(self) -> str:
        """One line: where, what, why, and how to fix it."""
        codes = f'[{",".join(self.codes)}]' if self.codes else ''
        marker = f'{self.kind.value}{codes}' if self.kind is not None else 'no suppression'
        accepted = f' Accepted forms: {" | ".join(self.forms)}' if self.forms else ''
        return f'{self.path}:{self.line}: {marker} -- {self.reason}{accepted}'

    def record(self) -> dict[str, object]:
        """The same finding as structured data, for ``--json``.

        Not the rendered line re-parsed, and not a second statement of what a
        finding IS: both come off the same fields, so a consumer reads values
        where a human reads a sentence and neither has to parse the other
        (heuristic 12).
        """
        return {
            'path': self.path,
            'line': self.line,
            'kind': None if self.kind is None else self.kind.value,
            'codes': list(self.codes),
            'reason': self.reason,
            'forms': list(self.forms),
        }


@dataclass(frozen=True)
class Classification:
    """The whole tree, classified.

    Attributes:
        classified: One entry per site, in scan order.
        violations: The exit-1 findings that do not depend on the baseline —
            the two disposition faults D6 names.  Ratchet violations are the
            ``--check`` verb's and are computed against the baseline.
        unowned: Rendered key to the entries that produced it, sorted by key.
            The COUNTS are derived from this rather than tracked beside it
            (heuristic 11): ``--check`` needs the entries behind an excess key
            in order to name their lines and to say why each one is a finding,
            and a count kept separately could drift from the list.
    """

    classified: tuple[Classified, ...]
    violations: tuple[Violation, ...]
    unowned: Mapping[str, tuple[Classified, ...]]

    @property
    def counts(self) -> dict[str, int]:
        """The multiset :class:`shared.ratchet.Enumeration` compares."""
        return {key: len(entries) for key, entries in self.unowned.items()}


def classify(
    scan: Scan, model: ConsumerModel, *, classes: Mapping[SuppressionClass, Policy]
) -> Classification:
    """Decide every site's ownership, and collect D6's two disposition faults.

    *classes* is the ratified class table this run classifies against.  It is
    required, so this reads no module global and every caller states the table
    it means.

    THE ORDER IS FIXED — consumer, then ratified class, then inline
    disposition — and it is the single mechanism that makes D8's "accepts no
    disposition" true.  A site nothing consumes is unowned whatever is written
    beside it, so boundary scenario 9 ("a disposition does not rescue it")
    falls out of the pipeline instead of needing a special case beside the
    disposition check.  It simultaneously satisfies D8's "grandfathered dead
    markers stay counted": such a site is in the multiset, so a baseline that
    already holds it yields no excess and the gate is green, while the report
    still counts it.

    The class table is second rather than first only because D8's prohibition
    is categorical whereas D9's valve is the operator's; recording that order is
    what stops a later reader flipping it by accident.

    A MALFORMED MARKER IS A VIOLATION AND LEAVES ITS SITES UNOWNED.  Both are
    true at once and neither substitutes for the other: the broken marker is a
    fault at the site (baseline-independent, as
    ``scripts/inline_suppressions.py::_verdict`` records), and
    the sites it failed to disposition are genuinely undisposed.

    THE PREFILTER BOUNDS THE SUPPRESSION-FREE-DISPOSITION FINDING: this layer
    sees only the comments of files whose raw bytes carried a KIND marker, so a
    disposition alone in a file with no suppression anywhere is never read
    (``scripts/tests/test_inline_suppression_classify.py::test_a_disposition_on_a_line_with_no_suppression_is_a_violation``
    says why that limit is the harmless side).
    """
    governed = import_shared('governed_exceptions')

    classified: list[Classified] = []
    violations: list[Violation] = []
    unowned: dict[str, list[Classified]] = {}

    for comment in scan.comments:
        try:
            disposition = governed.parse_disposition_marker(comment.text)
        except governed.MalformedDisposition as exc:
            disposition = None
            violations.append(
                Violation(
                    path=comment.path,
                    line=comment.line,
                    kind=None,
                    codes=(),
                    reason=(
                        f'the disposition marker in {comment.text!r} does not parse, so '
                        f'nothing here is dispositioned ({exc.__class__.__name__}).'
                    ),
                    forms=governed.INLINE_MARKER_FORMS,
                )
            )
        if disposition is not None and not comment.sites:
            violations.append(
                Violation(
                    path=comment.path,
                    line=comment.line,
                    kind=None,
                    codes=(),
                    reason=(
                        f'{comment.text!r} carries a disposition but the line holds no '
                        'suppression for it to answer for. A disposition names why a '
                        'silenced detector stays silent; with nothing silenced it says '
                        'nothing, and it will not be followed up. Remove it, or put it '
                        'on the line that carries the marker.'
                    ),
                )
            )

        for site in comment.sites:
            entry = _classify_site(site, disposition, model, classes)
            classified.append(entry)
            if entry.ownership is Ownership.UNOWNED:
                unowned.setdefault(key_for(site).render(), []).append(entry)

    return Classification(
        classified=tuple(classified),
        violations=tuple(violations),
        unowned=MappingProxyType(
            {key: tuple(entries) for key, entries in sorted(unowned.items())}
        ),
    )


def _classify_site(
    site: Site,
    disposition: Disposition | None,
    model: ConsumerModel,
    classes: Mapping[SuppressionClass, Policy],
) -> Classified:
    """One site, through the fixed order :func:`classify` documents."""
    consumer = model.consumer_for(site)
    if consumer is Consumer.NONE:
        return Classified(site=site, consumer=consumer, ownership=Ownership.UNOWNED)

    row = next(
        (row for row in classes if row.covers(site)),
        None,
    )
    if row is not None:
        return Classified(
            site=site,
            consumer=consumer,
            ownership=Ownership.CLASS,
            suppression_class=row,
        )

    if disposition is None:
        return Classified(site=site, consumer=consumer, ownership=Ownership.UNOWNED)

    return Classified(
        site=site,
        consumer=consumer,
        ownership=(
            Ownership.DEBT
            if isinstance(disposition, import_shared('governed_exceptions').Debt)
            else Ownership.POLICY
        ),
        disposition=disposition,
    )
