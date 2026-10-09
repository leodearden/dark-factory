"""INV-12's inline-suppression scanner, consumer model and multiset ratchet.

**What this is.**  Every ``# type: ignore``, ``# noqa``, ``# pyright: ignore``,
``# pragma: no cover`` and ``# nosec`` in the tracked Python of this repository
silences a detector.  INV-12 ("exceptions are owned or ratified") says each one
carries a *disposition* — a named owner who will remove it, or a recorded
operator ruling that it stays.  This module finds them, decides which are
owned, and ratchets the rest so the population can shrink but never grow.
``plans/inv12-exceptions-owned-or-ratified-prd.md``, decisions **D6**-**D9**
and **D12**.

**Layer order**, each layer reading only the layers listed before it:
refusal → kinds → scan → key → consumers → classify → ratchet/CLI.

**The key** (D7) is ``scripts/inline_suppression_key.py``'s subject.

**Exit codes** are the PRD Contract's 0/1/2 ladder, stated once in
:data:`_EPILOG` so that ``--help`` prints it.

**What it deliberately does NOT do.**

* *It commits no baseline.*  ``scripts/inline_suppression_baseline.json`` is
  seeded once, on main, at task 5607's operator cutover — not by this module's
  author and not by a test.  Until it exists the ratchet is not enforced and
  every run says ``advisory``.
* *It rules on nothing.*  :data:`RATIFIED_SUPPRESSION_CLASSES` is the
  operator's.
* *It parses no disposition grammar of its own.*  D6's forms live in
  ``shared.governed_exceptions`` and are reached only through
  ``parse_disposition_marker``.
* *It models no suppression kind outside*
  ``scripts/inline_suppression_kinds.py::KIND_SPECS``, whose docstring names
  the extension point.

**Authoring rule.**  Every suppression or disposition marker this scanner's
source mentions is written in a docstring or another string literal, never in
a ``#`` comment.  A comment quoting one IS one: ruff parses it and warns that
the directive is invalid, and this scanner reads it as a site of its own
corpus.  A docstring is a STRING, which is precisely what the token walk tells
apart.
``scripts/tests/test_inline_suppression_ratchet.py::test_the_live_tree_carries_no_disposition_faults``
enforces the rule tree-wide for disposition markers, and the ratchet enforces
it for suppression markers once the baseline is seeded.

**Consumers.**  Task 5602's exception register reads ``--json`` for the
closed-world check of inline ``ratified:`` ids and for the per-``(kind, code)``
table it renders; task 5607 runs ``--seed`` once at the cutover; task 5609 runs
``--tighten`` after the rulings land; task 5611's integration gate reads the
report.  ``--json`` is the report's only data source, so nothing downstream
re-implements the scan.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import TYPE_CHECKING

from inline_suppression_consumers import Consumer, ConsumerModel
from inline_suppression_key import key_for, key_params
from inline_suppression_kinds import Kind, Site
from inline_suppression_refusal import InstrumentFailure, import_shared
from inline_suppression_scan import Scan, scan_tree

if TYPE_CHECKING:
    from shared.governed_exceptions import Debt, Disposition, Policy
    from shared.ratchet import Enumeration

#: This checkout, resolved from ``__file__`` for the reason the bootstrap
#: comment above ``scripts/inline_suppression_refusal.py::_SHARED_SRC`` gives.
_REPO_ROOT = Path(__file__).resolve().parents[1]

#: How this scanner names itself in a refusal, so a reader of a bare exit 2 in
#: a merge log knows which instrument spoke.
_PROG = 'inline_suppressions.py'


# ---------------------------------------------------------------------------
# D9 — ratified classes, the pressure valve the OPERATOR holds.


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

    The table below is keyed by this TYPE rather than by its rendering, so the
    operator's rows are type-checked at import and a malformed row cannot
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
# Layer 3 — classification.


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
    fault at the site (baseline-independent, as :func:`_verdict` records), and
    the sites it failed to disposition are genuinely undisposed.

    THE PREFILTER BOUNDS THE SUPPRESSION-FREE-DISPOSITION FINDING: this layer
    sees only the comments of files whose raw bytes carried a KIND marker, so a
    disposition alone in a file with no suppression anywhere is never read
    (``scripts/tests/test_inline_suppression_ratchet.py::test_a_disposition_on_a_line_with_no_suppression_is_a_violation``
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


# ---------------------------------------------------------------------------
# Layer 4 — the ratchet: the baseline, the verbs and the exit ladder.

#: Where the committed baseline lives, relative to the repository root.  Named
#: here and nowhere else, so ``--baseline`` has a default that cannot drift from
#: the path task 5607 seeds and the merge gate reads.
BASELINE_PATH = 'scripts/inline_suppression_baseline.json'

RATIFIED_SUPPRESSION_CLASSES: Mapping[SuppressionClass, Policy] = MappingProxyType({})
"""Blanket rulings, keyed by class — SHIPPED EMPTY, and the OPERATOR's (PRD D9).

Rows are ruled by task 5603 and applied by task 5609.  An implementer adding a
row here would be ratifying a blanket exception on the operator's behalf, which
is the one thing D9 reserves.  :func:`main` hands it to :func:`classify`, and
nothing reads it as a global.
"""


#: The ``--json`` report's own schema version, and deliberately NOT
#: ``shared.ratchet.SCHEMA_VERSION`` — that one versions the BASELINE FILE's
#: format, which changes for entirely different reasons.  Two orthogonal schemas
#: get two numbers, so a report-shape change never reads as a baseline-format
#: change to the consumer that sees it.
REPORT_SCHEMA_VERSION = 1

class Status(Enum):
    """Which green a zero exit is — three states that are not interchangeable.

    :attr:`CLEAN` is the only one that means the WHOLE tree was measured against
    a real baseline and nothing was in excess.  The other two are green with a
    stated limit, and the label says which; boundary scenario 11's test says why
    a silent green would mislead.

    Red has no label here, and deliberately so: a run with violations reports
    how many, because a status word beside a finding would read as a verdict on
    the tree rather than on the run's reach.
    """

    CLEAN = 'clean'
    PARTIAL = 'partial'
    ADVISORY = 'advisory'


@dataclass(frozen=True)
class Request:
    """One invocation's resolved inputs, shared by every verb.

    Attributes:
        root: The checkout under measurement.
        baseline: The baseline file this run compares against or writes.
        classes: The ratified class table this run classifies against.
        scope: The positional ``PATH`` arguments, empty for a whole-tree run.
    """

    root: Path
    baseline: Path
    classes: Mapping[SuppressionClass, Policy]
    scope: tuple[str, ...] = ()

    @property
    def scoped(self) -> bool:
        """Whether this run measured only part of the tree."""
        return bool(self.scope)


def _status(request: Request) -> Status:
    """Which green *request*'s zero would be.

    ABSENCE IS DECIDED HERE, BY AN EXPLICIT EXISTENCE CHECK, and never by
    catching ``shared.ratchet.BaselineUnusable``.  That kernel refusal collapses
    absent, undecodable, unparseable, misshapen and wrong-schema into one case
    because they mean one thing to its callers; this consumer is the one place
    where they do not.  D12 makes absence a legitimate unseeded state, while a
    baseline that exists and cannot be read is a broken instrument.  Reaching the
    advisory path by catching the refusal would report a corrupt or truncated
    baseline as a clean tree, which is the silent fail-soft an empty baseline
    causes (INV-11).

    ABSENCE OUTRANKS SCOPE.  A scoped run with no baseline is enforcing nothing
    at all, which is the stronger of the two limits and therefore the one worth
    the label.
    """
    if not request.baseline.exists():
        return Status.ADVISORY
    return Status.PARTIAL if request.scoped else Status.CLEAN


@dataclass(frozen=True)
class Measurement:
    """One run's scan, its classification, and the model that resolved it.

    The model travels with the other two because ``--json`` publishes the
    configs it consulted; a report that re-resolved them could publish a
    different answer than the one the classification used.
    """

    scan: Scan
    classification: Classification
    model: ConsumerModel


def _measure(request: Request) -> Measurement:
    """Scan and classify *request*'s tree — the work every verb starts with."""
    scan = scan_tree(request.root, scope=request.scope)
    model = ConsumerModel(request.root)
    return Measurement(
        scan=scan,
        classification=classify(scan, model, classes=request.classes),
        model=model,
    )


def _enumeration(classification: Classification, kernel: ModuleType) -> Enumeration:
    """The multiset the kernel compares, under this scan's params.

    ``complete=True`` is asserted rather than computed because this scanner has
    no partial-success mode to report: every unreadable file has already been
    raised as an :class:`InstrumentFailure`, so a scan that returns at all read
    everything it enumerated.  The kernel's ``unreadable`` list is therefore
    always empty here, and the field that would carry names has none to carry.
    """
    return kernel.Enumeration(counts=classification.counts, params=key_params(), complete=True)


#: Why an undisposed suppression is a finding, in one clause.  The forms are
#: rendered beside it from the published grammar, never retyped.
_UNDISPOSED_REASON = (
    'this suppression is not in the baseline and names no owner. INV-12 requires every '
    'silenced detector to carry a disposition -- who will remove it, or the operator '
    'ruling that keeps it'
)

#: D8's rejection arm, in the same clause shape.  It names no remedy but
#: deletion, because there is no other one that works.
_DEAD_REASON = (
    'delete this marker -- no tool reads it, so it silences nothing and protects '
    'nothing. A disposition does not answer for it either: with no detector silenced '
    'there is nothing for an owner to own or for the operator to rule on'
)


def _violation_for(entry: Classified) -> Violation:
    """The exit-1 finding one excess *entry* renders.

    THE TWO REASONS ARE NOT INTERCHANGEABLE ADVICE.  An undisposed live
    suppression is fixed by writing a disposition, so the line publishes the
    accepted forms; a marker nothing consumes is fixed only by deleting it, so
    the line publishes none.  Handing an author forms that would not clear the
    finding is worse than handing them nothing: they would write one, rerun,
    and see the identical red with no idea why.

    Which of the two applies is read off the entry's consumer rather than
    recomputed, because the classification already resolved it — in the fixed
    order that made the site unowned in the first place.
    """
    dead = entry.consumer is Consumer.NONE
    return Violation(
        path=entry.site.path,
        line=entry.site.line,
        kind=entry.site.kind,
        codes=entry.site.codes,
        reason=_DEAD_REASON if dead else _UNDISPOSED_REASON,
        forms=() if dead else import_shared('governed_exceptions').INLINE_MARKER_FORMS,
    )


def _excess_violations(
    over: Mapping[str, int], classification: Classification
) -> tuple[Violation, ...]:
    """One violation per SITE of every key the baseline does not cover.

    EVERY SITE, NOT ``over[key]`` OF THEM, and the reason is structural: a key
    IS the content of its line, so two sites sharing one are indistinguishable
    by construction and the scanner cannot say which is the new one.  Naming
    them all is the only honest rendering — the alternative picks an arbitrary
    line and sends the reader to code that may have been there for a year.

    "ABSENT FROM THE BASELINE" IS MADE PRECISE AS "IN EXCESS", and that one
    substitution is what folds D8's rejection arm INTO the ratchet instead of
    standing it beside one.  A dead marker is not a separate rule with its own
    exemption list: it is an ordinary member of the multiset, so the baseline
    grandfathers the existing population, a second copy of a grandfathered line
    is over budget, and removing one shows up as slack — all without a second
    mechanism to keep in step with this one.

    Sorted by key so a rerun over one tree prints the same lines in the same
    order; within a key, scan order, which is path then line.
    """
    return tuple(
        _violation_for(entry) for key in sorted(over) for entry in classification.unowned[key]
    )


@dataclass(frozen=True)
class Verdict:
    """What the gate would say about this run.

    Attributes:
        status: Which green a zero exit would be.
        excess: Keys over budget, or ``None`` when no baseline was compared.
        slack: Headroom, or ``None`` when this run's view cannot honestly
            measure it — an advisory run has no baseline, and a scoped one would
            read every unscanned key as headroom.
        violations: Every exit-1 finding, disposition faults first.

    COMPUTED ONCE AND READ TWICE.  ``--check`` turns it into an exit code and
    ``--json`` publishes it, so a report reader sees the gate's answer without
    running the gate — and cannot see a DIFFERENT answer, which is what two
    scans of a tree that changed in between would give them.
    """

    status: Status
    excess: Mapping[str, int] | None
    slack: Mapping[str, int] | None
    violations: tuple[Violation, ...]

    @property
    def exit_code(self) -> int:
        """1 when this run found something, else 0."""
        return 1 if self.violations else 0


def _verdict(
    request: Request, classification: Classification, kernel: ModuleType
) -> Verdict:
    """What the gate would say about *classification*.

    The disposition faults come first and are baseline-INDEPENDENT: a marker that
    does not parse is a fault at the site whatever any baseline holds, so an
    advisory run still reports one.  Only the ratchet's half waits for a baseline.
    """
    status = _status(request)
    if status is Status.ADVISORY:
        return Verdict(
            status=status, excess=None, slack=None, violations=classification.violations
        )

    baseline = kernel.load(request.baseline)
    current = _enumeration(classification, kernel)
    over = dict(kernel.excess(current, baseline))
    return Verdict(
        status=status,
        excess=over,
        slack=None if request.scoped else dict(kernel.slack(current, baseline)),
        violations=classification.violations + _excess_violations(over, classification),
    )


def _total(counts: Mapping[str, int] | None) -> int | None:
    """A ratchet block's multiplicity total, preserving "not measured"."""
    return None if counts is None else sum(counts.values())


def _count(total: int, noun: str) -> str:
    """*total* and *noun*, pluralised by the only rule this report needs."""
    return f'{total} {noun}' if total == 1 else f'{total} {noun}s'


def _figure(total: int | None) -> str:
    """A ratchet figure, or ``n/a`` when this run's view cannot honestly give one.

    THE DISTINCTION IS NOT COSMETIC.  Zero means *measured, and nothing there*;
    ``n/a`` means *not measured*, which is what an advisory run's excess and a
    scoped run's slack both are.  Printing 0 for either would be a number that is
    wrong in the direction that reassures.
    """
    return 'n/a' if total is None else str(total)


def _headline(status: Status, violations: tuple[Violation, ...]) -> str:
    """The report's first words: which green this is, or how big the red is."""
    if not violations:
        return status.value
    return _count(len(violations), 'violation')


def _report_line(
    measured: Measurement,
    *,
    headline: str,
    excess_total: int | None,
    slack_total: int | None,
) -> str:
    """The one-line summary every verb prints to stdout.

    The counted-work figures lead because they are what a reader checks first
    when a run comes back suspiciously clean: a scan that enumerated nothing
    and a tree with nothing wrong are otherwise the same output.
    """
    counts = measured.classification.counts
    scan = measured.scan
    return (
        f'{headline}: {_count(len(scan.sites), "suppression site")} in '
        f'{scan.files_tokenized} of {scan.files_enumerated} tracked files; '
        f'{sum(counts.values())} unowned in {_count(len(counts), "key")}; '
        f'excess {_figure(excess_total)}, slack {_figure(slack_total)}'
    )


def _check(request: Request, kernel: ModuleType) -> int:
    """Compare *request*'s tree against its baseline — the merge gate's verb.

    The violations are :func:`_verdict`'s: the disposition faults D6 names plus
    the ratchet's excess.  Either alone is exit 1.
    """
    measured = _measure(request)
    verdict = _verdict(request, measured.classification, kernel)
    print(
        _report_line(
            measured,
            headline=_headline(verdict.status, verdict.violations),
            excess_total=_total(verdict.excess),
            slack_total=_total(verdict.slack),
        )
    )
    if verdict.status is Status.ADVISORY:
        print(_ADVISORY_NOTICE.format(baseline=request.baseline))
    for violation in verdict.violations:
        print(violation.render(), file=sys.stderr)
    return verdict.exit_code


def _require_whole_tree(request: Request, verb: str) -> None:
    """Refuse a scoped run of a baseline-WRITING verb, before any scan work.

    THE KERNEL'S ARITHMETIC CANNOT REFUSE THIS ONE, which is why the refusal
    lives here.  A scoped scan is honestly ``complete=True`` for its scope, so it
    passes ``_require_comparable`` and the tighten goes through — writing
    ``baseline ∩ scope`` and deleting every key outside it.  The kernel's own
    docstring names regenerating a baseline from a partial view as the single hole
    its arithmetic leaves; this is the consumer closing it.

    BEFORE THE SCAN, not after, for two reasons that both matter.  A refusal is
    not worth minutes of tokenizing, and — the load-bearing one — a tree that
    also holds an unreadable file would otherwise report THAT as the exit 2,
    sending the reader to fix a file when the invocation was the problem.

    A scoped ``--check`` is deliberately not refused: ``Counter`` subtraction is
    per-key and saturating, so a narrower current view can only lower a key's
    count and therefore only UNDER-report violations.  It can never manufacture
    one, which is what the ``partial`` label is telling the reader.
    """
    if request.scoped:
        raise InstrumentFailure(
            f'refusing a scoped {verb}: {" ".join(request.scope)} is part of a tree, and a '
            'baseline written from a partial view drops every key outside it -- which makes '
            'each of those suppressions a fresh violation on the next whole-tree run. Re-run '
            'without PATH arguments. A scoped --check is the supported early-feedback run.'
        )


def _seed(request: Request, kernel: ModuleType) -> int:
    """Write *request*'s tree as a fresh baseline — task 5607's one-time verb.

    BOTH REFUSALS PRECEDE ``dump``, and they have to: ``dump`` writes the
    enumeration it is handed and checks nothing about the file already at the
    path, because seeding a new baseline and carrying an honestly incomplete one
    across a file boundary are both legitimate uses of it.  Re-seeding an
    EXISTING baseline is therefore the one call in the kernel that widens the
    gate — by every key the current scan added — so the guard belongs to whoever
    calls it.

    The status is :attr:`Status.CLEAN` by construction rather than by
    measurement: a baseline written from this very scan has no excess over it,
    and the run is whole-tree because a scoped seed is refused.  Both ratchet
    figures are ``n/a``, because nothing was compared — printing 0 for an excess
    that was never computed would be the reassuring-direction error
    :func:`_figure` exists to refuse.
    """
    _require_whole_tree(request, '--seed')
    if request.baseline.exists():
        raise InstrumentFailure(
            f'refusing to seed over the baseline already at {request.baseline}: a baseline is '
            "seeded ONCE, as the operator's one-time cutover on main, and tightened "
            'thereafter. Re-seeding an existing one silently widens the gate by every key '
            'this scan added. Use --tighten to remove what the tree no longer needs.'
        )
    measured = _measure(request)
    kernel.dump(_enumeration(measured.classification, kernel), request.baseline)
    print(
        _report_line(
            measured,
            headline=f'{Status.CLEAN.value} -- seeded {request.baseline}',
            excess_total=None,
            slack_total=None,
        )
    )
    return 0


#: What a run with no baseline tells the reader, so the next question — "then
#: why is this green?" — is answered in the same output.
#:
#: IT SCOPES ITS CLAIM TO THE RATCHET, because the run it accompanies may still
#: exit 1 on a disposition fault, which :func:`_verdict` reports with or without
#: a baseline.
_ADVISORY_NOTICE = (
    'no baseline at {baseline}, so the RATCHET is not enforced yet: every suppression '
    'here is reported and none of them counts as excess. Disposition faults are '
    'reported regardless, because a marker that does not parse is a fault whatever a '
    'baseline holds. The baseline is seeded once, on main, with --seed'
)


def _tighten(request: Request, kernel: ModuleType) -> int:
    """Spend the baseline's headroom — task 5609's verb, run after the rulings land.

    THROUGH ``tighten_into``, NEVER ``tighten`` PLUS ``dump``.  The kernel makes
    the composition the SHORT way to write the safe call for a reason: persisting
    ``tighten``'s bare ``Counter`` by hand means re-supplying ``params``,
    ``complete`` and ``unreadable``, three chances to commit a baseline claiming
    more than the scan measured — sitting next to ``dump(current, path)``, a
    shorter line that widens the gate by every key this scan added.

    THE REMOVALS COME FROM THE RETURN VALUE, not from reading the file back.  The
    loaded pre-image and the tightened counter are both in hand, so the difference
    is exact; re-reading would report what the file says rather than what this run
    did, which are the same thing only when nothing went wrong.

    Excess is measured BEFORE the write and is not stale: tightening lowers a
    baseline key only where the current count is lower, which is precisely where
    excess is already zero, so ``excess`` is invariant under it.  Slack afterwards
    is zero by the same arithmetic — a pointwise minimum is elementwise ≤ the
    current scan — so the figure is measured, not assumed.

    It reports NO status word, deliberately.  ``--tighten`` does not gate: it
    returns 0 having done its maintenance even when the tree is still over
    budget, and a green label beside a non-zero excess would read as a verdict it
    never made.  The excess figure in the same line is what tells the reader the
    gate is still red.
    """
    _require_whole_tree(request, '--tighten')
    measured = _measure(request)
    baseline = kernel.load(request.baseline)
    current = _enumeration(measured.classification, kernel)
    excess_total = sum(kernel.excess(current, baseline).values())
    tightened = kernel.tighten_into(current, baseline, request.baseline)

    removed = {
        key: count - tightened.get(key, 0)
        for key, count in baseline.counts.items()
        if tightened.get(key, 0) < count
    }
    print(
        _report_line(
            measured,
            headline=f'tightened {request.baseline}',
            excess_total=excess_total,
            slack_total=0,
        )
    )
    for key, count in sorted(removed.items()):
        print(f'  -{count} {key}')
    return 0


def _site_record(site: Site) -> dict[str, object]:
    """One site, as every block of the report names it."""
    return {
        'path': site.path,
        'line': site.line,
        'kind': site.kind.value,
        'codes': list(site.codes),
    }


def _owner(debt: Debt) -> str:
    """A debt's owner as D6 spells it — ``task 5601`` or ``ticket tkt_…``.

    The same tail the inline marker carries, so a report entry and the line it
    came from read alike and a consumer following the owner up needs no second
    vocabulary.  The two nouns come from the two ref TYPES rather than from a
    string test, which is why a ticket can never be reported as a task.
    """
    noun = (
        'task'
        if isinstance(debt.owner, import_shared('governed_exceptions').TaskRef)
        else 'ticket'
    )
    return f'{noun} {debt.owner.id}'


def _tally(values: Iterable[str], vocabulary: Iterable[str]) -> dict[str, int]:
    """Count *values*, with every word of *vocabulary* present even at zero.

    THE ZERO ROWS ARE THE POINT.  A zero is a measurement, and a schema that
    omitted it would make "no markers of this kind" and "the scanner stopped
    looking for them" the same output.  Every kind, consumer and ownership
    state therefore appears in every report.
    """
    counted = Counter(values)
    return {word: counted.get(word, 0) for word in vocabulary}


def _by_kind_code(scan: Scan) -> list[dict[str, object]]:
    """The per-``(kind, code)`` table the exception register renders.

    A site with two codes contributes to BOTH rows, because the question this
    answers is "how many suppressions silence E402", and a marker silencing E402
    and F401 silences each of them.  The totals therefore need not sum to the
    site count, which is why they are published beside ``kind_totals`` rather
    than instead of it.

    A codeless marker gets ``None`` rather than an empty string: ``""`` is a
    value a reader could filter on and silently match nothing.
    """
    tally: Counter[tuple[str, str | None]] = Counter()
    for site in scan.sites:
        for code in site.codes or (None,):
            tally[(site.kind.value, code)] += 1
    return [
        {'kind': kind, 'code': code, 'sites': sites}
        for (kind, code), sites in sorted(
            tally.items(), key=lambda row: (row[0][0], row[0][1] or '')
        )
    ]


def _owned_blocks(classification: Classification) -> dict[str, object]:
    """The three site-level ownership blocks, built in one pass.

    GROUPED BY WHAT EACH ENTRY CARRIES rather than by its :class:`Ownership`
    label, and the two agree by construction: :func:`_classify_site` sets exactly
    one of ``disposition`` and ``suppression_class``, and sets NEITHER on an
    unowned site — including one whose comment did carry a disposition that D8
    refused.  Keying off the field removes the unreachable branch a label-first
    grouping would need, and the report then says what is actually written down.

    THESE ARE THE ONLY SITE-LEVEL BLOCKS, deliberately.  Each is bounded by what
    somebody wrote down, while the corpus itself is thousands of sites and is
    published as counts; a report enumerating every one would be a megabyte
    nobody reads and a second copy of the tree.  ``ratified`` carries its sites
    because the register's closed-world check cannot report an id it has no
    sites to point at, and ``classes`` carries its own so a blanket ruling cannot
    quietly absorb a growing population.
    """
    governed = import_shared('governed_exceptions')
    debt_type: type[Debt] = governed.Debt
    policy_type: type[Policy] = governed.Policy

    debt: list[dict[str, object]] = []
    ratified: dict[str, list[dict[str, object]]] = {}
    classes: dict[str, list[dict[str, object]]] = {}
    for entry in classification.classified:
        record = _site_record(entry.site)
        if isinstance(entry.disposition, debt_type):
            debt.append(record | {'owner': _owner(entry.disposition)})
        elif isinstance(entry.disposition, policy_type):
            ratified.setdefault(entry.disposition.ratified, []).append(record)
        if entry.suppression_class is not None:
            classes.setdefault(entry.suppression_class.render(), []).append(record)
    return {
        'debt': debt,
        'ratified': [{'id': key, 'sites': sites} for key, sites in sorted(ratified.items())],
        'classes': [{'class': key, 'sites': sites} for key, sites in sorted(classes.items())],
    }


def _report_dict(request: Request, measured: Measurement, verdict: Verdict) -> dict[str, object]:
    """The whole report, as plain JSON-expressible data.

    ONE BUILDER, so ``--json`` is the only place the report exists and nothing
    downstream re-implements the scan.  Every mapping is dumped with sorted keys
    and every list is built in a sorted order — the site-level ones in scan
    order, which is sorted path then ascending line — so two runs over one
    unchanged tree emit identical BYTES.  A report that reordered between runs
    would make its every consumer's diff pure noise.
    """
    classification = measured.classification
    return {
        'schema_version': REPORT_SCHEMA_VERSION,
        'status': verdict.status.value,
        'params': key_params(),
        'files_enumerated': measured.scan.files_enumerated,
        'files_tokenized': measured.scan.files_tokenized,
        'sites': len(measured.scan.sites),
        'kind_totals': _tally(
            (site.kind.value for site in measured.scan.sites),
            (kind.value for kind in Kind),
        ),
        'consumers': _tally(
            (entry.consumer.value for entry in classification.classified),
            (consumer.value for consumer in Consumer),
        ),
        'ownership': _tally(
            (entry.ownership.value for entry in classification.classified),
            (ownership.value for ownership in Ownership),
        ),
        'by_kind_code': _by_kind_code(measured.scan),
        **_owned_blocks(classification),
        'baseline': {
            'path': str(request.baseline),
            'present': verdict.status is not Status.ADVISORY,
            'excess': None if verdict.excess is None else dict(verdict.excess),
            'slack': None if verdict.slack is None else dict(verdict.slack),
        },
        'ruff_config': [
            {
                'pyproject': config.path,
                'select': [code.render() for code in config.select],
                'ignore': [code.render() for code in config.ignore],
            }
            for config in measured.model.resolved()
        ],
        'violations': [violation.record() for violation in verdict.violations],
    }


def _json(request: Request, kernel: ModuleType) -> int:
    """Publish the report on stdout — a REPORT verb, never a gate.

    IT EXITS 0 WHENEVER THE SCAN SUCCEEDED, including over a tree that is
    currently in breach, and that is the contract rather than an oversight: a
    consumer parsing this report must not also be gated by it, and the state the
    exception register most needs to read is precisely the one ``--check``
    refuses.  Only a broken instrument makes it 2, because then there is no
    report to publish.
    """
    measured = _measure(request)
    verdict = _verdict(request, measured.classification, kernel)
    print(json.dumps(_report_dict(request, measured, verdict), indent=2, sort_keys=True))
    return 0


_EPILOG = """exit codes:
  0  clean. A scoped run says `partial`; a run with no baseline yet says
     `advisory`. All three are green, and the label says which green it is.
  1  violations, one per line on stderr: site, kind, codes, the reason and the
     accepted disposition forms.
  2  instrument failure or a refusal to act -- a file that could not be read
     or tokenized, a baseline that exists but cannot be compared against, a
     config key this model does not implement, a missing import, a scope that
     matches no tracked file, a scoped --seed or --tighten, or --seed over an
     existing baseline. Never a finding.
"""


def _build_parser() -> argparse.ArgumentParser:
    """The CLI: one verb, a tree, a baseline, and an optional scope."""
    parser = argparse.ArgumentParser(
        prog=_PROG,
        description=(
            "INV-12's inline-suppression scanner and multiset ratchet: every tracked "
            'suppression is owned by a disposition, ratified by class, or held by the '
            'baseline, and the population can shrink but never grow.'
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    verbs = parser.add_mutually_exclusive_group()
    verbs.add_argument(
        '--check',
        action='store_true',
        help='compare the tree against the baseline (the default verb)',
    )
    verbs.add_argument(
        '--seed',
        action='store_true',
        help='write the tree as a fresh baseline; refuses if one already exists',
    )
    verbs.add_argument(
        '--tighten',
        action='store_true',
        help='remove the baseline headroom the tree no longer uses',
    )
    verbs.add_argument(
        '--json',
        action='store_true',
        help=(
            'publish the whole report on stdout and exit 0 unless the instrument broke. '
            'Mutually exclusive with the other verbs: --json --seed has two defensible '
            'readings and no way to say which was meant'
        ),
    )
    parser.add_argument(
        '--root',
        type=Path,
        default=_REPO_ROOT,
        help='the checkout to measure (default: the one holding this script)',
    )
    parser.add_argument(
        '--baseline',
        type=Path,
        default=None,
        help=f'the baseline file (default: <root>/{BASELINE_PATH})',
    )
    parser.add_argument(
        'paths',
        nargs='*',
        metavar='PATH',
        help=(
            'limit the scan to these files or directories. A scoped --check is early '
            'feedback and labels its green `partial`; the baseline-writing verbs refuse '
            'a scope outright'
        ),
    )
    return parser


def _refuse(exc: Exception) -> int:
    """Print *exc* as this scanner's exit-2 refusal and return 2.

    ONE PRINT SITE FOR EVERY BROKEN-INSTRUMENT MESSAGE, so the prefix a log
    reader greps for cannot differ between causes — which is the same argument
    :class:`InstrumentFailure` makes for there being one exception type.
    """
    print(f'{_PROG}: {exc}', file=sys.stderr)
    return 2


def _run(
    args: argparse.Namespace,
    kernel: ModuleType,
    classes: Mapping[SuppressionClass, Policy],
) -> int:
    """Perform the verb *args* selected against *classes*, as the exit code it returns.

    The kernel's whole error family is converted to this module's own
    instrument failure HERE, at the one place the kernel is reachable, because
    to a caller they mean the identical thing: nothing was compared.  That is
    the single ``except`` clause ``shared.ratchet``'s docstring says a consumer
    wants, spent once rather than at every call site.
    """
    root = Path(args.root)
    baseline = Path(args.baseline) if args.baseline is not None else root / BASELINE_PATH
    request = Request(root=root, baseline=baseline, classes=classes, scope=tuple(args.paths))
    try:
        if args.seed:
            return _seed(request, kernel)
        if args.tighten:
            return _tighten(request, kernel)
        if args.json:
            return _json(request, kernel)
        return _check(request, kernel)
    except kernel.RatchetError as exc:
        raise InstrumentFailure(
            f'the ratchet kernel refused this run -- {exc}'
        ) from exc


def main(
    argv: Sequence[str] | None = None,
    *,
    classes: Mapping[SuppressionClass, Policy] = RATIFIED_SUPPRESSION_CLASSES,
) -> int:
    """The 0/1/2 entry point, with every broken-instrument path landing on 2.

    *classes* is the operator's table, injectable so a test can run the CLI
    against a real non-empty table without patching a module global.
    """
    args = _build_parser().parse_args(argv)
    try:
        return _run(args, import_shared('ratchet'), classes)
    except InstrumentFailure as exc:
        return _refuse(exc)


if __name__ == '__main__':
    sys.exit(main())
