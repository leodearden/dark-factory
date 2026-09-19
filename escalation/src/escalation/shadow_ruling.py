"""Shadow-mode L2 adjudication rulings — MEASUREMENT ONLY, no authority granted.

``docs/escalation-standing-policy.md`` proposes classes of L2 escalation that an
adjudicating session could one day rule without waiting for the human. NONE of
them is adopted. This module is the measurement half of that proposal: a watcher
records what it WOULD have ruled, takes no action, and a weekly count compares
the recorded proposal against what the human actually did. Nothing here decides
anything, closes anything, or widens any caller's authority — the shape of that
authority is owned by ``escalation/src/escalation/authority.py`` and is untouched
by this file.

**Why the payload rides inside ``triage_note``.** The tool that writes it,
``escalation/src/escalation/server.py::stamp_triage``, has exactly one free
parameter — ``triage_note: str``. There is no structured field for a shadow
ruling and this task deliberately adds none: a new ``Escalation`` field plus a
new tool argument would be a schema and API change in the very subsystem whose
authority the task must not touch.

**Why the payload is a LINE rather than the whole note.** ``triage_note``
already carries a freshness contract — a named world-facing predicate plus the
probe used to check it (``skills/escalation-watcher/SKILL.md``, "Reading a
triage-ack annotation"). A shadow stamp must COMPOSE with that contract, not
replace it, so the payload is one ``x_shadow_ruling: {...}`` line among the
note's other lines. Reading it is ``line.startswith(marker)`` plus
``json.loads`` — real structured data behind a fixed token, never an ad-hoc
grammar over prose.

**Promotion path.** If a class is ever adopted, ``x_shadow_ruling`` becomes a
first-class ``Escalation`` field and this codec is DELETED, not extended. The
string-carried form is the price of changing nothing while the measurement runs;
it is not a design to build on.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType

from escalation.authority import L2_AUTO_CLOSE_DENY_CATEGORIES, L2_AUTO_CLOSE_DENY_ROLES
from escalation.classify import classify_resolver_tier
from escalation.models import Escalation
from escalation.queue import iter_all_escalation_paths

logger = logging.getLogger(__name__)

#: The fixed token that opens a shadow-ruling line inside a ``triage_note``.
#: A line qualifies only when it BEGINS with this token, so prose that merely
#: mentions it — this module's own guidance, or the skill's — is never mistaken
#: for a stamp.
SHADOW_RULING_MARKER = 'x_shadow_ruling:'

#: The first-tranche candidate classes, SHADOWED AND NOT ADOPTED. Each names a
#: recurring L2 shape the standing-policy skeleton proposes could be ruled by
#: the adjudicating session. Enumerated with their prose gloss in
#: ``docs/escalation-standing-policy.md``; that document and this frozenset are
#: held equal by ``tests/scripts/test_shadow_ruling_doc_contract.py``.
FIRST_TRANCHE_CLASSES: frozenset[str] = frozenset({
    'risk_identified_branch_behind_main',
    'risk_identified_recovery_veto_streak',
    'design_concern_semantic_collision',
    'infra_issue_transient_self_cleared',
})

#: The reversible actions the ratified skeleton permits. A SUPERSET of the C1
#: ``resolution_action`` vocabulary: ``close_only`` and ``resume`` are C1 values
#: a resolved record records, while ``add_dependency``, ``update_task_amendment``
#: and ``file_task`` are task-side operations that leave ``resolution_action``
#: unset — which is why the weekly count needs a third ``not_comparable``
#: outcome rather than folding them into agreement or divergence.
REVERSIBLE_ACTIONS: frozenset[str] = frozenset({
    'close_only',
    'resume',
    'add_dependency',
    'update_task_amendment',
    'file_task',
})

#: The seven classes of decision that stay with the human FOREVER, whatever any
#: shadow measurement shows. Only the first two have a record-level signal; see
#: :func:`mechanically_gated`. The authority for this list is
#: ``docs/escalation-standing-policy.md``, which glosses each one.
HUMAN_FOREVER_GATES: frozenset[str] = frozenset({
    'milestone_gate',
    'deterministic_runner_filing',
    'model_admission',
    'physical_operator_action',
    'irreversible_deletion',
    'spend_or_eval_launch',
    'post_breaker_resume_scheduler',
})

#: The two gate slugs :func:`mechanically_gated` can return. A PROPER subset of
#: :data:`HUMAN_FOREVER_GATES` — the remaining five gates are semantic and have
#: no record-level signal at all.
DETECTABLE_GATES: frozenset[str] = frozenset({'milestone_gate', 'deterministic_runner_filing'})

#: The categories and roles the detector matches against — THE IMPORTED
#: ``escalation.authority`` tables, bound under local names, never copies of
#: their literals. Re-exported so the SPOT property is assertable without
#: reaching into a private name: a member added to authority.py propagates here
#: for free, and replacing either binding with a literal fails the guard.
GATED_CATEGORIES: frozenset[str] = L2_AUTO_CLOSE_DENY_CATEGORIES
GATED_ROLES: frozenset[str] = L2_AUTO_CLOSE_DENY_ROLES

#: The reversible actions that are ALSO C1 ``resolution_action`` values, so an
#: observed outcome can be checked against the proposal. The rest of
#: :data:`REVERSIBLE_ACTIONS` is task-side and leaves ``resolution_action``
#: unset, which is what the report's ``not_comparable`` bucket counts.
#:
#: Must equal ``REVERSIBLE_ACTIONS & set(escalation.server.RESOLVE_ACTIONS)``,
#: and is pinned in lockstep by a cross-module TEST import rather than derived
#: by a production one — importing the MCP server here would invert the layer
#: direction and pull fastmcp into every reader of the archive. This mirrors the
#: convention ``escalation/src/escalation/authority.py`` already uses for the
#: watcher identity string and ``action_effects.py`` for its target statuses.
COMPARABLE_ACTIONS: frozenset[str] = frozenset({'close_only', 'resume'})

#: Wire keys of the JSON payload. ``class`` rather than ``ruling_class`` because
#: that is what a reader of the note sees; the Python attribute cannot be
#: spelled ``class``.
_PAYLOAD_KEYS = frozenset({'class', 'proposed_action', 'evidence', 'confidence'})


@dataclass(frozen=True)
class ShadowRuling:
    """What an adjudicating session WOULD have ruled, had the class been adopted.

    Validated on construction, so a rejected payload renders nothing and
    persists nothing. The four fields are exactly the four the standing policy
    requires of a proposal: which class it falls in, which reversible action it
    proposes, the evidence quoted verbatim, and how sure the adjudicator was.
    """

    ruling_class: str
    proposed_action: str
    evidence: str
    confidence: float

    def __post_init__(self) -> None:
        if self.ruling_class not in FIRST_TRANCHE_CLASSES:
            raise ValueError(
                f'unknown shadow ruling_class {self.ruling_class!r}; '
                f'expected one of {sorted(FIRST_TRANCHE_CLASSES)}'
            )
        if self.proposed_action not in REVERSIBLE_ACTIONS:
            raise ValueError(
                f'unknown shadow proposed_action {self.proposed_action!r}; '
                f'expected one of {sorted(REVERSIBLE_ACTIONS)}'
            )
        if not isinstance(self.evidence, str) or not self.evidence:
            raise ValueError(
                'shadow ruling evidence must be a non-empty string quoting the '
                'deciding evidence verbatim'
            )
        if not isinstance(self.confidence, int | float) or isinstance(self.confidence, bool):
            raise ValueError(f'shadow ruling confidence must be a number, got {self.confidence!r}')
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f'shadow ruling confidence {self.confidence!r} outside [0.0, 1.0]'
            )

    def to_note_line(self) -> str:
        """Render the marker line to append to an existing ``triage_note``.

        Always ONE line: JSON escaping folds any newline in *evidence* into
        ``\\n``, so the caller can append this to a note without breaking the
        note's other lines.
        """
        payload = {
            'class': self.ruling_class,
            'proposed_action': self.proposed_action,
            'evidence': self.evidence,
            'confidence': self.confidence,
        }
        return f'{SHADOW_RULING_MARKER} {json.dumps(payload, sort_keys=True)}'


def parse_shadow_ruling(triage_note: str) -> ShadowRuling | None:
    """Return the :class:`ShadowRuling` carried by *triage_note*, or ``None``.

    ``None`` means "no usable shadow ruling here" and is the answer for a note
    with no marker line, an empty note, a marker line whose payload is not a
    JSON object, and a payload whose values are outside the policy's
    vocabularies. It NEVER raises: the weekly count sweeps every escalation in
    the queue and archive, the overwhelming majority of which carry no stamp,
    so an exception would turn one malformed note into a failed measurement.

    A marker line that IS present but unusable is logged at WARNING — an
    unreadable stamp is a lost sample, and losing samples silently is how a
    class's agreement rate drifts without anybody noticing.
    """
    for line in triage_note.splitlines():
        if not line.startswith(SHADOW_RULING_MARKER):
            continue
        raw = line[len(SHADOW_RULING_MARKER):]
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning('unparsable %s payload: %s', SHADOW_RULING_MARKER, exc)
            return None
        if not isinstance(payload, dict) or set(payload) != _PAYLOAD_KEYS:
            logger.warning(
                '%s payload keys are %r; expected exactly %r',
                SHADOW_RULING_MARKER,
                sorted(payload) if isinstance(payload, dict) else type(payload).__name__,
                sorted(_PAYLOAD_KEYS),
            )
            return None
        try:
            return ShadowRuling(
                ruling_class=payload['class'],
                proposed_action=payload['proposed_action'],
                evidence=payload['evidence'],
                confidence=payload['confidence'],
            )
        except ValueError as exc:
            logger.warning('rejected %s payload: %s', SHADOW_RULING_MARKER, exc)
            return None
    return None


def mechanically_gated(record: Escalation) -> str | None:
    """Return the human-forever gate slug *record* trips, or ``None``.

    ``None`` means **no MECHANICAL gate was detected**, NEVER "not gated". Five
    of the seven gates in :data:`HUMAN_FOREVER_GATES` — model admission,
    physical operator actions, irreversible deletions, spend/eval launches and a
    post-breaker ``resume_scheduler`` — are semantic judgements with no signal
    on the record at all. ``docs/escalation-standing-policy.md`` remains the
    authority for the full list; this function covers only its detectable
    subset, and a caller that treats a ``None`` here as clearance is reading it
    wrong.

    Category and role are checked independently, mirroring the defence in depth
    ``escalation/src/escalation/authority.py`` already relies on: neither benign
    half can mask the other.

    THE ``design_concern`` TENSION, stated rather than hidden. The set this
    matches is authority.py's auto-close denylist, and ``design_concern`` is a
    member of it — so every ``design_concern`` shadow stamp lands in the weekly
    count's ``gated_stamps`` bucket, including one filed under the
    ``design_concern_semantic_collision`` first-tranche candidate. That is why
    the candidate is shadow-only. It is ALSO why adoption is a question about
    the INTERACTIVE arm rather than the auto arm: per authority.py's own
    docstring these tables constrain only identified callers in
    ``ROLE_LEVEL_ALLOWLIST`` — whose sole member is the auto-watcher identity —
    and a header-less interactive connection is never narrowed by that module.
    Extending a class to the AUTO watcher would need an authority.py change;
    adopting one for the interactive session would not.
    """
    if record.category in GATED_CATEGORIES:
        return 'milestone_gate'
    if record.agent_role in GATED_ROLES:
        return 'deterministic_runner_filing'
    return None


@dataclass(frozen=True)
class ClassAgreement:
    """How one shadowed class fared over the report window.

    ``agreed`` and ``diverged`` are the COMPARABLE outcomes: the stamped
    proposal was a C1 action, so the record's observed ``resolution_action``
    could be checked against it. ``not_comparable`` counts proposals whose
    action is task-side and leaves no C1 trace — kept as its own number rather
    than folded into either side, so the denominator the adoption threshold is
    read off is one a reader can see.
    """

    ruling_class: str
    agreed: int
    diverged: int
    not_comparable: int

    @property
    def comparable(self) -> int:
        """The denominator of :attr:`agreement_rate`."""
        return self.agreed + self.diverged

    @property
    def total(self) -> int:
        return self.agreed + self.diverged + self.not_comparable

    @property
    def agreement_rate(self) -> float | None:
        """Agreement over the comparable subset, or ``None`` when there is none.

        ``None`` rather than 0.0 or 1.0: a class whose proposals are all
        task-side is NOT YET MEASURABLE, and reporting either extreme would make
        it look decided.
        """
        return self.agreed / self.comparable if self.comparable else None


@dataclass(frozen=True)
class AgreementReport:
    """The weekly count over one window. Every field is a decided number.

    The three non-rate buckets are findings, not noise: each counts a stamp that
    must not contribute to any class's rate, and each is reported so a reader
    can tell a small sample from a thrown-away one.

    TWO OF THE THREE ARE IN-WINDOW AND THE THIRD CANNOT BE, and the field names
    say which: ``gated_stamps`` and ``self_resolved`` count records RESOLVED
    inside ``since..until``, exactly like ``agreed``/``diverged``, so they are
    comparable with the denominator printed beside them. ``unresolved_lifetime``
    counts stamps still pending — a record with no resolution instant to window
    on at all — so it is a standing backlog as of the sweep, not a number from
    this window, and is named for that. Windowing it on ``triaged_at`` instead
    was rejected: it would put a second time axis under one window header, and
    the operator-facing contract (``--since``/``--until`` help text) is that the
    window is read on ``resolved_at``.
    """

    since: datetime
    until: datetime
    classes: tuple[ClassAgreement, ...]
    gated_stamps: int
    self_resolved: int
    unresolved_lifetime: int
    resolver_tiers: Mapping[str, int]

    def for_class(self, ruling_class: str) -> ClassAgreement | None:
        """The row for *ruling_class*, or ``None`` when it had no counted record."""
        return next((c for c in self.classes if c.ruling_class == ruling_class), None)


def _as_aware(parsed: datetime) -> datetime:
    """Read a naive datetime as UTC; leave an offset-bearing one alone.

    THE single rule normalising both sides of the window comparison — the
    record's ``resolved_at`` and the operator's ``--since``/``--until``. Two
    copies that merely happened to agree is how the naive-argument
    ``TypeError`` got in: ``_resolved_at`` coerced and the CLI did not, so a
    perfectly valid ``--since 2026-09-01`` crashed the sweep on the first
    stamped record it reached.

    UTC rather than local time because every instant this module compares is
    written as UTC by ``escalation/src/escalation/queue.py::resolve``, so an
    unqualified operator argument means UTC too.
    """
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _resolved_at(record: Escalation) -> datetime | None:
    """*record*'s resolution instant as an aware datetime, or ``None``.

    Parsed, never string-compared: ``resolved_at`` is written by several call
    sites and an offset-bearing timestamp sorts differently as text than it does
    as an instant.
    """
    if record.resolved_at is None:
        return None
    try:
        parsed = datetime.fromisoformat(record.resolved_at)
    except ValueError:
        logger.warning('unparsable resolved_at %r on %s', record.resolved_at, record.id)
        return None
    return _as_aware(parsed)


def agreement_report(
    escalations_dir: Path | str, *, since: datetime, until: datetime,
) -> AgreementReport:
    """Count how often the shadow proposals matched what actually happened.

    Sweeps the queue root and its archive through
    ``escalation/src/escalation/queue.py::iter_all_escalation_paths``, which
    already handles root-wins-on-collision, archive-only multi-date duplicates
    and a missing directory (yields nothing rather than raising). Records with
    no parsable shadow ruling are skipped — the overwhelming majority carry
    none.

    THE ORDER OF CHECKS IS PART OF THE CONTRACT: unresolved -> out-of-window ->
    gated -> self_resolved -> non-human resolver -> not_comparable ->
    agreed/diverged. The first five are all "this record must not contribute to
    a rate at all"; putting any of them later would let an in-window matching
    close fall through to ``agreed`` first.

    THE WINDOW COMES BEFORE THE TWO EXCLUDED BUCKETS THAT CAN BE WINDOWED, so
    that every number printed under the window header is a number from that
    window. Counting them first made them lifetime totals rendered beside a
    weekly denominator: on the real archive they grow monotonically forever, so
    a weekly report would eventually show ``gated_stamps=140`` next to a
    four-item comparable denominator — the inverse of the "a small sample and a
    discarded one must not look alike" property the buckets exist to give.

    Pendingness is decided FIRST because a pending record has no ``resolved_at``
    to window on: it would otherwise be dropped by the window filter and vanish
    from every bucket. That order also decides where a pending GATED stamp
    lands — ``unresolved_lifetime``, not ``gated_stamps`` — which is the honest
    reading: nothing has been ruled on it yet.

    ``self_resolved`` — ``triaged_by is not None and triaged_by ==
    resolved_by`` — is the bucket task 5361 made necessary.
    ``escalation/src/escalation/classify.py::_HUMAN_RESOLVERS`` contains
    ``escalation-watcher``, so a watcher that stamps a proposal and then closes
    the record itself under that standing rule would otherwise be read back as
    a human agreeing with itself, inflating the very ``design_concern`` class
    this measurement exists to judge, toward its own adoption threshold.

    The limit of that check, stated honestly: attribution is server-enforced
    only for a header-bearing identity —
    ``escalation/src/escalation/server.py::stamp_triage`` overrides
    ``triaged_by`` from ``X-Escalation-Identity`` ONLY when the header is
    present — so for the header-less interactive channel this is a backstop over
    a convention rather than a guarantee. That is why
    ``skills/escalation-watcher/SKILL.md`` forbids the stamp in the first place
    and this bucket only makes a violation visible.
    """
    agreed: Counter[str] = Counter()
    diverged: Counter[str] = Counter()
    not_comparable: Counter[str] = Counter()
    tiers: Counter[str] = Counter()
    gated_stamps = 0
    self_resolved = 0
    unresolved_lifetime = 0

    for path in iter_all_escalation_paths(Path(escalations_dir)):
        try:
            record = Escalation.from_json(path.read_text())
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning('skipping unparsable escalation at %s: %s', path, exc)
            continue

        ruling = parse_shadow_ruling(record.triage_note)
        if ruling is None:
            continue

        if record.status == 'pending':
            unresolved_lifetime += 1
            continue

        resolved_at = _resolved_at(record)
        if resolved_at is None or not since <= resolved_at <= until:
            continue

        if mechanically_gated(record) is not None:
            gated_stamps += 1
            continue
        if record.triaged_by is not None and record.triaged_by == record.resolved_by:
            self_resolved += 1
            continue

        tier = classify_resolver_tier(record.resolved_by)
        tiers[tier] += 1
        if tier != 'human':
            continue

        if ruling.proposed_action not in COMPARABLE_ACTIONS:
            not_comparable[ruling.ruling_class] += 1
        elif record.resolution_action == ruling.proposed_action:
            agreed[ruling.ruling_class] += 1
        else:
            diverged[ruling.ruling_class] += 1

    counted = sorted(set(agreed) | set(diverged) | set(not_comparable))
    return AgreementReport(
        since=since,
        until=until,
        classes=tuple(
            ClassAgreement(
                ruling_class=slug,
                agreed=agreed[slug],
                diverged=diverged[slug],
                not_comparable=not_comparable[slug],
            )
            for slug in counted
        ),
        gated_stamps=gated_stamps,
        self_resolved=self_resolved,
        unresolved_lifetime=unresolved_lifetime,
        resolver_tiers=MappingProxyType(dict(sorted(tiers.items()))),
    )


#: How far back ``main`` looks when no window is given — "weekly", per
#: ``docs/escalation-standing-policy.md``.
_DEFAULT_WINDOW = timedelta(days=7)

_ROW = '{cls:<40} {agreed:>7} {diverged:>9} {not_comparable:>15} {comparable:>11} {rate:>8}'


def _as_json(report: AgreementReport) -> str:
    return json.dumps(
        {
            'since': report.since.isoformat(),
            'until': report.until.isoformat(),
            'classes': [
                {
                    'class': c.ruling_class,
                    'agreed': c.agreed,
                    'diverged': c.diverged,
                    'not_comparable': c.not_comparable,
                    'comparable': c.comparable,
                    'total': c.total,
                    'agreement_rate': c.agreement_rate,
                }
                for c in report.classes
            ],
            'gated_stamps': report.gated_stamps,
            'self_resolved': report.self_resolved,
            'unresolved_lifetime': report.unresolved_lifetime,
            'resolver_tiers': dict(report.resolver_tiers),
        },
        indent=2,
        sort_keys=True,
    )


def _as_table(report: AgreementReport) -> str:
    """Render the report so an operator can paste it and a reader can decide.

    Every number the adoption threshold needs is on the page: the comparable
    DENOMINATOR beside the rate, and the three excluded buckets — always, even
    at zero. A class whose stamps were mostly thrown out for self-agreement is
    not a class with a small sample, and the output must not let the two look
    alike.

    The excluded buckets sit on their own line under the window header, and the
    one that is NOT from the window says so in its own name
    (``unresolved_lifetime``) rather than relying on the reader to know which
    numbers on that line share the header's window.
    """
    lines = [
        f'shadow ruling agreement — window {report.since.isoformat()} .. '
        f'{report.until.isoformat()}',
        '',
    ]
    if report.classes:
        lines.append(_ROW.format(
            cls='class', agreed='agreed', diverged='diverged',
            not_comparable='not_comp', comparable='comparable', rate='rate',
        ))
        for c in report.classes:
            rate = 'n/a' if c.agreement_rate is None else f'{c.agreement_rate * 100:.1f}%'
            lines.append(_ROW.format(
                cls=c.ruling_class, agreed=c.agreed, diverged=c.diverged,
                not_comparable=c.not_comparable, comparable=c.comparable, rate=rate,
            ))
    else:
        lines.append('no shadow rulings in window')
    lines.extend([
        '',
        f'gated_stamps={report.gated_stamps} self_resolved={report.self_resolved} '
        f'unresolved_lifetime={report.unresolved_lifetime}',
        'resolver_tiers: ' + (
            ' '.join(f'{tier}={n}' for tier, n in report.resolver_tiers.items()) or '(none)'
        ),
    ])
    return '\n'.join(lines)


class _BadWindowArg(ValueError):
    """An unparsable ``--since``/``--until``, carrying the flag that carried it.

    Raised rather than exiting so the failure reaches ``main``'s own exit path
    beside the ``--queue-dir`` branch: this module's CLI discipline is to print
    to stderr and RETURN 2, which an argparse ``type=`` callback would break by
    raising ``SystemExit`` from inside ``parse_args``.
    """


def _parse_window_arg(value: str, flag: str) -> datetime:
    """Parse one window bound, naming the flag AND the value when it is junk.

    Naming both matters: an operator who typed ``--since yesterday`` needs to
    see which argument the parser rejected, not a bare isoformat complaint.
    """
    try:
        return _as_aware(datetime.fromisoformat(value))
    except ValueError as exc:
        raise _BadWindowArg(
            f'{flag} is not an ISO-8601 datetime: {value!r} ({exc})'
        ) from exc


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m escalation.shadow_ruling``.

    Invoked from the repo root as::

        uv run --directory escalation python -m escalation.shadow_ruling \\
            --queue-dir <project_root>/data/escalations

    Read-only: it opens escalation JSON and prints. It never resolves, stamps or
    moves anything.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Weekly shadow-ruling agreement count: how often a stamped proposal '
            'matched what actually happened. Measurement only — grants no authority.'
        ),
    )
    parser.add_argument(
        '--queue-dir', required=True, type=Path,
        help='Path to the escalation queue directory (parent of archive/).',
    )
    parser.add_argument(
        '--since', default=None,
        help='ISO-8601 window start on resolved_at, naive values read as UTC '
             '(default: 7 days ago).',
    )
    parser.add_argument(
        '--until', default=None,
        help='ISO-8601 window end on resolved_at, naive values read as UTC '
             '(default: now).',
    )
    parser.add_argument(
        '--json', action='store_true', default=False,
        help='Emit machine-readable JSON instead of the table.',
    )
    args = parser.parse_args(argv)

    # LOUD, not an all-zero table: an empty report and a misconfigured path must
    # not look identical, or a typo'd path reads back as a clean measurement.
    if not args.queue_dir.is_dir():
        print(f'queue-dir is not a directory: {args.queue_dir}', file=sys.stderr)
        return 2

    now = datetime.now(UTC)
    try:
        until = _parse_window_arg(args.until, '--until') if args.until else now
        since = (
            _parse_window_arg(args.since, '--since') if args.since
            else until - _DEFAULT_WINDOW
        )
    except _BadWindowArg as bad:
        print(bad, file=sys.stderr)
        return 2

    report = agreement_report(args.queue_dir, since=since, until=until)
    print(_as_json(report) if args.json else _as_table(report))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
