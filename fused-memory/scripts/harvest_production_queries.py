#!/usr/bin/env python3
"""Harvest the production query set from the reconciliation write journal (task 4004).

WHY THIS EXISTS
---------------
``bake_off_storage_shape.py``'s query set is blind-authored: the queries
were written to exercise the fixture corpus, not sampled from traffic. A
read transform that wins only on authored queries has no external
validity. This script samples the query shapes that ACTUALLY reach
``search`` in production, so the read-transform arms can be scored on real
traffic beside the authored set.

WHAT IT MEASURES
----------------
The orchestrator's briefing assembler asks memory a small fixed set of
queries per dispatched agent. Each is one CLASS here, tagged with its era:

  * CURRENT classes are rendered from
    ``shared/src/shared/briefing_queries.py::QUERY_SPECS``, so a reworded
    template moves the harvester with it. A template with literal text is
    matched by PATTERN: ``conventions and gotchas for {area}`` is ONE class
    however many areas it is rendered for. Matching it literally would
    scatter one high-traffic class across thousands of singleton tail
    entries and understate it to nearly zero.
  * The free-text task query ``{title} {area}`` has no literal text, so its
    pattern would match nearly every query. It is matched as a COMPANION
    instead: the next search from the same ``caller_agent_id`` after that
    caller's conventions-area query, within ``COMPANION_WINDOW`` of it. The
    caller is threaded by
    ``orchestrator/src/orchestrator/agents/memory_recall.py::MemoryRecall._search``
    (PRD D8). A spec with no literal text and no declared anchor is refused
    when the classes are built.
  * RETIRED classes are the four queries task 3659 retired, kept
    hand-spelled in ``RETIRED_LITERAL_TEMPLATES`` and
    ``RETIRED_TASK_TEMPLATE`` for journal rows written before it: their
    source no longer exists.

Every briefing query fires at ``limit=5``, not the E2 default of 10.

Everything no class claims is the residual long tail, which is sampled —
frequency-led head plus a seeded random remainder — so the committed
fixture is small, representative and exactly regenerable.

THE LIMIT IS MEASURED, NOT ASSUMED
----------------------------------
``BriefingQuerySpec.limit`` governs the briefing queries and NOTHING else.
The residual tail comes from arbitrary other callers, and the journal shows
those callers run at 3, 4, 5, 6, 8, 10, 15, 20, 30 and 50 — only about a
third of tail traffic is at 5. Stamping ``BRIEFING_SEARCH_LIMIT`` onto a
tail row would therefore publish a number nothing observed, under a field
named ``observed_limit``, into an artifact a selection gate reads.

So every row carries the limit ACTUALLY recorded in the journal's
``write_ops.params`` blob, which this module already parses for the query
text:

  * ``observed_limits`` — the full measured histogram, ``{limit: count}``,
    always present. This is the raw measurement; nothing is collapsed.
  * ``observed_limit`` — an int ONLY when every instance of that query
    agreed, and ``None`` otherwise. ``None`` is *no single observation*,
    never a defaulted or modal guess; a reader who wants a modal value can
    take it from the histogram and own that choice explicitly.

Even the retired briefing literals were not unanimous in the committed
harvest (each has one or two stray instances at 10/20 out of ~75k at 5),
so they too report ``None`` with a
histogram that makes the 99.99% concentration at 5 visible. The scoring
window downstream is consequently a stated CHOICE, not a reading.

UNLABELED BY CONSTRUCTION
-------------------------
Production queries have no ground truth: nobody recorded which memory
*should* have come back. The emitted rows therefore carry NO
``expects_claim_ids`` and NO ``expects_topic``. Downstream, claim recall
and canonical discoverability are ``None`` for these queries and render as
``—``. Inventing a label here would fabricate the very number the
measurement exists to establish.

READ-ONLY, LOUDLY
-----------------
The journal is a multi-gigabyte SQLite file a running fused-memory server
is actively writing to. Every connection is opened ``mode=ro`` via URI plus
``PRAGMA query_only=ON``, so a write is impossible rather than merely
unintended. An absent, schema-less or empty journal raises a NAMED error
and writes no fixture: a silently-empty sample would read downstream as
"production traffic looks like nothing", which is a fabricated measurement.

USAGE
-----
    uv run --project fused-memory python \
        fused-memory/scripts/harvest_production_queries.py \
        --journal /home/leo/src/dark-factory/data/reconciliation/write_journal.db \
        --out fused-memory/tests/fixtures/production_query_sample.jsonl

A run from a ``.worktrees/<id>`` lane records the same repo-relative
journal path as one from the main checkout.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import string
import sys
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from shared.briefing_queries import (
    CONVENTIONS_AREA,
    QUERY_SPECS,
    TASK_SEMANTIC,
    BriefingQuerySpec,
)

from fused_memory.models.scope import resolve_main_checkout


class HarvestError(RuntimeError):
    """Base class for every loud refusal in this module."""


class JournalUnavailableError(HarvestError):
    """The journal is absent, unreadable, or is not a write journal.

    Raised INSTEAD of returning an empty harvest, so a missing file can
    never be mistaken downstream for "production issues no searches".
    """


class EmptyHarvestError(HarvestError):
    """The journal is readable but carries no parseable search traffic."""


class UnmatchableTemplateError(HarvestError):
    """A briefing query spec cannot be recognised in the journal.

    Its template has no literal text, so its pattern would match nearly
    every query, and no anchor class was declared to pair it with instead.
    """


# --------------------------------------------------------------------------
# The briefing-assembler query classes
# --------------------------------------------------------------------------
#: The three literal briefing queries task 3659 retired, in their old firing
#: order. Their source is gone, so this is now their only home.
RETIRED_LITERAL_TEMPLATES: tuple[str, ...] = (
    'project overview architecture goals',
    'coding conventions and project norms',
    'recent decisions and rationale',
)

#: The retired parameterized query, kept in `str.format` shape so the
#: template string itself is what lands in the fixture and the report.
RETIRED_TASK_TEMPLATE = 'task {task_id} context and related decisions'


class BriefingEra(StrEnum):
    """Whether a class is fired by today's briefing or only by older journal rows."""

    RETIRED = 'retired'
    CURRENT = 'current'


class MatchKind(StrEnum):
    """How a class recognises its ops."""

    LITERAL = 'literal'
    PARAMETERIZED = 'parameterized'
    COMPANION = 'companion'


@dataclass(frozen=True)
class BriefingClass:
    """One briefing query class: its template and how an op is matched to it.

    A COMPANION class has no pattern. It names instead the template of the
    `anchor` class whose op, from the same caller, comes just before it.
    """

    template: str
    match: MatchKind
    era: BriefingEra
    pattern: re.Pattern[str] | None
    anchor: str | None = None

    def __post_init__(self) -> None:
        is_companion = self.match is MatchKind.COMPANION
        if is_companion != (self.pattern is None) or is_companion != (self.anchor is not None):
            raise ValueError(
                f'{self.template!r}: a companion class has an anchor and no '
                'pattern; every other class has a pattern and no anchor'
            )


#: Field patterns narrower than "anything".
#:
#: * A task id is any run of non-space characters: ids are numeric ('4004')
#:   but subtask ids carry a dot ('3.1'), so the class must not assume \d+.
#: * An area is what `shared/src/shared/briefing_queries.py::derive_area_terms`
#:   yields: lowercase alphanumeric terms joined by single spaces.
_FIELD_PATTERNS: dict[str, str] = {
    'task_id': r'\S+',
    'area': r'[a-z0-9]+(?: [a-z0-9]+)*',
}
_ANY_FIELD_PATTERN = '.+'

#: Fields that can render to nothing. `shared.briefing_queries.render_query`
#: drops every word already earlier in the query, so an area made only of the
#: template's own words vanishes together with the space before it.
_VANISHING_FIELDS: frozenset[str] = frozenset({'area'})


def _template_fields(template: str) -> tuple[str, ...]:
    return tuple(
        name for _, name, _, _ in string.Formatter().parse(template) if name is not None
    )


def _template_literal_text(template: str) -> str:
    return ''.join(literal for literal, _, _, _ in string.Formatter().parse(template))


def _template_pattern(template: str) -> re.Pattern[str]:
    """Render a `str.format` template into the pattern its instances fullmatch."""
    parts: list[str] = []
    for literal, name, _, _ in string.Formatter().parse(template):
        if name is None:
            parts.append(re.escape(literal))
            continue
        field_pattern = f'(?P<{name}>{_FIELD_PATTERNS.get(name, _ANY_FIELD_PATTERN)})'
        if name in _VANISHING_FIELDS and literal.endswith(' '):
            parts.append(f'{re.escape(literal[:-1])}(?: {field_pattern})?')
        else:
            parts.append(re.escape(literal) + field_pattern)
    return re.compile(''.join(parts))


def _pattern_class(template: str, era: BriefingEra) -> BriefingClass:
    return BriefingClass(
        template=template,
        match=MatchKind.PARAMETERIZED if _template_fields(template) else MatchKind.LITERAL,
        era=era,
        pattern=_template_pattern(template),
    )


def build_current_classes(
    specs: Iterable[BriefingQuerySpec],
    companion_anchors: Mapping[str, str],
) -> tuple[BriefingClass, ...]:
    """One current-era class per spec, in `specs` order.

    `companion_anchors` maps a spec slug to the slug of the spec fired just
    before it by the same caller. A template with no literal text cannot be
    told apart from an agent's own search by pattern, so it must be named
    there and is matched as that anchor's companion instead.
    """
    specs = tuple(specs)
    by_slug = {spec.slug: spec for spec in specs}
    classes: list[BriefingClass] = []
    for spec in specs:
        anchor_slug = companion_anchors.get(spec.slug)
        if anchor_slug is None:
            classes.append(_current_pattern_class(spec))
            continue
        anchor = by_slug.get(anchor_slug)
        if anchor is None or anchor.slug in companion_anchors:
            raise UnmatchableTemplateError(
                f'briefing spec {spec.slug!r} ({spec.text!r}) is declared the '
                f'companion of {anchor_slug!r}, which is not a pattern-matched '
                'spec in the table'
            )
        classes.append(
            BriefingClass(
                template=spec.text,
                match=MatchKind.COMPANION,
                era=BriefingEra.CURRENT,
                pattern=None,
                anchor=anchor.text,
            )
        )
    return tuple(classes)


def _current_pattern_class(spec: BriefingQuerySpec) -> BriefingClass:
    if not _template_literal_text(spec.text).strip():
        raise UnmatchableTemplateError(
            f'briefing spec {spec.slug!r} has template {spec.text!r}, which '
            'has no literal text to match on; declare the spec it is fired '
            'beside in companion_anchors'
        )
    return _pattern_class(spec.text, BriefingEra.CURRENT)


RETIRED_CLASSES: tuple[BriefingClass, ...] = tuple(
    _pattern_class(template, BriefingEra.RETIRED)
    for template in (*RETIRED_LITERAL_TEMPLATES, RETIRED_TASK_TEMPLATE)
)

#: The free-text task query is fired right after the area-scoped
#: conventions query, by the same caller.
_COMPANION_ANCHORS: dict[str, str] = {TASK_SEMANTIC.slug: CONVENTIONS_AREA.slug}

CURRENT_CLASSES: tuple[BriefingClass, ...] = build_current_classes(
    QUERY_SPECS, _COMPANION_ANCHORS
)

BRIEFING_CLASSES: tuple[BriefingClass, ...] = RETIRED_CLASSES + CURRENT_CLASSES

#: The limit every briefing query fires at (``BriefingQuerySpec.limit`` in
#: ``shared/src/shared/briefing_queries.py``), not the E2 default of 10.
#: This is a fact about the BRIEFING ASSEMBLER only. It is never stamped onto
#: a row as an observation — see "THE LIMIT IS MEASURED, NOT ASSUMED" above.
#: It survives as the documented default scoring window and as the sidecar's
#: ``scored_limit``, which is labelled a choice and pinned to the source's
#: limit by a test, so drift there forces a decision here.
BRIEFING_SEARCH_LIMIT = 5

#: Histogram key used when a search op recorded no usable integer ``limit``.
UNSPECIFIED_LIMIT = 'unspecified'

DEFAULT_JOURNAL = Path('/home/leo/src/dark-factory/data/reconciliation/write_journal.db')

#: sqlite's busy timeout for the journal connection, in SECONDS.
#: NAMED after sqlite3's ``connect(timeout=)`` kwarg, which despite that
#: spelling is the BUSY timeout and NOTHING else: it bounds how long a
#: statement waits for a concurrent writer's lock, never how long opening
#: the file may take.  A harvest does not give up 30s after failing to open
#: the journal; `_connect_readonly`'s docstring states the same axis.
#: sqlite3's implicit default is 5.0; this raises it because the journal
#: is a live multi-gigabyte file a running fused-memory server is
#: appending to, so a writer holding a lock is the expected case, not an
#: exceptional one.  A read that waits is strictly better than a read
#: that refuses -- and, before this, `database is locked` was reported
#: as a missing table (see the scan in `harvest`).
JOURNAL_CONNECT_TIMEOUT = 30.0

DEFAULT_TAIL_SAMPLE = 40
DEFAULT_SEED = 4004


# --------------------------------------------------------------------------
# Result shapes
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TemplateClass:
    """One briefing-assembler query class and its measured traffic share."""

    text: str
    """The concrete query text this class contributes to the fixture.

    The most-frequently-observed instance, so a parameterized or companion
    class carries real production text rather than a formatting
    placeholder; for a literal that IS the template. A class nobody fired
    reports its template, and contributes no fixture row.
    """

    template: str
    """The template the class was matched by (== `text` for literals)."""

    match: MatchKind
    """``'literal'``, ``'parameterized'`` (by pattern) or ``'companion'``
    (by following its anchor class's op from the same caller)."""

    era: BriefingEra
    """``'retired'`` or ``'current'``."""

    observed_count: int
    """Search ops in this class, summed across every instance."""

    traffic_share: float | None
    """Share of all parseable search ops, or None when there is no traffic.

    None is *no measurement*, never a measured zero — the discipline is
    inherited verbatim from the bake-off artifact.
    """

    distinct_instances: int = 1
    """Distinct concrete query strings that matched this class."""

    observed_limits: dict[str, int] = field(default_factory=dict)
    """Measured ``{limit: count}`` over every op in this class."""

    observed_limit: int | None = None
    """The limit when every op in this class agreed, else None."""


@dataclass(frozen=True)
class HarvestResult:
    """Everything measured in one read-only pass over the journal."""

    templates: list[TemplateClass]
    rows: list[dict[str, Any]]
    total_search_ops: int
    unparsed_search_ops: int
    tail_count: int
    tail_distinct: int
    tail_share: float | None
    literal_share: float | None
    """Share of every LITERAL-matched briefing class, in either era."""

    family_share: float | None
    """Share of EVERY briefing class, in either era: the whole briefing family.

    Not the share of a parameterized class, which the name suggests. In the
    committed sidecar the union is 0.645536, while the retired
    `task {task_id} ...` class alone is 0.125334 (its
    `TemplateClass.traffic_share`).  The CLI prints the union as
    `briefing family`.  The field keeps its name because
    `read_transform_selection` publishes the same union, summed
    independently from the template shares, as `family_share`.
    """

    journal_path: str
    tail_sample: int
    tail_top: int
    seed: int
    briefing_observed_limits: dict[str, int] = field(default_factory=dict)
    """Measured ``{limit: count}`` across every briefing class."""

    tail_observed_limits: dict[str, int] = field(default_factory=dict)
    """Measured ``{limit: count}`` across the WHOLE residual tail.

    Not just the sampled rows: this is the population the sample is drawn
    from, and it is what shows a reader that the tail does not run at the
    briefing's k.
    """

    harvested_at: str = field(default='')

    def provenance(self) -> dict[str, Any]:
        """The sidecar block: every count a reader needs to re-derive a share."""
        return {
            'journal_path': self.journal_path,
            'harvested_at': self.harvested_at,
            'total_search_ops': self.total_search_ops,
            'unparsed_search_ops': self.unparsed_search_ops,
            'literal_share': self.literal_share,
            'family_share': self.family_share,
            'tail_share': self.tail_share,
            'tail_count': self.tail_count,
            'tail_distinct': self.tail_distinct,
            'tail_sample': self.tail_sample,
            'tail_top': self.tail_top,
            'seed': self.seed,
            # NOT an observation: the briefing's limit governs the briefing
            # queries only.  The measured distributions sit beside it so the
            # difference between the choice and the reading is legible.
            'scored_limit': BRIEFING_SEARCH_LIMIT,
            'scored_limit_is_a_choice': True,
            'scored_limit_basis': (
                'BriefingQuerySpec.limit (shared/src/shared/briefing_queries.py) '
                'fires every briefing-assembler query at limit=5. It governs '
                'nothing else. The residual tail is '
                'arbitrary other callers running at 3-50, so scoring the '
                'tail at 5 is a CHOICE made for comparability with the '
                'briefing half, not a limit observed on those queries. Per-'
                'row observed_limits carry what was actually recorded.'
            ),
            'briefing_observed_limits': self.briefing_observed_limits,
            'tail_observed_limits': self.tail_observed_limits,
            'templates': [
                {
                    'text': t.text,
                    'template': t.template,
                    'match': t.match,
                    'era': t.era,
                    'observed_count': t.observed_count,
                    'traffic_share': t.traffic_share,
                    'distinct_instances': t.distinct_instances,
                }
                for t in self.templates
            ],
            'unlabeled': True,
            'unlabeled_reason': (
                'Production queries carry no ground truth: the journal records '
                'what was asked, never what should have been returned. Rows '
                'therefore carry no expects_claim_ids, and labeled metrics '
                'render as no-measurement downstream.'
            ),
        }


# --------------------------------------------------------------------------
# Read-only journal access
# --------------------------------------------------------------------------
def _connect_readonly(db_path: Path | str) -> sqlite3.Connection:
    """Open `db_path` read-only. Belt (``mode=ro``) and braces (``query_only``).

    ``mode=ro`` refuses at the VFS layer; ``PRAGMA query_only`` refuses at
    the statement layer. Both are set because this points at a live
    multi-gigabyte journal the fused-memory server is writing to.

    ``JOURNAL_CONNECT_TIMEOUT`` governs the third axis: not WHETHER a write
    is possible but how long a read WAITS for a concurrent writer before
    giving up. Against a journal being actively appended to, lock contention
    is the expected case, so a read that waits beats a read that refuses.
    """
    path = Path(db_path)
    if not path.exists():
        raise JournalUnavailableError(f'write journal not found: {path}')
    try:
        con = sqlite3.connect(
            f'file:{path}?mode=ro', uri=True, timeout=JOURNAL_CONNECT_TIMEOUT
        )
    except sqlite3.Error as exc: # pragma: no cover - OS-level failure
        raise JournalUnavailableError(f'cannot open {path} read-only: {exc}') from exc
    con.execute('PRAGMA query_only=ON')
    return con


@dataclass(frozen=True)
class _SearchOp:
    """One parsed ``search`` op from the journal."""

    text: str
    limit: int | None
    caller: str | None
    at: datetime | None


def _journal_time(created_at: object) -> datetime | None:
    """A ``write_ops.created_at`` value as an aware datetime, or None if it is not one.

    The journal stamps ``datetime.now(UTC).isoformat()``; a value without a
    zone is unreadable rather than guessed to be UTC.
    """
    if not isinstance(created_at, str):
        return None
    try:
        at = datetime.fromisoformat(created_at)
    except ValueError:
        return None
    return at if at.tzinfo is not None else None


def _search_op(params: str | None, created_at: object) -> _SearchOp | None:
    """Parse a ``write_ops.params`` JSON blob, or None when it carries no query.

    The limit rides in the SAME already-parsed dict as the text, so reading
    it costs nothing and discarding it is what forced the old fabricated
    ``observed_limit``.  ``None`` for the limit means the op recorded no
    usable integer one — ``bool`` is excluded explicitly because
    ``isinstance(True, int)`` is ``True`` in Python.  ``None`` for the
    caller means the op recorded no ``caller_agent_id``, and ``None`` for
    ``at`` an unreadable ``created_at``; either way the op cannot be paired.
    """
    if not params:
        return None
    try:
        parsed = json.loads(params)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    text = parsed.get('query')
    if not isinstance(text, str) or not text.strip():
        return None
    raw_limit = parsed.get('limit')
    limit = (
        raw_limit
        if isinstance(raw_limit, int) and not isinstance(raw_limit, bool) and raw_limit > 0
        else None
    )
    raw_caller = parsed.get('caller_agent_id')
    caller = raw_caller if isinstance(raw_caller, str) and raw_caller else None
    return _SearchOp(text=text, limit=limit, caller=caller, at=_journal_time(created_at))


#: Longest a companion op may be journalled after its anchor op.
#:
#: A search is journalled when it completes, and the briefing fires its task
#: query as soon as the area query returns, so a real pair is one search
#: apart. An anchor older than this is an orphan: its companion was never
#: journalled, and pairing it would claim an unrelated later search.
COMPANION_WINDOW = timedelta(seconds=60)


class _Classifier:
    """Classifies search ops read in journal order, for ONE `harvest` call.

    A pattern class claims every op its pattern fullmatches. A companion
    class claims the next op no pattern claims from a caller with an
    unpaired anchor op at most `COMPANION_WINDOW` older. Unpaired anchors
    are QUEUED per caller, because parallel dispatches can share one caller
    id and interleave their pairs.
    """

    def __init__(self, classes: tuple[BriefingClass, ...]) -> None:
        self._patterns = [(c, c.pattern) for c in classes if c.pattern is not None]
        self._companions = {c.anchor: c for c in classes if c.anchor is not None}
        self._unpaired: defaultdict[tuple[str, str], deque[datetime]] = defaultdict(deque)

    def classify(self, op: _SearchOp) -> BriefingClass | None:
        """The class `op` belongs to, or None if it is tail."""
        for cls, pattern in self._patterns:
            if pattern.fullmatch(op.text):
                if (
                    op.caller is not None
                    and op.at is not None
                    and cls.template in self._companions
                ):
                    self._unpaired[(op.caller, cls.template)].append(op.at)
                return cls
        if op.caller is None or op.at is None:
            return None
        for anchor, companion in self._companions.items():
            if self._claim_anchor(op.caller, anchor, op.at):
                return companion
        return None

    def _claim_anchor(self, caller: str, anchor: str, at: datetime) -> bool:
        """Consume `caller`'s oldest `anchor` op still inside the window at `at`."""
        pending = self._unpaired[(caller, anchor)]
        while pending and at - pending[0] > COMPANION_WINDOW:
            pending.popleft()
        if not pending:
            return False
        pending.popleft()
        return True


def _limit_histogram(counter: Counter[int | None]) -> dict[str, int]:
    """Render a measured limit counter as JSON-safe ``{limit: count}``.

    Keys are strings because JSON object keys are; they sort numerically
    (with ``unspecified`` last) so a re-harvest diffs cleanly.
    """
    def sort_key(item: tuple[int | None, int]) -> tuple[int, int]:
        limit = item[0]
        return (1, 0) if limit is None else (0, limit)

    return {
        (UNSPECIFIED_LIMIT if limit is None else str(limit)): count
        for limit, count in sorted(counter.items(), key=sort_key)
    }


def _unanimous_limit(counter: Counter[int | None]) -> int | None:
    """The observed limit when EVERY instance agreed, else ``None``.

    ``None`` is "no single limit was observed", never a modal pick and
    never a default: picking one would re-introduce the fabrication this
    function exists to prevent.
    """
    if len(counter) != 1:
        return None
    (only,) = counter
    return only


def _share(count: int, total: int) -> float | None:
    """count/total, or None when there is no traffic to take a share of."""
    if total <= 0:
        return None
    return round(count / total, 6)


def _template_class(
    cls: BriefingClass,
    instances: Counter[str],
    limits: Counter[int | None],
    total: int,
) -> TemplateClass:
    """Measure one class from its per-text instance counts and its limits."""
    observed = sum(instances.values())
    # The fixture text is the most-frequent real instance, so a parameterized
    # class carries production text rather than a '{field}' placeholder.
    text = (
        min(instances.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        if instances
        else cls.template
    )
    return TemplateClass(
        text=text,
        template=cls.template,
        match=cls.match,
        era=cls.era,
        observed_count=observed,
        traffic_share=_share(observed, total),
        distinct_instances=len(instances),
        observed_limits=_limit_histogram(limits),
        observed_limit=_unanimous_limit(limits),
    )


class _ClassTally:
    """Search ops split into briefing classes and the residual tail.

    Lives for exactly one `harvest` call.
    """

    def __init__(self, classes: tuple[BriefingClass, ...]) -> None:
        self._classes = classes
        self._instances: dict[BriefingClass, Counter[str]] = {c: Counter() for c in classes}
        self._limits: dict[BriefingClass, Counter[int | None]] = {
            c: Counter() for c in classes
        }
        self._tail_instances: Counter[str] = Counter()
        self._tail_limits: dict[str, Counter[int | None]] = {}

    def add(self, cls: BriefingClass | None, text: str, limit: int | None) -> None:
        """Count one op of `text` at `limit` in `cls`, or in the tail for None."""
        if cls is None:
            self._tail_instances[text] += 1
            self._tail_limits.setdefault(text, Counter())[limit] += 1
        else:
            self._instances[cls][text] += 1
            self._limits[cls][limit] += 1

    def template_classes(self, total: int) -> list[TemplateClass]:
        return [
            _template_class(cls, self._instances[cls], self._limits[cls], total)
            for cls in self._classes
        ]

    def briefing_limits(self) -> Counter[int | None]:
        return sum(self._limits.values(), Counter())

    def tail_counts(self) -> Counter[str]:
        """Tail ops per distinct query text."""
        return Counter(self._tail_instances)

    def tail_limits_of(self, text: str) -> Counter[int | None]:
        """The limits the tail ops of `text` ran at."""
        return Counter(self._tail_limits.get(text, Counter()))

    def tail_limits(self) -> Counter[int | None]:
        return sum(self._tail_limits.values(), Counter())


def _briefing_row(tpl: TemplateClass) -> dict[str, Any]:
    return {
        'query_id': _query_id(tpl.text, 'briefing_template'),
        'text': tpl.text,
        'source': 'briefing_template',
        'template': tpl.template,
        'match': tpl.match,
        'era': tpl.era,
        'observed_count': tpl.observed_count,
        'observed_limit': tpl.observed_limit,
        'observed_limits': tpl.observed_limits,
        'traffic_share': tpl.traffic_share,
        'distinct_instances': tpl.distinct_instances,
    }


def _main_checkout_root() -> Path:
    """The MAIN checkout's root, even when this file runs from a lane.

    The journal is untracked runtime data that only exists under the main
    checkout's gitignored `/data/`, so a `.worktrees/<id>` lane is the wrong
    anchor for it.  Falls back to this file's own checkout
    (scripts/ -> fused-memory/ -> root) when git cannot answer: not a
    checkout, or no git.
    """
    here = Path(__file__).resolve()
    try:
        return Path(resolve_main_checkout(here.parent))
    except ValueError:
        return here.parents[2]


def _repo_relative(path: Path | str) -> str:
    """`data/reconciliation/write_journal.db`, not somebody's home dir.

    An artifact naming an absolute checkout is neither reproducible nor
    readable by anyone else -- the rule
    `fused-memory/scripts/bake_off_storage_shape.py::fixture_digests`
    already states.  Anchored on the MAIN checkout (`_main_checkout_root`);
    a path outside it stays RESOLVED-ABSOLUTE so a journal parked elsewhere
    remains identifiable.  Nothing reads `journal_path` back.

    Sibling copies stay private (these scripts load via
    `_fm_helpers.load_script_module`, not as a package) and differ on anchor
    and fallback, so copy deliberately rather than by proximity: see
    `fused-memory/scripts/census_memory_metadata.py::_repo_relative` and
    `fused-memory/scripts/bake_off_storage_shape.py::_repo_relative`.
    """
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(_main_checkout_root()))
    except ValueError:
        return str(resolved)


def _query_id(text: str, source: str) -> str:
    """A stable, content-derived id, so a re-harvest keeps its row ids."""
    import hashlib  # noqa: PLC0415

    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]
    prefix = 'brief' if source == 'briefing_template' else 'tail'
    return f'prod-{prefix}-{digest}'


def harvest(
    db_path: Path | str,
    *,
    tail_sample: int = DEFAULT_TAIL_SAMPLE,
    tail_top: int | None = None,
    seed: int = DEFAULT_SEED,
    pin_tail_texts: list[str] | None = None,
) -> HarvestResult:
    """Measure the production query distribution in one read-only pass.

    Every search op is classified in journal order into a briefing class
    (`BRIEFING_CLASSES`) or the residual tail. `templates` lists every class,
    a measured zero included; fixture rows are emitted for observed classes
    and for the tail sample.

    The tail sample is deterministic given (journal contents, tail_sample,
    tail_top, seed): a frequency-led head (sorted by -count then text, so
    ties break stably) plus a seeded random draw over the sorted remainder.
    The head is seed-independent by construction, so the highest-traffic
    tail queries are always present regardless of seed.
    """
    if tail_top is None:
        tail_top = max(1, tail_sample // 2)
    tail_top = min(tail_top, tail_sample)

    classifier = _Classifier(BRIEFING_CLASSES)
    tally = _ClassTally(BRIEFING_CLASSES)
    total = 0
    unparsed = 0

    con = _connect_readonly(db_path)
    try:
        try:
            # PROBE for the table before scanning, so a genuine schema fault
            # is DISTINGUISHED from every other sqlite failure rather than
            # inferred from one.  Previously any `sqlite3.Error` here was
            # relabelled `has no readable write_ops table`, so `database is
            # locked` -- the likeliest failure against a journal a running
            # server is writing to -- sent an operator to diagnose a missing
            # table that was right there.
            present = con.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='write_ops'"
            ).fetchone()
            if present is None:
                # No `: {exc}` tail -- nothing raised.  An empty probe is a
                # READING, not a failure, and `JournalUnavailableError` is
                # not a `sqlite3.Error`, so this passes through the handler
                # below untouched.
                raise JournalUnavailableError(f'{db_path} has no write_ops table')
            # STREAMED, not `.fetchall()`ed: the live journal is
            # multi-gigabyte and the committed sidecar records 431,621 search
            # ops, so materializing the `params` blobs is a multi-hundred-MB
            # peak allocation for a stream consumed exactly once.
            #
            # The TRADE, stated in full: peak RSS is bounded, at the cost of
            # holding the read snapshot open across `json.loads` of every
            # blob rather than for the fetch alone.  The journal is WAL
            # (`fused-memory/src/fused_memory/services/write_journal.py::WriteJournal`),
            # so writers are never blocked -- but frames newer than the
            # oldest open reader snapshot cannot be reclaimed, so the
            # server's `PRAGMA wal_checkpoint(TRUNCATE)`
            # (`write_journal.py::WriteJournal.checkpoint`) is starved and
            # the WAL grows for the whole scan.  Worth it against a
            # multi-hundred-MB allocation, and `fetchmany()` chunking is not
            # an improvement: it would cap memory identically without
            # releasing the snapshot any earlier.
            #
            # In journal (rowid) order, because a companion op is recognised
            # by its position and time after its anchor.
            for params, created_at in con.execute(
                "SELECT params, created_at FROM write_ops WHERE operation = 'search' "
                "ORDER BY rowid"
            ):
                op = _search_op(params, created_at)
                if op is None:
                    unparsed += 1
                    continue
                total += 1
                tally.add(classifier.classify(op), op.text, op.limit)
        # Deliberately wrapped around the WHOLE loop, not just the
        # `execute()`: streaming moves the point of failure, so a `disk I/O
        # error` can now surface on any `next()` mid-scan.  Narrowing this
        # guard would let a mid-scan sqlite failure escape unnamed.
        except sqlite3.Error as exc:
            # Names the REAL failure and the path, with `from exc` keeping
            # the original message and traceback.  The named type is kept
            # deliberately: `JournalUnavailableError` already covers
            # "absent, unreadable, or is not a write journal", and a locked
            # or I/O-failing journal IS unreadable.  Only the wording lied.
            raise JournalUnavailableError(
                f'cannot read write_ops from {db_path}: {exc}'
            ) from exc
    finally:
        con.close()

    templates = tally.template_classes(total)
    tail_counts = tally.tail_counts()

    literal_total = sum(
        t.observed_count for t in templates if t.match is MatchKind.LITERAL
    )
    briefing_total = sum(t.observed_count for t in templates)
    tail_total = sum(tail_counts.values())

    # --- deterministic tail sample -------------------------------------
    ordered_tail = sorted(tail_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if pin_tail_texts is not None:
        # The journal is APPENDED TO by a running server, so a re-harvest
        # draws a different tail than the committed one and every new query
        # is a miss in the committed fetch cache — i.e. re-measuring one
        # field would silently demand a paid re-seed.  Pinning holds the
        # sampled QUERY SET fixed while every count, share and limit is
        # freshly measured, so a correction stays replayable offline.
        pinned = set(pin_tail_texts)
        missing = sorted(pinned - set(tail_counts))
        if missing:
            raise EmptyHarvestError(
                f'{len(missing)} pinned tail query/queries are absent from '
                f'{db_path}: {missing[:3]}. A pin may only narrow a harvest '
                'to queries the journal still carries; emitting a pinned row '
                'with no observations would fabricate its counts.'
            )
        ordered_tail = [kv for kv in ordered_tail if kv[0] in pinned]
        tail_top = len(ordered_tail)
        tail_sample = len(ordered_tail)
    head = ordered_tail[:tail_top]
    remainder = ordered_tail[tail_top:]
    want = max(0, tail_sample - len(head))
    if want and remainder:
        rng = random.Random(seed)
        drawn = rng.sample(remainder, min(want, len(remainder)))
    else:
        drawn = []
    head_texts = {text for text, _ in head}
    sampled = sorted(
        [*head, *drawn], key=lambda kv: kv[0]
    ) # emitted rows are text-sorted so the fixture diffs cleanly

    # A class nobody fired has no production text to score; its measured
    # zero stays visible in `templates` and the sidecar.
    fixture_rows: list[dict[str, Any]] = [
        _briefing_row(tpl) for tpl in templates if tpl.observed_count > 0
    ]
    for text, n in sampled:
        seen_limits = tally.tail_limits_of(text)
        row = {
            'query_id': _query_id(text, 'production_tail'),
            'text': text,
            'source': 'production_tail',
            'observed_count': n,
            # The tail is arbitrary other callers, NOT the briefing
            # assembler: its limits are whatever the journal recorded.
            'observed_limit': _unanimous_limit(seen_limits),
            'observed_limits': _limit_histogram(seen_limits),
            'traffic_share': _share(n, total),
        }
        if text in head_texts:
            # Frequency-led head members are seed-independent; recording the
            # rank is what lets a reader verify that without re-running.
            row['tail_rank'] = [t for t, _ in head].index(text)
        fixture_rows.append(row)

    return HarvestResult(
        templates=templates,
        rows=fixture_rows,
        total_search_ops=total,
        unparsed_search_ops=unparsed,
        tail_count=tail_total,
        tail_distinct=len(tail_counts),
        tail_share=_share(tail_total, total),
        literal_share=_share(literal_total, total),
        family_share=_share(briefing_total, total),
        journal_path=_repo_relative(db_path),
        tail_sample=tail_sample,
        tail_top=tail_top,
        seed=seed,
        briefing_observed_limits=_limit_histogram(tally.briefing_limits()),
        tail_observed_limits=_limit_histogram(tally.tail_limits()),
        harvested_at=datetime.now(UTC).isoformat(),
    )


# --------------------------------------------------------------------------
# Fixture I/O
# --------------------------------------------------------------------------
def write_fixture(result: HarvestResult, out_path: Path | str) -> Path:
    """Write the JSONL rows plus a `.provenance.json` sidecar.

    Refuses an empty harvest: a zero-row fixture would read downstream as a
    measured absence of production traffic.
    """
    out = Path(out_path)
    if not result.rows or result.total_search_ops <= 0:
        raise EmptyHarvestError(
            f'{result.journal_path} yielded no parseable search traffic; '
            'refusing to write an empty fixture'
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    body = ''.join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n' for row in result.rows
    )
    out.write_text(body, encoding='utf-8')
    sidecar = out.with_suffix('.provenance.json')
    sidecar.write_text(
        json.dumps(result.provenance(), indent=2, sort_keys=True, ensure_ascii=False)
        + '\n',
        encoding='utf-8',
    )
    return out


def read_fixture(path: Path | str) -> list[dict[str, Any]]:
    """Read the committed JSONL fixture back into rows."""
    text = Path(path).read_text(encoding='utf-8')
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', default=str(DEFAULT_JOURNAL))
    parser.add_argument('--out', required=True)
    parser.add_argument('--tail-sample', type=int, default=DEFAULT_TAIL_SAMPLE)
    parser.add_argument('--tail-top', type=int, default=None)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument(
        '--pin-tail-to', default=None,
        help='An existing sample fixture whose tail queries this harvest is '
             'restricted to. Counts, shares and limits are still measured '
             'fresh; only WHICH queries are emitted is held fixed, so a '
             'correction stays replayable against the committed fetch cache.',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pinned: list[str] | None = None
    if args.pin_tail_to:
        pinned = [
            json.loads(line)['text']
            for line in Path(args.pin_tail_to).read_text(encoding='utf-8').splitlines()
            if line.strip() and json.loads(line).get('source') == 'production_tail'
        ]
    result = harvest(
        args.journal,
        tail_sample=args.tail_sample,
        tail_top=args.tail_top,
        seed=args.seed,
        pin_tail_texts=pinned,
    )
    out = write_fixture(result, args.out)
    print(f'wrote {len(result.rows)} rows -> {out}')
    print(f'  total search ops  : {result.total_search_ops}')
    print(f'  literal templates : {result.literal_share}')
    print(f'  briefing family   : {result.family_share}')
    print(f'  residual tail     : {result.tail_share} over {result.tail_distinct} distinct')
    return 0


if __name__ == '__main__': # pragma: no cover
    sys.exit(main())
