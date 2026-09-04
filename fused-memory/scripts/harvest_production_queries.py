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
The orchestrator's briefing assembler fires a fixed family of four queries
per dispatched task (``orchestrator/src/orchestrator/agents/briefing.py``):

  * three literals   — ``project overview architecture goals`` (:1266),
    ``coding conventions and project norms`` (:1273),
    ``recent decisions and rationale`` (:1280)
  * one PARAMETERIZED family — ``task {task_id} context and related
    decisions`` (:1288-1290), which must be matched as a TEMPLATE. Matching
    it literally would scatter one high-traffic class across thousands of
    singleton tail entries and understate it to nearly zero.

All four fire at ``limit=5`` (briefing.py:1376), not the E2 default of 10.

Everything that is not one of those four is the residual long tail, which
is sampled — frequency-led head plus a seeded random remainder — so the
committed fixture is small, representative and exactly regenerable.

THE LIMIT IS MEASURED, NOT ASSUMED
----------------------------------
``briefing.py``:1376 governs the four briefing queries and NOTHING else.
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

Even the briefing literals are not unanimous (each has one or two stray
instances at 10/20 out of ~75k at 5), so they too report ``None`` with a
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
    ./harvest_production_queries.py \
        --journal /home/leo/src/dark-factory/data/reconciliation/write_journal.db \
        --out fused-memory/tests/fixtures/production_query_sample.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# The briefing-assembler query family
# --------------------------------------------------------------------------
#: The three literal briefing queries, in the order briefing.py fires them.
LITERAL_TEMPLATES: tuple[str, ...] = (
    'project overview architecture goals',
    'coding conventions and project norms',
    'recent decisions and rationale',
)

#: The parameterized fourth query, kept in `str.format` shape so the
#: template string itself is what lands in the fixture and the report.
TASK_TEMPLATE = 'task {task_id} context and related decisions'

#: The pattern the parameterized family is matched by. The task id is any
#: run of non-space characters: ids in this repo are numeric ('4004') but
#: subtask ids carry a dot ('3.1'), so the class must not assume \d+.
TASK_TEMPLATE_RE = re.compile(r'^task (?P<task_id>\S+) context and related decisions$')

#: briefing.py:1376 fires the family at limit=5, not the E2 default of 10.
#: This is a fact about the BRIEFING ASSEMBLER only. It is never stamped onto
#: a row as an observation — see "THE LIMIT IS MEASURED, NOT ASSUMED" above.
#: It survives as the documented default scoring window and as the sidecar's
#: ``scored_limit``, which is labelled a choice.
BRIEFING_SEARCH_LIMIT = 5

#: Histogram key used when a search op recorded no usable integer ``limit``.
UNSPECIFIED_LIMIT = 'unspecified'

DEFAULT_JOURNAL = Path('/home/leo/src/dark-factory/data/reconciliation/write_journal.db')
DEFAULT_TAIL_SAMPLE = 40
DEFAULT_SEED = 4004


class HarvestError(RuntimeError):
    """Base class for every loud refusal in this module."""


class JournalUnavailableError(HarvestError):
    """The journal is absent, unreadable, or is not a write journal.

    Raised INSTEAD of returning an empty harvest, so a missing file can
    never be mistaken downstream for "production issues no searches".
    """


class EmptyHarvestError(HarvestError):
    """The journal is readable but carries no parseable search traffic."""


# --------------------------------------------------------------------------
# Result shapes
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TemplateClass:
    """One briefing-assembler query class and its measured traffic share."""

    text: str
    """The concrete query text this class contributes to the fixture.

    For a literal this IS the template. For the parameterized family it is
    the most-frequently-observed concrete instance, so the fixture carries
    real production text rather than a formatting placeholder.
    """

    template: str
    """The template the class was matched by (== `text` for literals)."""

    match: str
    """``'literal'`` or ``'parameterized'``."""

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
    family_share: float | None
    journal_path: str
    tail_sample: int
    tail_top: int
    seed: int
    briefing_observed_limits: dict[str, int] = field(default_factory=dict)
    """Measured ``{limit: count}`` across all four briefing classes."""

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
            # NOT an observation: briefing.py:1376 governs the four briefing
            # queries only.  The measured distributions sit beside it so the
            # difference between the choice and the reading is legible.
            'scored_limit': BRIEFING_SEARCH_LIMIT,
            'scored_limit_is_a_choice': True,
            'scored_limit_basis': (
                'briefing.py:1376 fires the four briefing-assembler queries '
                'at limit=5. It governs nothing else. The residual tail is '
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
    """
    path = Path(db_path)
    if not path.exists():
        raise JournalUnavailableError(f'write journal not found: {path}')
    try:
        con = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    except sqlite3.Error as exc: # pragma: no cover - OS-level failure
        raise JournalUnavailableError(f'cannot open {path} read-only: {exc}') from exc
    con.execute('PRAGMA query_only=ON')
    return con


def _classify(text: str) -> tuple[str, str] | None:
    """Return (template, match_kind) for `text`, or None if it is tail.

    The parameterized family is matched by PATTERN. Matching it literally
    would scatter one class across thousands of singletons.
    """
    if text in LITERAL_TEMPLATES:
        return text, 'literal'
    if TASK_TEMPLATE_RE.match(text):
        return TASK_TEMPLATE, 'parameterized'
    return None


def _query_op(params: str | None) -> tuple[str, int | None] | None:
    """Pull ``(query_text, limit)`` out of a ``write_ops.params`` JSON blob.

    The limit rides in the SAME already-parsed dict as the text, so reading
    it costs nothing and discarding it is what forced the old fabricated
    ``observed_limit``.  ``None`` for the limit means the op recorded no
    usable integer one — ``bool`` is excluded explicitly because
    ``isinstance(True, int)`` is ``True`` in Python.
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
    return text, limit


def _query_text(params: str | None) -> str | None:
    """The text half of :func:`_query_op`, kept as the narrow reader."""
    op = _query_op(params)
    return None if op is None else op[0]


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

    The tail sample is deterministic given (journal contents, tail_sample,
    tail_top, seed): a frequency-led head (sorted by -count then text, so
    ties break stably) plus a seeded random draw over the sorted remainder.
    The head is seed-independent by construction, so the highest-traffic
    tail queries are always present regardless of seed.
    """
    if tail_top is None:
        tail_top = max(1, tail_sample // 2)
    tail_top = min(tail_top, tail_sample)

    counts: Counter[str] = Counter()
    # The limit rides in the same params blob as the text, so it is
    # accumulated in the same pass.  Discarding it is what produced the
    # fabricated `observed_limit` this keying replaces.
    limits: dict[str, Counter[int | None]] = {}
    unparsed = 0

    con = _connect_readonly(db_path)
    try:
        try:
            # STREAMED, not `.fetchall()`ed: the live journal is
            # multi-gigabyte and the committed sidecar records 431,621 search
            # ops, so materializing the `params` blobs is a multi-hundred-MB
            # peak allocation for a stream consumed exactly once.
            for (params,) in con.execute(
                "SELECT params FROM write_ops WHERE operation = 'search'"
            ):
                op = _query_op(params)
                if op is None:
                    unparsed += 1
                    continue
                text, limit = op
                counts[text] += 1
                limits.setdefault(text, Counter())[limit] += 1
        # Deliberately wrapped around the WHOLE loop, not just the
        # `execute()`: streaming moves the point of failure, so a `disk I/O
        # error` can now surface on any `next()` mid-scan.  Narrowing this
        # guard would let a mid-scan sqlite failure escape unnamed.
        except sqlite3.Error as exc:
            raise JournalUnavailableError(
                f'{db_path} has no readable write_ops table: {exc}'
            ) from exc
    finally:
        con.close()

    total = sum(counts.values())

    # Fold every observed query into its class.
    literal_counts: dict[str, int] = {t: 0 for t in LITERAL_TEMPLATES}
    family_counts: Counter[str] = Counter()
    tail_counts: Counter[str] = Counter()
    literal_limits: dict[str, Counter[int | None]] = {
        t: Counter() for t in LITERAL_TEMPLATES
    }
    family_limits: Counter[int | None] = Counter()
    tail_limits: Counter[int | None] = Counter()
    for text, n in counts.items():
        classified = _classify(text)
        seen_limits = limits.get(text, Counter())
        if classified is None:
            tail_counts[text] += n
            tail_limits.update(seen_limits)
        elif classified[1] == 'literal':
            literal_counts[text] += n
            literal_limits[text].update(seen_limits)
        else:
            family_counts[text] += n
            family_limits.update(seen_limits)

    templates: list[TemplateClass] = [
        TemplateClass(
            text=text,
            template=text,
            match='literal',
            observed_count=literal_counts[text],
            traffic_share=_share(literal_counts[text], total),
            observed_limits=_limit_histogram(literal_limits[text]),
            observed_limit=_unanimous_limit(literal_limits[text]),
        )
        for text in LITERAL_TEMPLATES
    ]
    family_total = sum(family_counts.values())
    # The family's fixture text is its most-frequent real instance, so the
    # fixture carries production text rather than a '{task_id}' placeholder.
    family_text = (
        min(family_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        if family_counts
        else TASK_TEMPLATE.format(task_id='0000')
    )
    templates.append(
        TemplateClass(
            text=family_text,
            template=TASK_TEMPLATE,
            match='parameterized',
            observed_count=family_total,
            traffic_share=_share(family_total, total),
            distinct_instances=len(family_counts),
            observed_limits=_limit_histogram(family_limits),
            observed_limit=_unanimous_limit(family_limits),
        )
    )

    literal_total = sum(literal_counts.values())
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

    fixture_rows: list[dict[str, Any]] = []
    for tpl in templates:
        fixture_rows.append(
            {
                'query_id': _query_id(tpl.text, 'briefing_template'),
                'text': tpl.text,
                'source': 'briefing_template',
                'template': tpl.template,
                'match': tpl.match,
                'observed_count': tpl.observed_count,
                'observed_limit': tpl.observed_limit,
                'observed_limits': tpl.observed_limits,
                'traffic_share': tpl.traffic_share,
                'distinct_instances': tpl.distinct_instances,
            }
        )
    for text, n in sampled:
        seen_limits = limits.get(text, Counter())
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
        family_share=_share(literal_total + family_total, total),
        journal_path=str(db_path),
        tail_sample=tail_sample,
        tail_top=tail_top,
        seed=seed,
        briefing_observed_limits=_limit_histogram(
            sum(literal_limits.values(), Counter()) + family_limits
        ),
        tail_observed_limits=_limit_histogram(tail_limits),
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
    print(f'  total search ops : {result.total_search_ops}')
    print(f'  3 literals       : {result.literal_share}')
    print(f'  4-template family: {result.family_share}')
    print(f'  residual tail    : {result.tail_share} over {result.tail_distinct} distinct')
    return 0


if __name__ == '__main__': # pragma: no cover
    sys.exit(main())
