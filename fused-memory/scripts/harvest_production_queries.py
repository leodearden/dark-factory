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

All four fire at ``limit=5`` (briefing.py:1376), not the E2 default of 10,
so the production arm is scored at k=5.

Everything that is not one of those four is the residual long tail, which
is sampled — frequency-led head plus a seeded random remainder — so the
committed fixture is small, representative and exactly regenerable.

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
BRIEFING_SEARCH_LIMIT = 5

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
            'search_limit': BRIEFING_SEARCH_LIMIT,
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


def _query_text(params: str | None) -> str | None:
    """Pull the query text out of a ``write_ops.params`` JSON blob."""
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
    return text


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

    con = _connect_readonly(db_path)
    try:
        try:
            rows = con.execute(
                "SELECT params FROM write_ops WHERE operation = 'search'"
            ).fetchall()
        except sqlite3.Error as exc:
            raise JournalUnavailableError(
                f'{db_path} has no readable write_ops table: {exc}'
            ) from exc
    finally:
        con.close()

    counts: Counter[str] = Counter()
    unparsed = 0
    for (params,) in rows:
        text = _query_text(params)
        if text is None:
            unparsed += 1
            continue
        counts[text] += 1

    total = sum(counts.values())

    # Fold every observed query into its class.
    literal_counts: dict[str, int] = {t: 0 for t in LITERAL_TEMPLATES}
    family_counts: Counter[str] = Counter()
    tail_counts: Counter[str] = Counter()
    for text, n in counts.items():
        classified = _classify(text)
        if classified is None:
            tail_counts[text] += n
        elif classified[1] == 'literal':
            literal_counts[text] += n
        else:
            family_counts[text] += n

    templates: list[TemplateClass] = [
        TemplateClass(
            text=text,
            template=text,
            match='literal',
            observed_count=literal_counts[text],
            traffic_share=_share(literal_counts[text], total),
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
        )
    )

    literal_total = sum(literal_counts.values())
    tail_total = sum(tail_counts.values())

    # --- deterministic tail sample -------------------------------------
    ordered_tail = sorted(tail_counts.items(), key=lambda kv: (-kv[1], kv[0]))
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
                'observed_limit': BRIEFING_SEARCH_LIMIT,
                'traffic_share': tpl.traffic_share,
                'distinct_instances': tpl.distinct_instances,
            }
        )
    for text, n in sampled:
        row = {
            'query_id': _query_id(text, 'production_tail'),
            'text': text,
            'source': 'production_tail',
            'observed_count': n,
            'observed_limit': BRIEFING_SEARCH_LIMIT,
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = harvest(
        args.journal,
        tail_sample=args.tail_sample,
        tail_top=args.tail_top,
        seed=args.seed,
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
