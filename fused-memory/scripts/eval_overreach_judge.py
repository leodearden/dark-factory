#!/usr/bin/env python3
"""Score the ruling-overreach judge against adjudicated verdicts (task 6137).

design.md §6.1's trigger, from shadow to enforce, needs the judge's
population-weighted precision and recall on hand-adjudicated edges. This
script computes both, and the trigger's verdict, in one of two modes:

* ``offline`` runs the live judge (services/overreach_guard.py::judge_edges)
  over the worksheet's episodes, one call per episode;
* ``online`` reads the labels the shadow already recorded in its sidecar
  (services/overreach_shadow.py) for a fresh frozen window.

The worksheet that scripts/audit_ruling_overreach.py --emit-worksheet writes
is the join: verdicts.json carries neither episode bodies nor strata, and
report.json's population weights are per (graph, stratum) cell. Verdict
validation, weighting and Wilson bounds are the audit's own, imported and never
re-derived. An edge the judge did not label counts against recall.

Read-only: it writes only --out.

Usage (from fused-memory/), offline, against the design's adjudicated window::

    uv run python scripts/eval_overreach_judge.py --mode offline \\
        --worksheet /tmp/ruling-overreach-worksheet-4716.jsonl \\
        --verdicts ../docs/ruling-overreach-guard-design-2026-10-01/verdicts.json \\
        --report ../docs/ruling-overreach-guard-design-2026-10-01/report.json \\
        --out /tmp/overreach-judge-eval.json

Online, per design.md §6.1 (2): a NEW frozen window, drawn after the shadow
was deployed with ``audit_ruling_overreach.py --window-start/--window-end
--emit-worksheet``, adjudicated under §2.6, with the weights from that
window's own ``report.json`` (the audit run with ``--verdicts``). The sidecar
is opened read-only and no LLM is consulted::

    uv run python scripts/eval_overreach_judge.py --mode online \\
        --sidecar <queue.data_dir>/overreach_shadow.db \\
        --worksheet /tmp/ruling-overreach-worksheet-online.jsonl \\
        --verdicts ../docs/<new-dated-dir>/verdicts.json \\
        --report ../docs/<new-dated-dir>/report.json \\
        --out /tmp/overreach-judge-eval-online.json

Exit codes: 0 ran, whether or not the trigger passes (``trigger.passes`` in
the JSON); 1 the judge or the sidecar is unavailable; 2 an input is invalid
(nothing is written).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fused_memory.backends.graphiti_client import build_llm_client
from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.services.overreach_guard import (
    OVERREACH,
    PROMPT_VERSION,
    JudgeEdge,
    StructuredLLM,
    judge_edges,
)
from fused_memory.services.overreach_shadow import OverreachShadowStore

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_ruling_overreach import (
    SampleDefinition,
    VerdictError,
    VerdictSet,
    load_verdicts,
    rate,
    weighted_rate,
)

logger = logging.getLogger('eval_overreach_judge')

EXIT_OK, EXIT_UNAVAILABLE, EXIT_BAD_INPUTS = 0, 1, 2

PRECISION_MIN = 0.80
"""design.md §6.1 trigger: weighted precision at least this."""
PRECISION_LOWER_MIN = 0.65
"""design.md §6.1 trigger: and its Wilson lower bound at least this."""
RECALL_MIN = 0.50
"""design.md §6.1 trigger (1), offline only: weighted recall at least this."""

EdgeKey = tuple[str, str]
"""``(graph, edge_uuid)``."""

Row = Mapping[str, Any]
"""A worksheet row, as audit_ruling_overreach.py::worksheet_rows yields it."""

Weights = Mapping[str, Mapping[str, float]]
"""graph -> stratum -> population weight."""

OUTCOMES = ('tp', 'fp', 'fn', 'tn')


class EvalInputError(ValueError):
    """An input the score must not be built from."""


@dataclass(frozen=True)
class Judged:
    """One mode's labels and rationales by edge, and the judges and prompts behind them."""

    labels: Mapping[EdgeKey, str]
    rationales: Mapping[EdgeKey, str]
    judge_models: Sequence[str | None]
    prompt_versions: Sequence[str]
    failures: Sequence[Mapping[str, Any]]


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


def load_worksheet(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_weights(report_obj: Any) -> dict[str, dict[str, float]]:
    try:
        weights = report_obj['adjudicated']['population_weighted']['weights']
        return {
            graph: {stratum: float(w) for stratum, w in by_stratum.items()}
            for graph, by_stratum in weights.items()
        }
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise EvalInputError(
            f'the report carries no adjudicated.population_weighted.weights: {exc!r}'
        ) from exc


def check_cells(rows: Iterable[Row], weights: Weights) -> None:
    weighted = {(graph, stratum) for graph, by_stratum in weights.items() for stratum in by_stratum}
    if missing := sorted({(row['graph'], row['stratum']) for row in rows} - weighted):
        raise EvalInputError(f'no population weight for worksheet cell(s) {missing}')


def _minted_keys(rows: Iterable[Row]) -> list[EdgeKey]:
    return [(row['graph'], edge['edge_uuid']) for row in rows for edge in row['minted']]


def validate(rows: Sequence[Row], verdicts_obj: Any) -> VerdictSet:
    """The audit's fail-closed verdict validation, against the worksheet's sample."""
    expected = {
        (row['graph'], edge['edge_uuid']): row['uuid'] for row in rows for edge in row['minted']
    }
    try:
        definition = SampleDefinition.from_dict(verdicts_obj['sample'])
    except (KeyError, TypeError, ValueError) as exc:
        raise EvalInputError(f'the verdict file carries no valid sample block: {exc!r}') from exc
    return load_verdicts(
        verdicts_obj, expected_edges=expected, existing_edges=expected.keys(),
        sampled_episodes=[(row['graph'], row['uuid']) for row in rows], definition=definition,
    )


# --------------------------------------------------------------------------- #
# Score and trigger
# --------------------------------------------------------------------------- #


def _outcome(verdict: str, judge: str | None) -> str:
    truth, flagged = verdict == OVERREACH, judge == OVERREACH
    if truth:
        return 'tp' if flagged else 'fn'
    return 'fp' if flagged else 'tn'


def _edge_outcomes(
    rows: Iterable[Row], verdict_set: VerdictSet,
    judged: Mapping[EdgeKey, str], rationales: Mapping[EdgeKey, str],
) -> list[dict[str, Any]]:
    verdicts = {verdict.key: verdict for verdict in verdict_set.verdicts}
    records = []
    for row in rows:
        for edge in row['minted']:
            key = (row['graph'], edge['edge_uuid'])
            verdict, judge = verdicts[key], judged.get(key)
            records.append({
                'graph': row['graph'], 'stratum': row['stratum'], 'episode_uuid': row['uuid'],
                'edge_uuid': edge['edge_uuid'], 'verdict': verdict.label, 'judge': judge,
                'outcome': _outcome(verdict.label, judge),
                'verdict_rationale': verdict.rationale, 'judge_rationale': rationales.get(key),
            })
    return records


def _confusion_by_cell(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, int]]]:
    cells: defaultdict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    for record in records:
        cell = cells[record['graph']].setdefault(record['stratum'], dict.fromkeys(OUTCOMES, 0))
        cell[record['outcome']] += 1
    return dict(cells)


def score(
    rows: Sequence[Row], verdict_set: VerdictSet, judged: Mapping[EdgeKey, str],
    weights: Weights, *, rationales: Mapping[EdgeKey, str] | None = None,
) -> dict[str, Any]:
    """Population-weighted precision and recall of *judged* against *verdict_set*."""
    records = _edge_outcomes(rows, verdict_set, judged, rationales or {})
    cells = _confusion_by_cell(records)
    weighted = [(weights[g][s], c) for g, by_stratum in cells.items() for s, c in by_stratum.items()]
    totals = Counter(record['outcome'] for record in records)
    return {
        'precision': weighted_rate([(w, c['tp'], c['tp'] + c['fp']) for w, c in weighted]),
        'recall': weighted_rate([(w, c['tp'], c['tp'] + c['fn']) for w, c in weighted]),
        'unweighted': {
            'precision': rate(totals['tp'], totals['tp'] + totals['fp']),
            'recall': rate(totals['tp'], totals['tp'] + totals['fn']),
        },
        'confusion_by_cell': cells,
        'coverage': {'offered': len(records), 'judged': sum(r['judge'] is not None for r in records)},
        'unjudged': [
            {k: r[k] for k in ('graph', 'episode_uuid', 'edge_uuid', 'verdict')}
            for r in records if r['judge'] is None
        ],
        'disagreements': [r for r in records if r['outcome'] in ('fp', 'fn')],
    }


def _at_least(value: float | None, floor: float) -> bool:
    return value is not None and value >= floor


def trigger(score: Mapping[str, Any], mode: str) -> dict[str, Any]:
    """design.md §6.1: precision and its lower bound in both modes; recall offline only."""
    precision, recall = score['precision'], score['recall']
    checks = {
        'precision': _at_least(precision['rate'], PRECISION_MIN),
        'precision_lower_bound': _at_least((precision['ci'] or [None])[0], PRECISION_LOWER_MIN),
    }
    if mode == 'offline':
        checks['recall'] = _at_least(recall['rate'], RECALL_MIN)
    return {'mode': mode, 'checks': checks, 'passes': all(checks.values())}


# --------------------------------------------------------------------------- #
# The live judge (offline mode)
# --------------------------------------------------------------------------- #


def _offered_edges(row: Row) -> tuple[JudgeEdge, ...]:
    return tuple(
        JudgeEdge(uuid=e['edge_uuid'], fact=e['fact'],
                  source_name=e['source_name'], target_name=e['target_name'])
        for e in row['minted']
    )


async def judge_live(rows: Iterable[Row], llm: StructuredLLM) -> Judged:
    """One judge call per row, in order; a failed row is recorded and left unjudged."""
    labels: dict[EdgeKey, str] = {}
    rationales: dict[EdgeKey, str] = {}
    failures: list[dict[str, Any]] = []
    for row in rows:
        try:
            judgements = await judge_edges(llm, row['content'] or '', _offered_edges(row))
        except Exception as exc:
            logger.warning('judge failed on %s/%s: %r', row['graph'], row['uuid'], exc)
            failures.append({
                'graph': row['graph'], 'episode_uuid': row['uuid'],
                'error_type': type(exc).__name__, 'error': str(exc),
            })
            continue
        for judgement in judgements:
            key = (row['graph'], judgement.edge_uuid)
            labels[key], rationales[key] = judgement.label, judgement.rationale
    return Judged(
        labels=labels, rationales=rationales, judge_models=[getattr(llm, 'model', None)],
        prompt_versions=[PROMPT_VERSION], failures=failures,
    )


# --------------------------------------------------------------------------- #
# The shadow's recorded labels (online mode)
# --------------------------------------------------------------------------- #


async def judge_from_sidecar(rows: Iterable[Row], db_path: str | Path) -> Judged:
    """The labels the shadow recorded for the worksheet's edges; an unrecorded edge is absent."""
    store = await OverreachShadowStore.open_read_only(db_path)
    try:
        stored = await store.judgements_for(_minted_keys(rows))
    finally:
        await store.close()
    return Judged(
        labels={key: judgement.label for key, judgement in stored.items()},
        rationales={key: judgement.rationale for key, judgement in stored.items()},
        judge_models=sorted({judgement.judge_model for judgement in stored.values()}),
        prompt_versions=sorted({judgement.prompt_version for judgement in stored.values()}),
        failures=[],
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Score the ruling-overreach judge against adjudicated verdicts.',
    )
    parser.add_argument('--mode', choices=('offline', 'online'), required=True)
    parser.add_argument('--worksheet', required=True, help='audit --emit-worksheet JSONL')
    parser.add_argument('--verdicts', required=True, help='adjudicated verdicts.json')
    parser.add_argument('--report', required=True, help='the audit report.json with weights')
    parser.add_argument('--out', required=True, help='where the score JSON is written')
    parser.add_argument('--sidecar', help='online only: the shadow sidecar overreach_shadow.db')
    return parser


def _check_mode_flags(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.mode == 'online' and not args.sidecar:
        parser.error('--mode online needs --sidecar')
    if args.mode == 'offline' and args.sidecar:
        parser.error('--sidecar is for --mode online only')


def _load_inputs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], VerdictSet, Weights]:
    rows = load_worksheet(args.worksheet)
    verdict_set = validate(rows, json.loads(Path(args.verdicts).read_text()))
    weights = load_weights(json.loads(Path(args.report).read_text()))
    check_cells(rows, weights)
    return rows, verdict_set, weights


def _write(path: str, result: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')


def _judge(args: argparse.Namespace, rows: Sequence[Row], llm: StructuredLLM | None) -> Judged | None:
    """The mode's labels, or None (logged) when its judge or sidecar is unavailable."""
    if args.mode == 'online':
        try:
            return asyncio.run(judge_from_sidecar(rows, args.sidecar))
        except (sqlite3.DatabaseError, FileNotFoundError) as exc:
            logger.error('the sidecar is unreadable (%s): %s', type(exc).__name__, exc)
            return None
    judge = llm if llm is not None else build_llm_client(FusedMemoryConfig())
    if judge is None:
        logger.error('no judge LLM is configured: the llm provider block or its api_key is missing')
        return None
    return asyncio.run(judge_live(rows, judge))


def run(argv: Sequence[str] | None = None, *, llm: StructuredLLM | None = None) -> int:
    """Parse *argv* (default ``sys.argv[1:]``) and run; *llm* replaces the configured offline judge."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    _check_mode_flags(parser, args)
    try:
        rows, verdict_set, weights = _load_inputs(args)
    except (VerdictError, EvalInputError, OSError, json.JSONDecodeError) as exc:
        logger.error('inputs rejected (%s): %s; nothing is written', type(exc).__name__, exc)
        return EXIT_BAD_INPUTS
    judged = _judge(args, rows, llm)
    if judged is None:
        return EXIT_UNAVAILABLE
    scored = score(rows, verdict_set, judged.labels, weights, rationales=judged.rationales)
    _write(args.out, {
        **scored, 'mode': args.mode, 'judge_models': judged.judge_models,
        'prompt_versions': judged.prompt_versions, 'failures': judged.failures,
        'trigger': trigger(scored, args.mode),
    })
    return EXIT_OK


def main() -> int:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    return run()


if __name__ == '__main__':
    sys.exit(main())
