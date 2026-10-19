#!/usr/bin/env python3
"""Migrate the open consolidation gates through the auto predicate (PRD θ, task 5244).

``plans/memory-auto-consolidation-prd.md`` D13 / §11 step 4. Every open
consolidation gate in a project's ``tasks.db`` is a topic cluster a human was
asked to fold. Many of them are clusters the auto predicate
(``fused_memory/reconciliation/consolidation_auto.py::evaluate_auto_predicate``)
would now pass. This script re-reads each gate's cluster live, rules on it,
and under ``--execute`` writes the consolidation through the executor's own
retain arm
(``fused_memory/reconciliation/consolidation_auto_executor.py::apply_auto_verdict``).
Gates the predicate fails are left to the curator sitting.

It never changes a task's status. The gate closes when its own runner resumes
it, and the report lists the calls that do that.

Usage
-----
  # Report what would happen, writing nothing.
  python scripts/migrate_open_consolidation_gates.py --dry-run

  # Mint a canonical where a topic has none: the claim comes from the sitting.
  python scripts/migrate_open_consolidation_gates.py --execute \\
      --claim some-topic='One assertion line the canonical leads with'

  # Another project, machine-readable.
  python scripts/migrate_open_consolidation_gates.py --dry-run \\
      --project-root /path/to/reify --json

Exit codes
----------
  0  the run completed.
  2  the run could not start: bad arguments, an unreadable tasks.db, or the
     canonical-slug list was unavailable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import secrets
import sqlite3
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import _checkout as _checkout
import check_consolidation_closure as closure_cli
from _task_db_scan import TaskDbUnreadable, connect_ro, decode_metadata, tasks_db_path
from fused_memory.reconciliation.consolidation_auto import (
    AutoOutcome,
    AutoProposal,
    AutoVerdict,
    evaluate_auto_predicate,
)
from fused_memory.reconciliation.consolidation_auto_executor import (
    ClusterFacts,
    RefusalCode,
    read_canonical_slugs,
    read_cluster_facts,
)
from fused_memory.reconciliation.consolidation_gate import (
    GATE_METADATA_KEY,
    declared_gate_topic,
    observed_members,
)
from fused_memory.reconciliation.curator_gate_resolution_sweep import (
    extract_open_gate_task_ids,
)
from fused_memory.server.consolidation import (
    ProposalLimits,
    ProposalShapeCode,
    proposal_shape_codes,
)
from fused_memory.utils.validation import is_full_uuid
from shared.project_id import project_id_for_root
from shared.project_layout import CheckoutRootError, default_project_root
from shared.task_statuses import ACTIVE

if TYPE_CHECKING:
    from fused_memory.config.schema import ConsolidationAutoConfig

logger = logging.getLogger('migrate_open_consolidation_gates')

_RATIONALE = 'Migrated from consolidation gate {task_id} by scripts/migrate_open_consolidation_gates.py'


class ExitCode(IntEnum):
    COMPLETED = 0
    COULD_NOT_RUN = 2


class Mode(StrEnum):
    DRY_RUN = 'dry_run'
    EXECUTE = 'execute'


class Disposition(StrEnum):
    """What this run did, or would do, with one gate."""

    REFUSED = 'refused'
    LEFT_TO_SITTING = 'left_to_sitting'
    NOOP = 'noop'
    WOULD_EXECUTE = 'would_execute'
    DEFERRED = 'deferred'
    EXECUTED = 'executed'


class MigrationCode(StrEnum):
    """Refusals only this script can name; the predicate, validator and executor name the rest."""

    GATE_TOPIC_UNUSABLE = 'gate_topic_unusable'
    MEMBER_SCROLL_UNAVAILABLE = 'member_scroll_unavailable'
    MEMBER_SCROLL_INCOMPLETE = 'member_scroll_incomplete'
    CATEGORY_UNDETERMINED = 'category_undetermined'
    CLAIM_REQUIRED = 'claim_required'
    CYCLE_CAP_REACHED = 'cycle_cap_reached'


class UsageError(Exception):
    """Anything that stops the run before the first gate: reported as COULD_NOT_RUN."""


@dataclass(frozen=True)
class Reason:
    code: str
    ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class OpenGate:
    task_id: str
    metadata: Mapping[str, Any]

    @property
    def block(self) -> Mapping[str, Any]:
        return self.metadata[GATE_METADATA_KEY]


@dataclass(frozen=True)
class GateReport:
    """One gate's outcome. ``reasons`` decided the disposition; ``verdict_reasons`` are the predicate's."""

    task_id: str
    topic: str | None
    disposition: Disposition
    reasons: tuple[Reason, ...] = ()
    verdict: AutoOutcome | None = None
    verdict_reasons: tuple[Reason, ...] = ()
    members: tuple[str, ...] = ()
    absent_observed: tuple[str, ...] = ()
    category: str | None = None

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(reason.code for reason in self.reasons)


@dataclass(frozen=True)
class MigrationReport:
    project_id: str
    mode: Mode
    run_id: str
    gates: tuple[GateReport, ...]


@dataclass(frozen=True)
class RunContext:
    memory: Any
    config: ConsolidationAutoConfig
    project_id: str
    run_id: str
    mode: Mode
    claims: Mapping[str, str]
    canonical_slugs: tuple[str, ...]


@dataclass(frozen=True)
class Assessment:
    """A gate's live cluster read and the predicate's ruling on it."""

    gate: OpenGate
    topic: str
    members: tuple[str, ...]
    absent_observed: tuple[str, ...]
    facts: ClusterFacts
    category: str | None
    verdict: AutoVerdict

    def report(self, disposition: Disposition, reasons: Sequence[Reason] = ()) -> GateReport:
        return GateReport(
            task_id=self.gate.task_id,
            topic=self.topic,
            disposition=disposition,
            reasons=tuple(reasons),
            verdict=self.verdict.outcome,
            verdict_reasons=_verdict_reasons(self.verdict),
            members=self.members,
            absent_observed=self.absent_observed,
            category=self.category,
        )


# --------------------------------------------------------------------------- #
# Enumeration
# --------------------------------------------------------------------------- #


def open_consolidation_gates(project_root: str, tag: str) -> tuple[OpenGate, ...]:
    """Every ACTIVE gate task carrying a consolidation-gate block, in task-id order."""
    active = [
        {'id': task_id, 'metadata': decode_metadata(raw)}
        for task_id, status, raw in _task_rows(project_root, tag)
        if status in ACTIVE
    ]
    gate_ids = set(extract_open_gate_task_ids(active))
    gates = [
        OpenGate(task_id=str(task['id']), metadata=task['metadata'])
        for task in active
        if str(task['id']) in gate_ids and isinstance(task['metadata'].get(GATE_METADATA_KEY), Mapping)
    ]
    return tuple(sorted(gates, key=lambda gate: _id_order(gate.task_id)))


def _task_rows(project_root: str, tag: str) -> list[tuple[Any, str, Any]]:
    db_path = tasks_db_path(project_root)
    try:
        con = connect_ro(db_path)
    except TaskDbUnreadable as refusal:
        raise UsageError(str(refusal)) from refusal
    except sqlite3.Error as exc:
        raise UsageError(f'could not open {db_path} read-only: {exc}') from exc
    try:
        return con.execute(
            'SELECT id, status, metadata FROM tasks WHERE tag = ?', (tag,),
        ).fetchall()
    except sqlite3.Error as exc:
        raise UsageError(f'could not read the tasks in {db_path}: {exc}') from exc
    finally:
        con.close()


def _id_order(task_id: str) -> tuple[int, str]:
    return (int(task_id), '') if task_id.isdigit() else (sys.maxsize, task_id)


# --------------------------------------------------------------------------- #
# Per-gate assessment
# --------------------------------------------------------------------------- #


async def migrate(ctx: RunContext, gates: Iterable[OpenGate]) -> tuple[GateReport, ...]:
    return tuple([await migrate_gate(ctx, gate) for gate in gates])


async def migrate_gate(ctx: RunContext, gate: OpenGate) -> GateReport:
    topic = declared_gate_topic(gate.metadata)
    if topic is None:
        return GateReport(
            task_id=gate.task_id,
            topic=None,
            disposition=Disposition.REFUSED,
            reasons=(Reason(MigrationCode.GATE_TOPIC_UNUSABLE),),
        )
    assessment = await assess(ctx, gate, topic)
    return _dispose(ctx, assessment)


async def assess(ctx: RunContext, gate: OpenGate, topic: str) -> Assessment:
    scrolled = await ctx.memory.get_memories_by_metadata(
        ctx.project_id, {'topic': topic}, limit=closure_cli.SCROLL_LIMIT,
    )
    scroll_ids = [str(row['id']) for row in scrolled if isinstance(row, Mapping) and row.get('id')]
    observed_only = _observed_only(gate.block, scroll_ids)
    facts = await read_cluster_facts(
        ctx.memory,
        project_id=ctx.project_id,
        topic=topic,
        member_ids=[*scroll_ids, *observed_only],
    )
    absent = tuple(i for i in observed_only if facts.members.get(i) is None)
    live = ClusterFacts(
        members={i: r for i, r in facts.members.items() if i not in absent},
        canonical_count=facts.canonical_count,
    )
    members = tuple(live.members)
    category = _gate_category(gate.block, topic, live.members)
    verdict = evaluate_auto_predicate(
        AutoProposal(topic=topic, member_ids=members, category=category or ''),
        members=live.members,
        canonical_count=live.canonical_count,
        open_gate_id=None,
        existing_canonical_slugs=ctx.canonical_slugs,
        config=ctx.config,
    )
    return Assessment(
        gate=gate,
        topic=topic,
        members=members,
        absent_observed=absent,
        facts=live,
        category=category,
        verdict=verdict,
    )


def _observed_only(block: Mapping[str, Any], scroll_ids: Sequence[str]) -> list[str]:
    """The gate's full-uuid observed members missing from the live scroll, deduped case-folded."""
    seen = {i.lower() for i in scroll_ids}
    extra: list[str] = []
    for raw in observed_members(block):
        if not is_full_uuid(raw) or str(raw).lower() in seen:
            continue
        seen.add(str(raw).lower())
        extra.append(str(raw))
    return extra


def _gate_category(
    block: Mapping[str, Any], topic: str, members: Mapping[str, Any],
) -> str | None:
    """The block's recorded proposal category, else the one category the readable peers share."""
    refusal = block.get('auto_refusal')
    if isinstance(refusal, Mapping) and isinstance(refusal.get('category'), str):
        return refusal['category']
    shared = {
        record['metadata'].get('category')
        for record in members.values()
        if isinstance(record, Mapping)
        and isinstance(record.get('metadata'), Mapping)
        and isinstance(record['metadata'].get('category'), str)
        and not _is_incumbent(record['metadata'], topic)
    }
    return shared.pop() if len(shared) == 1 else None


def _is_incumbent(metadata: Mapping[str, Any], topic: str) -> bool:
    return metadata.get('canonical') is True and metadata.get('topic') == topic


def _verdict_reasons(verdict: AutoVerdict) -> tuple[Reason, ...]:
    return tuple(Reason(r.code.value, tuple(r.ids)) for r in verdict.reasons)


def _dispose(ctx: RunContext, assessment: Assessment) -> GateReport:
    outcome = assessment.verdict.outcome
    if outcome is AutoOutcome.FAIL:
        return assessment.report(Disposition.LEFT_TO_SITTING, _verdict_reasons(assessment.verdict))
    if outcome is AutoOutcome.NOOP:
        return assessment.report(Disposition.NOOP, _verdict_reasons(assessment.verdict))
    refusals = _actionable_refusals(ctx, assessment)
    if refusals:
        return assessment.report(Disposition.REFUSED, refusals)
    return assessment.report(Disposition.WOULD_EXECUTE)


def _actionable_refusals(ctx: RunContext, assessment: Assessment) -> tuple[Reason, ...]:
    """Why a PASS / PASS_TAG_ONLY may still not be written, in the executor's terms."""
    category = assessment.category
    if category is None:
        return (Reason(MigrationCode.CATEGORY_UNDETERMINED),)
    if category not in ctx.config.category_weights:
        return (Reason(RefusalCode.CATEGORY_NOT_CONSOLIDATABLE),)
    claim = ctx.claims.get(assessment.topic)
    shape = proposal_shape_codes(
        topic=assessment.topic,
        member_ids=list(assessment.members),
        run_id=ctx.run_id,
        claim=claim,
        category=category,
        rationale=_RATIONALE.format(task_id=assessment.gate.task_id),
        limits=ProposalLimits.from_config(ctx.config),
    )
    reasons = [
        Reason(code)
        for code in shape
        if claim is not None or code is not ProposalShapeCode.CLAIM_NOT_INDEX_SHAPED
    ]
    if assessment.verdict.outcome is AutoOutcome.PASS and claim is None:
        reasons.append(Reason(MigrationCode.CLAIM_REQUIRED))
    return tuple(reasons)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def report_payload(report: MigrationReport) -> dict[str, Any]:
    return {
        'project_id': report.project_id,
        'mode': report.mode.value,
        'run_id': report.run_id,
        'gate_count': len(report.gates),
        'gates': [_gate_payload(gate) for gate in report.gates],
    }


def _gate_payload(gate: GateReport) -> dict[str, Any]:
    return {
        'task_id': gate.task_id,
        'topic': gate.topic,
        'disposition': gate.disposition.value,
        'codes': list(gate.codes),
        'reasons': [_reason_payload(r) for r in gate.reasons],
        'verdict': gate.verdict.value if gate.verdict is not None else None,
        'verdict_reasons': [_reason_payload(r) for r in gate.verdict_reasons],
        'members': list(gate.members),
        'absent_observed': list(gate.absent_observed),
        'category': gate.category,
    }


def _reason_payload(reason: Reason) -> dict[str, Any]:
    return {'code': reason.code, 'ids': list(reason.ids)}


def render_text(report: MigrationReport) -> str:
    lines = [
        f'project_id: {report.project_id}  mode: {report.mode.value}  '
        f'run_id: {report.run_id}  gates: {len(report.gates)}',
    ]
    for gate in report.gates:
        lines.extend(['', *_gate_lines(gate)])
    return '\n'.join(lines)


def _gate_lines(gate: GateReport) -> list[str]:
    verdict = gate.verdict.value if gate.verdict is not None else '-'
    lines = [
        f'gate {gate.task_id}  topic: {gate.topic or "-"}  '
        f'disposition: {gate.disposition.value}  verdict: {verdict}',
        f'  members ({len(gate.members)}): {", ".join(gate.members) or "-"}',
    ]
    if gate.absent_observed:
        lines.append(f'  absent observed: {", ".join(gate.absent_observed)}')
    lines.extend(f'  [{r.code}] {", ".join(r.ids) or "-"}' for r in gate.reasons)
    return lines


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Rule on every open consolidation gate with the auto predicate and, '
            'under --execute, consolidate the ones it passes.'
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true', help='report only; write nothing')
    mode.add_argument('--execute', action='store_true', help='write the passing consolidations')
    parser.add_argument(
        '--project-root', default=None, help='project root owning tasks.db (default: the main checkout)',
    )
    parser.add_argument(
        '--project-id', default=None, help='memory project_id (default: derived from --project-root)',
    )
    parser.add_argument('--tag', default='master', help='taskmaster tag (default: %(default)s)')
    parser.add_argument(
        '--claim',
        action='append',
        default=[],
        metavar='TOPIC=TEXT',
        help='the one-line claim a minted canonical leads with, per topic (repeatable)',
    )
    parser.add_argument('--json', action='store_true', dest='as_json', help='emit the report as JSON')
    return parser


def parse_claims(values: Sequence[str]) -> dict[str, str]:
    claims: dict[str, str] = {}
    for value in values:
        topic, _, text = value.partition('=')
        claims[topic.strip()] = text.strip()
    return claims


def _project_root(args: argparse.Namespace) -> str:
    if args.project_root is not None:
        return str(Path(args.project_root).resolve())
    try:
        return str(default_project_root())
    except CheckoutRootError as exc:
        raise UsageError(f'no --project-root given and {exc}') from exc


def _project_id(args: argparse.Namespace, project_root: str) -> str:
    if args.project_id is not None:
        return args.project_id
    try:
        return project_id_for_root(project_root)
    except ValueError as exc:
        raise UsageError(f'{exc}, or pass --project-id') from exc


def new_run_id(now: datetime) -> str:
    return f'migrate-gates-{now:%Y%m%dT%H%M%SZ}-{secrets.token_hex(4)}'


async def run(args: argparse.Namespace, *, memory: Any = None) -> int:
    mode = Mode.EXECUTE if args.execute else Mode.DRY_RUN
    try:
        claims = parse_claims(args.claim)
        project_root = _project_root(args)
        project_id = _project_id(args, project_root)
        gates = open_consolidation_gates(project_root, args.tag)
    except UsageError as exc:
        logger.error('could not run: %s', exc)
        return ExitCode.COULD_NOT_RUN
    if memory is None:
        memory = closure_cli.build_memory_service()
    slugs = await read_canonical_slugs(memory, project_id)
    if slugs is None:
        logger.error(
            'could not run: the canonical-slug list for %s is unavailable or may be partial',
            project_id,
        )
        return ExitCode.COULD_NOT_RUN
    ctx = RunContext(
        memory=memory,
        config=memory.config.consolidation_auto,
        project_id=project_id,
        run_id=new_run_id(datetime.now(UTC)),
        mode=mode,
        claims=claims,
        canonical_slugs=slugs,
    )
    report = MigrationReport(
        project_id=project_id, mode=mode, run_id=ctx.run_id, gates=await migrate(ctx, gates),
    )
    print(json.dumps(report_payload(report), indent=2) if args.as_json else render_text(report))
    return ExitCode.COMPLETED


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='%(levelname)s %(name)s: %(message)s')
    return asyncio.run(run(build_parser().parse_args(argv)))


if __name__ == '__main__':
    sys.exit(main())
