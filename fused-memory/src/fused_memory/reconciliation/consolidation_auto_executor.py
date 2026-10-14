"""The auto-consolidation executor post-step (``plans/memory-auto-consolidation-prd.md`` C4).

:func:`run_auto_consolidation` runs once per full recon Stage-1 cycle, after the
LLM turn has emitted its ``propose_consolidation`` rows. It ranks the rows under
the shared flood policy (``consolidation_flood_control.py``), reads each selected
cluster deterministically, rules with the pure predicate
(``consolidation_auto.py``), and executes, gates or records the verdict on the
proposal row (``consolidation_proposal.py``). Every stat it reports has one home,
:func:`auto_consolidation_initial_stats`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from fused_memory.models.reconciliation import StageId, StageReport
from fused_memory.reconciliation.consolidation_auto import (
    UNREADABLE,
    AutoOutcome,
    AutoProposal,
    AutoVerdict,
    build_auto_canonical,
    evaluate_auto_predicate,
)
from fused_memory.reconciliation.consolidation_flood_control import (
    FloodCandidate,
    rank_and_cap,
)
from fused_memory.reconciliation.consolidation_gate import (
    GATE_METADATA_KEY,
    ClosureVerdict,
    build_consolidation_gate_task,
    closure_exists_probe,
    declared_gate_topic,
    evaluate_closure,
    resolve_unstamped_live_ids,
)
from fused_memory.reconciliation.consolidation_proposal import (
    RECORD_KIND_CONSOLIDATION_PROPOSAL,
    ConsolidationProposal,
    ProposalOutcome,
    ProposalState,
    from_ledger_record,
    newest_per_topic,
    record_outcome,
)
from fused_memory.reconciliation.curator_gate_resolution_sweep import (
    extract_open_gate_task_ids,
)
from fused_memory.reconciliation.recon_ledger import ReconLedgerRecord
from fused_memory.services.consolidation_ops import (
    RetainArmApplied,
    RetainArmRefused,
    apply_retain_arm,
    read_topic_closure,
)

if TYPE_CHECKING:
    from fused_memory.config.schema import ConsolidationAutoConfig

logger = logging.getLogger(__name__)

#: Page size of the one canonical-slug scroll per cycle. A full page means the
#: slug list may be partial, so the cycle fails closed rather than let
#: ``slug_near_collision`` miss a twin topic.
CANONICAL_SLUG_LIMIT = 1000

_EXECUTOR_AGENT_ID = f'recon-stage-{StageId.memory_consolidator.value}'
_SOURCE = 'auto_consolidation'
_RUN_STAMP_KEY = 'x_auto_consolidated_run'
_CANONICAL_STAMP_KEY = 'x_auto_consolidated'
_SYSTEM_RECORD_KIND = 'auto_consolidation'
_SYSTEM_RECORD_FACTS_KEY = 'x_auto_consolidation'


def auto_consolidation_initial_stats() -> dict[str, Any]:
    """Every stat the executor reports, at its initial value."""
    return {
        'auto_consolidations': 0,
        'auto_consolidations_tag_only': 0,
        'auto_consolidations_dry_run': 0,
        'auto_consolidations_noop': 0,
        'auto_consolidations_skipped_project': 0,
        'auto_consolidation_skipped_unavailable': None,
        'auto_consolidation_refusals': {},
        'auto_consolidation_gates_filed': 0,
        'auto_consolidation_applied_cap': 0,
        'auto_consolidation_unexecuted': 0,
        'auto_consolidation_unfiled': 0,
        'auto_consolidation_backlog_tripped': False,
        'auto_consolidation_proposals_listed': 0,
        'auto_consolidation_proposals_expired': 0,
        'auto_consolidation_proposals_superseded': 0,
        'auto_consolidation_closure_failures': 0,
        'auto_consolidation_pointers': [],
    }


@dataclass
class _Cycle:
    """One cycle's inputs, plus the budgets and stats its walk consumes."""

    memory_service: Any
    taskmaster: Any
    escalation_queue: Any
    ledger: Any
    project_id: str
    project_root: str
    run_id: str
    config: ConsolidationAutoConfig
    now: datetime
    stats: dict[str, Any]
    open_gates: dict[str, str] = field(default_factory=dict)
    canonical_slugs: tuple[str, ...] = ()
    executions: int = 0

    def outcome(self, state: ProposalState, **fields: Any) -> ProposalOutcome:
        return ProposalOutcome(
            state=state, acted_run_id=self.run_id, acted_at=self.now.isoformat(), **fields,
        )


@dataclass(frozen=True)
class _Facts:
    members: dict[str, Any]
    canonical_count: int | None


async def run_auto_consolidation(
    memory_service: Any,
    taskmaster: Any,
    escalation_queue: Any,
    *,
    project_id: str,
    project_root: str,
    run_id: str,
    config: ConsolidationAutoConfig,
    active_tasks: list[dict[str, Any]] | None,
    report: StageReport,
    now: datetime | None = None,
) -> None:
    """Run one cycle's auto-consolidation; the full stat set lands in ``report.stats``.

    Fails closed: a project outside ``config.enabled_projects`` is skipped, and a
    missing ledger, task tree or canonical-slug list skips the cycle naming the
    missing input. With ``config.enabled`` false every would-be write is
    recorded as ``dry_run`` instead.
    """
    stats = auto_consolidation_initial_stats()
    try:
        await _run(
            memory_service,
            taskmaster,
            escalation_queue,
            project_id=project_id,
            project_root=project_root,
            run_id=run_id,
            config=config,
            active_tasks=active_tasks,
            stats=stats,
            now=now or datetime.now(UTC),
        )
    finally:
        report.stats.update(stats)


async def _run(
    memory_service: Any,
    taskmaster: Any,
    escalation_queue: Any,
    *,
    project_id: str,
    project_root: str,
    run_id: str,
    config: ConsolidationAutoConfig,
    active_tasks: list[dict[str, Any]] | None,
    stats: dict[str, Any],
    now: datetime,
) -> None:
    if project_id not in config.enabled_projects:
        stats['auto_consolidations_skipped_project'] = 1
        return
    ledger = getattr(memory_service, 'recon_ledger', None)
    if ledger is None:
        stats['auto_consolidation_skipped_unavailable'] = 'ledger'
        return
    if active_tasks is None:
        stats['auto_consolidation_skipped_unavailable'] = 'task_tree'
        return
    slugs = await _canonical_slugs(memory_service, project_id)
    if slugs is None:
        stats['auto_consolidation_skipped_unavailable'] = 'canonical_slugs'
        return

    cycle = _Cycle(
        memory_service=memory_service,
        taskmaster=taskmaster,
        escalation_queue=escalation_queue,
        ledger=ledger,
        project_id=project_id,
        project_root=project_root,
        run_id=run_id,
        config=config,
        now=now,
        stats=stats,
        open_gates=_open_gate_topics(active_tasks),
        canonical_slugs=slugs,
    )
    proposals = await _live_proposals(cycle)
    cap = config.max_auto_per_cycle + config.max_gate_filings_per_cycle
    plan = rank_and_cap(
        [
            FloodCandidate(
                key=topic,
                size=len(proposal.member_ids),
                category=proposal.category,
                canonical_absent=topic not in cycle.canonical_slugs,
            )
            for topic, (_, proposal) in proposals.items()
        ],
        cap=cap,
        config=config,
    )
    stats['auto_consolidation_applied_cap'] = plan.cap
    stats['auto_consolidation_unexecuted'] += len(plan.deferred)
    for candidate in plan.selected:
        record, proposal = proposals[candidate.key]
        try:
            await _evaluate(cycle, record, proposal)
        except Exception:
            logger.exception(
                'auto_consolidation: proposal for topic %r failed; row left as it was',
                proposal.topic,
                extra={'project_id': project_id, 'run_id': run_id, 'topic': proposal.topic},
            )


async def _canonical_slugs(memory_service: Any, project_id: str) -> tuple[str, ...] | None:
    """Every canonical's topic slug, or ``None`` when the list may be partial."""
    try:
        rows = await memory_service.get_memories_by_metadata(
            project_id=project_id, filters={'canonical': True}, limit=CANONICAL_SLUG_LIMIT,
        )
    except Exception:
        logger.warning(
            'auto_consolidation: the canonical-slug scroll failed; skipping the cycle',
            exc_info=True,
            extra={'project_id': project_id},
        )
        return None
    if not isinstance(rows, list) or len(rows) >= CANONICAL_SLUG_LIMIT:
        return None
    slugs = {_metadata(row).get('topic') for row in rows}
    return tuple(sorted(s for s in slugs if isinstance(s, str)))


def _open_gate_topics(active_tasks: list[dict[str, Any]]) -> dict[str, str]:
    """``{topic: gate task id}`` for every open consolidation gate."""
    gate_ids = set(extract_open_gate_task_ids(active_tasks))
    topics: dict[str, str] = {}
    for task in active_tasks:
        if not isinstance(task, dict) or str(task.get('id')) not in gate_ids:
            continue
        topic = declared_gate_topic(task.get('metadata'))
        if topic is not None:
            topics.setdefault(topic, str(task['id']))
    return topics


async def _live_proposals(
    cycle: _Cycle,
) -> dict[str, tuple[ReconLedgerRecord, ConsolidationProposal]]:
    """The newest unexpired decodable proposal per topic, keyed by topic.

    Expired rows are counted and left for ``ReconLedgerStore.gc``; older rows
    of a topic are moved to ``superseded``.
    """
    rows = await cycle.ledger.list_by_kind(
        cycle.project_id, RECORD_KIND_CONSOLIDATION_PROPOSAL, ProposalState.PROPOSED.value,
    )
    cycle.stats['auto_consolidation_proposals_listed'] = len(rows)
    live = [row for row in rows if not _expired(row, cycle.now)]
    cycle.stats['auto_consolidation_proposals_expired'] = len(rows) - len(live)
    keep, superseded = newest_per_topic(live)
    for row in superseded:
        if await record_outcome(cycle.ledger, row, cycle.outcome(ProposalState.SUPERSEDED)):
            cycle.stats['auto_consolidation_proposals_superseded'] += 1
    decoded: dict[str, tuple[ReconLedgerRecord, ConsolidationProposal]] = {}
    for row in keep:
        try:
            decoded[row.flag_type] = (row, from_ledger_record(row))
        except ValueError:
            logger.warning(
                'auto_consolidation: undecodable proposal row for topic %r',
                row.flag_type,
                exc_info=True,
                extra={'project_id': cycle.project_id, 'run_id': row.run_id},
            )
    return decoded


def _expired(row: ReconLedgerRecord, now: datetime) -> bool:
    if row.expires_at is None:
        return False
    try:
        return datetime.fromisoformat(row.expires_at) <= now
    except ValueError:
        return True


async def _evaluate(
    cycle: _Cycle, record: ReconLedgerRecord, proposal: ConsolidationProposal,
) -> None:
    facts = await _read_facts(cycle, proposal)
    verdict = evaluate_auto_predicate(
        AutoProposal(
            topic=proposal.topic,
            member_ids=proposal.member_ids,
            category=proposal.category,
        ),
        members=facts.members,
        canonical_count=facts.canonical_count,
        open_gate_id=cycle.open_gates.get(proposal.topic),
        existing_canonical_slugs=cycle.canonical_slugs,
        config=cycle.config,
    )
    if verdict.outcome is AutoOutcome.NOOP:
        await _record_noop(cycle, record, verdict)
    elif verdict.outcome in (AutoOutcome.PASS, AutoOutcome.PASS_TAG_ONLY):
        await _execute(cycle, record, proposal, verdict, facts)


async def _read_facts(cycle: _Cycle, proposal: ConsolidationProposal) -> _Facts:
    members: dict[str, Any] = {}
    for member_id in proposal.member_ids:
        try:
            members[member_id] = await cycle.memory_service.get_memory_by_id(
                project_id=cycle.project_id, memory_id=member_id,
            )
        except Exception:
            members[member_id] = UNREADABLE
    try:
        canonical_count = await cycle.memory_service.count_memories_by_metadata(
            project_id=cycle.project_id,
            filters={'topic': proposal.topic, 'canonical': True},
        )
    except Exception:
        canonical_count = None
    return _Facts(members=members, canonical_count=canonical_count)


def _verdict_fields(verdict: AutoVerdict) -> dict[str, Any]:
    return {
        'verdict': verdict.outcome,
        'reason_codes': tuple(reason.code.value for reason in verdict.reasons),
        'predicate_version': verdict.predicate_version,
        'stripped_ids': verdict.stripped_ids,
    }


async def _record_noop(cycle: _Cycle, record: ReconLedgerRecord, verdict: AutoVerdict) -> None:
    gate_ids = [
        reason.ids[0]
        for reason in verdict.reasons
        if reason.code.value == 'already_gated' and reason.ids
    ]
    state = ProposalState.ADDRESSED if cycle.config.enabled else ProposalState.DRY_RUN
    moved = await record_outcome(
        cycle.ledger,
        record,
        cycle.outcome(
            state,
            gate_task_id=gate_ids[0] if gate_ids else None,
            **_verdict_fields(verdict),
        ),
    )
    if moved:
        cycle.stats['auto_consolidations_noop'] += 1
        if state is ProposalState.DRY_RUN:
            cycle.stats['auto_consolidations_dry_run'] += 1


async def _execute(
    cycle: _Cycle,
    record: ReconLedgerRecord,
    proposal: ConsolidationProposal,
    verdict: AutoVerdict,
    facts: _Facts,
) -> None:
    if cycle.executions >= cycle.config.max_auto_per_cycle:
        cycle.stats['auto_consolidation_unexecuted'] += 1
        return
    cycle.executions += 1
    if not cycle.config.enabled:
        if await record_outcome(
            cycle.ledger,
            record,
            cycle.outcome(ProposalState.DRY_RUN, **_verdict_fields(verdict)),
        ):
            cycle.stats['auto_consolidations_dry_run'] += 1
        return

    arm = await _apply_arm(cycle, proposal, verdict, facts)
    if isinstance(arm, RetainArmRefused):
        logger.warning(
            'auto_consolidation: the retain arm refused topic %r: %s',
            proposal.topic,
            arm.response.get('error_type'),
            extra={'project_id': cycle.project_id, 'run_id': cycle.run_id},
        )
        return
    closure = await _closure(cycle, proposal)
    minted = verdict.outcome is AutoOutcome.PASS
    await _write_system_record(cycle, proposal, arm)
    fields = {
        **_verdict_fields(verdict),
        'canonical_id': arm.canonical_id,
        'canonical_minted': minted,
        'tagged_ids': tuple(arm.retained),
        'tag_failure_ids': tuple(str(f.get('id')) for f in arm.retain_failures),
        'closure_codes': tuple(str(r.get('code')) for r in closure.reasons),
    }
    if not closure.closed:
        cycle.stats['auto_consolidation_closure_failures'] += 1
        await record_outcome(cycle.ledger, record, cycle.outcome(ProposalState.GATED, **fields))
        return
    await record_outcome(cycle.ledger, record, cycle.outcome(ProposalState.ADDRESSED, **fields))
    cycle.stats['auto_consolidations' if minted else 'auto_consolidations_tag_only'] += 1
    cycle.stats['auto_consolidation_pointers'].append(
        {'topic': proposal.topic, 'canonical_id': arm.canonical_id, 'run_id': cycle.run_id},
    )


async def _apply_arm(
    cycle: _Cycle, proposal: ConsolidationProposal, verdict: AutoVerdict, facts: _Facts,
) -> RetainArmApplied | RetainArmRefused:
    """Mint (PASS) or resolve (PASS_TAG_ONLY) the canonical; tag only unstamped members."""
    to_tag = [
        member_id
        for member_id in verdict.retain_ids
        if _metadata(facts.members.get(member_id)).get('topic') != proposal.topic
    ]
    mint = verdict.outcome is AutoOutcome.PASS
    return await apply_retain_arm(
        cycle.memory_service,
        project_id=cycle.project_id,
        topic=proposal.topic,
        canonical_content=(
            build_auto_canonical(
                proposal.claim, proposal.topic, len(verdict.retain_ids), cycle.run_id,
            )
            if mint
            else None
        ),
        retain_ids=to_tag,
        category=proposal.category,
        agent_id=_EXECUTOR_AGENT_ID,
        run_id=cycle.run_id,
        extra_canonical_meta=(
            {
                _CANONICAL_STAMP_KEY: {
                    'run_id': cycle.run_id,
                    'predicate_version': verdict.predicate_version,
                    'members_count': len(verdict.retain_ids),
                    'at': cycle.now.isoformat(),
                },
            }
            if mint
            else None
        ),
        extra_peer_meta={_RUN_STAMP_KEY: cycle.run_id},
        source=_SOURCE,
    )


async def _closure(cycle: _Cycle, proposal: ConsolidationProposal) -> ClosureVerdict:
    """Task 4808's closure check over the topic after this run's writes."""
    closure = await read_topic_closure(
        cycle.memory_service,
        project_id=cycle.project_id,
        topic=proposal.topic,
        run_id=cycle.run_id,
    )
    gate_block = build_consolidation_gate_task(
        topic=proposal.topic, observed_members=proposal.member_ids,
    ).metadata[GATE_METADATA_KEY]
    try:
        live = await resolve_unstamped_live_ids(
            gate_block,
            members=closure.members,
            exists=closure_exists_probe(cycle.memory_service),
            project_id=cycle.project_id,
        )
    except Exception:
        logger.warning(
            'auto_consolidation: the unstamped-member probe failed for topic %r',
            proposal.topic,
            exc_info=True,
            extra={'project_id': cycle.project_id, 'run_id': cycle.run_id},
        )
        return evaluate_closure(
            gate_block,
            members=closure.members,
            scroll_total=closure.total,
            scroll_truncated=closure.truncated,
            scroll_available=False,
        )
    return evaluate_closure(
        gate_block,
        members=closure.members,
        scroll_total=closure.total,
        scroll_truncated=closure.truncated,
        scroll_available=closure.available,
        unstamped_live_ids=live,
    )


async def _write_system_record(
    cycle: _Cycle, proposal: ConsolidationProposal, arm: RetainArmApplied,
) -> None:
    """The run's pointer record. Its facts are nested so it never joins the topic."""
    await cycle.memory_service.add_system_record(
        content=(
            f'Auto-consolidation run {cycle.run_id} consolidated topic '
            f'{proposal.topic} under canonical {arm.canonical_id}.'
        ),
        project_id=cycle.project_id,
        agent_id=_EXECUTOR_AGENT_ID,
        category='observations_and_summaries',
        metadata={
            'record_kind': _SYSTEM_RECORD_KIND,
            _SYSTEM_RECORD_FACTS_KEY: {
                'topic': proposal.topic,
                'canonical_id': arm.canonical_id,
                'run_id': cycle.run_id,
                'member_ids': list(proposal.member_ids),
                'tagged_ids': list(arm.retained),
            },
        },
        _source=_SOURCE,
    )


def _metadata(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    metadata = record.get('metadata')
    return metadata if isinstance(metadata, dict) else {}
